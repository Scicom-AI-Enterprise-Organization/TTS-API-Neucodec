"""
LLM-based text normalizer: sends text to any OpenAI-compatible /chat/completions
endpoint (OPENAI_BASE_URL) with the few-shot prompt in app/prompt.py, constrained
to `{"normalized": "..."}` via `response_format` json_schema so the reply is always
just the normalized text, ready to feed the TTS LM.

This module must stay importable without the GPU stack (torch/NeuCodec) so its
unit tests run anywhere; only aiohttp + app.env + app.prompt are allowed here.
"""

import asyncio
import json
import re
import logging
from enum import Enum

import aiohttp

from app.env import OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL_NAME, OPENAI_TIMEOUT
from app.prompt import SYSTEM_PROMPT, EXAMPLES, JSON_SCHEMA


class NormalizerMode(str, Enum):
    rule = 'rule'        # legacy malaya-derived pipeline (app/normalizer)
    llm = 'llm'          # OpenAI-compatible LLM with the prompt in app/prompt.py
    spoken = 'spoken'    # rule-based replica of the LLM (app/spoken_normalizer), no network


class LLMNormalizerError(Exception):
    """LLM endpoint unreachable, non-200, or returned an unusable reply."""


_CODE_FENCE = re.compile(r'^```(?:json)?\s*(.*?)\s*```$', re.DOTALL)


def build_messages(text):
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    for before, after in EXAMPLES:
        messages.append({'role': 'user', 'content': before})
        messages.append({
            'role': 'assistant',
            'content': json.dumps({'normalized': after}, ensure_ascii=False),
        })
    messages.append({'role': 'user', 'content': text})
    return messages


def parse_normalized(content):
    """Extract the normalized text from a model reply. Guided decoding guarantees
    clean JSON, but stay defensive for backends that ignore response_format:
    unwrap code fences, accept a bare JSON string, else use the raw text."""
    content = (content or '').strip()
    fenced = _CODE_FENCE.match(content)
    if fenced:
        content = fenced.group(1).strip()
    try:
        out = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return content or None
    if isinstance(out, dict) and isinstance(out.get('normalized'), str):
        return out['normalized'].strip() or None
    if isinstance(out, str):
        return out.strip() or None
    return None


# ---------------------------------------------------------------------------
# "Nothing to normalize" fast path
# ---------------------------------------------------------------------------
# One LLM round trip is ~0.55 s of TTFB (gemma-4-31b behind the serverless proxy, measured
# from tm-h20 on 2026-09-04: mean 0.56 s, max 0.67 s over 21 sentences), and on plain
# conversational text it hands the input back unchanged (19/19 plain evalset sentences
# came back byte-identical). Everything SYSTEM_PROMPT asks the model to rewrite -- numbers,
# money, IC/phone numbers, dates, times, percentages, decimals, ordinals, units, emails,
# URLs -- plus what it expands in practice beyond that ("Dr." -> "Doctor", acronyms) leaves
# a lexical trace: a digit, a symbol, a dotted or ALL-CAPS token, or a known abbreviation.
# Text with none of those skips the call and gets the same pre/post cleanup the LLM path
# applies, so the output is identical to a no-op LLM reply. Conservative on purpose: any
# doubt still goes to the LLM (a wasted call costs latency, a wrong skip costs correctness).
# main.py gates this behind LLM_NORMALIZER_SKIP_PLAIN; bench/normalizer_gate_eval.py
# checks the assumption against the live LLM.

_PLAIN_PUNCT = ".,!?;:'’‘\"“”()\\-–—…、。，！？；：「」『』（）"
# Tamil vowel signs and virama (U+0BBE..U+0BCD) are combining marks, which Python's \w does
# not match -- without this every Tamil sentence looked like it contained symbols and went
# to the LLM regardless of the gate. Whole Tamil block treated as letters.
_TAMIL = '஀-௿'
_NEEDS_LLM = re.compile(
    r'\d'                                   # any digit (Unicode-aware)
    rf'|[^\w\s{_TAMIL}{_PLAIN_PUNCT}]'      # any symbol beyond plain punctuation: % $ @ & / + = # * ...
    r'|_'                                   # \w admits underscore (markdown, identifiers)
    r'|\w\.\w'                              # dotted tokens: site.com, e.g., U.S.
    r'|\b[A-Z]{2,}\b'                       # acronyms / all caps: IC, TNB, OTP, RM
    # abbreviations that are rarely ordinary words: match bare
    r'|(?i:\b(?:dr|mr|mrs|ms|prof|sdn|bhd|jln|tmn|hj|etc|eg|ie|vs|approx|dept|govt|tel|fax|'
    r'ext|acct|amt|mth|yr|wk|sept|km|cm|mm|ml|kg|mg|gb|mb|kb|tb|hz|khz|mhz|ghz|mph|kph|kmh|'
    r'lbs|oz|sq)\b)'
    # ... and ones that are ordinary words too ("no", "am", "sat"): only when written with a period
    r'|(?i:\b(?:no|st|rd|co|inc|ltd|corp|min|max|sec|hrs?|mins?|mon|tues?|wed|thur?s?|fri|sat|'
    r'sun|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec|am|pm|en|pn|tn|cik|ir|ref|acc|bal|ave|'
    r'blvd|ft)\.)'
)


def needs_normalization(text):
    """True if `text` may contain something the LLM normalizer would rewrite.

    False means the text is plain words and punctuation only, for which the LLM returns
    its input unchanged -- so the caller can skip the call. See the note above.
    """
    return bool(_NEEDS_LLM.search(text or ''))


_UNSPOKEN = re.compile(r'\d|[^\w\s' + _TAMIL + re.escape(_PLAIN_PUNCT) + r']|\w\.\w')


def has_unspoken(text):
    """Narrower than needs_normalization(): only what is unspeakable as written (a digit, a
    symbol, a dotted token). Acronyms and abbreviations do not count -- the TTS LM reads those
    and the LLM leaves most of them alone. Used by LLM_NORMALIZER_RULE_FIRST to decide whether
    the spoken normalizer left anything for the LLM to do."""
    return bool(_UNSPOKEN.search(text or ''))


async def llm_normalize(text, base_url=None, api_key=None, model=None, timeout=None):
    """Normalize `text` via the OpenAI-compatible endpoint. Raises LLMNormalizerError
    on any failure; callers decide whether to fall back to the rule-based path."""
    base_url = (OPENAI_BASE_URL if base_url is None else base_url).rstrip('/')
    api_key = OPENAI_API_KEY if api_key is None else api_key
    model = OPENAI_MODEL_NAME if model is None else model
    timeout = OPENAI_TIMEOUT if timeout is None else timeout

    if not base_url or not model:
        raise LLMNormalizerError('OPENAI_BASE_URL / OPENAI_MODEL_NAME not configured')

    payload = {
        'model': model,
        'messages': build_messages(text),
        'temperature': 0,
        # verbalization expands the text (digits -> words); chars overestimate tokens,
        # so this cap is safe.
        'max_tokens': min(4096, 256 + 2 * len(text)),
        'response_format': {'type': 'json_schema', 'json_schema': JSON_SCHEMA},
    }
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as session:

            async def post(p):
                async with session.post(
                    f'{base_url}/chat/completions', json=p, headers=headers
                ) as r:
                    return r.status, await r.text()

            status, body = await post(payload)
            if status == 400:
                # backend may not support response_format json_schema; retry unconstrained
                logging.warning(
                    f'llm normalizer got 400, retrying without response_format: {body[:200]}'
                )
                payload.pop('response_format')
                status, body = await post(payload)
            if status != 200:
                raise LLMNormalizerError(f'{base_url} returned {status}: {body[:500]}')
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        raise LLMNormalizerError(f'request to {base_url} failed: {e!r}') from e

    try:
        content = json.loads(body)['choices'][0]['message']['content']
    except (ValueError, LookupError, TypeError) as e:
        raise LLMNormalizerError(f'malformed completion response: {body[:500]}') from e

    normalized = parse_normalized(content)
    if normalized is None:
        raise LLMNormalizerError(f'unusable reply from model: {content[:500]!r}')
    return normalized
