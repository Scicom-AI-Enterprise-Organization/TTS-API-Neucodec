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
from app.prompt import SYSTEM_PROMPT, EXAMPLES, JSON_SCHEMA, CALLER_PROMPT_HEADER


class NormalizerMode(str, Enum):
    rule = 'rule'        # legacy malaya-derived pipeline (app/normalizer)
    llm = 'llm'          # OpenAI-compatible LLM with the prompt in app/prompt.py
    spoken = 'spoken'    # rule-based replica of the LLM (app/spoken_normalizer), no network


class LLMNormalizerError(Exception):
    """LLM endpoint unreachable, non-200, or returned an unusable reply."""


_CODE_FENCE = re.compile(r'^```(?:json)?\s*(.*?)\s*```$', re.DOTALL)


def build_messages(text, prompt=None):
    """`prompt` is the caller's extra context (the request's `prompt` field), put both at
    the end of the system prompt and in brackets above the text (why both: app/prompt.py);
    blank or None leaves the messages exactly as without it."""
    system = SYSTEM_PROMPT
    prompt = (prompt or '').strip()
    if prompt:
        system = f'{SYSTEM_PROMPT}\n\n{CALLER_PROMPT_HEADER}\n{prompt}'
    messages = [{'role': 'system', 'content': system}]
    for before, after in EXAMPLES:
        messages.append({'role': 'user', 'content': before})
        messages.append({
            'role': 'assistant',
            'content': json.dumps({'normalized': after}, ensure_ascii=False),
        })
    messages.append({'role': 'user', 'content': f'[{prompt}]\n{text}' if prompt else text})
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


# ---------------------------------------------------------------------------
# Replies that must not be spoken
# ---------------------------------------------------------------------------
# The LLM sometimes hands back a reply that reads well and is wrong in a way no listener can
# recover from: a number dropped, the digits of a read-back dropped or left as digits ("D 3 6 7 6"
# came back as written), or English text read with Chinese numerals ("8 8 0 1" -> "八 八 零 一"),
# which the TTS then speaks in Mandarin. Each leaves a trace that needs no second model to see. rejected_reason() names it; llm_normalize() raises
# on it, so the TTS path falls back to the spoken rules, which read every digit.
_HAN = '㐀-䶿一-鿿豈-﫿'
_CJK_TEXT = re.compile(f'[{_HAN}　-〿！-？]')
_HAN_CHAR = re.compile(f'[{_HAN}]')
_DIGIT_WORD = re.compile(
    r'(?i)(?<![a-z])(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|double|triple|'
    r'kosong|sifar|satu|dua|tiga|empat|lima|enam|tujuh|lapan|sembilan)(?![a-z])|[零〇一二两三四五六七八九]')
_NUMBER_WORD = re.compile(
    r'(?i)(?<![a-z])(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|'
    r'(?:thir|four|fif|six|seven|eigh|nine)teen|(?:twen|thir|for|fif|six|seven|eigh|nine)ty|'
    r'hundred|thousand|million|billion|half|quarter|first|second|third|'
    r'(?:four|fif|six|seven|eigh|nin|ten|eleven|twelf|(?:\w+teen)|(?:\w+tie)|hundred|thousand)th|'
    r'kosong|sifar|satu|dua|tiga|empat|lima|enam|tujuh|lapan|sembilan|sepuluh|sebelas|belas|'
    r'puluh|ratus|ribu|juta|bilion|seratus|seribu|sejuta|separuh|pertama|'
    r'ke(?:dua|tiga|empat|lima|enam|tujuh|lapan|sembilan|sepuluh|sebelas))(?![a-z])'
    r'|[零〇一二两三四五六七八九十百千万亿半]')
# Digits that are read one at a time: a 7+ digit run (not a round amount), single digits
# spaced out, or dash-joined groups of 9+ digits (IC, phone; a dashed date has 8).
_READ_BACK = re.compile(r'(?<![\d,.])(?:\d(?: \d(?!\d)){6,}|\d+(?:-\d+)+|\d{7,})(?![\d,.]\d)')


def _read_back_digits(text):
    need = 0
    for m in _READ_BACK.finditer(text):
        run = re.sub(r'\D', '', m.group(0))
        if '-' in m.group(0) and len(run) < 9:
            continue
        if ' ' not in m.group(0) and '-' not in m.group(0) and run.endswith('000'):
            continue
        need += len(run)
    return need


def _spoken_digits(out):
    words = [w.lower() for w in _DIGIT_WORD.findall(out)]
    return len(words) + words.count('triple')       # "double eight" is two words already


def rejected_reason(text, normalized, prompt=None):
    """Why the LLM's `normalized` reply to `text` must not be spoken, or None if it may.

    Skipped for Tamil (no digit-word list here). With a caller `prompt` the Chinese and
    leftover-digit checks are off, since a prompt may ask for exactly that."""
    if not re.search(r'\d', text) or re.search('[஀-௿]', text):
        return None
    if not re.search(r'\d', normalized) and not _NUMBER_WORD.search(normalized):
        return 'number dropped'
    if _spoken_digits(normalized) + len(re.findall(r'\d', normalized)) < _read_back_digits(text):
        return 'read-back digits dropped'
    if prompt:
        return None
    if re.search(r'\d', normalized):
        return 'digits left'
    if _HAN_CHAR.search(normalized) and not _CJK_TEXT.search(text):
        return 'Chinese numerals in non-Chinese text'
    return None


async def llm_normalize(text, base_url=None, api_key=None, model=None, timeout=None,
                        prompt=None):
    """Normalize `text` via the OpenAI-compatible endpoint, with the caller's `prompt`
    (see build_messages) if given. Raises LLMNormalizerError on any failure; callers
    decide whether to fall back to the rule-based path."""
    base_url = (OPENAI_BASE_URL if base_url is None else base_url).rstrip('/')
    api_key = OPENAI_API_KEY if api_key is None else api_key
    model = OPENAI_MODEL_NAME if model is None else model
    timeout = OPENAI_TIMEOUT if timeout is None else timeout

    if not base_url or not model:
        raise LLMNormalizerError('OPENAI_BASE_URL / OPENAI_MODEL_NAME not configured')

    payload = {
        'model': model,
        'messages': build_messages(text, prompt),
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
    reason = rejected_reason(text, normalized, prompt)
    if reason:
        raise LLMNormalizerError(f'rejected reply ({reason}): {normalized[:500]!r}')
    return normalized
