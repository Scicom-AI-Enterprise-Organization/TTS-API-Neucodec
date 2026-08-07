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
    rule = 'rule'
    llm = 'llm'


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
