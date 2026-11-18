"""`response_format: "tokens"` on /v1/audio/speech: return the LM's speech tokens instead of audio.

The request goes through the same normalization and the same prompt (interleave history included) as an audio
request, and the LM is called with the same sampling fields, but the NeuCodec decode is skipped: the response is
JSON with the `<|s_N|>` ids the LM produced. Uses: collecting the served model's own takes (DPO pairs built from
what production actually says), cheap A/B of sampling settings, and debugging a take without the codec.

No torch import here, so tests/test_speech_tokens.py runs anywhere.
"""
import json
import re
from typing import AsyncIterator, Awaitable, Callable, Optional, Tuple

TOKEN_RE = re.compile(r'<\|s_(\d+)\|>')


def parse_ids(text: str) -> list[int]:
    """The speech-token ids in an LM completion, in order (anything else is ignored)."""
    return [int(m) for m in TOKEN_RE.findall(text or '')]


async def collect(deltas: AsyncIterator[Tuple[str, Optional[str]]]) -> Tuple[str, Optional[str]]:
    """Join a stream of (text, finish_reason) deltas into (text, last finish_reason)."""
    parts, finish = [], None
    async for text, fr in deltas:
        if text:
            parts.append(text)
        finish = fr or finish
    return ''.join(parts), finish


async def http_deltas(session_post: Callable[..., Awaitable], url: str, headers: dict,
                      body: dict) -> AsyncIterator[Tuple[str, Optional[str]]]:
    """vLLM's OpenAI /v1/completions SSE stream as (text, finish_reason) deltas. `session_post` is an
    aiohttp-style `post` (async context manager); a non-200 raises RuntimeError with the body's start."""
    async with session_post(url, headers=headers, json={**body, 'stream': True}) as resp:
        if resp.status != 200:
            raise RuntimeError(f'backend returned {resp.status}: {(await resp.text())[:200]}')
        async for line in resp.content:
            if not line.startswith(b'data: '):
                continue
            payload = line[6:].strip()
            if payload == b'[DONE]':
                break
            try:
                choice = json.loads(payload)['choices'][0]
            except (ValueError, KeyError, IndexError):
                continue
            yield choice.get('text') or '', choice.get('finish_reason')


def response_body(*, model: str, voice: str, text_in: str, normalized: str, completion: str,
                  finish_reason: Optional[str], prompt: Optional[str] = None,
                  interleave: Optional[dict] = None) -> dict:
    """The JSON a tokens request returns. `prompt` only when the caller asked for it (it carries the
    interleave history's speech tokens, thousands of them)."""
    ids = parse_ids(completion)
    body = {
        'object': 'speech.tokens',
        'model': model,
        'voice': voice,
        'input': text_in,
        'normalized': normalized,
        'tokens': ids,
        'n_tokens': len(ids),
        'speech': ''.join(f'<|s_{i}|>' for i in ids),
        'finish_reason': finish_reason,
        # 24 kHz NeuCodec at 50 tokens a second: what the audio would have lasted
        'seconds': round(len(ids) / 50.0, 3),
    }
    if prompt is not None:
        body['prompt'] = prompt
    if interleave is not None:
        body['interleave'] = interleave
    return body
