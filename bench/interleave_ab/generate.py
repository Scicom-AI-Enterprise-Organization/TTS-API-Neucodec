#!/usr/bin/env python3
"""Generate the three conditions of the interleave A/B, straight from the LM.

For every corpus paragraph (`corpus.py`), the same text is rendered three ways:

  A `single`      one request for the whole paragraph                 -- the reference
  B `interleave`  one request per chunk, each prompted with the previous chunks'
                  text AND the speech tokens the LM produced for them (`build_prompt`)
  C `cold`        one request per chunk, each prompted on its own      -- today's behaviour

A is what the paragraph sounds like when the model gets it whole; C is what a LiveKit
`StreamAdapter` reply sounds like today; B is the same chunking with the interleaved
prompt. B and C see byte-identical chunk text, so any difference between them is the
prompt shape alone.

Talking to vLLM directly (rather than through `/v1/audio/speech`) is deliberate: it
removes the normalizer, the crossfade stitcher and the per-request loudness normalizer
from the comparison, leaving the LM. The prompt construction and the history bounds are
imported from `app/interleave.py`, so B is exactly the prompt the API would have built.

The `INTERLEAVE_FALLBACK` guard is NOT applied -- a collapsed chunk is recorded, not
retried, because the point is to measure how often the model collapses.

    PYTHONPATH=. python bench/interleave_ab/generate.py --api http://127.0.0.1:9086/v1/completions
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())
from app.interleave import Turn, build_prompt, fit_interleave, seconds_to_tokens  # noqa: E402

TOKEN_RE = re.compile(r'<\|s_(\d+)\|>')
TOKENS_PER_S = 50


def collapse_threshold(text: str) -> int:
    """`INTERLEAVE_FALLBACK`'s rule (app/main.py): a real rendering runs ~15-20 tokens
    per word, so a chunk that ends under 4x its word count did not render anything."""
    return max(10, min(50, 4 * len(text.split())))


async def complete(session, api, key, model, prompt, max_tokens, temperature, rep_pen):
    body = {'model': model, 'prompt': prompt, 'max_tokens': max_tokens,
            'temperature': temperature, 'repetition_penalty': rep_pen, 'stream': False}
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = f'Bearer {key}'
    t0 = time.perf_counter()
    async with session.post(api, json=body, headers=headers) as r:
        if r.status != 200:
            raise RuntimeError(f'LM HTTP {r.status}: {(await r.text())[:300]}')
        d = await r.json()
    ch = d['choices'][0]
    return {
        'tokens': [int(x) for x in TOKEN_RE.findall(ch['text'])],
        'finish_reason': ch.get('finish_reason'),
        'prompt_tokens': d.get('usage', {}).get('prompt_tokens'),
        'completion_tokens': d.get('usage', {}).get('completion_tokens'),
        'latency_s': round(time.perf_counter() - t0, 4),
    }


async def run_text(session, a, rec):
    """All three conditions for one paragraph. B is sequential by construction (chunk
    N+1's prompt needs chunk N's tokens); A and C are fired alongside it."""
    text, chunks, voice = rec['text'], rec['chunks'], a.voice
    out = {'id': rec['id'], 'lang': rec['lang'], 'text': text, 'chunks': chunks}

    async def single():
        r = await complete(session, a.api, a.key, a.model,
                           build_prompt([], voice, text), a.max_tokens, a.temperature, a.rep_pen)
        return [r]

    async def cold():
        rs = []
        for c in chunks:
            rs.append(await complete(session, a.api, a.key, a.model,
                                     build_prompt([], voice, c), a.max_tokens,
                                     a.temperature, a.rep_pen))
        return rs

    async def interleave():
        rs, history = [], []
        for c in chunks:
            turns, mt = fit_interleave(
                history, c, a.max_tokens, a.max_model_len,
                seconds_to_tokens(a.interleave_max_s), a.min_gen_tokens, a.max_retain)
            r = await complete(session, a.api, a.key, a.model,
                               build_prompt(turns, voice, c), mt, a.temperature, a.rep_pen)
            r['history_turns'] = len(turns)
            r['history_tokens'] = sum(len(t.tokens) for t in turns)
            rs.append(r)
            # Same rule as app/main.py: a `length` finish means the tokens stop
            # mid-text, so it is never stored as history.
            if r['finish_reason'] != 'length' and r['tokens']:
                history.append(Turn(text=c, tokens=r['tokens'], voice=voice, ts=time.time()))
        return rs

    res = await asyncio.gather(single(), interleave(), cold())
    for name, rs in zip(('single', 'interleave', 'cold'), res):
        out[name] = [
            {**r, 'n_tokens': len(r['tokens']),
             'collapsed': len(r['tokens']) < collapse_threshold(t)}
            for r, t in zip(rs, [text] if name == 'single' else chunks)
        ]
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default='bench/results/interleave_ab/corpus.json')
    ap.add_argument('--out', default='bench/results/interleave_ab/tokens.jsonl')
    ap.add_argument('--api', default='http://127.0.0.1:9086/v1/completions')
    ap.add_argument('--model', default='TTS-model')
    ap.add_argument('--voice', default='TM_English_Normal')
    ap.add_argument('--temperature', type=float, default=0.6)
    ap.add_argument('--rep-pen', type=float, default=1.15)
    ap.add_argument('--max-tokens', type=int, default=3072)
    ap.add_argument('--max-model-len', type=int, default=4096)
    ap.add_argument('--max-retain', type=int, default=5)
    ap.add_argument('--interleave-max-s', type=float, default=20.0)
    ap.add_argument('--min-gen-tokens', type=int, default=1000)
    ap.add_argument('--concurrency', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    a.key = os.environ.get('TTS_API_KEY', '')

    import aiohttp
    texts = json.loads(Path(a.corpus).read_text())['texts']
    if a.limit:
        texts = texts[:a.limit]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(a.concurrency)
    done = [0]
    t0 = time.time()

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as session:
        async def one(rec):
            async with sem:
                try:
                    r = await run_text(session, a, rec)
                except Exception as e:
                    print(f'[gen] {rec["id"]} FAILED: {type(e).__name__}: {e}', flush=True)
                    return None
                done[0] += 1
                if done[0] % 10 == 0 or done[0] == len(texts):
                    print(f'[gen] {done[0]}/{len(texts)} {time.time()-t0:.0f}s', flush=True)
                return r

        rows = [r for r in await asyncio.gather(*(one(t) for t in texts)) if r]

    rows.sort(key=lambda r: r['id'])
    with open(a.out, 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    for cond in ('single', 'interleave', 'cold'):
        n = [sum(x['n_tokens'] for x in r[cond]) for r in rows]
        col = sum(x['collapsed'] for r in rows for x in r[cond])
        tot = sum(len(r[cond]) for r in rows)
        print(f'[{cond:10s}] {len(rows)} texts, {tot} requests, '
              f'median {statistics.median(n)/TOKENS_PER_S:.2f}s audio/text, '
              f'collapsed {col}/{tot} ({100*col/tot:.1f}%)')
    print(f'[gen] wrote {a.out} in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
