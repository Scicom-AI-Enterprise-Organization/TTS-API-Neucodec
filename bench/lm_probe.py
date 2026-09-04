#!/usr/bin/env python3
"""
Direct vLLM single-stream probe: how fast does the LM hand us speech tokens?

Per request (times from POST start): t_headers, t_first_token (= prefill + scheduling),
steady-state tokens/s, and the time at which the N-th speech token arrived for the token
counts the stitcher gates its first chunk on (playback_speed*50 + overlap*50). That last
column is the LM's share of TTFB for a given first window, measured rather than inferred.

Reads TTS_API / TTS_API_KEY / MODEL_NAME from the environment (source the app's .env).

    set -a; source .env; set +a
    python bench/lm_probe.py --voice husein --reps 3 --gates 35,60,85,110
"""
import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evalset import EVAL_SET  # noqa: E402

TOKEN_RE = re.compile(r'<\|s_(\d+)\|>')


async def one(session, url, headers, model, prompt, temperature, rep_pen, max_tokens):
    body = {'model': model, 'prompt': prompt, 'max_tokens': max_tokens,
            'temperature': temperature, 'repetition_penalty': rep_pen, 'stream': True}
    t0 = time.perf_counter()
    tok_times = []
    async with session.post(url, headers=headers, json=body) as resp:
        t_headers = time.perf_counter() - t0
        if resp.status != 200:
            raise RuntimeError(f'HTTP {resp.status}: {(await resp.text())[:200]}')
        async for line in resp.content:
            if not line.startswith(b'data: '):
                continue
            data = line[6:].strip()
            if data == b'[DONE]':
                break
            try:
                text = json.loads(data)['choices'][0].get('text', '')
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            n = len(TOKEN_RE.findall(text))
            now = time.perf_counter() - t0
            tok_times.extend([now] * n)
    total = time.perf_counter() - t0
    return {'t_headers': t_headers, 'tok_times': tok_times, 'total': total}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--voice', default='husein')
    ap.add_argument('--ids', default='en_short_1,en_med_1,ms_short_1,ms_med_1,cs_1')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--temperature', type=float, default=0.6)
    ap.add_argument('--repetition-penalty', type=float, default=1.15)
    ap.add_argument('--max-tokens', type=int, default=3072)
    ap.add_argument('--gates', default='35,60,85,110',
                    help='token counts to report arrival time for (first window + overlap)')
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    url = os.environ.get('TTS_API', 'http://127.0.0.1:9093')
    if '/v1/completions' not in url:
        url += '/v1/completions'
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('TTS_API_KEY'):
        headers['Authorization'] = f"Bearer {os.environ['TTS_API_KEY']}"
    model = os.environ.get('MODEL_NAME', 'TTS-model')
    gates = [int(g) for g in args.gates.split(',')]
    ids = set(args.ids.split(','))
    texts = [(i, t) for i, _v, t in EVAL_SET if i in ids]

    rows = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        prompt = f'<|im_start|>{args.voice}: {texts[0][1]}<|speech_start|>'
        await one(session, url, headers, model, prompt, args.temperature,
                  args.repetition_penalty, args.max_tokens)  # warmup
        for rep in range(args.reps):
            for tid, text in texts:
                prompt = f'<|im_start|>{args.voice}: {text}<|speech_start|>'
                r = await one(session, url, headers, model, prompt, args.temperature,
                              args.repetition_penalty, args.max_tokens)
                tt = r['tok_times']
                n = len(tt)
                first = tt[0] if tt else None
                rate = (n - 1) / (tt[-1] - tt[0]) if n > 1 and tt[-1] > tt[0] else None
                row = {'id': tid, 'rep': rep, 'n_tokens': n, 't_headers': r['t_headers'],
                       't_first_token': first, 'tok_per_s': rate, 'total': r['total'],
                       'gate_t': {g: (tt[g - 1] if n >= g else None) for g in gates}}
                rows.append(row)
                gates_s = ' '.join(f"{g}:{row['gate_t'][g]:.3f}" if row['gate_t'][g] else f'{g}:-'
                                   for g in gates)
                print(f"{tid:<11} rep{rep} tokens {n:4d} hdr {r['t_headers']:.3f} first {first:.3f} "
                      f"tok/s {rate:6.1f} total {r['total']:.2f} | gate arrival {gates_s}", flush=True)

    def agg(key, f=lambda r, k: r[k]):
        xs = [f(r, key) for r in rows if f(r, key) is not None]
        return (round(statistics.mean(xs), 3), round(statistics.median(xs), 3)) if xs else None
    print('\naggregate (mean, median):')
    print('  t_headers      ', agg('t_headers'))
    print('  t_first_token  ', agg('t_first_token'))
    print('  tok/s          ', agg('tok_per_s'))
    for g in gates:
        print(f'  arrival of token {g:>3}', agg(g, lambda r, k: r['gate_t'][k]))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'args': vars(args), 'url': url, 'rows': rows}, f, indent=1)


if __name__ == '__main__':
    asyncio.run(main())
