"""Complete latency benchmark for the TTS API: TTFB, end-to-end and RTF, with the
full percentile spread, under closed-loop concurrency.

Why not `bench/bench.py`: it asks for `response_format=wav`, and a wav response emits
its 44-byte header before a single token has been decoded — so its "TTFB" is the time to
the header, not to audio, and reads ~0 whatever the stack is doing. This asks for `pcm`,
where the first byte IS audio. It also reports p10/p95/p99, which the throughput-shaped
bench does not.

Three latencies, because they answer different questions:

  ttfb   text in -> FIRST AUDIO BYTE out. What a caller hears as "did it respond?", and
         the only one a conversational agent is really judged on.
  e2e    text in -> LAST byte out. Matters for batch/offline use; for a streaming caller
         it is mostly a function of how much audio was asked for.
  rtf    e2e / seconds-of-audio-produced. Below 1.0 the stack generates faster than
         real time, so a stream never starves; it is the number that says whether a
         given concurrency is sustainable at all.

`lead` is the safety margin: audio produced minus wall time at the moment the stream
ends, i.e. how much buffer the client had. Negative means the client would have stalled.

    python bench/latency_bench.py --url http://127.0.0.1:9091 \
      --voice TM_English_Normal --concurrency 1,4,8,16,32 --out lat.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SR = 24_000
PCTS = (10, 50, 90, 95, 99)


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * len(xs) + 0.5)) - 1))
    return xs[k]


def spread(xs, nd=3):
    """mean / min / percentiles / max for one metric."""
    if not xs:
        return None
    out = {'n': len(xs), 'mean': round(statistics.mean(xs), nd), 'min': round(min(xs), nd)}
    for p in PCTS:
        out[f'p{p}'] = round(pct(xs, p), nd)
    out['max'] = round(max(xs), nd)
    return out


async def one_request(session, url, text, voice, temperature, max_tokens, timeout):
    """One streamed pcm request. Returns per-request latencies, or an error row."""
    payload = {'input': text, 'voice': voice, 'response_format': 'pcm', 'stream': True,
               'temperature': temperature, 'max_tokens': max_tokens}
    t0 = time.perf_counter()
    ttfb = None
    n = 0
    try:
        async with session.post(f'{url}/v1/audio/speech', json=payload, timeout=timeout) as r:
            if r.status != 200:
                return {'ok': False, 'error': f'HTTP {r.status}: {(await r.text())[:120]}'}
            async for block in r.content.iter_chunked(8192):
                if block:
                    if ttfb is None:
                        ttfb = time.perf_counter() - t0
                    n += len(block)
    except Exception as e:                       # noqa: BLE001 — one bad request, keep going
        return {'ok': False, 'error': f'{type(e).__name__}: {e}'}
    e2e = time.perf_counter() - t0
    audio_s = (n / 2) / SR
    if audio_s <= 0:
        return {'ok': False, 'error': 'empty audio'}
    return {'ok': True, 'ttfb': ttfb, 'e2e': e2e, 'audio_s': audio_s,
            'rtf': e2e / audio_s, 'lead': audio_s - e2e, 'bytes': n}


async def run_level(session, args, texts, conc):
    """`conc` workers pulling from one queue — closed loop, so the level is real."""
    total = max(args.min_requests, conc * args.per_conc_mult)
    q = asyncio.Queue()
    for i in range(total):
        q.put_nowait(texts[i % len(texts)])
    rows = []

    async def worker():
        while True:
            try:
                text = q.get_nowait()
            except asyncio.QueueEmpty:
                return
            rows.append(await one_request(session, args.url, text, args.voice,
                                          args.temperature, args.max_tokens, args.timeout))

    t0 = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(conc)))
    wall = time.perf_counter() - t0

    ok = [r for r in rows if r.get('ok')]
    bad = [r for r in rows if not r.get('ok')]
    audio = sum(r['audio_s'] for r in ok)
    return {
        'concurrency': conc, 'requests': total, 'ok': len(ok), 'errors': len(bad),
        'wall_s': round(wall, 2),
        'audio_s_total': round(audio, 1),
        'throughput_audio_s_per_s': round(audio / wall, 2) if wall else None,
        'requests_per_s': round(len(ok) / wall, 2) if wall else None,
        'ttfb_s': spread([r['ttfb'] for r in ok if r['ttfb'] is not None]),
        'e2e_s': spread([r['e2e'] for r in ok]),
        'rtf': spread([r['rtf'] for r in ok]),
        'audio_s': spread([r['audio_s'] for r in ok], 2),
        'lead_s': spread([r['lead'] for r in ok], 2),
        'stalls': sum(1 for r in ok if r['lead'] < 0),
        'error_samples': [r['error'] for r in bad[:3]],
    }


def table(levels):
    def row(lv, key, unit=''):
        m = lv[key]
        if not m:
            return f'{"":>9}' * 6
        return (f'{m["mean"]:>9.3f}{m["p10"]:>9.3f}{m["p50"]:>9.3f}'
                f'{m["p90"]:>9.3f}{m["p95"]:>9.3f}{m["p99"]:>9.3f}')

    for key, label in (('ttfb_s', 'TTFB (first audio byte), s'),
                       ('e2e_s', 'End-to-end, s'),
                       ('rtf', 'RTF (e2e / audio seconds)')):
        print(f'\n{label}')
        hdr = f'{"conc":>5}{"mean":>9}{"p10":>9}{"p50":>9}{"p90":>9}{"p95":>9}{"p99":>9}{"max":>9}'
        print(hdr)
        print('-' * len(hdr))
        for lv in levels:
            m = lv[key]
            print(f'{lv["concurrency"]:>5}' + (row(lv, key) if m else '') +
                  (f'{m["max"]:>9.3f}' if m else ''))

    print('\nThroughput and safety')
    hdr = (f'{"conc":>5}{"reqs":>6}{"err":>5}{"wall s":>9}{"audio s":>10}'
           f'{"audio-s/s":>11}{"req/s":>8}{"min lead s":>12}{"stalls":>8}')
    print(hdr)
    print('-' * len(hdr))
    for lv in levels:
        lead = lv['lead_s']
        print(f'{lv["concurrency"]:>5}{lv["requests"]:>6}{lv["errors"]:>5}{lv["wall_s"]:>9.2f}'
              f'{lv["audio_s_total"]:>10.1f}{lv["throughput_audio_s_per_s"]:>11.2f}'
              f'{lv["requests_per_s"]:>8.2f}'
              f'{(lead["min"] if lead else float("nan")):>12.2f}{lv["stalls"]:>8}')


async def main_async(args):
    import aiohttp

    if args.texts:
        texts = [l.strip() for l in open(args.texts, encoding='utf-8') if l.strip()]
    else:
        from evalset import EVAL_SET
        # EVAL_SET rows are (id, voice, text) — take the text and use OUR voice,
        # so the level is measured on the deployed speaker, not the fixture's.
        texts = [e[2] if isinstance(e, (list, tuple)) else e for e in EVAL_SET]
    print(f'{len(texts)} texts · voice {args.voice} · temp {args.temperature} · '
          f'server normalizer default\n')

    concs = [int(c) for c in args.concurrency.split(',')]
    conn = aiohttp.TCPConnector(limit=max(concs) * 2)
    levels = []
    async with aiohttp.ClientSession(connector=conn) as session:
        if args.warmup:
            await asyncio.gather(*(one_request(session, args.url, texts[i % len(texts)],
                                               args.voice, args.temperature,
                                               args.max_tokens, args.timeout)
                                   for i in range(args.warmup)))
        for c in concs:
            print(f'--- concurrency {c} …', flush=True)
            lv = await run_level(session, args, texts, c)
            levels.append(lv)
            print(f'    {lv["ok"]}/{lv["requests"]} ok · ttfb p50 '
                  f'{lv["ttfb_s"]["p50"] if lv["ttfb_s"] else float("nan"):.3f}s · '
                  f'rtf p50 {lv["rtf"]["p50"] if lv["rtf"] else float("nan"):.3f} · '
                  f'{lv["throughput_audio_s_per_s"]} audio-s/s', flush=True)

    table(levels)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'url': args.url, 'voice': args.voice,
                       'temperature': args.temperature, 'label': args.label,
                       'levels': levels}, f, indent=1)
        print(f'\nwrote {args.out}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--url', default='http://127.0.0.1:9091')
    p.add_argument('--voice', default='TM_English_Normal')
    p.add_argument('--concurrency', default='1,4,8,16,32')
    p.add_argument('--per-conc-mult', type=int, default=6,
                   help='requests per level = concurrency * this')
    p.add_argument('--min-requests', type=int, default=30)
    p.add_argument('--temperature', type=float, default=0.6)
    p.add_argument('--max-tokens', type=int, default=1024)
    p.add_argument('--timeout', type=float, default=300.0)
    p.add_argument('--warmup', type=int, default=6)
    p.add_argument('--texts', default='', help='one line per utterance; default: bench/evalset.py')
    p.add_argument('--label', default='run')
    p.add_argument('--out', default='')
    args = p.parse_args()
    args.url = args.url.rstrip('/')
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
