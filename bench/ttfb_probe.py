#!/usr/bin/env python3
"""
TTFB probe for the streaming TTS endpoint, with a client-stall check.

Per request (all times from the start of the POST):
  t_headers   response headers received
  ttfb        first non-empty AUDIO byte. Always measured on response_format=pcm: with wav
              the very first bytes are the 44-byte header, which the app yields before any
              decode has happened, so a wav TTFB says nothing (bench.py has that flaw).
  total       end of the stream
  audio_s     seconds of 16-bit/24 kHz audio received
  min_lead_s  the smallest playout buffer a client that starts playing at `ttfb` would have
              had, over every later chunk: (audio seconds received before chunk k) minus
              (arrival_k - ttfb). Negative => that client stalls: the LM/decoder fell behind
              playback. This is what bounds how small the first decode window may be.

`playback_speed` / `playback_overlap_speed` are request fields, so a grid over them needs
no restart; things like CUDA_GRAPH_BATCH or STREAM_CHUNK_GROWTH do (use --url per instance).

    python bench/ttfb_probe.py --url http://127.0.0.1:9087 --voice husein \
        --playback 2.0,1.5,1.0,0.75,0.5 --overlap 0.2,0.1 --reps 3 --out /tmp/ttfb.json
"""
import argparse
import asyncio
import json
import os
import statistics
import sys
import time

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evalset import EVAL_SET  # noqa: E402

SR = 24000
BYTES_PER_S = SR * 2


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * len(xs) + 0.5)) - 1))
    return xs[k]


async def one_request(session, url, text, voice, playback, overlap, mode, temperature,
                      speaking_rate=None, extra=None):
    payload = {
        'input': text, 'voice': voice, 'model': 'TTS-model',
        'response_format': 'pcm', 'stream': True, 'stream_format': 'audio',
        'playback_speed': playback, 'playback_overlap_speed': overlap,
        'mode': mode, 'temperature': temperature, 'normalize_malaysian': False,
    }
    if speaking_rate is not None:
        payload['speaking_rate'] = speaking_rate
    if extra:
        payload.update(extra)
    t0 = time.perf_counter()
    arrivals = []      # (t since t0, nbytes) for every non-empty read
    async with session.post(url + '/v1/audio/speech', json=payload) as resp:
        t_headers = time.perf_counter() - t0
        if resp.status != 200:
            raise RuntimeError(f'HTTP {resp.status}: {(await resp.text())[:200]}')
        async for chunk in resp.content.iter_any():
            if chunk:
                arrivals.append((time.perf_counter() - t0, len(chunk)))
    total = time.perf_counter() - t0
    if not arrivals:
        raise RuntimeError('empty stream')
    ttfb = arrivals[0][0]
    audio_before = 0.0
    min_lead = None
    for t, n in arrivals:
        if audio_before > 0:
            lead = audio_before - (t - ttfb)
            min_lead = lead if min_lead is None else min(min_lead, lead)
        audio_before += n / BYTES_PER_S
    return {
        't_headers': t_headers, 'ttfb': ttfb, 'total': total,
        'audio_s': audio_before, 'min_lead_s': min_lead, 'reads': len(arrivals),
    }


def summarize(rows):
    ttfb = [r['ttfb'] for r in rows]
    leads = [r['min_lead_s'] for r in rows if r['min_lead_s'] is not None]
    return {
        'n': len(rows),
        'ttfb_mean': round(statistics.mean(ttfb), 3),
        'ttfb_p50': round(pct(ttfb, 50), 3),
        'ttfb_p90': round(pct(ttfb, 90), 3),
        'ttfb_max': round(max(ttfb), 3),
        't_headers_mean': round(statistics.mean(r['t_headers'] for r in rows), 3),
        'min_lead_min': round(min(leads), 3) if leads else None,
        'min_lead_p10': round(pct(leads, 10), 3) if leads else None,
        'stalls': sum(1 for l in leads if l < 0),
        'audio_s_mean': round(statistics.mean(r['audio_s'] for r in rows), 2),
        'total_mean': round(statistics.mean(r['total'] for r in rows), 3),
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:9091')
    ap.add_argument('--voice', default='husein')
    ap.add_argument('--ids', default='en_short_1,en_med_1,ms_short_1,ms_med_1,cs_1',
                    help='evalset ids, comma separated (voice is overridden by --voice)')
    ap.add_argument('--playback', default='2.0,1.5,1.0,0.75,0.5')
    ap.add_argument('--overlap', default='0.2')
    ap.add_argument('--mode', default='rule', choices=['rule', 'llm', 'spoken'])
    ap.add_argument('--text', action='append', default=None,
                    help='probe this text instead of evalset ids (repeatable)')
    ap.add_argument('--temperature', type=float, default=0.6)
    ap.add_argument('--speaking-rate', type=float, default=None)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--out', default='')
    ap.add_argument('--label', default='')
    ap.add_argument('--api-key', default=os.environ.get('TTS_PROBE_API_KEY', ''),
                    help='Bearer token, for probing through an authenticating proxy')
    args = ap.parse_args()

    ids = set(args.ids.split(','))
    texts = [(f'text{i}', t) for i, t in enumerate(args.text)] if args.text else [(i, t) for i, _v, t in EVAL_SET if i in ids]
    if not texts:
        sys.exit(f'no evalset ids matched {args.ids}')
    playbacks = [float(x) for x in args.playback.split(',')]
    overlaps = [float(x) for x in args.overlap.split(',')]

    timeout = aiohttp.ClientTimeout(total=300)
    headers = {'Authorization': f'Bearer {args.api_key}'} if args.api_key else {}
    results = {}
    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        for _ in range(args.warmup):
            await one_request(session, args.url, texts[0][1], args.voice, playbacks[0],
                              overlaps[0], args.mode, args.temperature, args.speaking_rate)
        for ov in overlaps:
            for ps in playbacks:
                rows = []
                for rep in range(args.reps):
                    for tid, text in texts:
                        r = await one_request(session, args.url, text, args.voice, ps, ov,
                                              args.mode, args.temperature, args.speaking_rate)
                        r.update({'id': tid, 'rep': rep})
                        rows.append(r)
                key = f'ps={ps} ov={ov}'
                results[key] = {'playback_speed': ps, 'overlap': ov,
                                'summary': summarize(rows), 'rows': rows}
                s = results[key]['summary']
                print(f"{args.label:>10} {key:<16} ttfb mean {s['ttfb_mean']:.3f} p50 {s['ttfb_p50']:.3f} "
                      f"p90 {s['ttfb_p90']:.3f} max {s['ttfb_max']:.3f} | hdr {s['t_headers_mean']:.3f} | "
                      f"min_lead {s['min_lead_min']} p10 {s['min_lead_p10']} stalls {s['stalls']}/{s['n']} | "
                      f"audio {s['audio_s_mean']}s total {s['total_mean']}s", flush=True)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'args': vars(args), 'results': results}, f, indent=1)


if __name__ == '__main__':
    asyncio.run(main())
