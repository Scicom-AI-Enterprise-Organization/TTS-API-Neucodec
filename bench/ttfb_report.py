#!/usr/bin/env python3
"""
TTFB + end-to-end latency of the streaming TTS endpoint, per normalizer mode.

Measures, per request, from the start of the POST:
  t_norm     /v1/audio/normalize round trip for the same text and mode (the normalizer runs
             before the LM prompt is built, so it is TTFB one-for-one)
  ttfb       first non-empty AUDIO byte on a response_format=pcm stream (a wav stream emits
             its 44-byte header before any decode, so wav TTFB is meaningless)
  total      end of the stream = end-to-end latency
  audio_s    seconds of 16-bit / 24 kHz audio received
  rtf        total / audio_s (below 1.0 = faster than real time)
  min_lead_s smallest playout buffer a client starting at `ttfb` would ever hold; negative
             means that client stalls

Also samples a trivial GET to estimate the network round trip, so on-box cost can be
separated from the client's distance to the service.

    python bench/ttfb_report.py --url https://<host> --reps 5 --out /tmp/ttfb_report.json
"""
import argparse
import asyncio
import json
import os
import statistics
import sys
import time

import aiohttp

SR = 24000
BYTES_PER_S = SR * 2

# (id, mode-independent label, text). "covered" = the rule normalizer reads every token, so
# mode=llm skips the LLM call under LLM_NORMALIZER_RULE_FIRST; "forced" leaves a symbol the
# rules cannot speak, so mode=llm really calls the LLM.
TEXTS = [
    ('plain_en', 'plain', 'Thank you for calling, how can I help you today?'),
    ('plain_ms', 'plain', 'Selamat pagi, apa yang boleh saya bantu encik hari ini?'),
    ('covered_en', 'covered', 'Your balance is RM1,250.50 as of 15/3/2024 and the meeting is at 3pm.'),
    ('covered_ms', 'covered', 'Baki anda RM1,250.50 setakat 15/3/2024 dan mesyuarat pada pukul 3 petang.'),
    ('forced_en', 'forced', 'Meeting @ HQ at 9:30, agenda: Q3 P&L review and the 4.5/5 rating.'),
    ('forced_ms', 'forced', 'Baki RM50 & caj 6% dikenakan, rujuk Dept. #7 sebelum e.o.d.'),
    ('long_en', 'covered', 'Your order 4471 was shipped on 15/3/2024 and will arrive in 3 to 5 working days. '
                           'The total charged to your card was RM1,250.50, including 6% tax. '
                           'If you have any questions, call us at 03-1234 5678 between 9am and 5pm.'),
]
MODES = ['spoken', 'llm', 'rule']


async def rtt(session, url, n=6):
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            async with session.get(f'{url}/docs') as r:
                await r.read()
            out.append(time.perf_counter() - t0)
        except Exception:                                   # noqa: BLE001
            pass
    return out


async def normalize_once(session, url, text, mode):
    t0 = time.perf_counter()
    async with session.post(f'{url}/v1/audio/normalize',
                            json={'input': text, 'mode': mode, 'normalize_malaysian': False}) as r:
        d = await r.json()
    return time.perf_counter() - t0, d.get('output', '')


async def speech_once(session, url, text, mode, voice, playback, overlap, temperature):
    payload = {'input': text, 'voice': voice, 'model': 'TTS-model', 'response_format': 'pcm',
               'stream': True, 'stream_format': 'audio', 'playback_speed': playback,
               'playback_overlap_speed': overlap, 'mode': mode, 'temperature': temperature,
               'normalize_malaysian': False}
    t0 = time.perf_counter()
    ttfb = None
    nbytes = 0
    min_lead = None
    async with session.post(f'{url}/v1/audio/speech', json=payload) as r:
        r.raise_for_status()
        t_headers = time.perf_counter() - t0
        async for chunk in r.content.iter_any():
            if not chunk:
                continue
            now = time.perf_counter() - t0
            if ttfb is None:
                ttfb = now
            else:
                lead = nbytes / BYTES_PER_S - (now - ttfb)
                min_lead = lead if min_lead is None else min(min_lead, lead)
            nbytes += len(chunk)
    total = time.perf_counter() - t0
    audio_s = nbytes / BYTES_PER_S
    return {'t_headers': t_headers, 'ttfb': ttfb, 'total': total, 'audio_s': audio_s,
            'rtf': total / audio_s if audio_s else None, 'min_lead_s': min_lead}


def summarize(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {'n': len(vals), 'median': statistics.median(vals), 'mean': statistics.fmean(vals),
            'min': min(vals), 'max': max(vals)}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--voice', default='TM_English_Normal')
    ap.add_argument('--modes', default=','.join(MODES))
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--warmup', type=int, default=1)
    ap.add_argument('--playback', type=float, default=0.75)
    ap.add_argument('--overlap', type=float, default=0.2)
    ap.add_argument('--temperature', type=float, default=0.6)
    ap.add_argument('--timeout', type=float, default=180)
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    url = args.url.rstrip('/')
    modes = args.modes.split(',')
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    results = []
    async with aiohttp.ClientSession(timeout=timeout) as session:
        rtts = await rtt(session, url)
        print(f'network GET /docs: median {statistics.median(rtts) * 1000:.0f} ms over {len(rtts)} samples', flush=True)
        for _ in range(args.warmup):
            await speech_once(session, url, TEXTS[0][2], modes[0], args.voice, args.playback, args.overlap, args.temperature)
        for tid, kind, text in TEXTS:
            for mode in modes:
                for rep in range(args.reps):
                    try:
                        t_norm, norm_out = await normalize_once(session, url, text, mode)
                        row = await speech_once(session, url, text, mode, args.voice, args.playback, args.overlap, args.temperature)
                    except Exception as e:                  # noqa: BLE001
                        print(f'  {tid:<11} {mode:<7} rep {rep}: {type(e).__name__}: {str(e)[:80]}', flush=True)
                        continue
                    row.update({'id': tid, 'kind': kind, 'mode': mode, 'rep': rep, 't_norm': t_norm,
                                'text': text, 'normalized': norm_out})
                    results.append(row)
                    print(f'  {tid:<11} {mode:<7} rep {rep}  norm {t_norm * 1000:5.0f} ms  ttfb {row["ttfb"] * 1000:5.0f} ms  '
                          f'total {row["total"]:5.2f} s  audio {row["audio_s"]:5.2f} s  rtf {row["rtf"]:.2f}  lead {row["min_lead_s"]:+.2f}', flush=True)
    print('\n== median per text x mode ==')
    print(f'{"text":<12}{"mode":<8}{"norm ms":>9}{"ttfb ms":>9}{"total s":>9}{"audio s":>9}{"rtf":>7}{"lead s":>8}')
    table = {}
    for tid, kind, _ in TEXTS:
        for mode in modes:
            rs = [r for r in results if r['id'] == tid and r['mode'] == mode]
            if not rs:
                continue
            row = {k: summarize([r[k] for r in rs]) for k in ('t_norm', 'ttfb', 'total', 'audio_s', 'rtf', 'min_lead_s')}
            table[(tid, mode)] = row
            print(f'{tid:<12}{mode:<8}{row["t_norm"]["median"] * 1000:9.0f}{row["ttfb"]["median"] * 1000:9.0f}'
                  f'{row["total"]["median"]:9.2f}{row["audio_s"]["median"]:9.2f}{row["rtf"]["median"]:7.2f}{row["min_lead_s"]["median"]:+8.2f}')
    print('\n== median per kind x mode ==')
    for kind in ('plain', 'covered', 'forced'):
        for mode in modes:
            rs = [r for r in results if r['kind'] == kind and r['mode'] == mode]
            if rs:
                print(f'{kind:<9}{mode:<8} ttfb {statistics.median([r["ttfb"] for r in rs]) * 1000:5.0f} ms   '
                      f'norm {statistics.median([r["t_norm"] for r in rs]) * 1000:5.0f} ms   '
                      f'rtf {statistics.median([r["rtf"] for r in rs]):.2f}')
    if args.out:
        json.dump({'url_host_omitted': True, 'voice': args.voice, 'playback_speed': args.playback,
                   'overlap': args.overlap, 'temperature': args.temperature, 'reps': args.reps,
                   'rtt_docs_s': rtts, 'rows': results}, open(args.out, 'w'), indent=1, ensure_ascii=False)
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    asyncio.run(main())
