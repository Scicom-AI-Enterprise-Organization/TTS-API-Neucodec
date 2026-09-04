#!/usr/bin/env python3
"""
CER of the wavs bench/window_ab.py saved (--save-dir), through an OpenAI-compatible
/v1/audio/transcriptions endpoint (e.g. the Whisper vLLM engine), grouped by decode-window
config. Files are named `<evalset id>_oneshot.wav` and `<evalset id>_ps<p>_ov<o>.wav`; the
reference text is the evalset sentence. Because window_ab runs at temperature 0, every file
of one id carries the same speech tokens, so any CER difference between configs is the
decoder windowing alone.

    TTS_API_KEY=... python bench/cer_wavs.py --wav-dir /root/ttfb-test/ab_wavs \
        --stt-url http://127.0.0.1:9089/v1/audio/transcriptions --model whisper
"""
import argparse
import asyncio
import os
import re
import statistics
import sys
from collections import defaultdict

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evalset import EVAL_SET  # noqa: E402

TEXT = {i: t for i, _v, t in EVAL_SET}
NAME_RE = re.compile(r'^(?P<id>.+?)_(?P<cfg>oneshot|ps[\d.]+_ov[\d.]+)\.wav$')


def norm_text(s):
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', s.lower())).strip()


def cer(hyp, ref):
    hyp, ref = norm_text(hyp), norm_text(ref)
    if not ref:
        return 0.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(ref)


async def transcribe(session, url, model, path, language):
    form = aiohttp.FormData()
    form.add_field('file', open(path, 'rb'), filename=os.path.basename(path), content_type='audio/wav')
    form.add_field('model', model)
    form.add_field('response_format', 'json')
    if language:
        form.add_field('language', language)
    async with session.post(url, data=form) as resp:
        if resp.status != 200:
            raise RuntimeError(f'STT HTTP {resp.status}: {(await resp.text())[:200]}')
        d = await resp.json()
    return d.get('text', '') if isinstance(d, dict) else str(d)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wav-dir', required=True)
    ap.add_argument('--stt-url', default='http://127.0.0.1:9089/v1/audio/transcriptions')
    ap.add_argument('--model', required=True)
    ap.add_argument('--api-key', default=os.environ.get('TTS_API_KEY', ''))
    ap.add_argument('--language', default='', help='force a language; default lets the model detect')
    ap.add_argument('--reps', type=int, default=1, help='transcribe each file this many times (sampling noise)')
    args = ap.parse_args()

    headers = {'Authorization': f'Bearer {args.api_key}'} if args.api_key else {}
    files = sorted(f for f in os.listdir(args.wav_dir) if f.endswith('.wav'))
    per_cfg = defaultdict(list)
    transcripts = {}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600), headers=headers) as s:
        for f in files:
            m = NAME_RE.match(f)
            if not m or m['id'] not in TEXT:
                continue
            ref = TEXT[m['id']]
            vals = []
            for _ in range(args.reps):
                hyp = await transcribe(s, args.stt_url, args.model, os.path.join(args.wav_dir, f), args.language)
                vals.append(cer(hyp, ref))
            c = statistics.mean(vals)
            transcripts[(m['id'], m['cfg'])] = hyp
            per_cfg[m['cfg']].append((m['id'], c))
            print(f"{m['id']:<11} {m['cfg']:<14} cer {c:.3f}  | {hyp[:110]}")

    print('\n== CER by config (mean over ids; delta vs one-shot per id, mean) ==')
    oneshot = dict(per_cfg.get('oneshot', []))
    for cfg, rows in sorted(per_cfg.items(), key=lambda kv: (kv[0] != 'oneshot', kv[0])):
        cers = [c for _, c in rows]
        deltas = [c - oneshot[i] for i, c in rows if i in oneshot]
        same = sum(1 for i, _ in rows if norm_text(transcripts.get((i, cfg), '')) == norm_text(transcripts.get((i, 'oneshot'), '')))
        print(f"{cfg:<14} n={len(rows):>2} cer mean {statistics.mean(cers):.3f} median {statistics.median(cers):.3f} "
              f"| delta vs one-shot mean {statistics.mean(deltas) if deltas else 0:+.3f} | identical transcripts {same}/{len(rows)}")


if __name__ == '__main__':
    asyncio.run(main())
