#!/usr/bin/env python3
"""
Decode-window A/B at temperature 0: does a smaller first window change the audio?

At temperature 0 the LM is greedy, so one text gives the SAME speech tokens every time;
the only thing that differs between a `playback_speed=50` request (the whole utterance
decoded in one window = the no-seam, no-tilt reference) and a streamed request is how the
non-causal NeuCodec decoder windowed those tokens. Everything measured here is therefore
the cost of the streaming config alone, with sampling noise removed:

  env_med / env_p95 / env_max   |streamed - one-shot| per 50 ms frame, dB (frames with the
                                one-shot above -50 dBFS). Envelope tilt/ripple of windowing.
  offset                        mean(streamed - one-shot) dB: whole-utterance gain offset
  onset                         same over the first 0.5 s: the first (smallest) window
  click                         max |2nd difference| of the streamed signal / same for the
                                one-shot: >~3 means a seam click the one-shot does not have
  cer                           character error rate vs the input text through the STT
                                endpoint (--stt-url), streamed vs one-shot, when given

Raw (stream_normalize=false) isolates the decoder; normalized (stream_normalize=true)
adds the loudness normalizer, whose gain estimate comes from the first ~1 s of audio and
so also depends on the first window. Both are reported.

Identical token streams are assumed when the two outputs have the same length (a token
is exactly 480 samples); a length mismatch means vLLM's greedy path diverged (it can,
under concurrent batches) and that pair is dropped.

    python bench/window_ab.py --url http://127.0.0.1:9087 --voice husein \
        --configs 1.5:0.2,1.0:0.2,0.5:0.2,0.5:0.1 --stt-url http://127.0.0.1:9092/audio/transcriptions
"""
import argparse
import asyncio
import io
import json
import os
import re
import statistics
import sys
import wave

import aiohttp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evalset import EVAL_SET  # noqa: E402

SR = 24000
FRAME = SR // 20          # 50 ms
ONE_SHOT_PLAYBACK = 50.0  # 2500-token first window: the whole utterance in one decode


async def fetch_pcm(session, url, text, voice, playback, overlap, normalize):
    payload = {
        'input': text, 'voice': voice, 'model': 'TTS-model', 'response_format': 'pcm',
        'stream': True, 'stream_format': 'audio', 'temperature': 0.0,
        'playback_speed': playback, 'playback_overlap_speed': overlap,
        'stream_normalize': normalize, 'mode': 'rule', 'normalize_malaysian': False,
    }
    async with session.post(url + '/v1/audio/speech', json=payload) as resp:
        if resp.status != 200:
            raise RuntimeError(f'HTTP {resp.status}: {(await resp.text())[:200]}')
        b = await resp.read()
    return np.frombuffer(b, dtype=np.int16).astype(np.float64) / 32768.0


def frame_db(y):
    n = len(y) // FRAME
    f = y[:n * FRAME].reshape(n, FRAME)
    return 10 * np.log10(np.mean(f ** 2, axis=1) + 1e-12)


def active_rms_db(y):
    a = y[np.abs(y) > 10 ** (-50 / 20)]
    return 20 * np.log10(np.sqrt(np.mean(a ** 2)) + 1e-12) if len(a) else -120.0


def click_metric(y):
    return float(np.max(np.abs(np.diff(y, n=2)))) if len(y) > 2 else 0.0


def compare(y, ref):
    yd, rd = frame_db(y), frame_db(ref)
    n = min(len(yd), len(rd))
    yd, rd = yd[:n], rd[:n]
    mask = rd > -50
    d = (yd - rd)[mask]
    onset = (yd - rd)[:10][mask[:10]]
    ad = np.abs(d)
    return {
        'env_med': float(np.median(ad)), 'env_p95': float(np.percentile(ad, 95)),
        'env_max': float(np.max(ad)), 'offset': float(np.mean(d)),
        'onset': float(np.mean(onset)) if len(onset) else float('nan'),
        'click': click_metric(y) / max(click_metric(ref), 1e-9),
        'rms_db': active_rms_db(y), 'ref_rms_db': active_rms_db(ref),
        'peak': float(np.max(np.abs(y))), 'ref_peak': float(np.max(np.abs(ref))),
    }


def to_wav(y):
    bio = io.BytesIO()
    with wave.open(bio, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
    return bio.getvalue()


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


async def transcribe(session, stt_url, y):
    form = aiohttp.FormData()
    form.add_field('file', to_wav(y), filename='a.wav', content_type='audio/wav')
    form.add_field('response_format', 'json')
    async with session.post(stt_url, data=form) as resp:
        if resp.status != 200:
            raise RuntimeError(f'STT HTTP {resp.status}: {(await resp.text())[:200]}')
        d = await resp.json()
    if isinstance(d, dict):
        if 'text' in d:
            return d['text']
        if 'segments' in d:
            return ' '.join(s.get('text', '') for s in d['segments'])
    return str(d)


def agg(rows, key, f=statistics.median):
    xs = [r[key] for r in rows if r.get(key) is not None and not np.isnan(r[key])]
    return f(xs) if xs else float('nan')


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:9091')
    ap.add_argument('--voice', default='husein')
    ap.add_argument('--ids', default='en_short_1,en_med_1,en_long_1,ms_short_1,ms_med_1,cs_1,cs_2')
    ap.add_argument('--configs', default='1.5:0.2,1.0:0.2,0.75:0.2,0.5:0.2,0.5:0.1',
                    help='playback_speed:overlap pairs')
    ap.add_argument('--stt-url', default='')
    ap.add_argument('--out', default='')
    ap.add_argument('--save-dir', default='', help='write the wavs here for listening')
    args = ap.parse_args()

    ids = set(args.ids.split(','))
    texts = [(i, t) for i, _v, t in EVAL_SET if i in ids]
    configs = [tuple(float(x) for x in c.split(':')) for c in args.configs.split(',')]
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    results = {f'{ps}:{ov}': {'raw': [], 'norm': []} for ps, ov in configs}
    ref_cer = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
        for tid, text in texts:
            ref_raw = await fetch_pcm(session, args.url, text, args.voice, ONE_SHOT_PLAYBACK, 0.2, False)
            ref_norm = await fetch_pcm(session, args.url, text, args.voice, ONE_SHOT_PLAYBACK, 0.2, True)
            if len(ref_raw) != len(ref_norm):
                print(f'{tid}: one-shot raw/norm token mismatch ({len(ref_raw)} vs {len(ref_norm)}), skipping text')
                continue
            if args.save_dir:
                open(os.path.join(args.save_dir, f'{tid}_oneshot.wav'), 'wb').write(to_wav(ref_norm))
            if args.stt_url:
                ref_cer.append({'id': tid, 'cer': cer(await transcribe(session, args.stt_url, ref_norm), text)})
            for ps, ov in configs:
                key = f'{ps}:{ov}'
                y_raw = await fetch_pcm(session, args.url, text, args.voice, ps, ov, False)
                y_norm = await fetch_pcm(session, args.url, text, args.voice, ps, ov, True)
                if len(y_raw) != len(ref_raw) or len(y_norm) != len(ref_raw):
                    print(f'{tid} {key}: token mismatch (len {len(y_raw)}/{len(y_norm)} vs {len(ref_raw)}), dropped')
                    continue
                r_raw = compare(y_raw, ref_raw)
                r_norm = compare(y_norm, ref_norm)
                if args.stt_url:
                    r_norm['cer'] = cer(await transcribe(session, args.stt_url, y_norm), text)
                if args.save_dir:
                    open(os.path.join(args.save_dir, f'{tid}_ps{ps}_ov{ov}.wav'), 'wb').write(to_wav(y_norm))
                for r in (r_raw, r_norm):
                    r['id'] = tid
                    r['n_tokens'] = len(y_raw) // 480
                results[key]['raw'].append(r_raw)
                results[key]['norm'].append(r_norm)
                print(f"{tid:<11} {key:<9} raw: med {r_raw['env_med']:.2f} p95 {r_raw['env_p95']:.2f} "
                      f"max {r_raw['env_max']:.2f} off {r_raw['offset']:+.2f} onset {r_raw['onset']:+.2f} "
                      f"click {r_raw['click']:.1f} | norm: med {r_norm['env_med']:.2f} p95 {r_norm['env_p95']:.2f} "
                      f"off {r_norm['offset']:+.2f} rms {r_norm['rms_db']:.1f} vs {r_norm['ref_rms_db']:.1f} "
                      f"peak {r_norm['peak']:.2f}"
                      + (f" cer {r_norm['cer']:.3f}" if 'cer' in r_norm else ''), flush=True)

    print('\n== per config (median over texts unless noted) ==')
    print(f"{'config':<10} {'n':>2} | raw: {'env_med':>7} {'env_p95':>7} {'env_max(max)':>12} {'offset':>7} "
          f"{'onset':>7} {'click(max)':>10} | norm: {'env_med':>7} {'offset':>7} {'onset':>7} "
          f"{'rms-ref':>7} {'peak(max)':>9} {'cer':>6}")
    for key, r in results.items():
        raw, norm = r['raw'], r['norm']
        if not raw:
            print(f'{key:<10} no valid pairs')
            continue
        rms_gap = statistics.median(x['rms_db'] - x['ref_rms_db'] for x in norm)
        print(f"{key:<10} {len(raw):>2} | raw: {agg(raw,'env_med'):7.2f} {agg(raw,'env_p95'):7.2f} "
              f"{agg(raw,'env_max',max):12.2f} {agg(raw,'offset'):+7.2f} {agg(raw,'onset'):+7.2f} "
              f"{agg(raw,'click',max):10.1f} | norm: {agg(norm,'env_med'):7.2f} {agg(norm,'offset'):+7.2f} "
              f"{agg(norm,'onset'):+7.2f} {rms_gap:+7.2f} {agg(norm,'peak',max):9.2f} "
              f"{agg(norm,'cer',statistics.mean):6.3f}")
    if ref_cer:
        print(f"one-shot   cer mean {statistics.mean(x['cer'] for x in ref_cer):.3f}")
    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'args': vars(args), 'results': results, 'ref_cer': ref_cer}, f, indent=1)


if __name__ == '__main__':
    asyncio.run(main())
