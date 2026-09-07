#!/usr/bin/env python3
"""Perceptual + intelligibility scoring of the interleave A/B wavs.

CER/WER  through the box's Whisper vLLM engine (:9089) against the corpus text --
         catches the failure the acoustic metrics cannot see: a chunk that renders the
         wrong words, or renders nothing.
MOS      UTMOSv2 (:8300). Its TTA crops are stochastic (reps=1 spreads +-0.17 MOS on a
         bit-identical file), so `--reps 16`. Scored twice: as decoded, and after every
         clip is matched to a common active RMS -- UTMOSv2 mildly prefers louder audio
         and the three conditions do not come out at the same level, so the raw number
         alone cannot separate "more natural" from "louder".

    TTS_API_KEY=... python bench/interleave_ab/quality.py --wav-dir ... --out ...
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_DB = -23.0


def norm_text(s: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', s.lower())).strip()


def _lev(a: list, b: list) -> int:
    prev = list(range(len(a) + 1))
    for i, bc in enumerate(b, 1):
        cur = [i]
        for j, ac in enumerate(a, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ac != bc)))
        prev = cur
    return prev[-1]


def cer_wer(hyp: str, ref: str) -> tuple[float, float]:
    h, r = norm_text(hyp), norm_text(ref)
    if not r:
        return 0.0, 0.0
    return (_lev(list(h.replace(' ', '')), list(r.replace(' ', ''))) / max(1, len(r.replace(' ', ''))),
            _lev(h.split(), r.split()) / max(1, len(r.split())))


def transcribe(path, url, model, key, language, retries=3):
    import requests
    last = None
    for k in range(retries):
        try:
            with open(path, 'rb') as fh:
                r = requests.post(
                    url, timeout=300,
                    headers={'Authorization': f'Bearer {key}'} if key else {},
                    files={'file': (os.path.basename(path), fh, 'audio/wav')},
                    data={'model': model, 'response_format': 'json', 'language': language,
                          'temperature': '0'})
            if r.status_code != 200:
                raise RuntimeError(f'HTTP {r.status_code}: {r.text[:200]}')
            return r.json().get('text', '')
        except Exception as e:
            last = e
            time.sleep(1.5 * (k + 1))
    print(f'[stt] FAILED {path}: {last}', flush=True)
    return None


def utmos(path, url, reps, retries=4):
    import requests
    last = None
    for k in range(retries):
        try:
            with open(path, 'rb') as fh:
                r = requests.post(f'{url}/predict?reps={reps}', timeout=300,
                                  files={'file': (os.path.basename(path), fh, 'audio/wav')})
            if r.status_code != 200:
                raise RuntimeError(f'HTTP {r.status_code}: {r.text[:200]}')
            return float(r.json()['mos'])
        except Exception as e:
            last = e
            time.sleep(1.5 * (k + 1))
    print(f'[utmos] FAILED {path}: {last}', flush=True)
    return None


def level_match(src: str, dst: str) -> None:
    """Write `src` at a common active RMS so MOS cannot be won on loudness alone."""
    y, sr = sf.read(src, dtype='float32', always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    f = np.abs(y)
    active = y[f >= np.percentile(f, 60)]          # the speech, not the pauses
    rms = float(np.sqrt(np.mean(active ** 2))) if len(active) else 0.0
    g = 10 ** (TARGET_DB / 20) / rms if rms > 1e-6 else 1.0
    sf.write(dst, (y * g).astype(np.float32), sr, subtype='FLOAT')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wav-dir', default='bench/results/interleave_ab/wav')
    ap.add_argument('--out', default='bench/results/interleave_ab/quality.jsonl')
    ap.add_argument('--stt-url', default='http://127.0.0.1:9089/v1/audio/transcriptions')
    ap.add_argument('--stt-model', default='scicom-ai-enterprise/whisper-large-v3-turbo-2025-09-09')
    ap.add_argument('--utmos-url', default='http://127.0.0.1:8300')
    ap.add_argument('--reps', type=int, default=16)
    ap.add_argument('--concurrency', type=int, default=4)
    ap.add_argument('--norm-dir', default='')
    a = ap.parse_args()
    key = os.environ.get('TTS_API_KEY', '')

    index = json.loads(Path(a.wav_dir, 'index.json').read_text())
    norm_dir = Path(a.norm_dir or Path(a.wav_dir).parent / 'wav_norm')
    norm_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def one(rec):
        p = os.path.join(a.wav_dir, rec['wav'])
        pn = str(norm_dir / rec['wav'])
        level_match(p, pn)
        hyp = transcribe(p, a.stt_url, a.stt_model, key, rec['lang'])
        c, w = cer_wer(hyp, rec['text']) if hyp is not None else (None, None)
        return {'id': rec['id'], 'lang': rec['lang'], 'cond': rec['cond'],
                'hyp': hyp, 'cer': c, 'wer': w,
                'mos': utmos(p, a.utmos_url, a.reps),
                'mos_levelmatched': utmos(pn, a.utmos_url, a.reps)}

    with ThreadPoolExecutor(max_workers=a.concurrency) as ex, open(a.out, 'w') as f:
        for i, r in enumerate(ex.map(one, index)):
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            if (i + 1) % 30 == 0:
                print(f'[quality] {i+1}/{len(index)} {time.time()-t0:.0f}s', flush=True)
    print(f'[quality] wrote {a.out} in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
