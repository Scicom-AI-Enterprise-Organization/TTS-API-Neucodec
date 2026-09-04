#!/usr/bin/env python3
"""Score every arm: UTMOSv2 naturalness MOS + f0-continuity ("broken pitch") metrics.

MOS   : UTMOSv2 service. Its TTA crop choice is stochastic (reps=1 spreads ~+-0.17 MOS
        on a bit-identical file), so reps is pushed to 16 -- the server batches the
        crops, so it costs ~210 ms instead of 105 ms. Scored at each arm's NATIVE rate,
        which is the honest comparison; the widecodec_24k arm is the bandwidth control.
pitch : app-local `audiocheck.py` (the repo owner's own gates). Every arm is resampled
        to a COMMON 16 kHz before f0 tracking so sample rate is not a variable -- f0
        here is <=400 Hz, far inside 16 kHz.
"""
import argparse, glob, json, os, sys, threading, time
import urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

import librosa
import numpy as np
import soundfile as sf

import audiocheck

ANALYSIS_SR = 16_000


def _multipart(path):
    """Minimal multipart/form-data body -- keeps this stdlib-only."""
    b = os.urandom(16).hex()
    with open(path, 'rb') as f:
        data = f.read()
    body = (f'--{b}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="{os.path.basename(path)}"\r\n'
            f'Content-Type: audio/wav\r\n\r\n').encode() + data + f'\r\n--{b}--\r\n'.encode()
    return body, f'multipart/form-data; boundary={b}'


def utmos(path, url, reps, retries=4):
    body, ctype = _multipart(path)
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(f'{url}/predict?reps={reps}', data=body,
                                         headers={'Content-Type': ctype})
            with urllib.request.urlopen(req, timeout=180) as r:
                return float(json.loads(r.read())['mos'])
        except Exception as e:
            last = e
            time.sleep(1.5 * (k + 1))
    print(f'[utmos] FAILED {path}: {last}', flush=True)
    return None


def pitch(path):
    x, sr = sf.read(path, dtype='float32', always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    xr = librosa.resample(x, orig_sr=sr, target_sr=ANALYSIS_SR) if sr != ANALYSIS_SR else x
    m, _, _ = audiocheck.analyse(xr.astype(np.float64), ANALYSIS_SR)
    m['native_sr'] = sr
    m['dur_s'] = round(len(x) / sr, 4)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm-dir', action='append', required=True,
                    help='name=/path/to/wavdir (repeatable)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--utmos-url', default='http://127.0.0.1:8300')
    ap.add_argument('--reps', type=int, default=16)
    ap.add_argument('--concurrency', type=int, default=4)
    a = ap.parse_args()

    arms = {}
    for spec in a.arm_dir:
        name, _, d = spec.partition('=')
        arms[name] = d

    jobs = []
    for name, d in arms.items():
        for p in sorted(glob.glob(os.path.join(d, '*.wav'))):
            jobs.append((name, os.path.splitext(os.path.basename(p))[0], p))
    print(f'[score] {len(jobs)} clips over {len(arms)} arms', flush=True)

    lock = threading.Lock()
    done = [0]

    def work(j):
        name, uid, p = j
        rec = {'arm': name, 'id': uid, 'mos': utmos(p, a.utmos_url, a.reps)}
        try:
            rec.update(pitch(p))
        except Exception as e:
            print(f'[pitch] FAILED {p}: {e}', flush=True)
        with lock:
            done[0] += 1
            if done[0] % 50 == 0:
                print(f'[score] {done[0]}/{len(jobs)}', flush=True)
        return rec

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        recs = list(ex.map(work, jobs))
    with open(a.out, 'w') as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'[score] done in {time.time()-t0:.0f}s -> {a.out}', flush=True)


if __name__ == '__main__':
    main()
