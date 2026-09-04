#!/usr/bin/env python3
"""Level-matched copies of an arm: pure gain, no distortion.

Why this arm exists. The raw NeuCodec decode peaks at a median of 1.10 and exceeds full
scale on 145/200 clips (the loudness-in-the-tokens effect CLAUDE.md documents). If the two
decoders render level differently, native-level MOS partly scores loudness rather than
quality -- and in production BOTH arms would be loudness-normalized by the stitcher
(STREAM_NORMALIZE) anyway, so matched level is the production-representative comparison.

Gain = min(gain to hit TARGET_RMS on the active frames, gain to put peak at 0.99), so it
is always a single scalar multiply: no clipping, no limiter, no spectral change. Anything
UTMOSv2 then reports is not a level artifact.
"""
import glob, os, sys
import numpy as np
import soundfile as sf

TARGET_RMS_DB = -20.0
PEAK_CEIL = 0.99
ACTIVE_REL_DB = 40.0      # frames within 40 dB of the loudest count as speech


def active_rms(x, sr):
    hop = max(1, int(sr * 0.01))
    win = max(hop, int(sr * 0.025))
    n = max(1, (len(x) - win) // hop + 1)
    r = np.array([np.sqrt(np.mean(x[i * hop:i * hop + win] ** 2) + 1e-12) for i in range(n)])
    if not len(r):
        return float(np.sqrt(np.mean(x ** 2) + 1e-12))
    thr = r.max() * (10 ** (-ACTIVE_REL_DB / 20.0))
    a = r[r >= thr]
    return float(np.sqrt(np.mean(a ** 2))) if len(a) else float(r.max())


src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
stats = []
for p in sorted(glob.glob(os.path.join(src, '*.wav'))):
    x, sr = sf.read(p, dtype='float32')
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    rms = active_rms(x, sr)
    peak = float(np.abs(x).max()) or 1e-9
    g = min((10 ** (TARGET_RMS_DB / 20.0)) / max(rms, 1e-9), PEAK_CEIL / peak)
    y = x * g
    sf.write(os.path.join(dst, os.path.basename(p)), y.astype(np.float32), sr, subtype='FLOAT')
    stats.append((g, peak, float(np.abs(y).max())))
g = np.array([s[0] for s in stats])
print(f'[norm] {len(stats)} files {src} -> {dst}  '
      f'gain median={np.median(g):.3f} min={g.min():.3f} max={g.max():.3f}  '
      f'peak_in median={np.median([s[1] for s in stats]):.3f}  '
      f'peak_out max={max(s[2] for s in stats):.4f}', flush=True)
