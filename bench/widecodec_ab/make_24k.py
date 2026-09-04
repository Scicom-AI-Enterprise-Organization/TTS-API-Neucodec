#!/usr/bin/env python3
"""Bandwidth control: WideCodec's 44.1 kHz output resampled to 24 kHz.

If WideCodec's MOS win survives here, it is decoder quality; if it evaporates, the win
was mostly the extra bandwidth above 12 kHz.
"""
import glob, os, sys
import librosa, soundfile as sf

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
n = 0
for p in sorted(glob.glob(os.path.join(src, '*.wav'))):
    x, sr = sf.read(p, dtype='float32')
    if x.ndim > 1:
        x = x.mean(axis=1)
    y = librosa.resample(x, orig_sr=sr, target_sr=24_000)
    sf.write(os.path.join(dst, os.path.basename(p)), y, 24_000, subtype='FLOAT')
    n += 1
print(f'[24k] {n} files {src} -> {dst}', flush=True)
