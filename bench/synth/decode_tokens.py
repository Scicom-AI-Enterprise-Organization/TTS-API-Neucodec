"""Stage 2: speech tokens -> 24 kHz wav with the vendored NeuCodec decoder.

One-shot decode of the whole token stream per utterance (no streaming stitcher):
the NeuCodec decoder is non-causal, so a single big window has no chunk-edge
envelope tilt at all -- the best-quality decode the codec can give.

Two outputs per clip:
  <slug>/NN.wav      PCM_16, loudness-normalized the way the serving app does it
                     (app/main.py normalize_chunk: active-RMS -> TARGET_RMS_DB,
                     clamped to +/-MAX_GAIN_DB, boost capped by peak headroom, then a
                     tanh soft-knee limiter). Level-matched across models, no clipping.
  <slug>/raw/NN.wav  FLOAT (32-bit) WAV of the untouched decoder output. Raw peaks run
                     past 1.0 on hot utterances, which is exactly why the normalized
                     set exists; float keeps them intact for scoring.
"""
import argparse, json, os, sys
import numpy as np
import soundfile as sf
import torch

# `app/neucodec` is the vendored decoder: repo root when this runs from bench/synth/,
# or next to the script when only the payload was rsynced to a box.
_here = os.path.dirname(os.path.abspath(__file__))
for _root in (os.path.abspath(os.path.join(_here, '..', '..')), _here):
    if os.path.isdir(os.path.join(_root, 'app', 'neucodec')):
        sys.path.insert(0, _root)
        break
from app.neucodec import NeuCodec

ap = argparse.ArgumentParser()
ap.add_argument('--tokens', nargs='+', required=True, help='tokens json from gen_tokens.py')
ap.add_argument('--outdir', required=True)
ap.add_argument('--device', default='cuda')
ap.add_argument('--target-rms-db', type=float, default=-16.0)
ap.add_argument('--max-gain-db', type=float, default=12.0)
a = ap.parse_args()

SR = 24000
LIMITER_KNEE = 0.85
LIMITER_DRIVE = 1.4


def normalize(y, target_db, max_gain_db):
    """One-shot form of app/main.py's per-utterance `normalize_chunk`.

    The serving app locks its gain after ~1 s of voiced audio, so a whole-utterance
    estimate is the same trim it converges to -- computed here over the full clip.
    """
    active = y[np.abs(y) > 10 ** (-50 / 20)]          # gate out silence (< -50 dBFS)
    if not len(active):
        return y, 0.0
    est_db = 10 * np.log10(float(np.mean(active.astype(np.float64) ** 2)))
    gain_db = float(np.clip(target_db - est_db, -max_gain_db, max_gain_db))
    peak = float(np.abs(y).max())
    if peak > 0:
        gain_db = min(gain_db, 20 * np.log10(LIMITER_DRIVE / peak))
    y = y * (10 ** (gain_db / 20))
    over = np.abs(y) > LIMITER_KNEE
    if np.any(over):
        y = np.where(
            over,
            np.sign(y) * (LIMITER_KNEE + (1.0 - LIMITER_KNEE)
                          * np.tanh((np.abs(y) - LIMITER_KNEE) / (1.0 - LIMITER_KNEE))),
            y,
        )
    return np.clip(y, -1.0, 1.0), gain_db


codec = NeuCodec.from_pretrained('neuphonic/neucodec').eval().to(a.device)

summary = []
for tf in a.tokens:
    d = json.load(open(tf))
    slug = os.path.splitext(os.path.basename(tf))[0]
    od = os.path.join(a.outdir, slug)
    os.makedirs(os.path.join(od, 'raw'), exist_ok=True)
    man = {'meta': {**d['meta'], 'sample_rate': SR, 'decode': 'one-shot NeuCodec',
                    'loudness': {'target_rms_db': a.target_rms_db,
                                 'max_gain_db': a.max_gain_db,
                                 'limiter_knee': LIMITER_KNEE,
                                 'limiter_drive': LIMITER_DRIVE}},
           'clips': []}
    for r in d['records']:
        ids = r['tokens']
        base = {k: v for k, v in r.items() if k != 'tokens'}
        name = f"{r['index']:02d}.wav"
        if not ids:
            man['clips'].append({**base, 'file': None, 'error': 'no speech tokens'})
            print(slug, r['index'], 'EMPTY')
            continue
        with torch.no_grad():
            y = codec.decode_code(torch.tensor(ids)[None, None].to(a.device))
        y = y[0, 0].float().cpu().numpy()
        raw_peak = float(np.max(np.abs(y)))
        raw_rms = float(np.sqrt(np.mean(y ** 2)))
        yn, gain_db = normalize(y, a.target_rms_db, a.max_gain_db)
        sf.write(os.path.join(od, name), yn, SR, subtype='PCM_16')
        sf.write(os.path.join(od, 'raw', name), y, SR, subtype='FLOAT')
        man['clips'].append({**base, 'file': name, 'raw_file': f'raw/{name}',
                             'duration_s': round(len(y) / SR, 3),
                             'raw_peak': round(raw_peak, 4),
                             'raw_rms_db': round(20 * np.log10(raw_rms + 1e-12), 2),
                             'gain_db': round(gain_db, 2),
                             'out_peak': round(float(np.max(np.abs(yn))), 4),
                             'out_rms_db': round(
                                 20 * np.log10(float(np.sqrt(np.mean(yn ** 2))) + 1e-12), 2)})
        print(slug, r['index'], name, round(len(y) / SR, 2), 's raw_peak',
              round(raw_peak, 3), 'gain', round(gain_db, 2), 'dB')
    with open(os.path.join(od, 'manifest.json'), 'w') as f:
        json.dump(man, f, indent=2)
    summary.append((slug, len(man['clips'])))
print('SUMMARY', summary)
print('DECODE_DONE')
