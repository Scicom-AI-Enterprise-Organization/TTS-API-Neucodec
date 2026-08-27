"""Objective join-discontinuity numbers for bench/context_ab.py output.

At every chunk join (where one request's audio ends and the next request's begins) compare
the last voiced ~300 ms before the join with the first voiced ~300 ms after it:

  dF0   |median F0 after - median F0 before|  in semitones   (pitch register reset)
  dRMS  |RMS after - RMS before|              in dB          (energy reset)

Reported per condition as the median over all joins of all takes. A speaker continuing a
thought moves a little across a phrase boundary; a cold restart jumps. C (one request) has
no request joins, so its "joins" are the same time points in the one-shot audio -- what a
natural phrase boundary measures under this metric -- and is the reference.

    uv run --with librosa --with numpy -- python bench/context_ab_metrics.py --out audio/context_ab
"""

import argparse
import json
import os
import wave

import numpy as np
import librosa

SR = 24000
WIN_S = 0.30          # analysis window on each side of the join
FMIN, FMAX = 60, 400


def read_pcm(path):
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def voiced_f0_and_rms(y):
    """Median F0 (Hz) over voiced frames and RMS (dB) over non-silent frames, or None."""
    if len(y) < int(0.05 * SR):
        return None, None
    f0, voiced, _ = librosa.pyin(y, fmin=FMIN, fmax=FMAX, sr=SR, frame_length=1024, hop_length=256)
    f0 = f0[voiced & np.isfinite(f0)]
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    rms = rms[rms > 10 ** (-45 / 20)]
    return (float(np.median(f0)) if len(f0) >= 4 else None,
            float(20 * np.log10(np.median(rms))) if len(rms) else None)


def join_metrics(y, t_join):
    n = int(t_join * SR)
    w = int(WIN_S * SR)
    # widen each side until it has enough voiced frames (skip the pause at the join)
    for k in (1, 2, 3):
        before = y[max(0, n - k * w): n]
        after = y[n: n + k * w]
        f0b, rb = voiced_f0_and_rms(before)
        f0a, ra = voiced_f0_and_rms(after)
        if None not in (f0b, f0a, rb, ra):
            return {
                'df0_semitones': round(abs(12 * np.log2(f0a / f0b)), 2),
                'drms_db': round(abs(ra - rb), 2),
                'f0_before': round(f0b, 1), 'f0_after': round(f0a, 1),
            }
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='audio/context_ab')
    args = ap.parse_args()
    with open(os.path.join(args.out, 'results.json')) as f:
        doc = json.load(f)

    per_cond = {'nocontext': [], 'context': [], 'continue': [], 'oneshot': []}
    per_take = []
    for res in doc['results']:
        d = os.path.join(args.out, f"{res['case']}_take{res['take']}")
        # join times from the chunked conditions; the one-shot is measured at the
        # *context* condition's join times scaled to its own duration (same text, so
        # roughly the same phrase boundaries)
        joins = {}
        for key in ('nocontext', 'context', 'continue'):
            t, js = 0.0, []
            for ch in res['conditions'][key]['chunks'][:-1]:
                t += ch['audio_s']
                js.append(t)
            joins[key] = js
        y_one = read_pcm(os.path.join(d, 'oneshot.wav'))
        ctx_total = res['conditions']['context']['audio_s']
        joins['oneshot'] = [t / ctx_total * (len(y_one) / SR) for t in joins['context']]

        take = {'case': res['case'], 'take': res['take'], 'joins': {}}
        for key in ('nocontext', 'context', 'continue', 'oneshot'):
            y = read_pcm(os.path.join(d, f'{key}.wav'))
            ms = [m for m in (join_metrics(y, t) for t in joins[key]) if m]
            take['joins'][key] = ms
            per_cond[key].extend(ms)
        per_take.append(take)
        print(f"{res['case']} take{res['take']}: " + '  '.join(
            f"{k}: dF0 {np.median([m['df0_semitones'] for m in take['joins'][k]]):.1f} st, "
            f"dRMS {np.median([m['drms_db'] for m in take['joins'][k]]):.1f} dB"
            for k in take['joins'] if take['joins'][k]))

    summary = {}
    for key, ms in per_cond.items():
        if ms:
            summary[key] = {
                'joins': len(ms),
                'df0_semitones_median': round(float(np.median([m['df0_semitones'] for m in ms])), 2),
                'df0_semitones_p90': round(float(np.percentile([m['df0_semitones'] for m in ms], 90)), 2),
                'drms_db_median': round(float(np.median([m['drms_db'] for m in ms])), 2),
                'drms_db_p90': round(float(np.percentile([m['drms_db'] for m in ms], 90)), 2),
            }
    print('\nsummary (median over all joins):')
    for k, s in summary.items():
        print(f"  {k:9s} n={s['joins']:2d}  dF0 {s['df0_semitones_median']:.2f} st (p90 {s['df0_semitones_p90']:.2f})"
              f"  dRMS {s['drms_db_median']:.2f} dB (p90 {s['drms_db_p90']:.2f})")
    with open(os.path.join(args.out, 'metrics.json'), 'w') as f:
        json.dump({'window_s': WIN_S, 'summary': summary, 'takes': per_take}, f, indent=2)
    print(f"wrote {args.out}/metrics.json")


if __name__ == '__main__':
    main()
