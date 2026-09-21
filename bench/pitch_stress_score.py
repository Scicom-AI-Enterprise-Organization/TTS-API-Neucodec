"""Score bench/pitch_stress.py output for audible mid-utterance pitch/tone jumps.

The complaint this answers is "calm, even tone, then suddenly loud and excited", so the
headline is not a mean -- a mean of 438 joins is exactly what hid this. It is a RATE of
audible events and the worst one seen, per arm:

events     Inside one request, away from any chunk join: two adjacent 0.5 s windows of
           voiced speech where the level rises >= --db-thresh AND the register rises
           >= --st-thresh at the same time. Loud and excited, together, mid-utterance.
           Level or pitch alone moves all the time in normal speech; the conjunction is
           what a listener calls a change of tone.
joins      The same step measured AT a known chunk boundary (exact offsets from the
           generator, never inferred), via the interleave A/B's own probe so the numbers
           are comparable with bench/INTERLEAVE_AB.md.
opening    Level of the first second of voiced audio minus the level of the rest. This is
           where STREAM_NORMALIZE's pre-lock slew shows up: the gain settles over ~1 s,
           so a systematically quiet (or loud) opening is the stitcher, not the model.

Reading the arms: oneshot_raw is the model alone (M3); oneshot_norm - oneshot_raw is the
stitcher's gain (M2); chunked_norm - oneshot_norm is the chunk joins (M1);
chunked_interleave says what interleaving takes back.

    uv run --with librosa --with soundfile python bench/pitch_stress_score.py \
      --dir bench/results/pitch_stress --out bench/results/pitch_stress/scores.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'interleave_ab'))
from score import (                                          # noqa: E402  (shared probe)
    ANALYSIS_SR, HOP, MIN_PAUSE_S, SIL_REL_DB, _active_db, _frames, _gap_at, _probe,
    _silent_runs, _st, _voiced_median,
)

WIN_S = 0.5              # length of an event window
KEEPOUT_S = 0.6          # how far an event window must stay from a join or an edge
OPENING_S = 1.0          # the stretch STREAM_NORMALIZE spends settling its gain
# A window that is half pause reads as a huge level "jump" against a window that is not,
# which is a measurement artefact and not something a listener calls a change of tone.
# Both guards exist for that: most of the window has to be voiced, and the level is taken
# over the voiced frames only.
MIN_VOICED_FRAC = 0.7
MIN_JOIN_VOICED = 6      # voiced frames needed on each side of a join to trust its level


def _voiced_db(db, voiced, lo, hi, min_frames=1):
    """Level over the VOICED frames of a window -- immune to how much pause it contains."""
    d = db[max(0, lo):max(0, hi)][voiced[max(0, lo):max(0, hi)]]
    d = d[np.isfinite(d)]
    return float(np.mean(d)) if len(d) >= min_frames else None


def _windows(voiced, db, f0, sil, n_frames, blocked, win, step):
    """Non-overlapping voiced windows as (frame, register_st_ref, level_db)."""
    out = []
    for lo in range(0, n_frames - win, step):
        hi = lo + win
        if blocked[lo:hi].any():
            continue
        if voiced[lo:hi].mean() < MIN_VOICED_FRAC:
            continue
        f = _voiced_median(f0, voiced, lo, hi)
        d = _voiced_db(db, voiced, lo, hi, min_frames=int(MIN_VOICED_FRAC * win))
        if f is None or d is None:
            continue
        out.append((lo, f, d))
    return out


def score_one(rec, wav_dir, db_thresh, st_thresh):
    import librosa
    import soundfile as sf

    if rec.get('error'):
        return None
    y, sr = sf.read(os.path.join(wav_dir, rec['wav']), dtype='float32', always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if len(y) < sr // 2:
        return None
    ya = librosa.resample(y, orig_sr=sr, target_sr=ANALYSIS_SR) if sr != ANALYSIS_SR else y
    f0, voiced, db = _frames(ya.astype(np.float64))
    n = len(db)
    finite = db[np.isfinite(db)]
    if not len(finite):
        return None
    sil = db < (float(np.percentile(finite, 95)) - SIL_REL_DB)

    def to_frame(sample_at_sr):
        return int(round(sample_at_sr / rec['sr'] * ANALYSIS_SR / HOP))

    join_frames = [to_frame(j) for j in rec.get('joins', [])]

    # --- joins: the same probe the interleave A/B uses, at exactly known boundaries
    joins = []
    jw = int(0.25 * ANALYSIS_SR / HOP)           # same 0.25 s window the probe uses
    for fr in join_frames:
        if not 0 < fr < n:
            continue
        p = _probe(f0, voiced, db, sil, fr)
        p['frame'] = fr
        # Voiced-only level either side of the same gap, so a join that happens to sit
        # next to a pause cannot masquerade as a level jump.
        g = _gap_at(sil, fr, int(0.15 * ANALYSIS_SR / HOP))
        li, rj = (g[0], g[1] + 1) if g else (fr, fr)
        ld = _voiced_db(db, voiced, li - jw, li, MIN_JOIN_VOICED)
        rd = _voiced_db(db, voiced, rj, rj + jw, MIN_JOIN_VOICED)
        p['db_voiced_signed'] = None if (ld is None or rd is None) else round(rd - ld, 3)
        joins.append(p)

    # --- seams: every real pause in the audio, whether or not the boundaries are known.
    # Audio captured off a LiveKit track carries no join offsets, so this is the metric
    # the API-direct and through-LiveKit conditions can actually share -- and it is also
    # closer to what a listener hears, which is a step at a pause, not at a byte offset.
    seams = []
    edge = int(0.4 * ANALYSIS_SR / HOP)
    for i0, i1 in _silent_runs(sil, max(1, int(MIN_PAUSE_S * ANALYSIS_SR / HOP))):
        fr = (i0 + i1) // 2
        if fr < edge or fr > n - edge:
            continue
        q = _probe(f0, voiced, db, sil, fr)
        ld = _voiced_db(db, voiced, i0 - jw, i0, MIN_JOIN_VOICED)
        rd = _voiced_db(db, voiced, i1 + 1, i1 + 1 + jw, MIN_JOIN_VOICED)
        q['db_voiced_signed'] = None if (ld is None or rd is None) else round(rd - ld, 3)
        q['frame'] = fr
        seams.append(q)

    # --- events: inside a request, away from every join and both edges
    win = max(1, int(WIN_S * ANALYSIS_SR / HOP))
    keep = max(1, int(KEEPOUT_S * ANALYSIS_SR / HOP))
    blocked = np.zeros(n, dtype=bool)
    blocked[:keep] = blocked[max(0, n - keep):] = True
    for fr in join_frames:
        blocked[max(0, fr - keep): min(n, fr + keep)] = True
    wins = _windows(voiced, db, f0, sil, n, blocked, win, win)

    events, deltas = [], []
    for (fa, ra, da), (fb, rb, dbv) in zip(wins, wins[1:]):
        if fb != fa + win:                 # not adjacent (a join or a pause in between)
            continue
        d_st = _st(ra, rb)
        d_db = dbv - da
        deltas.append((d_st, d_db))
        if d_db >= db_thresh and d_st >= st_thresh:
            events.append({'t_s': round(fb * HOP / ANALYSIS_SR, 2),
                           'd_st': round(d_st, 2), 'd_db': round(d_db, 2)})

    # --- opening: STREAM_NORMALIZE's pre-lock slew, if any
    ofr = max(1, int(OPENING_S * ANALYSIS_SR / HOP))
    voiced_idx = np.flatnonzero(voiced & np.isfinite(db))
    opening_db = rest_db = None
    if len(voiced_idx) > ofr * 2:
        head, tail = voiced_idx[:ofr], voiced_idx[ofr:]
        opening_db = float(np.mean(db[head]))
        rest_db = float(np.mean(db[tail]))

    reg = _voiced_median(f0, voiced, 0, n, min_frames=20)
    return {
        'arm': rec['arm'], 'text_id': rec['text_id'], 'rep': rec['rep'],
        'concurrency': rec['concurrency'], 'wav': rec['wav'],
        'duration_s': rec['duration_s'], 'n_pieces': rec['n_pieces'],
        'joins': joins,
        'seams': seams,
        'n_event_windows': len(deltas),
        'events': events,
        'max_d_db': round(max((d for _, d in deltas), default=float('nan')), 2),
        'max_d_st': round(max((s for s, _ in deltas), default=float('nan')), 2),
        'opening_db': None if opening_db is None else round(opening_db, 2),
        'opening_minus_rest_db': (None if opening_db is None
                                  else round(opening_db - rest_db, 2)),
        'register_hz': None if reg is None else round(reg, 1),
        'level_db': round(_active_db(db) or float('nan'), 2),
    }


def _work(a):
    try:
        return score_one(*a)
    except Exception as e:                       # one bad wav must not lose the run
        return {'error': f'{type(e).__name__}: {e}', 'wav': a[0].get('wav')}


def _agg(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None
    vals = sorted(vals)
    return {'n': len(vals), 'mean': round(st.mean(vals), 3),
            'p95': round(vals[min(len(vals) - 1, int(0.95 * len(vals)))], 3),
            'max': round(vals[-1], 3)}


def summarize(rows, db_thresh, st_thresh):
    out = {}
    arms = sorted({r['arm'] for r in rows if r and 'arm' in r})
    for arm in arms:
        rs = [r for r in rows if r and r.get('arm') == arm]
        n_ev = sum(len(r['events']) for r in rs)
        n_win = sum(r['n_event_windows'] for r in rs)
        jl = [j for r in rs for j in r['joins']]
        sl = [j for r in rs for j in r.get('seams', [])]
        worst = max((e for r in rs for e in r['events']),
                    key=lambda e: (e['d_db'], e['d_st']), default=None)
        out[arm] = {
            'utterances': len(rs),
            'duration_s': round(sum(r['duration_s'] for r in rs), 1),
            'events': {
                'n': n_ev, 'windows': n_win,
                'per_utterance': round(n_ev / max(1, len(rs)), 3),
                'utterances_with_one': sum(1 for r in rs if r['events']),
                'worst': worst,
            },
            'within_request_step': {
                'd_db': _agg([r['max_d_db'] for r in rs]),
                'd_st': _agg([r['max_d_st'] for r in rs]),
            },
            'joins': {
                'n': len(jl),
                'db_signed': _agg([j['db_signed'] for j in jl]),
                'db_voiced_signed': _agg([j.get('db_voiced_signed') for j in jl]),
                'db_abs': _agg([j['db'] for j in jl]),
                'st_signed': _agg([j['f0_st_signed'] for j in jl]),
                'st_abs': _agg([j['f0_st'] for j in jl]),
                'reset_rate': (round(st.mean([j['reset'] for j in jl if j['reset'] is not None]), 3)
                               if any(j['reset'] is not None for j in jl) else None),
                'gap_s': _agg([j['gap_s'] for j in jl]),
            },
            'seams': {
                'n': len(sl),
                'db_signed': _agg([j['db_signed'] for j in sl]),
                'db_voiced_signed': _agg([j.get('db_voiced_signed') for j in sl]),
                'st_signed': _agg([j['f0_st_signed'] for j in sl]),
                'st_abs': _agg([j['f0_st'] for j in sl]),
                'reset_rate': (round(st.mean([j['reset'] for j in sl if j['reset'] is not None]), 3)
                               if any(j['reset'] is not None for j in sl) else None),
            },
            'opening_minus_rest_db': _agg([r['opening_minus_rest_db'] for r in rs]),
            'level_db': _agg([r['level_db'] for r in rs]),
        }
    return {'thresholds': {'d_db': db_thresh, 'd_st': st_thresh,
                           'window_s': WIN_S, 'keepout_s': KEEPOUT_S},
            'arms': out}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dir', required=True, help='pitch_stress.py --out directory')
    p.add_argument('--out', default=None)
    p.add_argument('--db-thresh', type=float, default=3.0)
    p.add_argument('--st-thresh', type=float, default=1.5)
    p.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 4) - 1))
    a = p.parse_args()

    with open(os.path.join(a.dir, 'records.jsonl')) as f:
        recs = [json.loads(l) for l in f]
    recs = [r for r in recs if not r.get('error')]
    print(f'scoring {len(recs)} wavs with {a.jobs} workers…')

    args = [(r, a.dir, a.db_thresh, a.st_thresh) for r in recs]
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        rows = list(ex.map(_work, args, chunksize=4))
    bad = [r for r in rows if r and r.get('error')]
    rows = [r for r in rows if r and not r.get('error')]
    if bad:
        print(f'{len(bad)} wavs failed to score, e.g. {bad[0]}')

    summary = summarize(rows, a.db_thresh, a.st_thresh)
    out = a.out or os.path.join(a.dir, 'scores.json')
    with open(out, 'w') as f:
        json.dump({'summary': summary, 'rows': rows}, f)

    print(f'\nthresholds: >= {a.db_thresh} dB AND >= {a.st_thresh} st between adjacent '
          f'{WIN_S}s windows, inside one request\n')
    hdr = (f'{"arm":<20}{"utts":>5}{"evt/utt":>9}{"utts hit":>9}'
           f'{"join dB":>9}{"join st":>9}{"reset":>7}{"open-rest dB":>14}')
    print(hdr); print('-' * len(hdr))
    for arm, s in summary['arms'].items():
        j, o = s['joins'], s['opening_minus_rest_db']
        print(f'{arm:<20}{s["utterances"]:>5}{s["events"]["per_utterance"]:>9.2f}'
              f'{s["events"]["utterances_with_one"]:>9}'
              f'{(j["db_signed"]["mean"] if j["db_signed"] else float("nan")):>9.2f}'
              f'{(j["st_signed"]["mean"] if j["st_signed"] else float("nan")):>9.2f}'
              f'{(j["reset_rate"] if j["reset_rate"] is not None else float("nan")):>7.2f}'
              f'{(o["mean"] if o else float("nan")):>14.2f}')
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
