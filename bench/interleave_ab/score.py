#!/usr/bin/env python3
"""Acoustic metrics for the interleave A/B: what happens where chunk N+1 meets chunk N.

That join is the whole reason interleaving exists. In LiveKit the reply is synthesized
utterance by utterance, and the request for N+1 carries no trace of N -- so the LM
picks a fresh pitch register, a fresh pace and a fresh energy for it, and the transition
sounds awkward however good each chunk is on its own. These metrics look at exactly that
transition, three ways:

seams      For every chunk boundary, the pitch step and level step from the speech
           immediately BEFORE it to the speech immediately AFTER it, plus the length of
           the silence sitting in between. Measured both as |step| (how far the prosody
           moved) and SIGNED (which way) -- the sign is the diagnostic: an utterance
           rendered as if it stood alone ends on a terminal fall and the next one opens
           in its own starting register, so cold chunking steps consistently UP, while
           continuous speech drifts gently DOWN across a phrase boundary. |step| alone
           cannot tell a register reset from ordinary intonation.
           Every probe is anchored the same way in all three conditions: the silent run
           straddling the boundary is located first, and the two windows are taken from
           its edges outward. Without that, `single` (whose boundaries are inferred) is
           probed at the start of a pause and the chunked conditions in the middle of
           one, and the level step is not comparable between them.
           The same probe is run at INTERIOR points -- 0.5 s-spaced, kept away from any
           boundary -- giving the within-utterance noise floor of that same audio;
           `excess = joins - interior` is what survives the fact that some sentences
           simply move around more than others.
trajectory Per-chunk register and level and their spread across the paragraph; the
           declination slope over the whole clip; and for every NON-FINAL chunk the
           pitch movement over its last 200 ms and the silence padding at its ends --
           a chunk with more text after it should not be signing off.
global     Duration, speaking rate, register, range, level.

`single` has no seams, so it is given VIRTUAL ones: the boundary is estimated by
character proportion (speech rate inside one utterance is uniform enough -- the same
assumption `trim_turn_tail` makes in app/interleave.py) and then snapped to the nearest
silence within +-0.5 s, because a chunk boundary is a punctuation boundary and those
carry a pause. Its row is the floor: what these metrics read at a transition no listener
can hear.

    python bench/interleave_ab/score.py --wav-dir ... --out ... --jobs 6
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ANALYSIS_SR = 16_000
HOP = 160                       # 10 ms
FRAME = 1024
FMIN, FMAX = 65.0, 400.0
SEAM_W = 0.25                   # s of speech looked at on each side of a boundary
GAP_SEARCH = 0.15               # s around a boundary in which a silent frame counts as its gap
SNAP = 0.5                      # s within which `single`'s inferred boundary snaps to silence
MIN_PAUSE_S = 0.04              # a silent run shorter than this is a stop closure, not a pause
INTERIOR_STEP = 0.5             # s between interior probe points
INTERIOR_KEEPOUT = 0.6          # s an interior point must stay away from a boundary/edge
SIL_REL_DB = 40.0               # a frame this far under the clip's loud level is silence
RESET_ST = 1.0                  # a signed pitch step above this counts as a register reset
TAIL_S = 0.20                   # s at the end of a chunk that carry its closing intonation


def _frames(y: np.ndarray):
    """f0 (Hz, nan where unvoiced), voiced flag, and frame level in dB."""
    import librosa
    f0, voiced, _ = librosa.pyin(y, fmin=FMIN, fmax=FMAX, sr=ANALYSIS_SR,
                                 frame_length=FRAME, hop_length=HOP, center=True)
    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP, center=True)[0]
    db = 20.0 * np.log10(np.maximum(rms, 1e-9))
    n = min(len(f0), len(db))
    return f0[:n], np.nan_to_num(voiced[:n], nan=False).astype(bool), db[:n]


def _st(a: float, b: float) -> float:
    return 12.0 * math.log2(b / a)


def _voiced_median(f0, voiced, lo, hi, min_frames=8):
    lo, hi = max(0, lo), max(0, hi)
    v = f0[lo:hi][voiced[lo:hi]]
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) >= min_frames else None


def _active_db(db, lo=None, hi=None):
    """Level of the loud half of the window -- the speech in it, not the pauses."""
    d = db if lo is None else db[max(0, lo):max(0, hi)]
    d = d[np.isfinite(d)]
    if not len(d):
        return None
    return float(np.mean(d[d >= np.percentile(d, 50)]))


def _gap_at(sil, frame, search):
    """The contiguous silent run straddling `frame`, as (first, last) frames, or None."""
    lo, hi = max(0, frame - search), min(len(sil), frame + search)
    if hi <= lo or not sil[lo:hi].any():
        return None
    seed = lo + int(np.argmax(sil[lo:hi]))
    i = j = seed
    while i > 0 and sil[i - 1]:
        i -= 1
    while j + 1 < len(sil) and sil[j + 1]:
        j += 1
    return i, j


def _silent_runs(sil, min_len):
    """[(first, last)] of every silent run at least `min_len` frames long."""
    out, i, n = [], 0, len(sil)
    while i < n:
        if not sil[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and sil[j + 1]:
            j += 1
        if j - i + 1 >= min_len:
            out.append((i, j))
        i = j + 1
    return out


def _probe(f0, voiced, db, sil, frame):
    """Pitch step, level step and gap across one boundary.

    The two windows start at the EDGES of the silence straddling the boundary, so what
    is compared is the speech before the pause against the speech after it -- the same
    geometry whether the pause is 20 ms or 300 ms, and whether the boundary was known
    exactly (chunked conditions) or inferred (`single`).
    """
    w = int(SEAM_W * ANALYSIS_SR / HOP)
    g = _gap_at(sil, frame, int(GAP_SEARCH * ANALYSIS_SR / HOP))
    if g is None:
        li, rj, gap = frame, frame, 0.0
    else:
        li, rj = g[0], g[1] + 1
        gap = (g[1] - g[0] + 1) * HOP / ANALYSIS_SR

    lf = _voiced_median(f0, voiced, li - w, li)
    rf = _voiced_median(f0, voiced, rj, rj + w)
    ld = _active_db(db, li - w, li)
    rd = _active_db(db, rj, rj + w)
    step = _st(lf, rf) if (lf and rf) else None
    dlev = (rd - ld) if (ld is not None and rd is not None) else None
    return {
        'f0_st': abs(step) if step is not None else None,
        'f0_st_signed': step,
        'reset': None if step is None else float(step > RESET_ST),
        'db': abs(dlev) if dlev is not None else None,
        'db_signed': dlev,
        'gap_s': gap,
    }


def score_one(rec, wav_dir):
    import librosa
    import soundfile as sf

    y, sr = sf.read(os.path.join(wav_dir, rec['wav']), dtype='float32', always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    ya = librosa.resample(y, orig_sr=sr, target_sr=ANALYSIS_SR) if sr != ANALYSIS_SR else y
    f0, voiced, db = _frames(ya.astype(np.float64))
    nfr = len(db)
    dur = len(y) / sr
    sil_floor = float(np.percentile(db[np.isfinite(db)], 95)) - SIL_REL_DB
    sil = db < sil_floor

    def to_frame(sample_at_sr):
        return int(round(sample_at_sr / sr * ANALYSIS_SR / HOP))

    if rec['cond'] == 'single':
        w = [len(c) for c in rec['chunks']]
        cum, tot = np.cumsum(w[:-1]), float(sum(w))
        snap = int(SNAP * ANALYSIS_SR / HOP)
        minrun = max(1, int(MIN_PAUSE_S * ANALYSIS_SR / HOP))
        runs = _silent_runs(sil, minrun)
        joins = []
        for c in cum:
            f = min(max(to_frame(c / tot * len(y)), 0), nfr - 1)
            # prefer a real pause near the estimate; a plosive closure is also "silence"
            # by the level floor, and snapping onto one puts the probe inside a word.
            near = [r for r in runs if abs((r[0] + r[1]) // 2 - f) <= snap]
            if near:
                i, j = min(near, key=lambda r: abs((r[0] + r[1]) // 2 - f))
                joins.append((i + j) // 2)
            else:
                joins.append(f)
    else:
        joins = [to_frame(s) for s in rec['joins']]
    joins = [j for j in joins if 0 < j < nfr]

    keep = int(INTERIOR_KEEPOUT * ANALYSIS_SR / HOP)
    step = int(INTERIOR_STEP * ANALYSIS_SR / HOP)
    interior = [p for p in range(keep, max(keep + 1, nfr - keep), step)
                if all(abs(p - j) >= keep for j in joins)]

    seam_j = [_probe(f0, voiced, db, sil, j) for j in joins]
    seam_i = [_probe(f0, voiced, db, sil, p) for p in interior]

    # per-chunk trajectory over the same boundaries the seams were measured at
    bounds = [0] + list(joins) + [nfr]
    tail = int(TAIL_S * ANALYSIS_SR / HOP)
    segs = []
    for k in range(len(bounds) - 1):
        lo, hi = bounds[k], bounds[k + 1]
        if hi - lo < 5:
            continue
        smed = _voiced_median(f0, voiced, lo, hi, min_frames=5)
        send = _voiced_median(f0, voiced, max(lo, hi - tail), hi, min_frames=3)
        lead = int(np.argmin(sil[lo:hi])) if sil[lo:hi].any() else 0
        rev = sil[lo:hi][::-1]
        trail = int(np.argmin(rev)) if rev.any() else 0
        segs.append({'f0': smed, 'db': _active_db(db, lo, hi),
                     # positive = the chunk ends ABOVE its own register (continuation
                     # intonation), negative = it ends on a fall (it signed off)
                     'end_move_st': (_st(smed, send) if (smed and send) else None),
                     'lead_sil_s': lead * HOP / ANALYSIS_SR,
                     'tail_sil_s': trail * HOP / ANALYSIS_SR,
                     'dur_s': (hi - lo) * HOP / ANALYSIS_SR,
                     'words': len(rec['chunks'][k].split()) if k < len(rec['chunks']) else None})

    v = f0[voiced & np.isfinite(f0)]
    f0_med = float(np.median(v)) if len(v) > 20 else None
    f0_iqr_st = (_st(float(np.percentile(v, 25)), float(np.percentile(v, 75)))
                 if len(v) > 20 else None)
    slope = None
    if len(v) > 50:
        t = np.flatnonzero(voiced & np.isfinite(f0)) * HOP / ANALYSIS_SR
        slope = float(np.polyfit(t, 12 * np.log2(v / np.median(v)), 1)[0])

    sf0 = [s['f0'] for s in segs if s['f0']]
    sdb = [s['db'] for s in segs if s['db'] is not None]
    nonfinal = segs[:-1]
    move = [s['end_move_st'] for s in nonfinal if s['end_move_st'] is not None]
    pad = [s['tail_sil_s'] for s in nonfinal] + [s['lead_sil_s'] for s in segs[1:]]
    return {
        'id': rec['id'], 'lang': rec['lang'], 'cond': rec['cond'],
        'dur_s': round(dur, 4), 'n_tokens': rec['n_tokens'],
        'words': len(rec['text'].split()),
        'words_per_s': round(len(rec['text'].split()) / dur, 4) if dur else None,
        'f0_med': f0_med, 'f0_iqr_st': f0_iqr_st, 'f0_slope_st_per_s': slope,
        'active_db': _active_db(db),
        'voiced_frac': float(voiced.mean()),
        'sil_frac': float(sil.mean()),
        'seg_f0_spread_st': (_st(min(sf0), max(sf0)) if len(sf0) > 1 else None),
        'seg_db_std': (float(np.std(sdb)) if len(sdb) > 1 else None),
        'nonfinal_end_move_st': (float(np.mean(move)) if move else None),
        'nonfinal_pad_s': (float(np.mean(pad)) if pad else None),
        'segs': segs,
        'seams': {'join': seam_j, 'interior': seam_i},
    }


def _work(args):
    rec, wav_dir = args
    try:
        return score_one(rec, wav_dir)
    except Exception as e:
        return {'id': rec['id'], 'cond': rec['cond'], 'error': f'{type(e).__name__}: {e}'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wav-dir', default='bench/results/interleave_ab/wav')
    ap.add_argument('--out', default='bench/results/interleave_ab/acoustics.jsonl')
    ap.add_argument('--jobs', type=int, default=6)
    a = ap.parse_args()

    index = json.loads(Path(a.wav_dir, 'index.json').read_text())
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex, open(a.out, 'w') as f:
        for i, r in enumerate(ex.map(_work, [(rec, a.wav_dir) for rec in index], chunksize=1)):
            f.write(json.dumps(r) + '\n')
            if (i + 1) % 30 == 0:
                print(f'[score] {i+1}/{len(index)} {time.time()-t0:.0f}s', flush=True)
    print(f'[score] wrote {a.out} in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
