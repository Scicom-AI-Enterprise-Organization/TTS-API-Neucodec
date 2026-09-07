#!/usr/bin/env python3
"""Aggregate the interleave A/B into the tables of `bench/INTERLEAVE_AB.md`.

Every number is PAIRED: the three conditions render the same 80 paragraphs, so the
comparison that matters is the per-paragraph difference, not the difference of the
means. Medians of paired differences come with a bootstrap 95% CI and a win rate
(how many of the 80 paragraphs the condition wins), which together say whether an
effect is real at n=80 -- the sampling temperature is 0.6, and the mean of a noisy
per-utterance metric moves around a lot more than its paired median.

    python bench/interleave_ab/analyze.py --dir bench/results/interleave_ab > report.md
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
from pathlib import Path

CONDS = ['single', 'interleave', 'cold']
LABEL = {'single': 'A single', 'interleave': 'B interleave', 'cold': 'C cold'}


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(x, n=2):
    return '-' if x is None else f'{x:.{n}f}'


def pct(x, n=2):
    """A ratio as a percentage, surviving an all-None column."""
    return '-' if x is None else f'{100 * x:.{n}f}'


def boot_ci(diffs, reps=4000, seed=7):
    """Bootstrap 95% CI of the median paired difference."""
    d = [x for x in diffs if x is not None]
    if len(d) < 5:
        return (None, None)
    rng = random.Random(seed)
    ms = sorted(st.median([d[rng.randrange(len(d))] for _ in range(len(d))]) for _ in range(reps))
    return ms[int(0.025 * reps)], ms[int(0.975 * reps)]


def boot_mean_ci(d, reps=4000, seed=11):
    """Bootstrap 95% CI of the MEAN of `d` (absolute bounds, not offsets)."""
    d = [x for x in d if x is not None]
    if len(d) < 5:
        return (None, None)
    rng = random.Random(seed)
    ms = sorted(sum(d[rng.randrange(len(d))] for _ in range(len(d))) / len(d) for _ in range(reps))
    return ms[int(0.025 * reps)], ms[int(0.975 * reps)]


def paired(rows_by_cond, key, a, b, better='lower'):
    """(median diff a-b, CI, win rate of `a`, n) over the ids both conditions have."""
    ids = sorted(set(rows_by_cond[a]) & set(rows_by_cond[b]))
    d = []
    for i in ids:
        x, y = rows_by_cond[a][i].get(key), rows_by_cond[b][i].get(key)
        if x is not None and y is not None:
            d.append(x - y)
    if not d:
        return None, (None, None), None, 0
    wins = sum(1 for x in d if (x < 0 if better == 'lower' else x > 0))
    return st.median(d), boot_ci(d), wins / len(d), len(d)


def seam_stats(rec, which, key):
    return [s[key] for s in rec['seams'][which] if s.get(key) is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='bench/results/interleave_ab')
    ap.add_argument('--lang', default='', help='restrict to one language')
    a = ap.parse_args()
    d = Path(a.dir)

    ac = [json.loads(l) for l in open(d / 'acoustics.jsonl') if 'error' not in l]
    qu = [json.loads(l) for l in open(d / 'quality.jsonl')] if (d / 'quality.jsonl').exists() else []
    tok = [json.loads(l) for l in open(d / 'tokens.jsonl')]
    if a.lang:
        ac = [r for r in ac if r['lang'] == a.lang]
        qu = [r for r in qu if r['lang'] == a.lang]
        tok = [r for r in tok if r['lang'] == a.lang]

    q_by = {(r['id'], r['cond']): r for r in qu}
    # one flat record per (id, cond): acoustics + quality + per-seam summaries
    rows = {c: {} for c in CONDS}
    for r in ac:
        q = q_by.get((r['id'], r['cond']), {})
        rec = dict(r)
        rec.update({k: q.get(k) for k in ('cer', 'wer', 'mos', 'mos_levelmatched')})
        for which in ('join', 'interior'):
            for key in ('f0_st', 'f0_st_signed', 'reset', 'db', 'db_signed', 'gap_s'):
                # signed steps are averaged, not medianed: the question is whether the
                # steps at this clip's joins have a consistent DIRECTION, and a mean is
                # what cancels when they do not.
                rec[f'{which}_{key}'] = (mean(seam_stats(r, which, key))
                                         if ('signed' in key or key == 'reset')
                                         else med(seam_stats(r, which, key)))
        for key in ('f0_st', 'f0_st_signed', 'reset', 'db', 'db_signed', 'gap_s'):
            j, i = rec[f'join_{key}'], rec[f'interior_{key}']
            rec[f'excess_{key}'] = None if (j is None or i is None) else j - i
        rows[r['cond']][r['id']] = rec

    dur = {c: {i: r['dur_s'] for i, r in rows[c].items()} for c in CONDS}
    for c in ('interleave', 'cold'):
        for i, r in rows[c].items():
            if i in dur['single'] and dur['single'][i]:
                r['dur_ratio'] = r['dur_s'] / dur['single'][i]
    for i, r in rows['single'].items():
        r['dur_ratio'] = 1.0

    n = len(rows['single'])
    langs = sorted({r['lang'] for r in ac})
    print(f'<!-- n={n} paragraphs, langs={langs} -->\n')

    # ---------------------------------------------------------------- seams
    print('### Seam behaviour (median over paragraphs; joins vs interior points of the same clip)\n')
    print('| Condition | joins/clip | f0 step at join (st) | interior (st) | **f0 excess** | '
          'level step join (dB) | interior (dB) | **level excess** | gap at join (s) | interior (s) | **gap excess** |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for c in CONDS:
        rs = list(rows[c].values())
        nj = mean([len(r['seams']['join']) for r in rs])
        print(f'| {LABEL[c]} | {fmt(nj,1)} | '
              + ' | '.join(fmt(med([r[k] for r in rs]), 2 if 'gap' not in k else 3)
                           for k in ('join_f0_st', 'interior_f0_st', 'excess_f0_st',
                                     'join_db', 'interior_db', 'excess_db',
                                     'join_gap_s', 'interior_gap_s', 'excess_gap_s')) + ' |')
    print()

    print('### Direction of the step at a join, and what each chunk ends on (mean over paragraphs)\n')
    print('| Condition | signed f0 step at join (st) | same at interior points (st) | '
          'joins that step UP >1 st | signed level step at join (dB) | '
          'pitch movement over last 200 ms of a NON-FINAL chunk (st) | silence padding per boundary (s) |')
    print('|---|---|---|---|---|---|')
    for c in CONDS:
        rs = list(rows[c].values())
        cells = [fmt(mean([r.get(k) for r in rs]), 3)
                 for k in ('join_f0_st_signed', 'interior_f0_st_signed')]
        cells.append(pct(mean([r.get('join_reset') for r in rs]), 0) + '%')
        cells += [fmt(mean([r.get(k) for r in rs]), 3)
                  for k in ('join_db_signed', 'nonfinal_end_move_st', 'nonfinal_pad_s')]
        print(f'| {LABEL[c]} | ' + ' | '.join(cells) + ' |')
    print()

    # ---------------------------------------------------------------- global
    print('### Whole-paragraph prosody (median)\n')
    keys = [('dur_s', 'duration s', 2), ('dur_ratio', 'dur vs A', 3),
            ('words_per_s', 'words/s', 2), ('f0_med', 'f0 median Hz', 1),
            ('f0_iqr_st', 'f0 IQR st', 2), ('f0_slope_st_per_s', 'declination st/s', 3),
            ('active_db', 'active level dB', 1), ('sil_frac', 'silence frac', 3),
            ('seg_f0_spread_st', 'chunk f0 spread st', 2), ('seg_db_std', 'chunk level SD dB', 2)]
    print('| Condition | ' + ' | '.join(k[1] for k in keys) + ' |')
    print('|---' * (len(keys) + 1) + '|')
    for c in CONDS:
        rs = list(rows[c].values())
        print(f'| {LABEL[c]} | ' + ' | '.join(fmt(med([r.get(k) for r in rs]), p) for k, _l, p in keys) + ' |')
    print()

    # ---------------------------------------------------------------- quality
    if qu:
        print('### Intelligibility and naturalness\n')
        print('| Condition | CER % | WER % | UTMOSv2 | UTMOSv2 level-matched |')
        print('|---|---|---|---|---|')
        for c in CONDS:
            rs = list(rows[c].values())
            print(f'| {LABEL[c]} | {pct(med([r["cer"] for r in rs]))} | '
                  f'{pct(med([r["wer"] for r in rs]))} | '
                  f'{fmt(mean([r["mos"] for r in rs]),3)} | '
                  f'{fmt(mean([r["mos_levelmatched"] for r in rs]),3)} |')
        print()

    # ---------------------------------------------------------------- paired
    print('### Paired differences (same paragraph, same chunking)\n')
    print('| Metric | better | B-A median [95% CI] | B wins | C-A median [95% CI] | C wins | '
          'B-C median [95% CI] | B wins vs C |')
    print('|---|---|---|---|---|---|---|---|')
    metrics = [('excess_f0_st', 'lower', 3), ('excess_db', 'lower', 3), ('excess_gap_s', 'lower', 4),
               ('join_f0_st_signed', 'lower', 3), ('join_reset', 'lower', 3),
               ('nonfinal_pad_s', 'lower', 4), ('dur_ratio', 'lower', 4),
               ('seg_f0_spread_st', 'lower', 3), ('seg_db_std', 'lower', 3),
               ('f0_slope_st_per_s', 'lower', 4), ('cer', 'lower', 4),
               ('mos_levelmatched', 'higher', 3), ('mos', 'higher', 3)]
    for key, better, prec in metrics:
        cells = []
        for x, y in (('interleave', 'single'), ('cold', 'single'), ('interleave', 'cold')):
            m, (lo, hi), win, k = paired(rows, key, x, y, better)
            cells.append(f'{fmt(m,prec)} [{fmt(lo,prec)}, {fmt(hi,prec)}]')
            cells.append('-' if win is None else f'{100*win:.0f}% ({k})')
        print(f'| `{key}` | {better} | ' + ' | '.join(cells) + ' |')
    print()

    # -------------------------------------------- chunk-to-chunk (whole-chunk)
    # The seam probe reads 0.25 s on each side of a boundary, so it needs to know where
    # the boundary IS -- fine for B and C, inferred for A. This asks the same question
    # from whole-chunk medians instead: how does chunk N+1's register sit against chunk
    # N's? No boundary probing, so all three conditions are on equal footing.
    print('### Chunk N+1 against chunk N (whole-chunk medians)\n')
    print('| Condition | signed register step (st) | \\|register step\\| (st) | '
          'signed level step (dB) | \\|level step\\| (dB) | steps up >1 st | n |')
    print('|---|---|---|---|---|---|---|')
    for c in CONDS:
        f0d, dbd = [], []
        for r in rows[c].values():
            sg = r['segs']
            for k in range(len(sg) - 1):
                x, y = sg[k], sg[k + 1]
                if x['f0'] and y['f0']:
                    f0d.append(12 * math.log2(y['f0'] / x['f0']))
                if x['db'] is not None and y['db'] is not None:
                    dbd.append(y['db'] - x['db'])
        up = mean([1.0 if x > 1.0 else 0.0 for x in f0d])
        print(f'| {LABEL[c]} | {fmt(mean(f0d),3)} | {fmt(mean([abs(x) for x in f0d]),3)} | '
              f'{fmt(mean(dbd),3)} | {fmt(mean([abs(x) for x in dbd]),3)} | '
              f'{pct(up,0)}% | {len(f0d)} |')
    print()
    def chunk_steps(rec, key):
        sg = rec['segs']
        out = []
        for k in range(len(sg) - 1):
            x, y = sg[k][key], sg[k + 1][key]
            out.append(None if (x is None or y is None)
                       else (12 * math.log2(y / x) if key == 'f0' else y - x))
        return out

    print('| Metric | pair | signed diff [95% CI] | \\|step\\| diff [95% CI] | n |')
    print('|---|---|---|---|---|')
    for key, label, prec in (('f0', 'register step (st)', 3), ('db', 'level step (dB)', 3)):
        for x, y in (('interleave', 'cold'), ('interleave', 'single'), ('cold', 'single')):
            ids = sorted(set(rows[x]) & set(rows[y]))
            signed, absolute = [], []
            for i in ids:
                for b, c in zip(chunk_steps(rows[x][i], key), chunk_steps(rows[y][i], key)):
                    if b is None or c is None:
                        continue
                    signed.append(b - c)
                    absolute.append(abs(b) - abs(c))
            cells = []
            for d in (signed, absolute):
                lo, hi = boot_mean_ci(d)
                cells.append(f'{fmt(mean(d),prec)} [{fmt(lo,prec)}, {fmt(hi,prec)}]')
            tag = f'{LABEL[x].split()[0]}-{LABEL[y].split()[0]}'
            print(f'| {label} | {tag} | ' + ' | '.join(cells) + f' | {len(signed)} |')
    print()

    # ------------------------------------------------------- per-join (paired)
    # B and C chunk the same text at the same places and no chunk came back empty, so
    # join k of paragraph t exists in both -- 440 matched pairs instead of 80 paragraph
    # medians. This is the unit the effect lives in, and the only place there is enough
    # power to put a CI on a per-seam difference. `single` cannot join this table: its
    # boundaries are inferred, so its joins do not pair with anything.
    print('### Per-join, pooled over every chunk boundary\n')
    keys = [('f0_st_signed', 'signed f0 step (st)', 3), ('f0_st', '|f0 step| (st)', 3),
            ('reset', 'steps up >1 st', 3), ('db_signed', 'signed level step (dB)', 3),
            ('db', '|level step| (dB)', 3), ('gap_s', 'silence at the join (s)', 4)]
    per_join = {c: {} for c in CONDS}
    for c in CONDS:
        for i, r in rows[c].items():
            for k, sm in enumerate(r['seams']['join']):
                per_join[c][(i, k)] = sm
    print('| Metric | ' + ' | '.join(LABEL[c] for c in CONDS) + ' | B-C paired mean [95% CI] | B closer to A |')
    print('|---|---|---|---|---|---|')
    for key, label, prec in keys:
        cells = []
        for c in CONDS:
            xs = [sm[key] for sm in per_join[c].values() if sm.get(key) is not None]
            cells.append(fmt(mean(xs), prec))
        shared = sorted(set(per_join['interleave']) & set(per_join['cold']))
        d = [per_join['interleave'][j][key] - per_join['cold'][j][key] for j in shared
             if per_join['interleave'][j].get(key) is not None
             and per_join['cold'][j].get(key) is not None]
        m = mean(d)
        lo, hi = boot_mean_ci(d)
        ci = ('-' if m is None or lo is None
              else f'{m:+.{prec}f} [{lo:+.{prec}f}, {hi:+.{prec}f}]')
        # "closer to A" = |B - A| < |C - A| on the pooled means
        m_a, m_b, m_c = (mean([sm[key] for sm in per_join[c].values()
                               if sm.get(key) is not None]) for c in CONDS)
        closer = ('-' if None in (m_a, m_b, m_c) else
                  ('**yes**' if abs(m_b - m_a) < abs(m_c - m_a) else 'no'))
        print(f'| {label} | ' + ' | '.join(cells) + f' | {ci} | {closer} |')
    print(f'\n{len(sorted(set(per_join["interleave"]) & set(per_join["cold"])))} matched B/C joins; '
          f'A has {len(per_join["single"])} inferred ones.\n')

    # ---------------------------------------------------------------- LM / cost
    print('### LM behaviour and cost\n')
    print('| Condition | requests | tokens/word | collapsed chunks | finish=length | '
          'median prompt tok | median LM latency s |')
    print('|---|---|---|---|---|---|---|')
    for c in CONDS:
        reqs = [x for r in tok for x in r[c]]
        wc = [(x['n_tokens'], len((r['text'] if c == 'single' else r['chunks'][i]).split()))
              for r in tok for i, x in enumerate(r[c])]
        col = sum(x['collapsed'] for x in reqs)
        ln = sum(1 for x in reqs if x['finish_reason'] == 'length')
        print(f'| {LABEL[c]} | {len(reqs)} | '
              f'{fmt(med([t/max(1,w) for t, w in wc]),1)} | '
              f'{col} ({100*col/max(1,len(reqs)):.1f}%) | {ln} | '
              f'{fmt(med([x.get("prompt_tokens") for x in reqs]),0)} | '
              f'{fmt(med([x["latency_s"] for x in reqs]),3)} |')
    print()

    # ---------------------------------------------------------------- per language
    if len(langs) > 1 and not a.lang:
        print('### Per language (median excess at joins / chunk level SD / CER %)\n')
        print('| Lang | ' + ' | '.join(f'{LABEL[c]}' for c in CONDS) + ' |')
        print('|---|---|---|---|')
        for lang in langs:
            for name, key, prec, scale in (('f0 excess st', 'excess_f0_st', 3, 1),
                                           ('gap excess s', 'excess_gap_s', 4, 1),
                                           ('chunk level SD dB', 'seg_db_std', 3, 1),
                                           ('CER %', 'cer', 2, 100)):
                cells = []
                for c in CONDS:
                    m = med([r[key] for r in rows[c].values() if r['lang'] == lang])
                    cells.append('-' if m is None else fmt(m * scale, prec))
                print(f'| {lang} {name} | ' + ' | '.join(cells) + ' |')
        print()


if __name__ == '__main__':
    main()
