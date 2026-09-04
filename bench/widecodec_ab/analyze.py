#!/usr/bin/env python3
"""Paired analysis of the codec arms. Same tokens per id => every comparison is paired."""
import json, math, sys, collections

RES = sys.argv[1] if len(sys.argv) > 1 else 'results.jsonl'
TOK = sys.argv[2] if len(sys.argv) > 2 else 'tokens.jsonl'

rows = [json.loads(l) for l in open(RES)]
by = collections.defaultdict(dict)          # id -> arm -> rec
for r in rows:
    by[r['id']][r['arm']] = r
toks = {json.loads(l)['id']: json.loads(l) for l in open(TOK)} if TOK else {}

ARMS = ['neucodec', 'widecodec', 'widecodec_24k']


def stats(v):
    v = [x for x in v if x is not None]
    if not v:
        return None
    v = sorted(v)
    n = len(v)
    mean = sum(v) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (n - 1)) if n > 1 else 0.0
    q = lambda p: v[min(n - 1, max(0, int(round(p * (n - 1)))))]
    return dict(n=n, mean=mean, sd=sd, p10=q(.10), med=q(.50), p90=q(.90),
                mn=v[0], mx=v[-1])


def paired(a, b, key):
    """b - a, over ids where both arms have the metric."""
    d = []
    for uid, m in by.items():
        if a in m and b in m and m[a].get(key) is not None and m[b].get(key) is not None:
            d.append(m[b][key] - m[a][key])
    if len(d) < 2:
        return None
    n = len(d)
    mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1))
    se = sd / math.sqrt(n)
    t = mean / se if se else float('inf')
    wins = sum(1 for x in d if x > 0)
    ties = sum(1 for x in d if x == 0)
    return dict(n=n, mean=mean, sd=sd, se=se, t=t, ci=(mean - 1.96 * se, mean + 1.96 * se),
                win_rate=wins / n, ties=ties)


def fmt(s, p=3):
    if s is None:
        return 'n/a'
    return (f"n={s['n']:3d} mean={s['mean']:.{p}f} sd={s['sd']:.{p}f} "
            f"p10={s['p10']:.{p}f} med={s['med']:.{p}f} p90={s['p90']:.{p}f} "
            f"max={s['mx']:.{p}f}")


print('=' * 100)
print('CORPUS')
print('=' * 100)
if toks:
    nt = stats([t['n_tokens'] for t in toks.values()])
    print(f"  utterances={len(toks)}  tokens/utt: {fmt(nt,1)}")
    print(f"  total audio = {sum(t['n_tokens'] for t in toks.values())/50/60:.1f} min")
    fr = collections.Counter(t['finish_reason'] for t in toks.values())
    print(f"  finish_reason: {dict(fr)}")
    spk = collections.Counter(t['speaker'] for t in toks.values())
    print(f"  speakers: {dict(spk)}")
print(f"  scored ids={len(by)}  arms={sorted({r['arm'] for r in rows})}")

print()
print('=' * 100)
print('UTMOSv2 MOS  (higher = better; reps=16, native sample rate)')
print('=' * 100)
for arm in ARMS:
    s = stats([by[u][arm]['mos'] for u in by if arm in by[u]])
    print(f'  {arm:16s} {fmt(s)}')
print()
for a, b in [('neucodec', 'widecodec'), ('neucodec', 'widecodec_24k'),
             ('widecodec_24k', 'widecodec')]:
    p = paired(a, b, 'mos')
    if p:
        print(f'  {b} - {a}:  Δ={p["mean"]:+.3f}  95% CI [{p["ci"][0]:+.3f},{p["ci"][1]:+.3f}]  '
              f't={p["t"]:+.1f}  wins {p["win_rate"]*100:.0f}% of {p["n"]}')

print()
print('=' * 100)
print('MOS by speaker  (widecodec - neucodec, paired)')
print('=' * 100)
spk_of = {u: toks[u]['speaker'] for u in by if u in toks}
for spk in sorted(set(spk_of.values())):
    d = [by[u]['widecodec']['mos'] - by[u]['neucodec']['mos']
         for u in by if spk_of.get(u) == spk
         and 'widecodec' in by[u] and 'neucodec' in by[u]
         and by[u]['widecodec']['mos'] is not None and by[u]['neucodec']['mos'] is not None]
    if not d:
        continue
    n = len(d); mean = sum(d) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1)) if n > 1 else 0
    nc = stats([by[u]['neucodec']['mos'] for u in by if spk_of.get(u) == spk and 'neucodec' in by[u]])
    wc = stats([by[u]['widecodec']['mos'] for u in by if spk_of.get(u) == spk and 'widecodec' in by[u]])
    print(f'  {spk:20s} neu={nc["mean"]:.3f}  wide={wc["mean"]:.3f}  '
          f'Δ={mean:+.3f} (sd {sd:.3f}, n={n}, wins {sum(1 for x in d if x>0)/n*100:.0f}%)')

print()
print('=' * 100)
print('BROKEN PITCH  (f0 metrics, all arms resampled to a common 16 kHz for analysis)')
print('=' * 100)
PITCH = [
    ('f0_excursion_ms', 'longest contiguous stretch >9 st off clip median (register break)', 1),
    ('f0_excursion_st', 'how far that excursion sat from the median', 3),
    ('f0_step_max_st',  'largest step across 50 ms of CONTIGUOUS voicing (seam)', 3),
    ('f0_step_long_st', 'same but needs 200 ms voicing each side', 3),
    ('f0_jump_max_st',  'max frame-to-frame jump (loose backstop)', 3),
    ('f0_jump_frac',    'fraction of adjacent voiced frames jumping >4 st', 4),
    ('f0_iqr_st',       'pitch spread, IQR in semitones', 3),
    ('f0_outlier_frac', 'fraction of frames >6 st from median', 4),
    ('f0_med',          'median f0 (Hz) - sanity, should match across arms', 2),
]
for key, desc, p in PITCH:
    print(f'\n  -- {key}  ({desc})')
    for arm in ['neucodec', 'widecodec']:
        s = stats([by[u][arm].get(key) for u in by if arm in by[u]])
        print(f'     {arm:16s} {fmt(s,p)}')
    pr = paired('neucodec', 'widecodec', key)
    if pr:
        print(f'     paired Δ(wide-neu) = {pr["mean"]:+.{p}f}  '
              f'95% CI [{pr["ci"][0]:+.{p}f},{pr["ci"][1]:+.{p}f}]  t={pr["t"]:+.1f}')

print()
print('=' * 100)
print('BROKEN-PITCH FLAG RATES  (thresholds from audiocheck.py\'s own documented evidence)')
print('=' * 100)
GATES = [
    ('f0_excursion_ms', 100.0, 'register break >=100 ms  (the module\'s own broken exemplar sat 100-200 ms)'),
    ('f0_excursion_ms', 200.0, 'register break >=200 ms  (severe)'),
    ('f0_step_max_st',    6.0, 'pitch step >=6 st across contiguous voicing'),
    ('f0_step_long_st',   4.0, 'sustained step >=4 st (seam / "changes person")'),
    ('f0_jump_frac',     0.02, '>2% of adjacent voiced frames jump >4 st (warble)'),
    ('f0_outlier_frac',  0.05, '>5% of frames sit >6 st off median'),
]
print(f'  {"gate":62s} {"neucodec":>12s} {"widecodec":>12s}')
for key, thr, desc in GATES:
    line = {}
    for arm in ['neucodec', 'widecodec']:
        v = [by[u][arm].get(key) for u in by if arm in by[u]]
        v = [x for x in v if x is not None]
        line[arm] = (sum(1 for x in v if x >= thr), len(v))
    a, b = line['neucodec'], line['widecodec']
    print(f'  {desc:62s} {a[0]:4d}/{a[1]:<3d}={a[0]/max(a[1],1)*100:4.1f}% '
          f'{b[0]:4d}/{b[1]:<3d}={b[0]/max(b[1],1)*100:4.1f}%')

# Utterances where the two arms disagree most on register breaks -- listening shortlist.
print()
print('=' * 100)
print('WORST OFFENDERS  (largest excursion per arm; for listening)')
print('=' * 100)
for arm in ['neucodec', 'widecodec']:
    top = sorted((by[u][arm] for u in by if arm in by[u]),
                 key=lambda r: -(r.get('f0_excursion_ms') or 0))[:6]
    print(f'  {arm}:')
    for r in top:
        print(f"     {r['id']}  excursion={r.get('f0_excursion_ms')}ms @{r.get('f0_excursion_st')}st  "
              f"step_max={r.get('f0_step_max_st')}  mos={r['mos'] if r['mos'] is None else round(r['mos'],2)}  "
              f"f0_med={r.get('f0_med')}Hz  spk={spk_of.get(r['id'],'?')}")
