"""Figures for the bench reports.

Theme copied from Multilingual-TTS/tts-evaluation (plot_results.py): white ground,
#203882 primary, #27ae60 accent, dashed #e0e0e0 grid, #1a1a2e titles, 300 dpi.

Measured series come from bench/results/*.json where they exist. Series that were only
ever recorded in a report table are declared inline below, the same way plot_results.py
carries CER_DATA — edit the constant, not the plot.

    uv run --with matplotlib --with numpy python bench/plots/make_figures.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, 'bench', 'results')
IMG = os.path.join(ROOT, 'docs', 'img')
os.makedirs(IMG, exist_ok=True)

S = dict(bg='#ffffff', grid='#e0e0e0', primary='#203882', accent='#27ae60',
         warn='#c0392b', neutral='#7f8c8d', title='#1a1a2e', label='#333333',
         tick='#444444', caption='#666666', spine='#cccccc', dpi=300)

def axis(ax, title, ylab, xlab=''):
    ax.set_facecolor(S['bg'])
    for sp in ax.spines.values():
        sp.set_color(S['spine'])
    ax.grid(True, color=S['grid'], linewidth=0.8, linestyle='--', alpha=0.6, zorder=1)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=11.5, color=S['title'], fontweight='bold', pad=9)
    ax.set_ylabel(ylab, fontsize=9.5, color=S['label'])
    if xlab:
        ax.set_xlabel(xlab, fontsize=9, color=S['label'])
    ax.tick_params(colors=S['tick'], labelsize=8.5)

def save(fig, name, caption):
    fig.text(0.5, 0.004, caption, ha='center', fontsize=8.8,
             color=S['caption'], style='italic')
    plt.tight_layout(rect=[0, 0.028, 1, 0.96])
    out = os.path.join(IMG, name)
    plt.savefig(out, dpi=S['dpi'], bbox_inches='tight', facecolor=S['bg'])
    plt.close(fig)
    print('wrote', os.path.relpath(out, ROOT))

# ─────────────────────────────────────────────── 1. latency percentiles
def fig_latency():
    d = json.load(open(f'{RES}/latency-2026-09-21/latency.json'))
    lv = d['levels']
    c = [x['concurrency'] for x in lv]
    fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.4), dpi=S['dpi'])
    fig.patch.set_facecolor(S['bg'])
    for ax, key, title, ylab in (
            (axs[0], 'ttfb_s', 'TTFB — first audio byte', 'seconds'),
            (axs[1], 'e2e_s', 'End-to-end', 'seconds'),
            (axs[2], 'rtf', 'RTF (e2e ÷ audio seconds)', 'RTF')):
        axis(ax, title, ylab, 'concurrency')
        for pct, col, lw, ls in (('p50', S['primary'], 2.0, '-'),
                                 ('p90', S['accent'], 1.5, '--'),
                                 ('p99', S['warn'], 1.3, ':')):
            ax.plot(c, [x[key][pct] for x in lv], color=col, lw=lw, ls=ls,
                    marker='o', ms=4, label=pct, zorder=3)
        ax.set_xscale('log', base=2); ax.set_xticks(c); ax.set_xticklabels(c)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=S['label'])
    if lv:
        axs[2].axhline(1.0, color=S['warn'], lw=1.0, ls='-', alpha=0.5)
        axs[2].text(c[0], 1.02, 'real time', fontsize=8, color=S['warn'])
    fig.suptitle('Latency under closed-loop concurrency (app on one H20, TP=4 LM on another host)',
                 fontsize=13.5, color=S['title'], fontweight='bold', y=0.99)
    save(fig, 'latency_percentiles.png',
         '396 requests, 0 errors. ~5.2 s of audio per request. bench/latency_bench.py')

# ─────────────────────────────────────────────── 2. throughput + saturation
SAT = dict(conc=[16, 32, 64, 96], audio=[127, 208, 284, 304],
           codec_gpu=[37, 74, 82, 96], codec_bw=[2, 4, 5, 6],
           lead=[1.93, 1.68, 1.20, 0.19])
def fig_saturation():
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.4), dpi=S['dpi'])
    fig.patch.set_facecolor(S['bg'])
    a = axs[0]; axis(a, 'Throughput saturates on the codec GPU', 'audio-seconds / wall-second', 'concurrency')
    a.plot(SAT['conc'], SAT['audio'], color=S['primary'], lw=2.2, marker='o', ms=5, zorder=3)
    a2 = a.twinx()
    a2.plot(SAT['conc'], SAT['codec_gpu'], color=S['accent'], lw=1.6, ls='--', marker='s', ms=4)
    a2.plot(SAT['conc'], SAT['codec_bw'], color=S['neutral'], lw=1.3, ls=':', marker='^', ms=4)
    a2.set_ylabel('% ', fontsize=9.5, color=S['label']); a2.set_ylim(0, 100)
    a2.tick_params(colors=S['tick'], labelsize=8.5)
    a2.text(64, 90, 'GPU util', fontsize=8.5, color=S['accent'], fontweight='bold')
    a2.text(64, 11, 'memory bandwidth', fontsize=8.5, color=S['neutral'])
    a = axs[1]; axis(a, 'Client buffer collapses as it saturates', 'seconds of audio buffered', 'concurrency')
    a.plot(SAT['conc'], SAT['lead'], color=S['primary'], lw=2.2, marker='o', ms=5, zorder=3)
    a.axhline(0, color=S['warn'], lw=1.2)
    a.fill_between(SAT['conc'], 0, SAT['lead'], color=S['accent'], alpha=0.10)
    a.annotate('0.19 s left', xy=(96, 0.19), xytext=(55, 0.45), fontsize=9.5, color=S['warn'],
               fontweight='bold', ha='center',
               arrowprops=dict(arrowstyle='->', color=S['warn'], lw=1.1,
                               connectionstyle='arc3,rad=-0.25'))
    fig.suptitle('Where the stack runs out: 96% GPU at 6% bandwidth = launch-bound',
                 fontsize=13.5, color=S['title'], fontweight='bold', y=0.99)
    save(fig, 'saturation.png',
         'Codec GPU is the only resource that climbs with load. 0 errors at every level.')

# ─────────────────────────────────────────────── 3. the padding bug
PAD = [('same-length\nbatch', 72.3, S['accent']),
       ('padded to\nbucket 675', 4.3, S['warn']),
       ('padded to\nbucket 500', 4.7, S['warn']),
       ('batched with a\nlonger request', -1.3, S['warn']),
       ('AFTER THE FIX', 223.0, S['primary'])]
def fig_padding():
    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=S['dpi'])
    fig.patch.set_facecolor(S['bg'])
    axis(ax, 'Decode SNR against an unpadded reference', 'SNR (dB)')
    names = [p[0] for p in PAD]; vals = [p[1] for p in PAD]; cols = [p[2] for p in PAD]
    b = ax.bar(names, vals, color=cols, zorder=3, width=0.62)
    for r, v in zip(b, vals):
        ax.text(r.get_x()+r.get_width()/2, v + (6 if v > 0 else -14), f'{v:.1f}',
                ha='center', fontsize=9.5, fontweight='bold', color=S['title'],
                fontfamily='monospace')
    ax.axhline(0, color=S['spine'], lw=1.0)
    ax.axhspan(-10, 20, color=S['warn'], alpha=0.07, zorder=0)
    ax.text(0.02, 12, 'audible corruption', fontsize=8.5, color=S['warn'], style='italic')
    ax.set_ylim(-20, 250)
    fig.suptitle('Padding a decode window corrupts a non-causal decoder',
                 fontsize=13.5, color=S['title'], fontweight='bold', y=0.99)
    save(fig, 'padding_bug.png',
         'Equal lengths are bit-exact. Padding to a CUDA-graph bucket costs 4.7 dB SNR; '
         'batching a short window with a long one costs more than the signal itself.')

# ─────────────────────────────────────────────── 4. graphs vs eager, by load
GR = dict(conc=[8, 32, 64], eager=[66.7, 132.2, 138.8],
          lazy=[65.8, 190.6, 213.4], prefix=[67.6, 209.4, 296.2])
def fig_graphs():
    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=S['dpi'])
    fig.patch.set_facecolor(S['bg'])
    axis(ax, 'CUDA graphs only pay once the codec GPU is the constraint',
         'audio-seconds / wall-second', 'concurrency')
    x = np.arange(len(GR['conc'])); w = 0.26
    ax.bar(x-w, GR['eager'], w, label='eager (prod default)', color=S['neutral'], zorder=3)
    ax.bar(x,   GR['lazy'],  w, label='lazy graphs (correct)', color=S['primary'], zorder=3)
    ax.bar(x+w, GR['prefix'], w, label='pre-fix graphs (wrong audio)', color=S['warn'],
           alpha=0.55, zorder=3)
    for i, (e, l) in enumerate(zip(GR['eager'], GR['lazy'])):
        ax.text(i, max(e, l)+12, f'{l/e:.2f}×', ha='center', fontsize=10,
                fontweight='bold', color=S['primary'], fontfamily='monospace')
    ax.set_xticks(x); ax.set_xticklabels([f'c={c}' for c in GR['conc']])
    ax.legend(frameon=False, fontsize=8.8, labelcolor=S['label'], loc='upper left')
    fig.suptitle('Graph speedup, and what correctness cost',
                 fontsize=13.5, color=S['title'], fontweight='bold', y=0.99)
    save(fig, 'cuda_graphs.png',
         'Interleaved passes on one GPU. At c=8 graphs are worth nothing; at c=64, 1.54×. '
         'The red bar is faster still — and 10.6 dB SNR.')

# ─────────────────────────────────────────────── 5. precision / quantization
PREC = [('bf16', [0.92, 0.99, 1.31], 35.0), ('TF32', [1.00, 0.97, 1.25], 64.0),
        ('fold weight_norm', [0.97, 0.99, 1.00], 224.0),
        ('cudnn.benchmark', [0.99, 0.99, 1.00], 224.0),
        ('int8 weight-only', [0.76, 0.76, 1.09], 34.0),
        ('int8 dyn act+w', [0.06, 0.05, 0.07], 28.0),
        ('torch.compile', [1.92, 1.49, 1.16], 80.0),
        ('torch.compile\n+ reduce-overhead', [2.26, 1.58, 1.19], 80.0)]
def fig_precision():
    """Horizontal bars: speedup, coloured by how faithful the output stayed.

    A scatter over SNR needs a log axis spanning 28 -> 224 dB and reads terribly; the
    question is really "did any arm beat 1.0x, and at what cost", which is a bar chart.
    """
    fig, ax = plt.subplots(figsize=(10.4, 5.0), dpi=S['dpi'])
    fig.patch.set_facecolor(S['bg'])
    axis(ax, 'Every arithmetic optimisation of the codec (window 121)',
         '', 'speedup vs fp32 eager')
    rows = sorted(PREC, key=lambda r: r[1][1])
    names = [f'{n}' for n, _, _ in rows]
    vals = [sp[1] for _, sp, _ in rows]
    snrs = [snr for _, _, snr in rows]
    cols = [S['accent'] if q > 200 else (S['warn'] if q < 60 else S['primary']) for q in snrs]
    y = np.arange(len(rows))
    ax.barh(y, vals, color=cols, zorder=3, height=0.58)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9.5, color=S['label'])
    ax.axvline(1.0, color=S['title'], lw=1.6, zorder=4)
    # sits just above the x-axis, well clear of the subplot title
    ax.text(1.0, -0.52, 'fp32 baseline', ha='center', va='center', fontsize=9,
            color=S['title'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=S['bg'],
                      edgecolor=S['spine'], lw=0.8))
    for i, (v, q) in enumerate(zip(vals, snrs)):
        tag = 'exact' if q > 200 else f'{q:.0f} dB'
        ax.text(v + 0.03, i, f'{v:.2f}x   ({tag})', va='center', fontsize=9,
                color=S['title'], fontfamily='monospace')
    ax.set_xlim(0, 2.35); ax.set_ylim(-1.0, len(rows) - 0.35)
    ax.text(1.92, 2.6, 'fp16: will not run\n(cuFFT rejects the ISTFT\ndims in half precision)',
            fontsize=9.5, color=S['warn'], fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#fdf0ee', edgecolor=S['warn'], lw=1.0))
    handles = [plt.Rectangle((0,0),1,1, color=S['accent']),
               plt.Rectangle((0,0),1,1, color=S['primary']),
               plt.Rectangle((0,0),1,1, color=S['warn'])]
    ax.legend(handles, ['bit-exact', '64-80 dB SNR (inaudible)', 'under 40 dB SNR (audible)'],
              frameon=False, fontsize=8.5, labelcolor=S['label'],
              loc='lower right', bbox_to_anchor=(1.0, 0.03))
    fig.suptitle('Only FUSION beats fp32 eager — no precision or quantization arm does',
                 fontsize=13.5, color=S['title'], fontweight='bold', y=0.98)
    save(fig, 'precision_matrix.png',
         'One 121-token decode issues 609 CUDA kernel launches across 51 distinct ops. That is '
         'the cost, so fusing the graph pays and cheaper arithmetic does not.')

# ─────────────────────────────────────────────── 6. pitch/tone: direct vs LiveKit
TONE = [('one request\nnormalizer off', 30.8, S['primary']),
        ('one request\nnormalizer on', 27.0, S['primary']),
        ('through LiveKit', 55.4, S['warn']),
        ('LiveKit +\ninterleave_id', 51.4, S['accent'])]
def fig_tone():
    fig, ax = plt.subplots(figsize=(9.2, 4.4), dpi=S['dpi'])
    fig.patch.set_facecolor(S['bg'])
    axis(ax, 'Audible mid-utterance tone jumps', 'events per 1000 voiced windows')
    names = [t[0] for t in TONE]; vals = [t[1] for t in TONE]; cols = [t[2] for t in TONE]
    b = ax.bar(names, vals, color=cols, zorder=3, width=0.6)
    for r, v in zip(b, vals):
        ax.text(r.get_x()+r.get_width()/2, v+1.2, f'{v:.1f}', ha='center',
                fontsize=10, fontweight='bold', color=S['title'], fontfamily='monospace')
    ax.axhline(30.8, color=S['neutral'], lw=1.2, ls='--')
    ax.text(2.6, 32.4, 'floor: the model alone', fontsize=8.6, color=S['neutral'], style='italic')
    fig.suptitle('Half of what a caller hears is chunking; the other half is the model',
                 fontsize=13, color=S['title'], fontweight='bold', y=0.99)
    save(fig, 'pitch_tone.png',
         'An event = level rises ≥3 dB AND register ≥1.5 st together, between adjacent '
         '0.5 s voiced windows. 840 utterances.')


# ─────────────────────────────────────────────── 7. interleave A/B
ILV = dict(metric=['register step |st|', 'level step |dB|', 'chunk-to-chunk\nregister wander (st)',
                   'paragraph declination\n(st/s, ×-10)'],
           one_shot=[1.60, 1.35, 3.77, 1.61], interleave=[1.24, 1.16, 3.00, 1.37],
           cold=[1.65, 1.61, 3.62, 0.82])
def fig_interleave():
    fig, ax = plt.subplots(figsize=(10.2, 4.6), dpi=S['dpi'])
    fig.patch.set_facecolor(S['bg'])
    axis(ax, 'What the join between two chunks sounds like', 'lower is smoother')
    x = np.arange(len(ILV['metric'])); w = 0.26
    ax.bar(x-w, ILV['one_shot'], w, label='one-shot (no real joins)', color=S['neutral'], zorder=3)
    ax.bar(x,   ILV['interleave'], w, label='interleaved', color=S['accent'], zorder=3)
    ax.bar(x+w, ILV['cold'], w, label='cold chunking (today)', color=S['warn'], zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(ILV['metric'], fontsize=8.8)
    ax.legend(frameon=False, fontsize=8.8, labelcolor=S['label'])
    fig.suptitle('interleave_id cuts the chunk-join jump about a quarter',
                 fontsize=13.5, color=S['title'], fontweight='bold', y=0.99)
    save(fig, 'interleave_ab.png',
         '80 paragraphs, 438 matched chunk pairs. Declination is negated so lower is better '
         'on every bar. bench/INTERLEAVE_AB.md')

# ─────────────────────────────────────────────── 8. normalizer agreement
NRM = dict(lang=['en', 'ms', 'zh', 'ta', 'code-switch', 'overall'],
           pct=[89, 87, 88, 67, 59, 82])
def fig_normalizer():
    fig, ax = plt.subplots(figsize=(9.2, 4.3), dpi=S['dpi'])
    fig.patch.set_facecolor(S['bg'])
    axis(ax, 'Rule-based normalizer vs the LLM it replicates', '% verbatim agreement')
    cols = [S['primary']]*5 + [S['title']]
    cols[3] = cols[4] = S['warn']
    b = ax.bar(NRM['lang'], NRM['pct'], color=cols, zorder=3, width=0.6)
    for r, v in zip(b, NRM['pct']):
        ax.text(r.get_x()+r.get_width()/2, v+1.5, f'{v}%', ha='center', fontsize=10,
                fontweight='bold', color=S['title'], fontfamily='monospace')
    ax.set_ylim(0, 100)
    ax.axhline(82, color=S['neutral'], lw=1.2, ls='--')
    fig.suptitle('mode=spoken reproduces the LLM on 82% of 497 sentences',
                 fontsize=13.5, color=S['title'], fontweight='bold', y=0.99)
    save(fig, 'normalizer.png',
         'Every digit is read in all 497. Of the 88 residual diffs, 28 are LLM errors and 50 '
         'are the LLM contradicting itself. bench/NORMALIZER.md')

# ─────────────────────────────────────────────── 9. widecodec A/B
WC = dict(cond=['TTS speech tokens', 'real-audio resynthesis'],
          neucodec=[0.0, 0.0], wide=[-0.212, 0.011])
def fig_widecodec():
    fig, ax = plt.subplots(figsize=(8.6, 4.3), dpi=S['dpi'])
    fig.patch.set_facecolor(S['bg'])
    axis(ax, 'WideCodec minus NeuCodec, UTMOSv2', 'ΔMOS (positive = WideCodec better)')
    cols = [S['warn'] if v < -0.05 else S['accent'] for v in WC['wide']]
    b = ax.bar(WC['cond'], WC['wide'], color=cols, zorder=3, width=0.5)
    for r, v in zip(b, WC['wide']):
        ax.text(r.get_x()+r.get_width()/2, v + (0.012 if v > 0 else -0.028),
                f'{v:+.3f}', ha='center', fontsize=11, fontweight='bold',
                color=S['title'], fontfamily='monospace')
    ax.axhline(0, color=S['title'], lw=1.4)
    ax.set_ylim(-0.27, 0.07)
    fig.suptitle('An LM/decoder pairing effect, not codec quality',
                 fontsize=13.5, color=S['title'], fontweight='bold', y=0.99)
    save(fig, 'widecodec_ab.png',
         'WideCodec loses 0.212 MOS on our TTS tokens but ties on real audio — the LM was '
         'trained against NeuCodec. Verdict: keep NeuCodec. reps=16, scoring is stochastic.')

# ─────────────────────────────────────────────── 10. LiveKit: the agent + WebRTC path
LKD = f'{RES}/livekit-2026-09-23'

def _lk(arm):
    """{concurrency: {'summary':…, 'rows':[…]}} for one arm."""
    out = {}
    for f in os.listdir(LKD):
        if f.startswith(f'lk2_{arm}_c') and f.endswith('.json'):
            d = json.load(open(f'{LKD}/{f}'))
            out[d['summary']['concurrency']] = d
    return dict(sorted(out.items()))

def _direct():
    d = json.load(open(f'{LKD}/direct_9095.json'))
    return {lv['concurrency']: lv for lv in d['levels']}

def _p(xs, q):
    xs = sorted(xs)
    return xs[max(0, min(len(xs)-1, int(round(q*len(xs)+0.5))-1))]

def fig_livekit():
    cold, ilv, dir_ = _lk('cold'), _lk('interleave'), _direct()
    # c=32 is a rig ceiling (partial connects), not a serving number — plot the clean levels
    cs = [c for c in cold if cold[c]['summary'].get('errors', 0) == 0]
    rms = lambda d, c: [r['rms_db'] for r in d[c]['rows'] if r.get('ok')]
    fig, axs = plt.subplots(1, 3, figsize=(15.8, 4.5), dpi=S['dpi'])
    fig.patch.set_facecolor(S['bg'])

    # (a) TTFB: what the agent + WebRTC add on top of the API
    a = axs[0]; axis(a, 'TTFB — text sent → first audible frame', 'seconds', 'concurrent rooms')
    for data, col, lab in ((cold, S['primary'], 'LiveKit'),
                           (ilv, S['accent'], 'LiveKit + interleave_id')):
        xs = [c for c in cs if c in data]
        a.plot(xs, [data[c]['summary']['ttfb_p50'] for c in xs], color=col, lw=2.2,
               marker='o', ms=5, label=f'{lab} p50', zorder=3)
        a.plot(xs, [data[c]['summary']['ttfb_p95'] for c in xs], color=col, lw=1.3,
               ls='--', marker='s', ms=3.5, alpha=0.7, label=f'{lab} p95', zorder=3)
    a.plot(cs, [dir_[c]['ttfb_s']['p50'] for c in cs], color=S['neutral'], lw=1.8,
           ls=':', marker='^', ms=4.5, label='HTTP direct p50', zorder=3)
    a.fill_between(cs, [dir_[c]['ttfb_s']['p50'] for c in cs],
                   [cold[c]['summary']['ttfb_p50'] for c in cs],
                   color=S['primary'], alpha=0.08, zorder=2)
    a.annotate('agent + WebRTC ≈ +130 ms, flat', xy=(4, 0.155), fontsize=8.4,
               color=S['primary'], style='italic', ha='center')
    a.set_xscale('log', base=2); a.set_xticks(cs); a.set_xticklabels(cs)
    a.set_ylim(0, 1.35)
    a.legend(frameon=False, fontsize=7.8, labelcolor=S['label'], loc='upper left', ncol=1)

    # (b) loudness. sd and p95-p5 — NOT max-min, which is an extreme-value statistic and
    # grows with sample count alone (n=8 at c=1 vs n=128 at c=16).
    b = axs[1]; axis(b, 'Per-utterance loudness consistency', 'dB', 'concurrent rooms')
    x = np.arange(len(cs)); w = 0.35
    b.bar(x - w/2, [np.std(rms(cold, c)) for c in cs], w, color=S['primary'],
          label='LiveKit — sd', zorder=3)
    b.bar(x + w/2, [np.std(rms(ilv, c)) if c in ilv else 0 for c in cs], w,
          color=S['accent'], label='+ interleave_id — sd', zorder=3)
    b.plot(x, [_p(rms(cold, c), .95) - _p(rms(cold, c), .05) for c in cs], color=S['warn'],
           lw=1.5, ls='--', marker='o', ms=4, label='LiveKit — p95 − p5', zorder=4)
    b.set_xticks(x); b.set_xticklabels(cs)
    b.axhline(2.0, color=S['neutral'], lw=1.0, ls=':')
    b.text(-0.42, 2.1, 'STREAM_NORMALIZE holds sd under 2 dB', fontsize=8,
           color=S['neutral'], style='italic')
    b.set_ylim(0, 8.2)
    b.legend(frameon=False, fontsize=8, labelcolor=S['label'], loc='upper left', ncol=3)

    # (c) the tail, per utterance, at the top clean level
    c_ = axs[2]; axis(c_, 'Every utterance at 16 rooms (n=128 each)', 'TTFB, seconds')
    data = [[r['ttfb'] for r in cold[16]['rows'] if r.get('ok')],
            [r['ttfb'] for r in ilv[16]['rows'] if r.get('ok')]]
    bp = c_.boxplot(data, vert=True, widths=0.42, patch_artist=True, whis=(5, 95),
                    tick_labels=['LiveKit', '+ interleave_id'], showfliers=True,
                    flierprops=dict(marker='o', ms=2.6, mfc=S['neutral'],
                                    mec='none', alpha=0.55))
    for patch, col in zip(bp['boxes'], [S['primary'], S['accent']]):
        patch.set_facecolor(col); patch.set_alpha(0.75); patch.set_edgecolor(col)
    for el in ('whiskers', 'caps', 'medians'):
        for ln in bp[el]:
            ln.set_color(S['title']); ln.set_linewidth(1.2)
    c_.axhline(dir_[16]['ttfb_s']['p50'], color=S['neutral'], lw=1.3, ls=':')
    c_.text(2.44, dir_[16]['ttfb_s']['p50'] + 0.012, 'HTTP direct p50', fontsize=8,
            color=S['neutral'], ha='right', style='italic')
    c_.set_ylim(0, 0.95)

    fig.suptitle('Through a real LiveKit agent: TTFB flat to 16 rooms, loudness held, 0 errors',
                 fontsize=13.5, color=S['title'], fontweight='bold', y=0.99)
    save(fig, 'livekit_bench.png',
         '8 utterances per room, TM_English_Normal, patched app on one H20 + TP=4 LM. '
         'Boxes are p25–p75, whiskers p5–p95. bench/livekit/ · bench/LIVEKIT.md')

# ─────────────────────────────────────────────── 11. the load client was the bottleneck
def fig_livekit_client():
    fig, axs = plt.subplots(1, 2, figsize=(12.6, 4.4), dpi=S['dpi'])
    fig.patch.set_facecolor(S['bg'])

    a = axs[0]; axis(a, '16 rooms, one client process vs two', 'TTFB p50, seconds')
    names = ['1 client process\n(16 rooms)', '2 client processes\n(8 rooms each)']
    vals = [14.08, 0.307]
    bars = a.bar(names, vals, color=[S['warn'], S['accent']], width=0.5, zorder=3)
    for r, v in zip(bars, vals):
        a.text(r.get_x()+r.get_width()/2, v*1.35, f'{v:.3f} s', ha='center', fontsize=11,
               fontweight='bold', color=S['title'], fontfamily='monospace')
    a.set_yscale('log'); a.set_ylim(0.1, 60)
    a.text(0.5, 0.155, '45× — and nothing server-side changed', fontsize=9,
           color=S['title'], ha='center', style='italic',
           bbox=dict(boxstyle='round,pad=0.35', facecolor='#ffffff',
                     edgecolor=S['spine'], linewidth=0.8))

    b = axs[1]; axis(b, 'Who was actually busy during the 16-room run', '')
    who = ['load_client.py\n(1 process)', 'agent\n(20 processes)', 'TTS app :9095']
    cpu = [267, 275, 19]
    cols = [S['warn'], S['neutral'], S['accent']]
    bars = b.barh(who, cpu, color=cols, height=0.5, zorder=3)
    for r, v in zip(bars, cpu):
        b.text(v + 8, r.get_y()+r.get_height()/2, f'{v}%', va='center', fontsize=10.5,
               fontweight='bold', color=S['title'], fontfamily='monospace')
    b.set_xlim(0, 330)
    b.invert_yaxis()
    b.set_ylabel('')
    b.set_xlabel('mean %CPU over the run (ps pcpu) — 100% = one core',
                 fontsize=9, color=S['label'])
    b.text(120, 2.36, 'the service under test was idle', fontsize=8.6,
           color=S['accent'], style='italic')

    fig.suptitle('The "LiveKit collapses at 16 rooms" result was the measuring rig',
                 fontsize=13.5, color=S['title'], fontweight='bold', y=0.99)
    save(fig, 'livekit_client_trap.png',
         'Each room coroutine scans its frame buffer and runs numpy RMS on the shared event '
         'loop. load_client.py now shards across processes (--procs, default 1 per 8 rooms).')

if __name__ == '__main__':
    fig_latency(); fig_saturation(); fig_padding()
    fig_graphs(); fig_precision(); fig_tone()
    fig_interleave(); fig_normalizer(); fig_widecodec()
    fig_livekit(); fig_livekit_client()
