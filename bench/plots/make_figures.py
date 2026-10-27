"""Figures for the bench reports, one chart per file.

Style: the Scicom preprint style of Voice-Isolator/preprint/plot_figs.py (itself from Penjajaran-Daya/preprint).
Every figure is a single panel in Inter, with the legend under the axes, in the Scicom platform's chart palette: its
eight series colours in a fixed order, muted text and border greys, thin solid grid lines. No title or caption is drawn
into an image: the report that shows a figure says what it is and where the numbers come from.

Measured series come from bench/results/ where they exist. Series that were only ever recorded in a report table are
declared inline below. Edit the constant, not the plot.

    uv run --with matplotlib --with numpy python bench/plots/make_figures.py
"""
import json
import os
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402


def _inter():
    """Inter ships with TeX Live (the preprint's text face); register its files with matplotlib."""
    try:
        reg = subprocess.run(['kpsewhich', 'Inter-Regular.otf'], capture_output=True, text=True).stdout.strip()
    except OSError:
        reg = ''
    if not reg:
        return 'DejaVu Sans'
    for f in Path(reg).parent.glob('Inter-*.otf'):
        font_manager.fontManager.addfont(str(f))
    return 'Inter'


FONT = _inter()
matplotlib.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': [FONT, 'DejaVu Sans'], 'mathtext.fontset': 'custom',
    'mathtext.rm': FONT, 'mathtext.it': f'{FONT}:italic', 'mathtext.bf': f'{FONT}:bold', 'mathtext.sf': FONT,
    'mathtext.cal': FONT, 'axes.unicode_minus': True, 'legend.frameon': False,
})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, 'bench', 'results')
IMG = os.path.join(ROOT, 'docs', 'img')
os.makedirs(IMG, exist_ok=True)
DPI = 200

# Scicom platform tokens (light theme)
FG, TEXT, MUTED = '#18181b', '#3f3f46', '#6b7280'
BORDER, RULE, BAND = '#e5e7eb', '#d4d4d8', '#f3f4f6'
REF = '#9ca3af'
S1, S2, S3, S4, S5, S6, S7, S8 = ('#3b82f6', '#0095a2', '#21763c', '#c67700', '#7a4aba', '#d44d8c', '#79821d', '#414aa1')
GRID = dict(color=BORDER, linewidth=0.6, linestyle='-')
GOOD, BAD, NEUTRAL = S3, S4, MUTED     # fixed / harmful / reference arms
W = 6.4                                # every panel is the preprint's single-column width


def frame(ax, xlab=None, ylab=None, axis='both'):
    ax.grid(True, axis=axis, **GRID)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(RULE)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=0, pad=4)
    if xlab:
        ax.set_xlabel(xlab, fontsize=8.5, color=TEXT, labelpad=7)
    if ylab:
        ax.set_ylabel(ylab, fontsize=8.5, color=TEXT, labelpad=7)


def better(label, lower=True, word='better'):
    """An axis label that says which way is better, with an arrow to the better end. A y label is turned a quarter, so
    the same arrows point down (lower) and up (higher) there."""
    return f'{label}  ← {word}' if lower else f'{label}  {word} →'


def legend_below(ax, ncol=2, pad=0.2, **kw):
    """Legends go under the axes: inside the frame they cover the lines a reader came for."""
    leg = ax.legend(fontsize=7.8, ncol=ncol, loc='upper center', bbox_to_anchor=(0.5, -pad), borderaxespad=0.0,
                    handlelength=1.8, columnspacing=1.6, **kw)
    for t in leg.get_texts():
        t.set_color(TEXT)


def note(ax, x, y, s, **kw):
    kw = {'fontsize': 7, 'color': TEXT, **kw}
    return ax.text(x, y, s, **kw)


def label_bars(ax, bars, vals, fmt, dy=0.0, **kw):
    for r, v in zip(bars, vals):
        note(ax, r.get_x() + r.get_width() / 2, v + dy, fmt.format(v), ha='center',
             va='bottom' if v >= 0 else 'top', **kw)


def save(fig, name):
    out = os.path.join(IMG, name)
    fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print('wrote', os.path.relpath(out, ROOT))


FIGS = []


def figure(fn):
    FIGS.append(fn)
    return fn


def log2_concurrency(ax, xs):
    ax.set_xscale('log', base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(xs)
    ax.minorticks_off()


# ---- 1. latency percentiles (bench/latency_bench.py, bench/TTFB.md) ------------------------------------------------
def _latency(key, name, ylab, rt_line=False):
    lv = json.load(open(f'{RES}/latency-2026-09-21/latency.json'))['levels']
    c = [x['concurrency'] for x in lv]
    fig, ax = plt.subplots(figsize=(W, 3.2))
    for pct, col in (('p50', S1), ('p90', S2), ('p99', S4)):
        ax.plot(c, [x[key][pct] for x in lv], color=col, lw=1.6, marker='o', ms=3.5, label=pct, zorder=3)
    if rt_line:
        ax.axhline(1.0, color=REF, lw=0.9, ls='--')
        note(ax, c[0], 1.02, 'real time', color=MUTED, va='bottom')
    log2_concurrency(ax, c)
    frame(ax, 'concurrent requests', better(ylab))
    legend_below(ax, ncol=3)
    save(fig, name)


@figure
def fig_latency_ttfb():
    _latency('ttfb_s', 'latency_ttfb.png', 'time to first audio byte (s)')


@figure
def fig_latency_e2e():
    _latency('e2e_s', 'latency_e2e.png', 'end-to-end time (s)')


@figure
def fig_latency_rtf():
    _latency('rtf', 'latency_rtf.png', 'real-time factor (e2e / audio s)', rt_line=True)


# ---- 2. throughput and saturation (bench/TTFB.md) ------------------------------------------------------------------
SAT = dict(conc=[16, 32, 64, 96], audio=[127, 208, 284, 304],
           codec_gpu=[37, 74, 82, 96], codec_bw=[2, 4, 5, 6],
           lead=[1.93, 1.68, 1.20, 0.19])


@figure
def fig_saturation_throughput():
    fig, ax = plt.subplots(figsize=(W, 3.0))
    ax.plot(SAT['conc'], SAT['audio'], color=S1, lw=1.6, marker='o', ms=3.5, zorder=3)
    for x, y in zip(SAT['conc'], SAT['audio']):
        note(ax, x, y + 8, f'{y}', ha='center', va='bottom')
    ax.set_xticks(SAT['conc'])
    ax.set_ylim(0, 340)
    frame(ax, 'concurrent requests', better('audio seconds per wall second', lower=False))
    save(fig, 'saturation_throughput.png')


@figure
def fig_saturation_codec():
    fig, ax = plt.subplots(figsize=(W, 3.0))
    ax.plot(SAT['conc'], SAT['codec_gpu'], color=S1, lw=1.6, marker='o', ms=3.5, label='codec GPU utilisation', zorder=3)
    ax.plot(SAT['conc'], SAT['codec_bw'], color=S4, lw=1.6, marker='s', ms=3.5, label='codec memory bandwidth',
            zorder=3)
    ax.set_xticks(SAT['conc'])
    ax.set_ylim(0, 100)
    frame(ax, 'concurrent requests', '% of the codec GPU')
    legend_below(ax, ncol=2)
    save(fig, 'saturation_codec.png')


@figure
def fig_saturation_buffer():
    fig, ax = plt.subplots(figsize=(W, 3.0))
    ax.plot(SAT['conc'], SAT['lead'], color=S1, lw=1.6, marker='o', ms=3.5, zorder=3)
    ax.axhline(0, color=RULE, lw=0.8)
    ax.annotate('0.19 s left', xy=(96, 0.19), xytext=(66, 0.55), fontsize=7, color=TEXT, ha='center',
                arrowprops=dict(arrowstyle='->', color=MUTED, lw=0.8))
    ax.set_xticks(SAT['conc'])
    ax.set_ylim(-0.1, 2.2)
    frame(ax, 'concurrent requests', better('client buffer when the stream ends (s)', lower=False))
    save(fig, 'saturation_buffer.png')


# ---- 3. the padding bug (bench/PADDING_BUG.md) ---------------------------------------------------------------------
PAD = [('same-length\nbatch', 72.3, GOOD),
       ('padded to\nbucket 675', 4.3, BAD),
       ('padded to\nbucket 500', 4.7, BAD),
       ('batched with a\nlonger request', -1.3, BAD),
       ('after the fix', 223.0, S1)]


@figure
def fig_padding():
    fig, ax = plt.subplots(figsize=(W, 3.2))
    names, vals, cols = zip(*PAD)
    b = ax.bar(names, vals, color=cols, width=0.6, zorder=3)
    label_bars(ax, b, vals, '{:.1f}', dy=3)
    ax.axhline(0, color=RULE, lw=0.8)
    ax.axhspan(-10, 20, color=BAND, zorder=0)
    note(ax, 2.5, 22, 'audible corruption below 20 dB', color=MUTED, ha='center', va='bottom')
    ax.set_ylim(-20, 250)
    frame(ax, None, better('SNR against an unpadded decode (dB)', lower=False), axis='y')
    save(fig, 'padding_bug.png')


# ---- 4. CUDA graphs against eager, by load (bench/PADDING_BUG.md) --------------------------------------------------
GR = dict(conc=[8, 32, 64], eager=[66.7, 132.2, 138.8],
          lazy=[65.8, 190.6, 213.4], prefix=[67.6, 209.4, 296.2])


@figure
def fig_graphs():
    fig, ax = plt.subplots(figsize=(W, 3.2))
    x, w = np.arange(len(GR['conc'])), 0.26
    ax.bar(x - w, GR['eager'], w, label='eager (prod default)', color=NEUTRAL, zorder=3)
    ax.bar(x, GR['lazy'], w, label='lazy graphs (correct)', color=S1, zorder=3)
    ax.bar(x + w, GR['prefix'], w, label='graphs before the fix (wrong audio)', color=BAD, zorder=3)
    for i, (e, lz) in enumerate(zip(GR['eager'], GR['lazy'])):
        note(ax, i, lz + 8, f'{lz / e:.2f}x', ha='center', va='bottom')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c} concurrent' for c in GR['conc']])
    frame(ax, None, better('audio seconds per wall second', lower=False), axis='y')
    legend_below(ax, ncol=3, pad=0.12)
    save(fig, 'cuda_graphs.png')


# ---- 5. precision and quantization (bench/MEGAKERNEL.md) -----------------------------------------------------------
PREC = [('bf16', [0.92, 0.99, 1.31], 35.0), ('TF32', [1.00, 0.97, 1.25], 64.0),
        ('fold weight_norm', [0.97, 0.99, 1.00], 224.0),
        ('cudnn.benchmark', [0.99, 0.99, 1.00], 224.0),
        ('int8 weight-only', [0.76, 0.76, 1.09], 34.0),
        ('int8 dyn act+w', [0.06, 0.05, 0.07], 28.0),
        ('torch.compile', [1.92, 1.49, 1.16], 80.0),
        ('torch.compile\n+ reduce-overhead', [2.26, 1.58, 1.19], 80.0)]


@figure
def fig_precision():
    """Horizontal bars: speedup at window 121, coloured by how faithful the output stayed. A scatter over SNR needs a
    log axis from 28 to 224 dB and reads badly; the question is "did any arm beat 1.0x, at what cost"."""
    rows = sorted(PREC, key=lambda r: r[1][1])
    fig, ax = plt.subplots(figsize=(W, 0.42 * len(rows) + 1.2))
    y = np.arange(len(rows))
    vals = [sp[1] for _, sp, _ in rows]
    snrs = [q for _, _, q in rows]
    cols = [GOOD if q > 200 else (BAD if q < 60 else S1) for q in snrs]
    ax.barh(y, vals, color=cols, height=0.58, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([n for n, _, _ in rows])
    ax.axvline(1.0, color=REF, lw=0.9, ls='--', zorder=4)
    note(ax, 1.02, len(rows) - 0.45, 'fp32 eager', color=MUTED, ha='left', va='bottom')
    for i, (v, q) in enumerate(zip(vals, snrs)):
        note(ax, v + 0.03, i, f'{v:.2f}x  ({"exact" if q > 200 else f"{q:.0f} dB SNR"})', va='center')
    note(ax, 2.3, 1.0, 'fp16 does not run:\ncuFFT rejects the ISTFT\ndims in half precision', color=MUTED,
         ha='right', va='center')
    ax.set_xlim(0, 2.35)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (GOOD, S1, BAD)]
    ax.legend(handles, ['bit-exact', '64 to 80 dB SNR (inaudible)', 'under 40 dB SNR (audible)'], fontsize=7.8,
              ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.16), borderaxespad=0.0, handlelength=1.4)
    frame(ax, better('speedup over fp32 eager, window 121', lower=False), None, axis='x')
    save(fig, 'precision_matrix.png')


# ---- 6. pitch and tone: one request against LiveKit (bench/PITCH_TONE_AB.md) ---------------------------------------
TONE = [('one request,\nnormalizer off', 30.8, S1),
        ('one request,\nnormalizer on', 27.0, S1),
        ('through\nLiveKit', 55.4, BAD),
        ('LiveKit +\ninterleave_id', 51.4, GOOD)]


@figure
def fig_tone():
    fig, ax = plt.subplots(figsize=(W, 3.2))
    names, vals, cols = zip(*TONE)
    b = ax.bar(names, vals, color=cols, width=0.56, zorder=3)
    label_bars(ax, b, vals, '{:.1f}', dy=0.8)
    ax.axhline(30.8, color=REF, lw=0.9, ls='--')
    note(ax, 1.3, 31.6, 'floor: the model alone', color=MUTED, ha='center', va='bottom')
    ax.set_ylim(0, 64)
    frame(ax, None, better('audible tone jumps per 1000 voiced windows'), axis='y')
    save(fig, 'pitch_tone.png')


# ---- 7. interleave A/B (bench/INTERLEAVE_AB.md) --------------------------------------------------------------------
ILV = dict(metric=['register step\n|st|', 'level step\n|dB|', 'chunk-to-chunk\nregister wander (st)',
                   'paragraph declination\n(st/s, x -10)'],
           one_shot=[1.60, 1.35, 3.77, 1.61], interleave=[1.24, 1.16, 3.00, 1.37],
           cold=[1.65, 1.61, 3.62, 0.82])


@figure
def fig_interleave():
    fig, ax = plt.subplots(figsize=(W, 3.2))
    x, w = np.arange(len(ILV['metric'])), 0.26
    ax.bar(x - w, ILV['one_shot'], w, label='one request (no real joins)', color=NEUTRAL, zorder=3)
    ax.bar(x, ILV['interleave'], w, label='interleaved chunks', color=S1, zorder=3)
    ax.bar(x + w, ILV['cold'], w, label='cold chunks', color=BAD, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(ILV['metric'])
    frame(ax, None, better('size of the change at a join'), axis='y')
    legend_below(ax, ncol=3, pad=0.22)
    save(fig, 'interleave_ab.png')


# ---- 8. normalizer agreement (bench/NORMALIZER.md) -----------------------------------------------------------------
NRM = dict(lang=['en', 'ms', 'zh', 'ta', 'code-switch', 'overall'],
           pct=[89, 87, 88, 67, 59, 82])


@figure
def fig_normalizer():
    fig, ax = plt.subplots(figsize=(W, 3.0))
    cols = [S1, S1, S1, BAD, BAD, S2]
    b = ax.bar(NRM['lang'], NRM['pct'], color=cols, width=0.56, zorder=3)
    label_bars(ax, b, NRM['pct'], '{}%', dy=1.2)
    ax.axhline(82, color=REF, lw=0.9, ls='--')
    ax.set_ylim(0, 100)
    frame(ax, None, better('verbatim agreement with the LLM (%)', lower=False), axis='y')
    save(fig, 'normalizer.png')


# ---- 9. WideCodec A/B (bench/WIDECODEC_AB.md) ----------------------------------------------------------------------
WC = dict(cond=['TTS speech tokens', 'real-audio resynthesis'], wide=[-0.212, 0.011])


@figure
def fig_widecodec():
    fig, ax = plt.subplots(figsize=(W, 2.9))
    cols = [BAD if v < -0.05 else GOOD for v in WC['wide']]
    b = ax.bar(WC['cond'], WC['wide'], color=cols, width=0.46, zorder=3)
    for r, v in zip(b, WC['wide']):
        note(ax, r.get_x() + r.get_width() / 2, v + (0.006 if v > 0 else -0.006), f'{v:+.3f}', ha='center',
             va='bottom' if v > 0 else 'top')
    ax.axhline(0, color=RULE, lw=0.8)
    ax.set_ylim(-0.27, 0.07)
    frame(ax, None, better('UTMOSv2, WideCodec minus NeuCodec', lower=False), axis='y')
    save(fig, 'widecodec_ab.png')


# ---- 10. LiveKit: the agent + WebRTC path (bench/LIVEKIT.md) -------------------------------------------------------
LKD = f'{RES}/livekit-2026-09-23'


def _lk(arm):
    """{concurrency: {'summary': ..., 'rows': [...]}} for one arm."""
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
    return xs[max(0, min(len(xs) - 1, int(round(q * len(xs) + 0.5)) - 1))]


def _clean_levels(cold):
    # 32 rooms is a ceiling of the rig (partial connects), not a serving number: plot the clean levels
    return [c for c in cold if cold[c]['summary'].get('errors', 0) == 0]


@figure
def fig_livekit_ttfb():
    cold, ilv, dir_ = _lk('cold'), _lk('interleave'), _direct()
    cs = _clean_levels(cold)
    fig, ax = plt.subplots(figsize=(W, 3.3))
    for data, col, lab in ((cold, S1, 'LiveKit'), (ilv, S2, 'LiveKit + interleave_id')):
        xs = [c for c in cs if c in data]
        ax.plot(xs, [data[c]['summary']['ttfb_p50'] for c in xs], color=col, lw=1.6, marker='o', ms=3.5,
                label=f'{lab}, p50', zorder=3)
        ax.plot(xs, [data[c]['summary']['ttfb_p95'] for c in xs], color=col, lw=1.1, ls='--', marker='s', ms=3,
                label=f'{lab}, p95', zorder=3)
    ax.plot(cs, [dir_[c]['ttfb_s']['p50'] for c in cs], color=NEUTRAL, lw=1.3, ls=':', marker='^', ms=3.5,
            label='HTTP to the API, p50', zorder=3)
    ax.fill_between(cs, [dir_[c]['ttfb_s']['p50'] for c in cs], [cold[c]['summary']['ttfb_p50'] for c in cs],
                    color=BAND, zorder=2)
    note(ax, 4, 0.17, 'agent + WebRTC: about +130 ms, flat', ha='center', va='bottom')
    log2_concurrency(ax, cs)
    ax.set_ylim(0, 1.35)
    frame(ax, 'concurrent rooms', better('text sent to first audible frame (s)'))
    legend_below(ax, ncol=2, pad=0.22)
    save(fig, 'livekit_ttfb.png')


@figure
def fig_livekit_loudness():
    """sd and p95 - p5, never max - min: that is an extreme-value statistic and grows with the sample count alone
    (8 utterances at one room, 128 at sixteen)."""
    cold, ilv = _lk('cold'), _lk('interleave')
    cs = _clean_levels(cold)
    rms = lambda d, c: [r['rms_db'] for r in d[c]['rows'] if r.get('ok')]  # noqa: E731
    fig, ax = plt.subplots(figsize=(W, 3.2))
    x, w = np.arange(len(cs)), 0.34
    ax.bar(x - w / 2, [np.std(rms(cold, c)) for c in cs], w, color=S1, label='LiveKit, sd', zorder=3)
    ax.bar(x + w / 2, [np.std(rms(ilv, c)) if c in ilv else 0 for c in cs], w, color=S2,
           label='LiveKit + interleave_id, sd', zorder=3)
    ax.plot(x, [_p(rms(cold, c), .95) - _p(rms(cold, c), .05) for c in cs], color=BAD, lw=1.3, ls='--', marker='o',
            ms=3.5, label='LiveKit, p95 - p5', zorder=4)
    ax.axhline(2.0, color=REF, lw=0.9, ls='--')
    note(ax, -0.45, 2.1, 'STREAM_NORMALIZE holds sd under 2 dB', color=MUTED, va='bottom')
    ax.set_xticks(x)
    ax.set_xticklabels(cs)
    ax.set_ylim(0, 8.2)
    frame(ax, 'concurrent rooms', better('spread of per-utterance loudness (dB)'), axis='y')
    legend_below(ax, ncol=3)
    save(fig, 'livekit_loudness.png')


@figure
def fig_livekit_tail():
    cold, ilv, dir_ = _lk('cold'), _lk('interleave'), _direct()
    data = [[r['ttfb'] for r in cold[16]['rows'] if r.get('ok')],
            [r['ttfb'] for r in ilv[16]['rows'] if r.get('ok')]]
    fig, ax = plt.subplots(figsize=(W, 3.2))
    bp = ax.boxplot(data, widths=0.42, patch_artist=True, whis=(5, 95),
                    tick_labels=[f'LiveKit (n={len(data[0])})', f'LiveKit + interleave_id (n={len(data[1])})'],
                    showfliers=True, flierprops=dict(marker='o', ms=2.4, mfc=MUTED, mec='none', alpha=0.6))
    for patch, col in zip(bp['boxes'], (S1, S2)):
        patch.set_facecolor(col)
        patch.set_alpha(0.25)
        patch.set_edgecolor(col)
    for el in ('whiskers', 'caps', 'medians'):
        for ln in bp[el]:
            ln.set_color(TEXT)
            ln.set_linewidth(1.0)
    ax.axhline(dir_[16]['ttfb_s']['p50'], color=NEUTRAL, lw=1.0, ls=':')
    note(ax, 2.45, dir_[16]['ttfb_s']['p50'] + 0.012, 'HTTP to the API, p50', color=MUTED, ha='right', va='bottom')
    ax.set_ylim(0, 0.95)
    frame(ax, None, better('every utterance at 16 rooms: TTFB (s)'), axis='y')
    save(fig, 'livekit_tail.png')


# ---- 11. the load client was the bottleneck (bench/LIVEKIT.md) -----------------------------------------------------
@figure
def fig_livekit_client_ttfb():
    fig, ax = plt.subplots(figsize=(W, 3.0))
    names = ['1 client process\n(16 rooms)', '2 client processes\n(8 rooms each)']
    vals = [14.08, 0.307]
    b = ax.bar(names, vals, color=[BAD, GOOD], width=0.46, zorder=3)
    for r, v in zip(b, vals):
        note(ax, r.get_x() + r.get_width() / 2, v * 1.25, f'{v:.3f} s', ha='center', va='bottom')
    ax.set_yscale('log')
    ax.set_ylim(0.1, 60)
    note(ax, 0.5, 38, '45x, and nothing server-side changed', ha='center', va='center', color=MUTED)
    frame(ax, None, better('TTFB p50 at 16 rooms (s, log)'), axis='y')
    save(fig, 'livekit_client_ttfb.png')


@figure
def fig_livekit_client_cpu():
    fig, ax = plt.subplots(figsize=(W, 2.6))
    who = ['load_client.py\n(1 process)', 'agent\n(20 processes)', 'TTS app']
    cpu = [267, 275, 19]
    b = ax.barh(who, cpu, color=[BAD, NEUTRAL, GOOD], height=0.5, zorder=3)
    for r, v in zip(b, cpu):
        note(ax, v + 6, r.get_y() + r.get_height() / 2, f'{v}%', va='center')
    note(ax, 60, 2, 'the service under test was idle', va='center', color=MUTED)
    ax.set_xlim(0, 330)
    ax.invert_yaxis()
    frame(ax, 'mean CPU over the 16-room run (100% = one core)', None, axis='x')
    save(fig, 'livekit_client_cpu.png')


# ---- 12. torch.compile / megakernel decode (bench/MEGAKERNEL.md) ---------------------------------------------------
MK = dict(win=['47 tokens', '121 tokens', '269 tokens'],
          eager=[6.24, 6.29, 9.39],
          arms=[('compile, default', [3.24, 4.22, 8.07], NEUTRAL),
                ('compile, reduce-overhead', [2.75, 3.98, 7.88], S1),
                ('compile, max-autotune', [2.77, 3.95, 7.85], S2)],
          shapes=[47, 121, 269, 187, 95],
          cost_static=[11.24, 9.49, 10.38, 11.51, 11.74],
          cost_dynamic=[12.66, 9.17, 10.02, 9.38, 9.46])


@figure
def fig_megakernel_speedup():
    fig, ax = plt.subplots(figsize=(W, 3.2))
    x, w = np.arange(len(MK['win'])), 0.26
    for k, (lab, ms, col) in enumerate(MK['arms']):
        sp = [e / m for e, m in zip(MK['eager'], ms)]
        b = ax.bar(x + (k - 1) * w, sp, w, color=col, label=lab, zorder=3)
        label_bars(ax, b, sp, '{:.2f}', dy=0.03, fontsize=6.5)
    ax.axhline(1.0, color=REF, lw=0.9, ls='--')
    note(ax, 2.45, 1.03, 'fp32 eager', color=MUTED, va='bottom')
    ax.set_xticks(x)
    ax.set_xticklabels(MK['win'])
    ax.set_xlim(-0.55, 2.9)
    ax.set_ylim(0, 2.6)
    frame(ax, 'decode window, batch 1', better('speedup per decode over fp32 eager', lower=False), axis='y')
    legend_below(ax, ncol=3)
    save(fig, 'megakernel_speedup.png')


@figure
def fig_megakernel_compile():
    fig, ax = plt.subplots(figsize=(W, 3.0))
    x, w = np.arange(len(MK['shapes'])), 0.36
    ax.bar(x - w / 2, MK['cost_static'], w, color=S1, label='dynamic=False', zorder=3)
    ax.bar(x + w / 2, MK['cost_dynamic'], w, color=NEUTRAL, label='dynamic=True', zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(MK['shapes'])
    ax.set_ylim(0, 15)
    frame(ax, 'decode length (tokens), in the order first seen', better('compile time for a new shape (s)'),
          axis='y')
    legend_below(ax, ncol=2)
    save(fig, 'megakernel_compile.png')


# ---- 13. first-samples fade-in (bench/fade_bench.py, bench/FADE_IN.md) ---------------------------------------------
FD = f'{RES}/fade-2026-09-23'
FADE_LEVELS = [1, 8, 16]


def _fade_rows(arm, c):
    return [json.loads(l) for l in open(f'{FD}/{arm}_c{c}.jsonl') if l.strip()]


@figure
def fig_fade_clicks():
    def rate(arm, c, k):
        rs = [r for r in _fade_rows(arm, c) if r.get('onset_db') is not None]
        return 100 * sum(r[k] for r in rs) / len(rs)
    cs = FADE_LEVELS
    fig, ax = plt.subplots(figsize=(W, 3.2))
    x, w = np.arange(len(cs)), 0.34
    nf = [rate('nofade', c, 'click') for c in cs]
    fd = [rate('fade', c, 'click') for c in cs]
    b1 = ax.bar(x - w / 2, nf, w, color=BAD, label='FADE_IN_MS=0', zorder=3)
    b2 = ax.bar(x + w / 2, fd, w, color=GOOD, label='FADE_IN_MS=10', zorder=3)
    label_bars(ax, list(b1) + list(b2), nf + fd, '{:.1f}%', dy=0.12)
    hn, hf = [rate('nofade', c, 'hot') for c in cs], [rate('fade', c, 'hot') for c in cs]
    ax.plot(x, [(p + q) / 2 for p, q in zip(hn, hf)], color=NEUTRAL, lw=1.2, ls='--', marker='o', ms=3.5,
            label='the model started voiced (both arms)', zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(cs)
    ax.set_ylim(0, 9)
    frame(ax, 'concurrent requests', better('responses that start with a click (%)'), axis='y')
    legend_below(ax, ncol=3)
    save(fig, 'fade_in_clicks.png')


@figure
def fig_fade_first_ms():
    fig, ax = plt.subplots(figsize=(W, 3.2))
    for arm, col, lab in (('nofade', BAD, 'FADE_IN_MS=0'), ('fade', GOOD, 'FADE_IN_MS=10')):
        v = np.sort([r['first1ms'] for c in FADE_LEVELS for r in _fade_rows(arm, c) if r.get('first1ms') is not None])
        v = np.maximum(v, 1e-5)
        ax.plot(v, 1 - np.arange(len(v)) / len(v), color=col, lw=1.6, label=f'{lab} (n={len(v)})', zorder=3)
    ax.axvline(0.05, color=REF, lw=0.9, ls='--')
    note(ax, 0.058, 0.2, 'click\nthreshold\n0.05', color=MUTED)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-5, 1)
    ax.set_ylim(1e-3, 1.05)
    frame(ax, better('loudest sample in the first millisecond (full scale)'), 'share of responses at or above')
    legend_below(ax, ncol=2)
    save(fig, 'fade_in_first_ms.png')


@figure
def fig_fade_example():
    ex = json.load(open(f'{FD}/example_onset.json'))
    n = int(0.030 * ex['sr'])
    t = np.arange(n) / ex['sr'] * 1000
    fig, ax = plt.subplots(figsize=(W, 2.9))
    ax.axvspan(0, 10, color=BAND, zorder=1)
    ax.plot(t, np.array(ex['raw'][:n]) / 32768, color=BAD, lw=1.0, label='as generated', zorder=3)
    ax.plot(t, np.array(ex['faded'][:n]) / 32768, color=GOOD, lw=1.2, label='with the 10 ms fade', zorder=4)
    note(ax, 0.4, -0.93, 'ramp', color=MUTED)
    note(ax, 11, -0.93, 'identical after 10 ms', color=MUTED)
    ax.set_ylim(-1.0, 1.0)
    frame(ax, 'time from the first sample (ms)', 'amplitude (full scale)')
    legend_below(ax, ncol=2)
    save(fig, 'fade_in_example.png')


# ---- 14. playback_speed sweep (bench/PLAYBACK_SPEED.md) ------------------------------------------------------------
PB = f'{RES}/playback_sweep'
PB_TODAY = 0.75
PB_PICK = (0.35, 0.55)      # the recommended 0.4 to 0.5


def _pb():
    full = json.load(open(f'{PB}/pb_full.json'))
    chunks = {r['ps']: r for r in json.load(open(f'{PB}/pb_chunks.json'))}
    ps = [r['ps'] for r in full]
    mean = lambda r, k: float(np.mean([x[k] for x in r['per']]))  # noqa: E731
    return ps, full, chunks, mean


def _pb_axes(ax, ylab, legend=False):
    ax.axvspan(*PB_PICK, color=BAND, zorder=0)
    ax.axvline(PB_TODAY, color=REF, lw=0.9, ls='--', zorder=1)
    frame(ax, 'playback_speed (first decode window = value x 50 tokens)', ylab)
    if legend:
        legend_below(ax, ncol=2)


@figure
def fig_playback_ttfb():
    ps, full, _, mean = _pb()
    fig, ax = plt.subplots(figsize=(W, 3.0))
    ax.plot(ps, [1000 * mean(r, 'ttfb') for r in full], color=S1, lw=1.6, marker='o', ms=3.2, zorder=3)
    note(ax, PB_TODAY + 0.03, 45, 'today 0.75', color=MUTED)
    note(ax, 0.45, 185, 'recommended\n0.4 to 0.5', color=MUTED, ha='center')
    _pb_axes(ax, better('time to first audio byte (ms)'))
    save(fig, 'playback_ttfb.png')


@figure
def fig_playback_loudness():
    ps, full, _, mean = _pb()
    fig, ax = plt.subplots(figsize=(W, 3.0))
    ax.axhline(-17, color=REF, lw=0.9, ls=':')
    ax.plot(ps, [mean(r, 'rms') for r in full], color=S1, lw=1.6, marker='o', ms=3.2, zorder=3)
    ax.annotate('the normalizer breaks: +6.3 dB', xy=(0.1, -10.7), xytext=(0.5, -12.0), fontsize=7, color=TEXT,
                arrowprops=dict(arrowstyle='->', color=MUTED, lw=0.8))
    _pb_axes(ax, 'active RMS (dBFS), target -17')
    save(fig, 'playback_loudness.png')


@figure
def fig_playback_pitch():
    ps, full, _, mean = _pb()
    fig, ax = plt.subplots(figsize=(W, 3.0))
    ax.plot(ps, [mean(r, 'f0') for r in full], color=S1, lw=1.6, marker='o', ms=3.2, zorder=3)
    note(ax, 1.25, 197.5, 'flat: the window does not move pitch', color=MUTED, ha='center')
    ax.set_ylim(190, 200)
    _pb_axes(ax, 'median f0 (Hz)')
    save(fig, 'playback_pitch.png')


@figure
def fig_playback_step():
    ps, _, chunks, _ = _pb()
    fig, ax = plt.subplots(figsize=(W, 3.0))
    ax.plot(ps, [chunks[p]['step_mean'] for p in ps], color=S1, lw=1.6, marker='o', ms=3.2, label='mean', zorder=3)
    ax.plot(ps, [chunks[p]['step_p95'] for p in ps], color=S2, lw=1.3, ls='--', marker='s', ms=3, label='p95',
            zorder=3)
    _pb_axes(ax, better('level step between emitted chunks (dB)'), legend=True)
    save(fig, 'playback_step.png')


@figure
def fig_playback_seams():
    ps, _, chunks, _ = _pb()
    fig, ax = plt.subplots(figsize=(W, 3.0))
    ax.plot(ps, [chunks[p]['over_3db'] for p in ps], color=S1, lw=1.6, marker='o', ms=3.2, zorder=3)
    _pb_axes(ax, better('chunk joins stepping more than 3 dB (%)'))
    save(fig, 'playback_seams.png')


@figure
def fig_playback_mos():
    ps, full, _, _ = _pb()
    fig, ax = plt.subplots(figsize=(W, 3.0))
    ax.errorbar(ps, [r['mos'] for r in full], yerr=[r['mos_sd'] for r in full], color=S1, lw=1.6, marker='o',
                ms=3.2, ecolor=BORDER, elinewidth=2.0, capsize=0, zorder=3)
    _pb_axes(ax, better('UTMOSv2 (mean, bars = sd over texts)', lower=False))
    save(fig, 'playback_mos.png')


if __name__ == '__main__':
    for fn in FIGS:
        fn()
