# NeuCodec vs WideCodec — codec A/B on TTS speech tokens (tm-h20)

Does swapping the vocoder for **[`Scicom-intl/WideCodec`](https://huggingface.co/Scicom-intl/WideCodec)**
(44.1 kHz, 0.8 kbps, single codebook) improve the audio the TTS API serves, versus the
**[`neuphonic/neucodec`](https://github.com/neuphonic/neucodec)** decoder the app ships today
(24 kHz, vendored under `app/neucodec/`)?

Measured on **1× H20-3e** (tm-h20, GPU 4 + GPU 7), slurm jobs 599 / 600 / 602, 2026-09-02.

## TL;DR

| | NeuCodec 24 kHz | WideCodec 44.1 kHz | verdict |
|---|---|---|---|
| UTMOSv2 MOS, **TTS-LM tokens** | **3.152** | 2.940 | **NeuCodec +0.212** (t = −15.9) |
| UTMOSv2 MOS, **real-audio resynthesis** | 2.503 | 2.514 | **tie** (+0.011, t = +0.4) |
| Pitch warble (`f0_jump_frac`) | **0.0060** | 0.0084 | NeuCodec cleaner (t = +7.3) |
| Median decode / utterance | **30 ms** | 38 ms | NeuCodec 1.27× faster |
| Decoder params | **185.6 M** | 287.9 M | — |

**Keep NeuCodec.** WideCodec is *not* a worse codec — it ties NeuCodec on codes from real audio, at
~1/10 the bitrate of NVIDIA NeMo-44k. It loses on **our** tokens because our LM was trained to emit
tokens *NeuCodec's* decoder renders well. The MOS gap is an **LM/decoder pairing effect**, not codec
quality. The pitch jitter, however, is intrinsic and shows up in both conditions.

## Why this A/B is clean

WideCodec is a **decoder-only finetune** of NeuCodec: same frozen FSQ codebook (`levels=[4]×8`,
`num_quantizers=1` → 16 bits/frame), same frozen encoder, same 50 tokens/s, identical
`decode_code([B,1,F]) -> [B,1,T]` signature. So the harness **generates the LM speech tokens once and
decodes the byte-identical stream through both decoders.**

That matters because sampling noise would otherwise swamp the effect: at temp 0.6 the same text spreads
3–10 dB of active-RMS and shifts prosody run to run (see the `STREAM_NORMALIZE` note in `CLAUDE.md`).
Generating per-arm would have measured the LM, not the codec.

Integrity checks that passed:

- **Max |duration difference| across all 200 pairs = 0.0000 s** — both codecs are exactly 50 tok/s
  (24000/480 and 44100/882), so pairing is exact.
- `sample_rate` asserted per arm (24000/480, 44100/882) and decoder param counts logged. This is
  load-bearing: WideCodec's bundled loader drops shape-mismatched keys and calls
  `load_state_dict(strict=False)`, so a wrong checkpoint would load "successfully" and emit noise.
- Checkpoint byte-exact: 3,703,305,959 B, matching the HF file listing.
- 200/200 generations ended `finish_reason=stop` — no truncation, no early stops.

## Method

| | |
|---|---|
| Texts | 200 utterances, 50 each × `TM_English` / `TM_English_Normal` / `TM_Malay` / `TM_Mandarin`, seeded (`random.Random(1024)`) from `texts_medium.jsonl`. 35.1 min audio, 407–876 tokens each (median 519) |
| LM | live engine `:9093`, `Scicom-intl/Multilingual-Expressive-TTS-1.7B-TMVoice-Synthetic`, temp 0.6, `repetition_penalty` 1.15 — the app's defaults |
| Arm A | vendored `app/neucodec` + `neuphonic/neucodec` → 24 kHz, GPU 4 |
| Arm B | WideCodec's own bundled package, `decoder_depth=20` → 44.1 kHz, GPU 7 |
| Arm C | Arm B resampled to 24 kHz — **bandwidth control** |
| Arm D/E | Arms A/B level-matched — **loudness control** |
| MOS | UTMOSv2 (`tm-h20-utmosv2`, internal `:8300`), **`reps=16`** |
| Pitch | `audiocheck.py` f0 gates, every arm resampled to a **common 16 kHz** so sample rate is not a variable |

Decode is **one-shot** (whole utterance, single window), not streamed — so this measures each codec's
intrinsic rendering rather than the crossfade stitcher's window seams. See *Limitations*.

### `reps=16` is not optional

UTMOSv2 picks TTA crops at random. On a **bit-identical** file:

| reps | repeated calls | spread | server cost |
|---|---|---|---|
| 1 | 2.656 / 2.984 / 2.641 | **±0.17** | 105 ms |
| 4 | 2.566 / 2.684 / 2.582 | ±0.06 | 107 ms |
| **16** | 2.662 / 2.605 / 2.646 | **±0.03** | 209 ms |

±0.17 MOS is larger than the effect being measured, so `reps=1` would have produced noise. `reps=16`
is nearly free because the server batches the crops (`max_batch=16`, which is also the cap).

## Result 1 — MOS on TTS-LM tokens: NeuCodec wins

| Arm | MOS mean | sd | p10 | median | p90 |
|---|---|---|---|---|---|
| **NeuCodec 24 kHz** | **3.152** | 0.161 | 2.960 | 3.164 | 3.346 |
| WideCodec 44.1 kHz | 2.940 | 0.220 | 2.640 | 2.950 | 3.181 |
| WideCodec → 24 kHz | 2.942 | 0.215 | 2.668 | 2.958 | 3.202 |

Paired (n = 200):

| comparison | Δ | 95% CI | t | WideCodec win rate |
|---|---|---|---|---|
| WideCodec − NeuCodec | **−0.212** | [−0.238, −0.186] | **−15.9** | 14% |
| WideCodec@24k − NeuCodec | −0.210 | [−0.237, −0.183] | −15.2 | 15% |
| WideCodec − WideCodec@24k | −0.002 | [−0.013, +0.009] | −0.4 | 48% |

Consistent across every voice:

| voice | NeuCodec | WideCodec | Δ | win rate |
|---|---|---|---|---|
| TM_English | 3.282 | 3.112 | −0.169 | 12% |
| TM_English_Normal | 3.102 | 2.938 | −0.164 | 20% |
| TM_Malay | 3.148 | 2.899 | −0.249 | 16% |
| TM_Mandarin | 3.076 | 2.810 | −0.266 | 10% |

**Both confounds are ruled out.**

- *Not bandwidth.* Resampling WideCodec to 24 kHz moves MOS by −0.002 (t = −0.4). The advantage
  above 12 kHz contributes nothing here.
- *Not loudness.* Raw NeuCodec peaks at a median of **1.100** and exceeds full scale on **145/200**
  clips (WideCodec: 1.066, 137/200) — the loudness-in-the-tokens effect. Level-matching both arms
  with a single scalar gain (common active-RMS, 0.99 peak guard, no limiter) leaves
  **Δ = −0.198 (t = −14.3)**. Both arms needed near-identical gain (median 0.643 vs 0.656).
  UTMOSv2 mildly prefers louder audio — matching cost NeuCodec 0.049 and WideCodec 0.036.

## Result 2 — on real audio they tie, which reframes Result 1

Same two decoders, but codes from the **frozen production encoder** over 131 real multilingual clips
instead of LM sampling (`encode_refs.py`):

| Arm | MOS mean | vs ground truth | t |
|---|---|---|---|
| ground truth (real audio) | 2.303 | — | — |
| NeuCodec resynthesis | 2.503 | +0.200 | +5.2 |
| WideCodec resynthesis | **2.514** | **+0.211** | +6.1 |
| **WideCodec − NeuCodec** | — | **+0.011** (95% CI [−0.048, +0.070]) | **+0.4**, 49% of 131 |

WideCodec is **statistically indistinguishable** from NeuCodec on real-audio codes, and both score
*above* ground truth — reproducing the model card's above-GT claim.

So nothing here contradicts WideCodec's published benchmark, which measures resynthesis. The −0.212
on TTS tokens is a **distribution/pairing effect**: our LM emits token sequences tuned to NeuCodec's
decoder; WideCodec's decoder was finetuned on codes from clean real audio.

## Result 3 — broken pitch: WideCodec is jumpier, and this part is intrinsic

`audiocheck.py`, all arms analysed at a common 16 kHz. Paired over 200 utterances:

| metric | NeuCodec | WideCodec | Δ (wide−neu) | t |
|---|---|---|---|---|
| `f0_jump_frac` — frames jumping >4 st (warble) | **0.0060** | 0.0084 | +0.0024 | **+7.3** |
| `f0_jump_max_st` | **11.93** | 13.13 | +1.20 | +3.0 |
| `f0_step_max_st` — step across 50 ms contiguous voicing | **11.96** | 12.81 | +0.86 | +2.4 |
| `f0_outlier_frac` — frames >6 st off median | **0.095** | 0.099 | +0.0033 | +2.6 |
| `f0_excursion_st` | **10.75** | 11.33 | +0.58 | +2.2 |
| `f0_excursion_ms` — register-break plateau | 81.8 | 83.2 | +1.4 | +0.8 (ns) |
| `f0_step_long_st` — sustained seam | 7.32 | 7.14 | −0.18 | −1.2 (ns) |
| `f0_iqr_st` — pitch spread | 4.483 | 4.479 | −0.003 | −0.1 (ns) |
| `f0_med` (Hz) — sanity | 171.31 | 172.52 | +1.21 | +7.3 |

The **shape** of this is specific and worth reading carefully: **identical median f0 and identical
IQR** — both decoders track the same voice over the same range — but WideCodec's track is measurably
less stable frame to frame. `f0_step_long_st` (the sustained "voice changes person" seam) does **not**
differ. This is **micro-warble, not register breaks.** Clearest single gate: clips exceeding 2% warble
frames go **3.5% → 10.0%**.

Unlike the MOS gap, this persists on real audio. Scored as **per-clip deviation from ground truth**
(lower = renders the true pitch track more faithfully), NeuCodec is closer on every metric:

| metric | \|NeuCodec − GT\| | \|WideCodec − GT\| | Δ | t |
|---|---|---|---|---|
| `f0_med` (Hz) | **2.47** | 5.74 | +3.27 | **+3.5** |
| `f0_step_max_st` | **4.14** | 5.39 | +1.25 | +2.7 |
| `f0_excursion_ms` | **31.1** | 44.8 | +13.7 | +2.7 |
| `f0_jump_frac` | **0.0067** | 0.0090 | +0.0023 | +2.4 |
| `f0_iqr_st` | **0.674** | 0.894 | +0.220 | +1.2 (ns) |
| `f0_jump_max_st` | **4.54** | 4.89 | +0.35 | +0.7 (ns) |

Interestingly the two err in *opposite directions* relative to real speech: ground truth sits at
`f0_jump_frac` 0.0116, NeuCodec **under**-renders pitch micro-variation (0.0089, smoother than real)
while WideCodec **over**-renders it (0.0159).

## Recommendation

**Maintain NeuCodec** for the TTS serving path:

1. **Quality** — +0.212 MOS with the current LM, winning 86% of paired utterances, plus a cleaner
   pitch track. No config recovers it (not bandwidth, not level).
2. **Switching cost is structural, not a model swap** — 44.1 kHz moves `samples_per_token` 480 → 882,
   which touches every `CUDA_GRAPH_BATCH` bucket, the crossfade window math in
   `audio_stream_crossfade()`, WSOLA, and the loudness stitcher.
3. **Throughput** — 287.9 M vs 185.6 M params, 38 ms vs 30 ms median decode. The codec decode *is*
   the documented bottleneck (`bench/OPTIMIZATION.md`), so that cost lands directly on throughput.
4. **The bandwidth is unusable in our delivery path** — prod serves LiveKit agents over WebRTC, which
   downsamples before anyone hears it. Arm C already showed the >12 kHz content adds no MOS anyway.

**Do not write WideCodec off.** It ties NeuCodec on real-audio resynthesis at ~0.8 kbps, which is
strong for a single-codebook codec, and it is the right tool where codes come from real audio rather
than LM sampling.

**What would flip this:** finetune WideCodec's decoder on TTS-sampled codes (or adapt the LM against
WideCodec's codes) to fix the pairing. Re-running is then one `sbatch`; the seeded 200-utterance set
makes new numbers directly comparable to the tables above.

## Limitations

- **One-shot decode, so streaming seams are untested.** The non-causal decoder's window effects and
  the crossfade stitcher are exactly where broken pitch bites in production, and WideCodec is not
  wired into the streaming path. A streaming variant is the obvious follow-up.
- **`audiocheck.py`'s absolute gates do not discriminate on this content.** `f0_step_max_st ≥ 6`,
  `f0_step_long_st ≥ 4` and `f0_outlier_frac > 0.05` fire on **85–90% of clips in both arms** —
  those thresholds were tuned on short clips, and these are 10 s multi-sentence utterances. Use the
  paired continuous deltas; the warble gate is the one that separates.
- **MOS is reference-free.** UTMOSv2 scores perceived naturalness, not fidelity — it rates both
  codecs *above* real audio. Pair with a reference metric (PESQ/mel-L1) before any decision that
  hinges on fidelity.
- **Resynthesis refs were 24 kHz**, so WideCodec's 44.1 kHz headroom could not be fully exercised
  there. Arm C makes this a minor concern for the MOS conclusion, but a ≥44.1 kHz ref set would be
  the stricter test.
- No text normalization was applied (identical text reaches both arms, so it cannot bias the A/B).

## Reproducing

Harness in `bench/widecodec_ab/`, raw results in `bench/results/widecodec_ab_*.jsonl`.

```bash
# on tm-h20, work dir /mnt/data/codec-ab  (NOT / -- rootfs runs ~98% full)
sbatch run.sbatch        # fetch WideCodec, generate tokens once, decode both arms, score
sbatch run_norm.sbatch   # level-matched rescore
sbatch run_resyn.sbatch  # real-audio resynthesis diagnostic
python analyze.py results.jsonl tokens.jsonl
```

GPUs are pinned with `CUDA_VISIBLE_DEVICES` and no `--gres`, matching `jobs/tm-h20/tts-api.yaml`.

### Gotchas hit building this

- **`encode_code(path)` is dead on torch 2.9** — `_prepare_audio` uses `torchaudio.load`, which now
  raises `TorchCodec is required for load_with_torchcodec`. Pass a `[1,1,T]` tensor instead, and
  resample to **16 kHz** yourself (the tensor branch does not resample).
- **That tensor must stay on CPU** — `encode_code` hands it to the HF feature extractor (numpy) and
  moves things to the device itself; a CUDA tensor gives
  `can't convert cuda:0 device type tensor to numpy`.
- **`NeuCodec._from_pretrained` ignores a `snapshot_download(local_dir=…)` copy** — it calls
  `hf_hub_download(repo_id=…)`, re-fetching 3.7 GB into `HF_HOME`. The local snapshot is still needed
  for the bundled `neucodec/` package. Use `local_ckpt_path=` or point `cache_dir` at the copy.
- **WideCodec's bundled package cannot build the 24 kHz baseline** — its `_from_pretrained` hardcodes
  `cls(44_100, 882, …)`. Load Arm A from the repo's vendored `app/neucodec` (`cls(24_000, 480)`).
- **Restrict the HF fetch with `allow_patterns`** — the WideCodec repo also holds `last.ckpt` (6.3 GB)
  and ~300 GB of sample wavs.
