# Interleaved multi-turn generation: does chunk N+1 sound like it continues chunk N?

Measured **2026-09-07** on **tm-h20 ("box 1024")**, against a **private interleave-trained
checkpoint** — an in-house 1.7B TTS LM packed with interleaved documents, deliberately not named
here because this document is public. It is the first checkpoint actually trained that way, and so
the first one on which `interleave_id` ([`INTERLEAVE.md`](../INTERLEAVE.md),
[`app/interleave.py`](../app/interleave.py)) can be judged on its merits instead of on a model that
never saw the shape.

> **The interleave fine-tune exists only on our own private checkpoints — no open-source TTS LM has
> it.** The interleaved-document packing (`pack_stage1.py --interleave_style full`) is an in-house
> training recipe, and the checkpoint measured here is a private model repo (the box's `HF_TOKEN` is
> what can read it; a laptop token gets 401). So **everything in this report is a property of that
> training, not of the prompt shape on its own** — see §7.2. Dropping an open-source TTS LM behind
> this API and switching `interleave_id` on buys nothing, and on the evidence of the earlier A/B it
> costs a 7.5% collapse rate.

Deployed as a **separate** vLLM (GPU 6, port 9086) with the scoring job on GPU 7. Prod `tts-api`
(:9091) and `vllm-tts` (:9093) were not touched.

---

## Verdict

**Interleaving works, and it works on exactly the thing it was built for.** Over 438 matched
consecutive-chunk pairs it cuts the prosodic discontinuity between chunk N and chunk N+1 by about a
quarter — **−0.41 semitones of register step (−25%)** and **−0.45 dB of level step (−28%)**, both
with a 95% CI well clear of zero. Against the one-shot rendering as the yardstick, cold chunking is
**significantly worse** on the loudness step between consecutive chunks (+0.26 dB [+0.12, +0.41])
while interleaving is **significantly better than one-shot** on both (−0.36 st, −0.19 dB). It also
recovers most of the paragraph-level pitch declination that cold chunking throws away. It costs **no latency** (the extra ~400 tokens of prefill are free),
**no intelligibility** (CER 0.44% vs 0.58%), and it never collapsed once in 518 chunks, which means
the `INTERLEAVE_FALLBACK` guard is dead weight on this checkpoint.

The one real cost is **~30 ms more silence at every join** (73 ms vs 43 ms), which is why an
interleaved paragraph runs ~1.7% longer than the same text spoken in one request.

| What the join sounds like | A one-shot | B interleave | C cold (today) |
|---|---|---|---|
| Register step from chunk N to N+1, \|st\| | 1.60 | **1.24** | 1.65 |
| Level step from chunk N to N+1, \|dB\| | 1.35 | **1.16** | 1.61 |
| Chunk-to-chunk register wander over the paragraph, st | 3.77 | **3.00** | 3.62 |
| Chunk-to-chunk level SD over the paragraph, dB | 0.99 | **0.96** | 1.20 |
| Paragraph pitch declination, st/s | −0.161 | **−0.137** | −0.082 |
| Silence at each join, s | 0.051 | 0.073 | **0.043** |

---

## 1. The question

In LiveKit the reply is not synthesized as one utterance. The `openai.TTS` plugin is
non-streaming, so `livekit-agents` wraps it in a `StreamAdapter` that cuts the LLM's reply into
sentence-ish chunks and sends **each chunk as its own `/v1/audio/speech` request**, one after the
other. The request for chunk N+1 carries no trace of chunk N, so the LM opens it cold: a fresh pitch
register, a fresh pace, a fresh energy, sampled independently at temperature 0.6. Each chunk is fine
on its own and the transition between them sounds awkward.

Interleaved prompting hands the LM the previous chunks' text **and the speech tokens it produced for
them** — the document shape the model was packed with (`pack_stage1.py --interleave_style full`):

```
<|im_start|>TM_English_Normal: Selamat pagi semua orang,<|speech_start|><|s_2551|>…<|s_53404|><|im_end|>
<|im_start|>TM_English_Normal: Nama saya husein.<|speech_start|>
                                                               ↑ the LM generates from here
```

So: **is a chunked paragraph rendered that way closer to the whole-paragraph rendering than a
cold-chunked one is?**

## 2. Design

Three conditions over the **same** 80 paragraphs with the **same** chunking:

| | Condition | Prompt per request |
|---|---|---|
| **A** | `single` | the whole paragraph in one request — the reference: what the model does when it gets everything |
| **B** | `interleave` | one request per chunk, each carrying the previous chunks' text + speech tokens (`build_prompt` / `fit_interleave`, retain 5 turns / 20 s — the API's own defaults) |
| **C** | `cold` | one request per chunk, each on its own — today's behaviour |

B and C see byte-identical chunk text, so **anything that differs between them is the prompt shape
alone**. Both are compared against A, which needs no chunking at all.

**Corpus** ([`bench/interleave_ab/corpus.py`](interleave_ab/corpus.py), cached to
`bench/results/interleave_ab/corpus.json`): 40 English + 40 Malaysian-Malay paragraphs written by
the `OPENAI_*` LLM, 37–55 words each, on everyday assistant topics, with real commas and full stops
so they chunk where a `StreamAdapter` would chunk them. Digits and abbreviations are deliberately
excluded — the text goes to the LM verbatim (this A/B bypasses the normalizer), so anything
unspeakable would be read as garbage in all three conditions and only add noise. `chunk_text()` is a
copy of what the `StreamAdapter` does: split at sentence ends, then at commas, merging pieces until
each is ≥20 chars (its `min_sentence_len` default). Result: **6.5 chunks per paragraph**, 518
requests per chunked condition.

**Generation** ([`generate.py`](interleave_ab/generate.py)) talks to vLLM's `/v1/completions`
directly, not through `/v1/audio/speech`, which takes the normalizer, the crossfade stitcher and the
per-request loudness normalizer out of the comparison and leaves the LM. Sampling is prod's:
temperature 0.6, `repetition_penalty` 1.15, `max_tokens` 3072, voice `TM_English_Normal`. The prompt
construction and the history bounds are **imported from `app/interleave.py`**, so B is exactly the
prompt the API would have built. `INTERLEAVE_FALLBACK` is deliberately **not** applied: a collapsed
chunk is recorded, not retried, because how often the model collapses is one of the things being
measured.

**Decode** ([`decode.py`](interleave_ab/decode.py)): for B and C the chunks' token lists are
concatenated and decoded in a **single** NeuCodec window, exactly like A. The streaming stitcher's
growing windows, crossfade and loudness normalization are already characterized
([CLAUDE.md](../CLAUDE.md), [TTFB.md](TTFB.md)) and would otherwise sit on top of the effect being
measured; the codec's own boundary behaviour is identical for B and C, so it cancels. What is left
in the audio is the LM's prosody: identical decoder, identical window, identical everything but the
token stream.

## 3. How the join is measured

Two independent instruments, because they have different weaknesses
([`score.py`](interleave_ab/score.py), `librosa.pyin` at 16 kHz, 10 ms hop):

**Whole-chunk medians (the robust one).** Per chunk, the median voiced f0 and the active level; then
the step from chunk N to chunk N+1. No knowledge of where the boundary sits within a few hundred
milliseconds is needed, so all three conditions are on equal footing, and a ±100 ms error in A's
inferred boundaries barely moves a median taken over a ~2.3 s chunk. **This is the table to trust.**

**A seam probe (the sharp but fragile one).** At each boundary, locate the silent run straddling it
and read 250 ms of speech outward from each edge of that silence — so what is compared is the speech
*before* the pause against the speech *after* it, with the same geometry whether the pause is 20 ms
or 300 ms. Reported both as |step| (how far the prosody moved) and **signed** (which way), because
the sign is the diagnostic: a chunk rendered as if it stood alone ends on a terminal fall and the
next opens in its own starting register, so cold chunking should step consistently **up**, whereas
continuous speech drifts gently **down** across a phrase boundary. The same probe runs at interior
points (0.5 s spaced, kept 0.6 s clear of any boundary) to give the within-utterance noise floor of
that same clip.

A has no seams, so it is given **virtual** ones: the boundary is estimated by character proportion
(speech rate inside one utterance is uniform enough — the same assumption `trim_turn_tail` makes)
and snapped to the nearest silent run of ≥40 ms within ±0.5 s, since a chunk boundary is a
punctuation boundary and those carry a pause. The 40 ms floor matters: without it the snap lands on
plosive closures and puts the probe inside a word.

**A's seam-probe row is not a trustworthy reference.** Changing the snapping rule moved A's mean
signed step by 0.4 st — the same order as the B↔C difference being measured. So the seam probe is
used for **B vs C** (both exact, paired 1:1 since they chunk the same text at the same places) and
the whole-chunk metrics are used whenever A is in the comparison.

Also scored: Whisper-large-v3-turbo CER/WER (the box's `:9089` engine) against the corpus text, and
UTMOSv2 naturalness MOS at `reps=16` (`:8300`), both raw and after level-matching every clip to a
common active RMS.

## 4. Results

### 4a. Chunk N+1 against chunk N — whole-chunk medians, 438 pairs

| Condition | signed register step (st) | **\|register step\| (st)** | signed level step (dB) | **\|level step\| (dB)** | steps up >1 st |
|---|---|---|---|---|---|
| A single | −0.372 | 1.604 | −0.235 | 1.348 | 24% |
| B interleave | −0.274 | **1.241** | −0.222 | **1.157** | 20% |
| C cold | −0.182 | 1.648 | −0.159 | 1.608 | 28% |

Paired over the 438 matched pairs, bootstrap 95% CI on the mean difference:

| Metric | pair | signed diff [95% CI] | **\|step\| diff [95% CI]** |
|---|---|---|---|
| register step (st) | **B−C** | −0.091 [−0.298, +0.115] | **−0.407 [−0.556, −0.261]** |
| register step (st) | B−A | +0.098 [−0.088, +0.279] | **−0.363 [−0.501, −0.226]** |
| register step (st) | C−A | +0.190 [−0.024, +0.395] | +0.044 [−0.100, +0.189] |
| level step (dB) | **B−C** | −0.063 [−0.263, +0.139] | **−0.451 [−0.586, −0.320]** |
| level step (dB) | B−A | +0.013 [−0.150, +0.176] | **−0.191 [−0.309, −0.069]** |
| level step (dB) | C−A | +0.076 [−0.150, +0.296] | **+0.260 [+0.119, +0.407]** |

This is the headline, and it reads cleanly in three parts:

- **Interleaving cuts the chunk-to-chunk discontinuity by a quarter** (−0.41 st, −0.45 dB, both CIs
  clear of zero).
- **Cold chunking is measurably worse than one-shot on loudness** (+0.26 dB, CI clear of zero): the
  loudness jump between consecutive chunks is bigger than anything the model does inside a
  continuous utterance. On pitch it merely matches one-shot (+0.04 st, not significant).
- **Interleaving is tighter than even the one-shot reference** (−0.36 st, −0.19 dB, both significant),
  which makes sense rather than being too good to be true: A is one continuous utterance with genuine
  expressive movement between its phrases, while B pins each new chunk to the register it just left.

What changes is the **size** of the jump, not its direction: every paired *signed* difference is
non-significant. The absolute signed values are all negative (−0.37 / −0.27 / −0.18 st) — that is
declination — and their ordering A < B < C tells the same story from another angle: the one-shot
rendering descends most across its chunks, cold chunking descends least, interleaving lands in
between and nearer A.

### 4b. The seam probe, pooled over every boundary

| Metric | A single (inferred) | B interleave | C cold | B−C paired mean [95% CI] |
|---|---|---|---|---|
| signed f0 step (st) | +0.426 | **−0.353** | +0.529 | −0.923 [−1.585, −0.283] |
| \|f0 step\| (st) | 4.357 | 5.253 | 4.737 | +0.558 [+0.115, +1.008] |
| joins stepping up >1 st | 44.9% | 50.1% | 56.3% | −6.5 pp [−12.0, −1.2] |
| signed level step (dB) | +5.812 | +4.896 | +3.023 | +1.873 [+1.059, +2.698] |
| \|level step\| (dB) | 9.439 | 7.563 | 4.947 | +2.616 [+2.021, +3.217] |
| silence at the join (s) | 0.0510 | 0.0731 | 0.0427 | +0.0304 [+0.0245, +0.0363] |
| interior signed f0 step (st) | −0.800 | −0.905 | −0.670 | |

This probe reads the 250 ms either side of the pause, so it sees what happens *at* the boundary,
where §4a's whole-chunk medians average it away. Read together the two are coherent, not
contradictory:

- **The upward register reset is real and interleaving removes it.** Cold joins step **up** +0.53 st
  on average and 56% of them step up by more than a semitone. Interleaved joins step **down** −0.35 st
  — the same direction the interior of the utterance drifts (−0.67 to −0.91 st). Paired difference
  −0.92 st, CI clear of zero. This is the local, boundary-adjacent half of the effect; §4a is the
  whole-chunk half.
- **But B's boundaries are locally *more* dynamic, not less** (|f0 step| 5.25 vs 4.74 st, |level
  step| 7.56 vs 4.95 dB). Cold chunks are flatter at their seams than *natural speech is* — C's
  level excess over its own interior is 2.12 dB against A's 3.45 dB. That is the tell of uniform
  blandness: every cold chunk is rendered in the same canonical sentence contour, so consecutive
  ones meet smoothly by both being unremarkable. Interleaving produces genuine intonation across the
  boundary while keeping the two chunks in one register, which is what 4a measures and what
  continuous speech does.
- **B pauses longer at joins**: 73 ms against 43 ms for C and 51 ms for A. This is the one place C is
  closest to the reference, and it accounts for B's extra duration.

### 4c. Whole paragraph

| Condition | duration s | dur vs A | words/s | f0 median Hz | f0 IQR st | **declination st/s** | active level dB | **chunk f0 spread st** | **chunk level SD dB** |
|---|---|---|---|---|---|---|---|---|---|
| A single | 14.98 | 1.000 | 2.89 | 184.4 | 3.91 | **−0.161** | −17.1 | 3.77 | 0.99 |
| B interleave | 15.58 | 1.017 | 2.92 | 186.0 | 3.94 | **−0.137** | −17.4 | **3.00** | **0.96** |
| C cold | 15.06 | 1.028 | 2.83 | 192.0 | 4.10 | **−0.082** | −16.9 | 3.62 | 1.20 |

- **Declination.** A paragraph spoken as one utterance drifts down in pitch, and that drift is a
  large part of why it sounds like one thought. Cold chunking loses half of it (−0.082 vs −0.161 st/s);
  interleaving recovers ~70% (−0.137). Paired: C−A **+0.065 [+0.033, +0.080]** (significantly flatter
  than the reference), B−C **−0.062 [−0.083, −0.011]**, B wins 64% of paragraphs. B−A is not
  significant.
- **Register.** C sits **7.6 Hz (≈0.7 st) higher** than the one-shot rendering — each chunk opening
  in its own fresh, slightly raised register. B lands within 0.15 st of A.
- **Chunk-to-chunk loudness consistency** (level SD across a paragraph's chunks): B−C
  **−0.207 dB [−0.447, −0.065]**, B wins **71%** of paragraphs. C−A is +0.292 [−0.012, 0.348] —
  marginal at the paragraph level, but the same effect measured per chunk pair in §4a is clear
  (+0.26 dB [+0.12, +0.41]). Note this is all *before* the API's own `STREAM_NORMALIZE`, which
  applies one gain per request and would partly mask it.
- **Register wander across the paragraph**: B 3.00 st vs C 3.62 st, paired −0.575 [−0.950, −0.200],
  B wins 65%.
- **Duration**: both chunked conditions run longer than one-shot — B +1.7%, C +2.8% — and are
  indistinguishable from each other (paired ratio +0.005 [−0.016, +0.015]).

### 4d. Intelligibility, naturalness, and what it costs

| Condition | CER % | WER % | UTMOSv2 | UTMOSv2 level-matched | requests | collapsed | finish=length | median prompt tok | median LM latency s |
|---|---|---|---|---|---|---|---|---|---|
| A single | 0.50 | 2.25 | 3.181 | 2.817 | 80 | 0 | 0 | 80 | 2.762 |
| B interleave | 0.44 | 2.13 | 3.180 | 2.776 | 518 | **0 (0.0%)** | 0 | **418** | **0.423** |
| C cold | 0.58 | 2.11 | 3.226 | 2.835 | 518 | 0 (0.0%) | 0 | 19 | 0.427 |

- **No collapses at all.** `INTERLEAVE.md` §2b records 3/40 chunks where the pre-interleave model,
  handed history, decided the utterance was already over and emitted end-of-speech after a handful
  of tokens. On this checkpoint: **0 of 518**. `INTERLEAVE_FALLBACK` never fires — it is now
  insurance, not a working part.
- **Prefill is free.** History costs a median of **+399 prompt tokens** (418 vs 19) and the LM
  latency is unchanged: 0.423 s vs 0.427 s. Exactly what `INTERLEAVE.md` predicted, now measured.
  History reached the 5-turn cap on 121 of 518 requests, at a median of 344 and a maximum of 888
  speech tokens; nothing ever hit `max_tokens`.
- **No intelligibility cost.** CER 0.44% (B) vs 0.58% (C) vs 0.50% (A); the paired median difference
  is 0.0000 in every pairing. Whisper hears the same words.
- **MOS says nothing, and cannot.** B−C level-matched is −0.048 [−0.216, +0.033]: not significant.
  B−A is −0.101 [−0.161, −0.020], a real but tiny deficit against the one-shot rendering. Treat all
  of this as near-uninformative here: **UTMOSv2 scores random crops of a single clip, so it is
  structurally blind to continuity across a boundary** — the one thing this experiment is about. It
  also mildly prefers louder audio, which is why the level-matched column exists.

### 4e. Per language

| | A single | B interleave | C cold |
|---|---|---|---|
| en — chunk level SD dB | 0.955 | **0.973** | 1.331 |
| ms — chunk level SD dB | 1.066 | **0.954** | 1.136 |
| en — seam f0 excess st | 1.550 | 3.149 | 2.924 |
| ms — seam f0 excess st | 2.827 | 3.048 | 2.238 |
| en — silence at join, excess s | 0.040 | 0.080 | 0.030 |
| ms — silence at join, excess s | 0.033 | 0.058 | 0.035 |
| en — CER % | 0.25 | 0.00 | 0.00 |
| ms — CER % | 0.68 | 0.79 | 0.70 |

The loudness-consistency win holds in both languages and is larger in Malay (0.95 vs 1.14 dB).
The seam-probe numbers split by language the same way they split overall — English shows the bigger
|step| increase under B, Malay the smaller — and with A's row unreliable at seam level there is
nothing here that changes the verdict either way.

## 5. Caveats

- **A's boundaries are inferred.** Everything in §4b that involves A is soft; §4a and §4c are not.
- **n = 80 paragraphs / 438 joins, one voice, one temperature.** `TM_English_Normal` at 0.6 only.
  Per-paragraph medians are underpowered for a per-seam effect, which is why the headline numbers are
  pooled per join with a bootstrap CI over joins.
- **The serving path is excluded on purpose.** No normalizer, no crossfade stitcher, no
  `STREAM_NORMALIZE`, no `speaking_rate`. In production `STREAM_NORMALIZE` applies one loudness gain
  per request, which partly masks the chunk-to-chunk level spread measured in §4c — so C's real-world
  loudness inconsistency is smaller than 1.20 dB, and carrying the locked gain through the interleave
  store (the follow-up `INTERLEAVE.md` §9 already names) is the way to close the rest.
- **The metric is not the ear.** A 0.4 st register step and a 0.45 dB level step are small in
  absolute terms; whether they are the difference between "awkward" and "continuous" is a listening
  question. Level-matched samples for four paragraphs × three conditions are in
  `audio/interleave_ab/` (gitignored) — `en000` and `ms005` are among the largest B-over-C
  improvements, `ms012` is the median case, and `en037` is one of the few where C came out tighter.

## 6. Reproducing

```bash
# 0. a SEPARATE vLLM for the interleave checkpoint (GPU 6, :9086) -- never prod's :9093
ssh $BOX 'mkdir -p /mnt/data/interleave-ab/logs; cd /mnt/data/interleave-ab && \
  PORT=9086 GPU=6 sbatch --nodelist=$(hostname) vllm_interleave.sbatch'   # bench/interleave_ab/

# 1. the corpus (laptop; cached, one-off)
set -a; source .env; set +a
uv run --with aiohttp python bench/interleave_ab/corpus.py --per-lang 40

# 2. everything else on the box (generate -> decode -> score -> quality -> analyze)
rsync -az --exclude .git --exclude bench/results ./ $BOX:/mnt/data/interleave-ab/repo/
scp bench/results/interleave_ab/corpus.json $BOX:/mnt/data/interleave-ab/out/
ssh $BOX 'cd /mnt/data/interleave-ab && GPU=7 LM=http://127.0.0.1:9086/v1/completions \
  VOICE=TM_English_Normal sbatch --nodelist=$(hostname) repo/bench/interleave_ab/run.sbatch'
# STAGES="score analyze" re-runs just the analysis; ~30 s generate, 16 s decode,
# 70 s score, 170 s quality for all 240 clips.
```

Raw results in `bench/results/interleave_ab/`: `corpus.json`, `tokens.jsonl` (every request's
tokens, prompt size, latency, finish reason), `acoustics.jsonl`, `quality.jsonl`,
`report_tables.md` (the generated tables this report is written from).

## 7. What to do with it

1. **Ship it.** The feature is already on `main` and defaults on with `MAX_RETAIN_INTERLEAVE=5`;
   nothing in these numbers argues against those defaults, and the prefill is free.
2. **Deploy an interleave-trained checkpoint if the feature is wanted — there is no open-source
   substitute.** The effect is a property of the interleaved-document *training*, which only our own
   private checkpoints have: the earlier A/B on a model packed without it
   ([`INTERLEAVE.md`](../INTERLEAVE.md) §9, branch `feat/speech-context`) found no measurable
   improvement **and** a 7.5% collapse rate, and both changed here purely because the checkpoint
   changed. Treat `interleave_id` as a feature gated on the served model: on an interleave-trained
   checkpoint it is a free win, on anything else (an open-source TTS LM, or an older in-house one
   packed without `--interleave_style full`) it is a regression risk and `INTERLEAVE_STORE=off` is
   the right setting.
3. **`INTERLEAVE_FALLBACK` can be left on but no longer earns its keep** (0/518). Keep it as
   insurance for other checkpoints; it costs nothing when it does not fire.
4. **Next**: carry the locked `STREAM_NORMALIZE` gain through the turn store, so the loudness
   continuity the LM now provides is not re-broken by a per-request gain; and look at the extra 30 ms
   of join silence — trimming trailing-silence tokens from stored turns is the obvious lever, and
   `INTERLEAVE.md` §9 already lists it.
