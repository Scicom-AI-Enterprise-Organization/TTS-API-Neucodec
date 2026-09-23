# CLAUDE.md

Guidance for working in this repository — a **streaming Text-to-Speech (TTS) and Voice-Conversion (VC) API**.

## Architecture (read this first)

Two GPU services **colocated on a single GPU**:

1. **vLLM LM server** (`vllm.yaml`, `Dockerfile_vllm`) — serves the autoregressive TTS model
   `Scicom-intl/Multilingual-TTS-1.7B-Base`, a **Qwen3-1.7B** continued-pretrained to emit discrete
   *speech tokens* (`<|s_NNNN|>`). Listens on `:9093` (OpenAI `/v1/completions`). This is the
   text → speech-token stage and is **not** the bottleneck.
2. **FastAPI app** (`app/main.py`, `Dockerfile`) — the public API (`:9091`). Normalizes text, streams
   a prompt to vLLM, parses returned `<|s_NNNN|>` tokens, and uses **NeuCodec** (vendored under
   `app/neucodec/`) to decode tokens → 24 kHz audio. The speech-token → waveform stage **is** the
   bottleneck and is where optimization effort belongs.

```
client → app:9091 ──prompt──▶ vLLM:9093 (Qwen3-1.7B) ──speech tokens──▶ app ──NeuCodec decode──▶ PCM/WAV
        (normalize)            autoregressive LM                          (dynamic batch + CUDA graphs)
```

Both share one GPU: vLLM is capped with `--gpu-memory-utilization`; NeuCodec uses the rest.

## Key files

- `app/main.py` — the whole serving app: `/v1/audio/speech` (TTS), `/v1/audio/vc` (voice conversion),
  `/v1/audio/normalize`, `/v1/audio/speaker`. Holds the dynamic-batching + CUDA-graph decode pipeline.
- `app/env.py` — all runtime config, read from environment / `.env`.
- `app/interleave.py` — **interleaved generation, needs an interleave-trained LM** (`interleave_id` on `/v1/audio/speech`,
  aliases `request_id`/`context_id`, header `X-Interleave-Id`/`X-Context-Id`). Requests sharing
  an id are prompted with the previous turns' text + the speech tokens the LM produced for them,
  in the document shape the model was packed with (GPUPlatform `pack_stage1.py
  --interleave_style full`): `<|im_start|>{spk}: {text_i}<|speech_start|>{audio_i}<|im_end|>` ×N,
  then the new turn opened at `<|speech_start|>`. Why: LiveKit's `StreamAdapter` cuts a reply into
  ~5-word chunks and sends each as its own request, so the LM at T+1 had no idea it continued T —
  cold pitch/pace/energy at every join. History is prefill only (tens of ms); the codec path is
  untouched. Retention is `MAX_RETAIN_INTERLEAVE` turns (**default 5**, per-request field
  `max_retain_interleave`) *and* `INTERLEAVE_MAX_S` seconds, left-trimmed (oldest whole turns
  first, then the oldest survivor cut to its tail with its text shortened in proportion);
  `fit_interleave()` clamps `max_tokens` so prompt+generation fit `LM_MAX_MODEL_LEN` (vLLM 400s
  rather than truncating). Turns are stored per id as one JSON file in `/dev/shm` (atomic replace
  + a directory-wide flock) so all uvicorn workers share them — a chunk's turn is committed
  *before* the stitcher's end-of-stream marker, so chunk N is stored before its response
  completes and N+1 sees it on any worker. Only clean finishes are stored (`finish_reason=length`
  means the tokens stop mid-text). `INTERLEAVE_FALLBACK` regenerates a chunk cold if the LM stops
  after a handful of tokens. Voice switches start cold. `GET`/`DELETE /v1/audio/interleave/{id}`
  inspect/forget an id. No torch import — `tests/test_interleave.py` (44) runs anywhere.
  **Measured (2026-09-07, 80 paragraphs / 438 matched chunk pairs, `INTERLEAVE.md` §2c +
  `bench/INTERLEAVE_AB.md`): the jump between consecutive chunks drops −0.41 st of pitch register
  [−0.56, −0.26] and −0.45 dB of level [−0.59, −0.32] (≈−25%), the upward register reset at the
  join (+0.53 st cold) becomes the −0.35 st downward drift continuous speech has, and ~70% of the
  reply's pitch declination is recovered. Free: +399 prefill tokens, 0.423 s vs 0.427 s LM latency,
  CER unchanged. Costs ~30 ms more silence per join. `INTERLEAVE_FALLBACK` fired 0/518 (3/40 on the
  pre-interleave model).** Full write-up: `INTERLEAVE.md`.
- `app/wrapper.py` — `CUDAGraphsWrapper`: captures one CUDA graph per `(batch, token-length)` bucket.
- `app/timestretch.py` — `WSOLA`: streaming pitch-preserving time stretch behind the `speaking_rate`
  request field (alias `speed`; default `DEFAULT_SPEAKING_RATE=1.0`, range 0.5–2.0). The LM has no
  rate control, so this runs on the stitched PCM in `stream_speech()` (`time_stretch_pcm16`, after
  crossfade + loudness normalization, before pcm/wav/SSE), stateful across chunks with ~55 ms
  lookahead; 1.0 bypasses it. Pure numpy — `tests/test_timestretch.py` runs without GPU. Output
  is bit-identical regardless of chunking, which the tests rely on; keep it that way.
- `app/neucodec/` — **vendored** NeuCodec (`from app.neucodec import NeuCodec`; *not* the pip package).
- `app/normalizer/`, `app/rules.py` — Malaysian/multilingual text normalization + markdown sanitization.
- `app/llm_normalizer.py`, `app/prompt.py` — **LLM-based normalizer** (`mode: "llm"` on
  `/v1/audio/normalize` and TTS requests; default `mode: "rule"` is the pipeline above). Calls any
  OpenAI-compatible `/chat/completions` (`OPENAI_BASE_URL`/`OPENAI_API_KEY`/`OPENAI_MODEL_NAME`,
  timeout `OPENAI_TIMEOUT`) with the few-shot prompt in `prompt.py`, output constrained to
  `{"normalized": "..."}` via `response_format` json_schema (retries once without it on 400 for
  backends lacking guided decoding). Importable without torch/GPU — keep it that way so
  `tests/test_llm_normalizer.py` runs anywhere. On LLM failure: `/v1/audio/normalize` returns 502
  (400 if unconfigured); the TTS path falls back to rule-based so speech is still produced.
  Request-field defaults are env-driven: `DEFAULT_NORMALIZER_MODE` (`rule`|`llm`) and
  `DEFAULT_NORMALIZE_MALAYSIAN` (bool) set what requests get when they omit `mode` /
  `normalize_malaysian`.
  **TTFB shortcut (`LLM_NORMALIZER_SKIP_PLAIN`, default on):** the LLM round trip is ~0.55 s and
  sits entirely before the first audio byte, yet on plain conversational text the model returns
  its input unchanged. `needs_normalization()` in `llm_normalizer.py` skips the call when the
  pre-normalized text has no digit, symbol, ALL-CAPS/dotted token or known abbreviation; the
  output is identical (51/51 sentences vs the live LLM, `bench/normalizer_gate_eval.py` — rerun
  it after touching the gate or the prompt). Conservative: anything doubtful still goes to the LLM.
- `app/spoken_normalizer/` — **rule-based replica of the LLM normalizer** (`mode: "spoken"`), pure Python,
  importable anywhere: `numbers.py` (cardinals/ordinals/years/digits for en, ms, zh, ta — Tamil with sandhi,
  lakhs below 10⁷, `ta_attach()` glues case suffixes: 2024ல் → …நான்கில்), `lang.py` (script → zh/ta;
  Malay-vs-English by marker words per sentence, and per number in code-switched sentences: distance-weighted
  neighbour vote + sentence prior), `core.py` (ordered regex handlers: email, url, negative sign, IC, numeric
  dates, month-name dates, phone, military time, Malay period ranges, time ranges, times, dotted times with a
  cue, versions, percent/money/unit ranges, money, percent, units, `/unit`, ordinals, decades, hyphenated
  compounds, `10k`/`2x`, digit groups, fractions & slashes & d/m dates, ranges & scores, `#N`, alphanumeric
  ids, years, plain numbers, abbreviations). Built against the LLM's own outputs (`bench/normalizer_corpus.py`
  → `bench/normalizer_truth.py` → `bench/results/normalizer_truth.jsonl`, 497 sentences: 300 everyday +
  197 harder cases added 2026-09-04) and scored by `bench/normalizer_agreement.py`: 82% verbatim agreement
  (en 89 / ms 87 / zh 88 / ta 67 / code-switch 59; the original 300 still 85%). Of the 88 residual diffs, 28
  are LLM errors (it returned 8 Tamil sentences with digits in them), 50 the LLM contradicting itself, 10
  code-switch hybrids; every digit is read in all 497. Full write-up in `bench/NORMALIZER.md`, 1,726 tests,
  ~45 µs. Also the fallback for `mode=llm`, and with `LLM_NORMALIZER_RULE_FIRST=true` the LLM is only
  called for what the rules leave unspeakable. Regex gotchas that each cost a whole language or category:
  Python's `\w`/`\b` treat CJK and Tamil letters as word characters, so digit boundaries must be ASCII
  classes (`(?<![A-Za-z0-9_])`); trailing lookaheads must allow a sentence-final `.` (`(?![...])(?!\.\d)`,
  never a class containing `.`); a letter glued to a digit (`3A`) must not vote in language detection; a
  bare 7-digit run is a phone number *unless* it is round (`1000000`); and a new time/number shape needs a
  cue word or it will eat decimals (`at 3.25 per annum` is not a time).
- `app/tracing.py` — OpenTelemetry spans on the hot path (`ENABLE_TRACING_SPANS`, **default on**,
  but gated on an exporter actually being configured; off, no exporter, or no opentelemetry ⇒
  every helper is a shared `nullcontext()` / a `None`-returning no-op, so the GIL-bound decode
  loop pays nothing). Exports spans through the provider
  `wan.patch()` installs, so they share the trace id with the JSON log lines.
  (`wan` is the observability library, ex-`fastapi-loki-tempo` — renamed repo *and* package.)
  Span tree and the reasoning behind it: module docstring + README "Tracing (Loki + Tempo)".
- `vllm.yaml` / `docker-compose.yaml` — the two services, sharing external docker network `tts-network`.
- `bench/` — benchmark + Whisper-CER harness, RunPod deploy scripts, and recorded results (see `bench/OPTIMIZATION.md`).
- `bench/widecodec_ab/` — codec A/B harness: decodes ONE LM token stream through two decoders, so the
  codec is isolated from temp-0.6 sampling noise. Verdict (`bench/WIDECODEC_AB.md`): **keep NeuCodec**.
  `Scicom-intl/WideCodec` (44.1 kHz decoder-only finetune, shared frozen FSQ codebook) scores −0.212
  UTMOSv2 on our TTS tokens yet **ties** NeuCodec on real-audio resynthesis (+0.011) — an LM/decoder
  pairing effect, not codec quality, since the LM was trained against NeuCodec's decoder. Its pitch
  track is intrinsically jumpier in both conditions (warble clips 3.5% → 10.0%), though sustained
  seams don't differ. UTMOSv2 scoring is stochastic — **use `reps=16`** (`reps=1` spreads ±0.17 MOS on
  a bit-identical file, larger than the effect).
- `bench/multilingual_normalizer/` — **written → spoken dataset generator for fine-tuning a small normalizer LLM**,
  16 locales (en ms id zh ta ta-LK si tl ar fr es de it pt nl pl). Two sources, tagged per row: `template` (LLM-written
  sentence templates with typed slots `{money} {date} {phone} …`, filled with random locale-formatted values and
  verbalized **deterministically** — `verbalize.py`: num2words for ar/fr/id/es/de/it/pt/nl/pl, own tables for tl/si,
  `app.spoken_normalizer` for en/ms/zh/ta; `SAFE_SLOTS` keeps grammar-sensitive shapes out of pl/ar/si/tl) and `llm`
  (natural sentences written and normalized by the OPENAI_* LLM with a multilingual prompt + 2 deterministic few-shots,
  kept only if no digit survives, right script, ≥80% words preserved). Plus **6 Malaysian code-switched pairs**
  (`codeswitch.py`: ms-en, en-ms, zh-en, zh-ms, ta-en, ta-ms, 1,500 rows each) — hand-written frames that tag the
  read-language on every slot (`{money:ms}`, `{date:en}`) because in rojak the number is read in the language of the
  fragment it sits in; the spoken side is `app.spoken_normalizer` with the language **forced per slot**, normalized
  together with the carrier words next to it and stripped again (the cue is what decides: `704251` is a quantity,
  `nombor rujukan anda 704251` is digit by digit). The LLM is **not** the teacher here — for `bil anda RM250` it said
  "ringgit malaysia dua ratus lima puluh". Output in `bench/results/multilingual_normalizer/`
  (`train/val/test.jsonl`, `*_sft.jsonl`, `stats.md`; build 2026-09-06: 52,698 rows, 49k template incl. 9k code-switch
  + 3.7k LLM), published as **[Scicom-intl/Multilingual-Normalizer](https://huggingface.co/datasets/Scicom-intl/Multilingual-Normalizer)**;
  split by template id so no frame leaks across splits. README there
  has the per-locale grammar caveats (si/tl/ar/pl and the Tamil code-switch rows need native review). Commands:
  `templates_llm` → `generate --per-locale N --cs-per-locale N` (free) → `llm_pairs` → `build --sft`, all via
  `python -m bench.multilingual_normalizer.<step>` with `uv run --with num2words --with aiohttp`; every LLM stage is
  cached and incremental. `synthetic-normalizer/` is a standalone copy of the same pipeline (own `data/`, `SN_*` env)
  — **edit both or neither**.
- `bench/interleave_ab/` — **interleave A/B harness**: the same 80 paragraphs (40 en + 40 ms) rendered
  three ways — one-shot, interleaved chunks, cold chunks — through a *separately deployed* vLLM, decoded
  one-shot so the stitcher cannot confound it, then scored for chunk-to-chunk pitch/level continuity,
  declination, CER and UTMOSv2. Verdict (`bench/INTERLEAVE_AB.md`, on the private interleave-trained
  checkpoint — the repo is public, so the model is named only in the untracked launcher and in
  `bench/results/interleave_ab/`): **interleaving works** — the chunk N→N+1
  register/level jump drops ~25% (−0.41 st, −0.45 dB, n=438 paired), cold chunking loses half the
  paragraph declination, prefill is free (+399 tokens, no latency), CER unchanged, and `INTERLEAVE_FALLBACK`
  never fired (0/518 vs 3/40 on the pre-interleave model). Costs ~30 ms extra silence per join.
  **The interleave fine-tune only exists on our private checkpoints** (in-house `--interleave_style full`
  packing; no open-source TTS LM has it), so this is a property of the *training*, not of the prompt shape:
  `interleave_id` is a free win on an interleave-trained checkpoint and a regression risk on anything else
  (`INTERLEAVE_STORE=off` there). Two traps:
  the one-shot arm has no real boundaries so it is only usable for whole-chunk metrics, and |step| alone
  flatters cold chunking (it is *blander* at its seams than natural speech) — the **sign** is what exposes
  the register reset.
- `bench/synth/` — **render a sentence file through N checkpoints** for listening/scoring A/Bs
  (`gen_tokens.py` → `<|s_N|>` ids with offline vLLM, one process per checkpoint, kept as
  `tokens/<slug>.json`; `decode_tokens.py` → 24 kHz wav with the vendored NeuCodec; `run_models.sh`
  = both, LM on one GPU and codec on another). Deliberately *not* the serving app — no normalizer,
  no stitcher — so only the weights differ between models. Three choices that keep it a checkpoint
  comparison: **one-shot decode** of the whole token stream (the streaming stitcher's ~0.7–1.5 dB
  envelope tilt is the non-causal decoder, not the model); **prod's loudness treatment applied
  identically** — raw peaks exceed full scale on ~half the utterances (1.25–1.40 measured), so
  `normalize_chunk` from `app/main.py` is re-applied in one-shot form and the untouched output kept
  as float32 in `raw/`, otherwise a PCM_16 write clips and the louder checkpoint wins on volume;
  and **tokens are kept**, so a decode change re-runs without the LM. `suspect rows` in the log
  (finish_reason≠stop, or <10 tokens) is the gate to read before trusting a set. Env pins are
  load-bearing: **vLLM 0.10.2 needs `transformers==4.56.2`** (5.x ⇒ `Qwen2Tokenizer has no attribute
  all_special_tokens_extended`, after the weights load) and **numpy ≥2.1, not the repo's 1.26.4**
  (scipy's `np.long` kills `import vllm`). Driving it on a shared GPU box: skill `tts-synth-checkpoints`.
  First set: `ucc_ai_research/evaluation/tts/synthetic-audio/2026-09-15/` (20 TM voicebot sentences
  × 3 interleave checkpoints, `TM_English_Normal`, temp 0.6 / rep 1.15).
- `app/batching.py` — **one decode per distinct token length**, and the reason: NeuCodec's decoder is
  non-causal, so padding a short window up to a longer one in the same batch feeds it right-context
  the unpadded decode never had and changes the samples that are kept (−1.3 dB SNR; equal lengths
  are bit-exact). Imports nothing, so `tests/test_decode_batching.py` runs anywhere. Full story:
  `bench/PADDING_BUG.md`.
- `bench/latency_bench.py` — **TTFB + end-to-end + RTF with the full percentile spread**
  (p10/p50/p90/p95/p99) under closed-loop concurrency. Use this and not `bench/bench.py` for
  latency: that one requests `wav`, and a wav response emits its 44-byte header before a single
  token is decoded, so its "TTFB" times the header and reads ~0 whatever the stack is doing. This
  streams `pcm`, where the first byte IS audio. Also reports `lead` — audio produced minus wall
  time when the stream ends, i.e. the client's buffer; negative means a caller would have
  stalled. Latest numbers and the topology they were taken on: `bench/TTFB.md` (2026-09-21).
- `bench/pitch_stress.py` + `bench/pitch_stress_score.py` — **pitch / tone / volume stress test**, built
  for the demo report of "calm, even tone, then suddenly loud and excited part-way through". Arms
  separate the three mechanisms that sound identical: one request with `stream_normalize=false` is the
  LM alone, +normalizer is the stitcher's gain, +chunking is the chunk joins, +`interleave_id` is what
  interleaving takes back. The headline is a **rate of audible events** (two adjacent 0.5 s voiced
  windows where level rises ≥3 dB *and* register ≥1.5 st together) — a mean is what hides a one-off
  jump. Verdict (`bench/PITCH_TONE_AB.md`, 840 utterances, 0 errors): **the normalizer is not the
  cause** (gain off is slightly *worse*), chunk joins step **+2.4 dB and +1.7 st upward at two-thirds
  of joins**, and the floor is the LM itself (31 events/1k in a single un-chunked request). Through
  LiveKit 42/120 utterances carry an audible jump; `X-Interleave-Id` from the agent takes that to
  35/120 for +141 ms TTFB. Two traps: an event window that is half pause reads as a huge fake level
  jump (hence 70%-voiced windows and voiced-only levels — without it the same run reports 3× the
  events), and **librosa 1.0.0 segfaults in `pyin`** on numba 0.67 / numpy 2.2 (exit 139, no
  traceback) — pin `librosa==0.11.0`.
- `bench/livekit/` — LiveKit agent stress rig (no-STT/no-LLM agent + load client): measures TTFB and
  per-utterance loudness through a real agent + WebRTC path, and with `--wav-dir` saves one wav per
  utterance for `bench/pitch_stress_score.py`. `TTS_INTERLEAVE=true` (default) makes the agent send
  `X-Interleave-Id: <room>` so the chunks of one reply share prosody (`INTERLEAVE.md` §6);
  `AGENT_IDLE_PROCESSES` sizes the warm job-process pool, which is an admission limit under bursts.
  **Measured 2026-09-23 (`bench/LIVEKIT.md`): the agent + WebRTC cost a flat +111 to +136 ms over raw
  HTTP at every load** — TTFB p50 0.232 s at 1 room, 0.258 s at 16, against the API's own 0.102 /
  0.147 s; 0 errors in 688 utterances; loudness sd 0.92–1.26 dB throughout, so `STREAM_NORMALIZE`
  survives the transport. `interleave_id` costs ~64 ms of prefill (shrinking under load: 78 ms at
  1 room, 49 at 16) and at 16 rooms tightens sd 1.05 → 0.92 dB and TTFB p95 0.736 → 0.594 s. 32 rooms
  is a ceiling of the single-node rig (ICE for 64 peers through one dev-config server), **not** the
  API, which is clean to concurrency 64 (TTFB p50 0.920 s, 179 audio-s/s, 0 errors).
  **Two rig traps, both first recorded as server results:** (1) one python process cannot drive more
  than ~8 rooms — each room's coroutine scans its frame buffer and runs numpy RMS on the shared event
  loop, so 16 rooms from one process reported TTFB p50 14.08 s while the same 16 over two processes
  reported 0.307 s and the app sat at 19% of one core (`load_client.py --procs` now shards, default
  one process per 8 rooms); (2) each room opens several WebRTC sockets, so the default 1024-fd limit
  makes the *client* fail at 16 rooms with `Too many open files` and inflates TTFB on the survivors.
  Before believing a latency cliff, check what the service under test was doing — if it is idle, the
  cliff is yours. Agent-side, `await ctx.connect()` must come **before** `session.start()`: the wrong
  order works at low concurrency and drops jobs under a burst, because livekit-server kills a job
  whose room is not connected within 10 s of `job_entry`.

## Decode pipeline internals (`app/main.py`)

Streaming decode is the hot path. Per request, `audio_stream_crossfade()` accumulates speech tokens and
decodes **growing windows**: the first is `chunk_size = playback_speed * 50` tokens (default 2 s), each
later one ×`STREAM_CHUNK_GROWTH` capped at `STREAM_MAX_CHUNK_S` — plus `STREAM_PAST_CONTEXT_S` of past
tokens (free, sliced off) and `playback_overlap_speed*50` future tokens each side for crossfade. Bigger
windows + real history matter because the NeuCodec decoder is non-causal: small isolated windows tilt the
loudness envelope ~0.7–1.5 dB vs one-shot decode. Each decode flows through:

`decode_speech_token` → `dynamic_batch_queue` → `dynamic_batching()` (collects ≤`MAX_BATCH_SIZE` every
`MICROSLEEP`s) → `batch_thread_fn` (pads tokens to the next CUDA-graph bucket, pinned async H2D copy) →
`compute_thread_fn` (replays the bucketed CUDA graph, D2H, resolves the per-request futures).

CUDA-graph buckets come from `CUDA_GRAPH_BATCH` (each fraction ×50 = a token-length bucket), captured for
every batch size `1..MAX_BATCH_SIZE`. Empty `CUDA_GRAPH_BATCH` ⇒ eager decode (slow).

## Performance: where the time goes & what to turn

Measured on **1× H100 SXM (80GB)** with both services colocated (see `bench/OPTIMIZATION.md` for the full
writeup + raw JSON in `bench/results/`). *Numbers predate the 2026-08 stitcher rework (growing windows,
past context, loudness normalization) — per-request decode count dropped ~2.5×, so throughput shape may
differ slightly; the bottleneck analysis still holds.*

- **The LM is not the bottleneck.** vLLM alone sustains ~12,800 speech-tokens/s (≈255 audio-s/s) at
  concurrency 50. The end-to-end pipeline is gated by **codec decode + the Python serving loop**.
- The serving process is **single-threaded / GIL-bound**: once CUDA graphs make the codec GPU work cheap,
  one event loop maxes ~1 core at ~85% and the **GPU sits idle** at high concurrency. The fix is
  multiple worker processes sharing the GPU via **NVIDIA MPS**.

### Throughput (audio-seconds produced per wall-second), eval set ≈4.6 s audio/request

| Concurrency | Baseline (eager, 1 worker) | + CUDA graphs (1 worker) | + CUDA graphs + 4 workers + MPS |
|---|---|---|---|
| 1  | 6.9  | 7.8  | 6.5  |
| 4  | 14.6 | 26.3 | 24.8 |
| 16 | 18.4 | 31.8 | **71.5** |
| 50 | 18.7 | 28.7 | **83.9** |

At concurrency 50: throughput **18.7 → 83.9 audio-s/s (4.5×)**, mean latency **12.3 s → 2.6 s**,
RTF (p50) **2.87 → 0.70** (faster than real-time even at 50 concurrent). Single-request latency ≈0.6 s
for ~4.6 s of audio (RTF ≈0.15).

### Knobs (`.env` unless noted)

| Var | Effect |
|---|---|
| `DYNAMIC_BATCHING` (default `true`) | Batch concurrent decode calls. Essential for concurrency, free at concurrency 1. |
| `CUDA_GRAPH_LAZY` (default `true`), `CUDA_GRAPH_MAX_SHAPES` (`64`) | Capture one CUDA graph per **exact** decode shape, on first sight, up to the cap. Replaces the fixed buckets below, which only worked by padding a window up to a bucket — and that padding corrupted the audio (`bench/PADDING_BUG.md`). Measured: 0 of 150 real decodes ever landed on a configured bucket (the stitcher produces lengths like 47/121/269), while a lazy cache on the exact shape hits 80–84%. Costs one capture per new shape (~0.06 GB each). |
| `CUDA_GRAPH_BATCH=[...]` | **Legacy.** Token-length buckets (×50), used only when `CUDA_GRAPH_LAZY=false`. Measured interleaved on one GPU: graphs are worth **1.0× at c=8, 1.44× at c=32, 1.54× at c=64** — they only pay once the codec GPU is the constraint, so benchmarking them at low concurrency measures nothing. That broadly confirms the "~1.7×" below. Removing the padding costs 3% at c=8 and **28% at c=64** (grouping by length fragments batches); see `bench/PADDING_BUG.md`. Empty = eager. |
| `MAX_BATCH_SIZE` | Max requests/decode-batch and largest CUDA-graph batch dim. Bigger = more graph memory (~0.06 GB/graph; graphs = `MAX_BATCH_SIZE × len(CUDA_GRAPH_BATCH)`). |
| `DEFAULT_PLAYBACK_SPEED` (default `2.0`) | Size of the **first** decode window only (×50 tokens ⇒ 2 s); later windows grow via `STREAM_CHUNK_GROWTH`/`STREAM_MAX_CHUNK_S`, with `STREAM_PAST_CONTEXT_S` of past tokens primed into every window. Larger first window ⇒ higher first-chunk latency, better first-window decode. **This gate is the TTFB** (the LM does ~530 tok/s single-stream): 2.0 ⇒ ~220 ms, 1.5 ⇒ ~170 ms, **0.75 ⇒ ~100 ms** (deploy example), 0.5 ⇒ ~80 ms but skews the loudness normalizer +1 dB. See *Time to first byte* below. |
| `TORCH_COMPILE=true` | Use `torch.compile` instead of CUDA graphs (alternative codepath). **Measured 2026-09-23 (`bench/MEGAKERNEL.md`): the only arm that beats fp32 eager** — the decoder is launch-bound (609 kernel launches / 51 ops per w121 decode, 6% memory bandwidth), so bf16/TF32/int8 do nothing and fusion does: `reduce-overhead` 2.27× at w47, 1.58× at w121, 1.19× at w269, ~80 dB SNR (not bit-exact). **Not safe to enable as-is:** ~10 s compile per new decode length, `dynamic=True` does not amortise it, and the stitcher emits many lengths — needs a fixed, pre-compiled window schedule (without padding) first. |
| `LLM_NORMALIZER_SKIP_PLAIN` (default `true`) | Skip the `mode=llm` normalizer call when the text has nothing to normalize (see `app/llm_normalizer.py` above). Saves ~0.55 s of TTFB on plain text; output identical. |
| `LLM_NORMALIZER_RULE_FIRST` (default `false`) | `mode=llm`: run `app/spoken_normalizer` first, call the LLM only if a digit/symbol/dotted token survives. Removes the LLM round trip from practically every request; changes output on the ~14% of sentences where rules and LLM differ. `DEFAULT_NORMALIZER_MODE=spoken` skips the LLM outright. |
| `DEFAULT_SPEAKING_RATE` (default `1.0`) | Default for the `speaking_rate` request field: WSOLA time stretch of the output audio (1.3 = 30% faster, pitch unchanged). Not a decode knob — the token count and every decode are the same; only the emitted PCM is shorter/longer. ~1–2 ms CPU per 2 s chunk on the event loop when ≠ 1.0. |
| `MAX_RETAIN_INTERLEAVE` (default `5`) | Interleaved generation (`app/interleave.py`, `INTERLEAVE.md`): previous turns retained per `interleave_id` and put in the prompt, so LiveKit-style chunked replies keep one prosody. **Only works on an interleave-trained LM (private in-house checkpoints; no open-source TTS model has the packing) — `INTERLEAVE_STORE=off` on any other model.** Bounded also by `INTERLEAVE_MAX_S` (20 s) and the LM window; `INTERLEAVE_STORE_DIR` must be shared by every worker. Prefill-only, so decode cost is unchanged; TTFB grows by the prefill of ~750 extra tokens (tens of ms). |
| `STREAM_NORMALIZE` (default `true`) | Per-utterance loudness normalization in the crossfade stitcher (running active-RMS → gain toward `TARGET_RMS_DB`, slew `GAIN_SLEW_DB`/chunk, clamp `MAX_GAIN_DB`); per-request override via `stream_normalize` on `/v1/audio/speech`. The LM's sampled tokens carry loudness — same text at temp 0.6–0.7 spreads 3–10 dB active-RMS (and different voices sit ~5 dB apart in natural level), and ~half of hot utterances clip at full scale; this collapses the spread to <2 dB (verified through a LiveKit agent at concurrency 8, see `bench/livekit/`). Boosts are capped by running-peak headroom and peaks are rounded by a tanh soft-knee limiter (knee 0.85, drive 1.4) — RMS-boosting a peaky voice (crest ~19 dB) through the old hard clip crackled audibly. The gain locks after the first ~1s of voiced audio (one static trim per utterance): a continuously-adapting causal AGC drifted ±0.75 dB mid-utterance, audible as "damping". Residual streaming loudness ripple (~0.7–1.5 dB envelope tilt vs a single big decode window) is the non-causal decoder's window effect — shrink it with larger `playback_speed`/`playback_overlap_speed` (e.g. `playback_speed=10` for non-realtime requests). |
| `FADE_IN_MS` (default `10`) | Raised-cosine fade-in over the first N ms of every response (`app/fade.py`, applied last, after crossfade / loudness / time stretch). ~5% of requests start mid-waveform because the LM's first tokens are already voiced, and 3.3–3.8% opened on an audible click (→ 0.0% with the fade). Present at concurrency 1 and with the padding-fixed batcher too, so it is the model, not load. It removes the click, not the cause: the first word can still come out garbled when the LM starts mid-sound. Numbers in `bench/FADE_IN.md`. Split-invariant across chunk boundaries (`tests/test_fade.py`). 0 disables. |
| uvicorn `--workers N` (deploy) | Run N app processes to beat the GIL ceiling. **Requires NVIDIA MPS** to share the GPU without context-thrash collapse at high concurrency. |
| `TRACE_ASGI_MESSAGE_SPANS` (default `false`) / `DISCONNECT_POLL_S` (`0.25`) | Suppress OTel's per-ASGI-message `http receive`/`http send` spans, and throttle the disconnect poll that generates them. See the gotcha below — without these one streaming request emits ~500 empty spans. |
| `ENABLE_TRACING_SPANS` (default `true`, **and** needs an exporter: `OTLP_ENDPOINT`/`OTEL_EXPORTER_OTLP_*`/`JAEGER_HOST`/`ENABLE_CONSOLE_SPAN_EXPORTER`, else spans are not built — waive with `TRACING_SPANS_REQUIRE_EXPORTER=false`) | Hot-path spans (`app/tracing.py`): `codec.batch_wait`/`batch_prep`/`compute_wait` (dynamic batching), `lm.connect`/`lm.first_token` + `tts.lm_wait_s` (waiting on vLLM), `codec.gpu_decode` + `tts.decode_wait_s` (decoding speech tokens), per emitted chunk. ~5 extra spans **per decode**, so pair with `TRACING_SAMPLE<1` under load; An SDK with no span processor still *builds* every span before dropping it (~292 us/request measured), which is why no exporter ⇒ no spans. `false` = `nullcontext`, no timing taken at all (2.6 us). |

### Time to first byte (TTFB)

**Full write-up: `bench/TTFB.md`** (a 2026-09-21 update at the end covers the split-box
deployment — app on one H20, LM a **TP=4 vLLM on another host** — measured on-box with
`bench/latency_bench.py`: TTFB p50 **102 ms** single-stream and **195 ms at concurrency 32**,
RTF p50 0.096→0.152, **181.7 audio-s/s at c=32** with 0 errors and eager decode, i.e. 2.2× the
H100 row below at lower concurrency because the codec GPU contends with nothing. `lm_probe`
splits that 102 ms as prefill 9 ms / **autoregressive generation of the first 47 tokens 90 ms
(88%)** / codec+stitcher+HTTP ~12 ms — so TTFB work has to aim at the LM. The TP sweep is **done**
(2026-09-22, same node, same client): **TP=1 469 tok/s, TP=2 498, TP=4 557** — TP=4 is the right
setting and there is no free win there. At batch 1 the ranks run at **98% occupancy and 14%
memory-bandwidth utilisation**, with the co-tenant STT engine idle and rank 3 no slower than
rank 0 — so decode is launch/sync-bound (many tiny kernels per token), not bandwidth-bound and
not contended. Through a real LiveKit
agent on the same box TTFB p50 is **225-237 ms** (LiveKit adds ~130 ms, and fattens p95/p50 from
~1.15× to 2.2-2.5×), against 462 ms on the previous single-box stack.) (measured 2026-09-05 against the staging deployment: vLLM TP=2 on
2× H20-3e + the app on 1× H20-3e with 4 workers). Headline: the rule path and `mode=llm` are
indistinguishable at **~235 ms wall / ~115 ms on-box** because `SKIP_PLAIN` + `RULE_FIRST` skip the
LLM on 99.8% of the 497-sentence corpus; when a request does reach the LLM it costs **+633 ms**.
The post-header pipeline (LM first window + first codec decode) is a constant **~113 ms**.
End-to-end RTF 0.16–0.23, and 34 audio-s/wall-s at concurrency 8. `bench/ttfb_report.py` reproduces
it. Note two different rule paths: `mode=spoken` is correct, legacy `mode=rule` misreads
`RM1,250.50` as "ringgit. five zero".

Earlier decomposition, measured 2026-09-04 on H20 (8× H20-3e; prod vLLM, TP=2) with `bench/ttfb_probe.py`, which times the
first **audio** byte on a `pcm` stream (WAV emits its header before any decode, so `bench.py`'s wav TTFB
is meaningless). Staging's ~1 s decomposed as:

| Stage | On the box | Notes |
|---|---|---|
| LLM normalizer (`mode=llm`, gemma-4-31b via serverless proxy) | **~560 ms** | headers only after it returns; 19/21 sentences came back unchanged |
| LM: first `playback_speed*50 + overlap*50` tokens | 173 ms @1.5 s window | prefill 11 ms, then ~530 tok/s; 85 tokens ⇒ 166 ms |
| First codec decode + emit | ~7 ms (CUDA graph) / ~17 ms (eager) | not a TTFB factor on H20 |
| Serverless proxy hop (from a laptop) | +125 ms | and it paces the stream: 3/9 runs would have stalled a client (`min_lead_s<0`) |

What moved it (both output-preserving, so the CER guardrail cannot regress): `LLM_NORMALIZER_SKIP_PLAIN`
(above) and `DEFAULT_PLAYBACK_SPEED=0.75`. Result, on-box `mode=llm` plain text: **754 ms → 103 ms**; text
with digits/abbreviations keeps the LLM call (~660 ms). The window change was checked at temperature 0
(identical tokens) against a one-shot `playback_speed=50` decode with `bench/window_ab.py`: envelope
median |Δ| 0.11 dB (1.5 s: 0.16), no seam clicks (2nd-difference ratio ≤1.7), loudness-normalizer offset
−0.22 dB (1.5 s: −0.50), and Whisper transcripts identical 7/7 (`bench/cer_wavs.py`). 0.5 s is too small:
+1.0 dB normalizer bias. The LM at 10× realtime keeps ≥0.5 s of client buffer even at 0.75 s
(`min_lead_s`), so no underrun risk from the smaller first window. Not fixable here: the LLM's own
~0.55 s when text *does* need normalizing, and the proxy's +125 ms / buffering.

Reproducing on a shared slurm GPU box: `bench/deploy/test_instance.sbatch` runs one app process from
an rsynced checkout + an env file next to whatever is already on the node (own port/GPU, `OVERRIDES=`
for per-instance knobs); `bench/lm_probe.py` times the LM alone, `bench/normalizer_probe.py` the
normalizer alone.

### Accuracy guardrail (Whisper-large-v3 CER, 16 sentences, temp 0.6)

| Config | CER mean | CER median | WER mean |
|---|---|---|---|
| Baseline | 1.21% | 0% | 1.97% |
| Optimized (run 1) | 1.97% | 0% | 5.07% |
| Optimized (run 2) | 1.73% | 0% | 3.10% |

~~All optimizations (CUDA graphs, MPS, multi-worker) are **bit-identical decode operations** … so
accuracy cannot regress by construction.~~ **This was wrong — see `bench/PADDING_BUG.md` (2026-09-22).**
The graph *replay* is bit-identical, but enabling buckets changed the decoder's *input*: the batcher
padded every window up to the next bucket with speech token id 0, and NeuCodec's decoder is
non-causal, so that padding altered the samples that were kept — 4.7 dB SNR against an unpadded
decode, worst in the MIDDLE of the window. The same bug fired with graphs off whenever dynamic
batching put different-length windows together (−1.3 dB). Fixed: one decode per distinct length
(`app/batching.py`), and graphs are now captured lazily on the exact shape. The CER guardrail could
never have caught it — it runs at concurrency 1, where nothing is padded.
The run-to-run spread (~0.2–0.8% CER, same config) matches the baseline↔optimized gap, confirming the
difference is temperature-0.6 sampling noise, not a regression. Median CER is 0% in every config.

## Deploying the optimized stack (single GPU, colocated)

See `bench/` for ready scripts. Summary:

```bash
# 1. NVIDIA MPS (lets vLLM + N codec workers share one GPU without time-slice collapse)
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
nvidia-cuda-mps-control -d

# 2. vLLM (cap max-num-seqs; the 217K speech-token vocab makes the sampler warmup memory-heavy)
vllm serve Scicom-intl/Multilingual-TTS-1.7B-Base --dtype bfloat16 --port 9093 \
  --gpu-memory-utilization 0.4 --max-model-len 4096 --max-num-seqs 64 --served-model-name TTS-model

# 3. App with CUDA graphs + multiple workers (CUDA_GRAPH_BATCH set in .env, TTS_API=http://localhost:9093)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uvicorn app.main:app --host 0.0.0.0 --port 9091 --workers 4
```

Budget GPU memory: each codec worker ≈6–7 GB (mostly CUDA-graph pools) + vLLM (~33 GB at 0.4). 4 workers +
vLLM ≈60 GB on an 80 GB card.

> **Disk: never use `/workspace`.** On RunPod pods it is a **network-backed volume — slow**. Put the repo,
> the HF cache (`HF_HOME=/root/hf`), and the venvs (`/opt`) on the **local container disk (`/`)**. Loading
> multi-GB weights or reading code from `/workspace` cripples startup and skews benchmarks. Verify with
> `df -h /` (overlay = local/fast).

## Common commands

```bash
python -m pytest tests/ -v                        # 623 pass / 35 skip with a live API + OPENAI_* set
python -m pytest tests/test_sanitize_markdown.py -v   # no GPU deps
python -m pytest tests/test_timestretch.py -v         # no GPU deps (speaking_rate WSOLA)
uv run --with pytest -- pytest tests/test_interleave.py -v   # no GPU deps (interleaved generation)
uv run --with pytest --with opentelemetry-sdk -- pytest tests/test_tracing.py -v  # no GPU deps
uv run --with aiohttp --with pytest -- pytest tests/test_llm_normalizer.py -v  # no GPU deps; `set -a; source .env; set +a` first to include the live LLM tests

# TTFB / first-window tooling (talks to a running app; see "Time to first byte")
uv run --with aiohttp python bench/latency_bench.py --url http://127.0.0.1:9091 --concurrency 1,4,8,16,32   # TTFB + e2e + RTF, p10/p50/p90/p95/p99 (pcm, so TTFB is the first AUDIO byte)
python bench/ttfb_probe.py --url http://127.0.0.1:9091 --playback 1.5,0.75 --overlap 0.2 --reps 3   # first-audio-byte latency + client stall margin
uv run --with aiohttp python bench/ttfb_report.py --url http://127.0.0.1:9091 --reps 5   # TTFB + end-to-end per normalizer mode (rules vs llm); writes bench/TTFB.md's numbers
python bench/lm_probe.py --gates 35,60,85,110          # vLLM alone: prefill, tok/s, arrival of the N-th token (source .env first)
python bench/window_ab.py --url ... --configs 1.5:0.2,0.75:0.2   # temp-0 streamed vs one-shot decode: envelope, clicks (+ --save-dir, then bench/cer_wavs.py)
PYTHONPATH=. python bench/normalizer_gate_eval.py --old-url ... --new-url ...   # LLM-skip gate: outputs must be identical
uv run --with pytest pytest tests/test_spoken_normalizer.py -q   # rule normalizer (1,726 tests, no deps)
PYTHONPATH=. python bench/normalizer_agreement.py               # rule vs LLM ground truth, per language/category
uv run --with fasttext-wheel --with "numpy<2" --with huggingface_hub --with aiohttp python bench/langid_compare.py --llm   # marker words vs fastText vs LLM language detection (results in bench/NORMALIZER.md)
set -a; source .env; set +a; uv run --with aiohttp python bench/normalizer_truth.py   # extend the ground truth (new corpus ids only)

# local docker stack
docker network create tts-network
TTS_MODEL=Scicom-intl/Multilingual-TTS-1.7B-Base GPU_MEM_UTIL=0.7 docker compose -f vllm.yaml up -d
docker compose up --build

# smoke test
curl -X POST localhost:9091/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input":"Hello there","voice":"husein","response_format":"wav","stream":false}' -o out.wav

# benchmark + CER (see bench/)
# checkpoint A/B listening set: one sentence file -> N checkpoints (GPU box, own venv; bench/synth/README.md)
bash bench/synth/run_models.sh sentences.txt /out slug1:Scicom-intl/<repo> slug2:Scicom-intl/<repo>

python bench/bench.py --concurrency 1,4,16,50 --out /tmp/bench.json
python bench/cer_eval.py --wav-dir /tmp/eval --out /tmp/cer.json   # needs faster-whisper + jiwer
```

## Gotchas (these will bite)

- **Pin `uvicorn==0.35.x`.** `main.py` calls `asyncio.create_task()` / `get_running_loop()` at module top
  level; uvicorn ≥0.36 eagerly imports the app *outside* the event loop → `RuntimeError: no running event
  loop`. (Pinned in `requirements.txt`.)
- **vLLM + new FastAPI/Starlette.** vLLM only pins `fastapi>=0.115`; with FastAPI 0.138/Starlette 1.x its
  `prometheus-fastapi-instrumentator` middleware 500s on every request
  (`'_IncludedRouter' object has no attribute 'path'`). Pin `fastapi==0.115.6` in the vLLM env.
- **vLLM `--max-num-seqs` too high OOMs at sampler warmup** because the speech-token vocab is ~217K. Cap it
  (64 is ample — the codec, not the LM, is the throughput limit).
- **Multiple GPU processes without MPS collapse under load** (CUDA context time-slicing): throughput swings
  wildly and p99 latency explodes at high concurrency. Always run multi-worker + colocated vLLM under MPS.
- **OTel's ASGI instrumentation spans every ASGI *message*, which streaming turns into a flood.**
  `request.is_disconnected()` is a real ASGI receive, and aiohttp yields two lines per SSE event, so
  polling it per line produced ~2 `http receive` spans per speech token (measured: 505 spans vs 19
  real ones for one request; 25 spans / 0 noise after the fix). Fixed on both ends —
  `DISCONNECT_POLL_S` throttles the poll (505 noise spans → 9 on its own), and
  `_suppress_asgi_message_spans()` defaults `OpenTelemetryMiddleware.__init__` to
  `exclude_spans=['receive','send']` for the rest. Two traps if you touch it: the fastapi
  instrumentation ≥0.50 does **not** register that middleware via `add_middleware` (it wraps
  `build_middleware_stack` and constructs it directly, so editing `app.user_middleware` patches
  nothing), and it passes `exclude_spans=None` explicitly, so a `setdefault` never fires. Don't
  instrument the app yourself before `patch()` either — on older versions that flips the
  middleware order and drops `traceID` from the `type=request` log line.
- **Batch-queue items are 3-tuples: `(future, payload, meta)`.** `meta` is the tracing carrier
  (`tracing.stage_meta()`, `None` when tracing is off) that the batch/compute threads mutate to
  time each hop; `compute_queue` items are `(uuid, tokens, lens, futures, metas)`. Adding a field
  means touching all of `dynamic_batching` / `_batch_one` / `_compute_one` (and the `vc_*` twins)
  — an arity mismatch there wedges every decode with the error only visible in the worker log.
- **Interleaved generation is per-host state.** The turn store is a directory (`/dev/shm` by default), so it is shared by the uvicorn workers on one box and by nothing else: behind a multi-host load balancer chunk N+1 lands on a box that never saw chunk N and silently generates cold — `X-Interleave-Turns: 0` on the response is the tell. Pin one call's requests to a host (or add a redis `InterleaveStore`). In docker, `/dev/shm` defaults to 64 MB (~10k ids); `INTERLEAVE_STORE_DIR` must point at the same directory for every worker.
- NeuCodec downloads `facebook/w2v-bert-2.0` + `neuphonic/neucodec` from HF on first start — cache them.
- **`MODEL_NAME` ≠ `OPENAI_MODEL_NAME`.** `MODEL_NAME` is the TTS model vLLM serves (`TTS-model`);
  the LLM normalizer's model goes in `OPENAI_MODEL_NAME`. Setting `MODEL_NAME=google/gemma-...` in
  `.env` silently breaks every TTS request (vLLM rejects the unknown model).
- Killing the stack: vLLM's engine-core child has comm `VLLM::EngineCor` (uppercase) — a `pkill -f vllm`
  (lowercase) misses it and leaks GPU memory. Match case-insensitively or kill by PID.
- **On RunPod, never run from `/workspace`** — it's slow network storage. Keep code, `HF_HOME`, and venvs
  on the local container disk (`/`, e.g. `/root`, `/opt`). See the disk note in the deploy section.
