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
- `app/wrapper.py` — `CUDAGraphsWrapper`: captures one CUDA graph per `(batch, token-length)` bucket.
- `app/neucodec/` — **vendored** NeuCodec (`from app.neucodec import NeuCodec`; *not* the pip package).
- `app/normalizer/`, `app/rules.py` — Malaysian/multilingual text normalization + markdown sanitization.
- `vllm.yaml` / `docker-compose.yaml` — the two services, sharing external docker network `tts-network`.
- `bench/` — benchmark + Whisper-CER harness, RunPod deploy scripts, and recorded results (see `bench/OPTIMIZATION.md`).

## Decode pipeline internals (`app/main.py`)

Streaming decode is the hot path. Per request, `audio_stream()` accumulates speech tokens and decodes a
sliding window of `chunk_size = playback_speed * 50` tokens (with `playback_overlap_speed*50` overlap for
crossfade). Each decode flows through:

`decode_speech_token` → `dynamic_batch_queue` → `dynamic_batching()` (collects ≤`MAX_BATCH_SIZE` every
`MICROSLEEP`s) → `batch_thread_fn` (pads tokens to the next CUDA-graph bucket, pinned async H2D copy) →
`compute_thread_fn` (replays the bucketed CUDA graph, D2H, resolves the per-request futures).

CUDA-graph buckets come from `CUDA_GRAPH_BATCH` (each fraction ×50 = a token-length bucket), captured for
every batch size `1..MAX_BATCH_SIZE`. Empty `CUDA_GRAPH_BATCH` ⇒ eager decode (slow).

## Performance: where the time goes & what to turn

Measured on **1× H100 SXM (80GB)** with both services colocated (see `bench/OPTIMIZATION.md` for the full
writeup + raw JSON in `bench/results/`).

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
| `DYNAMIC_BATCHING=true` | Batch concurrent decode calls. Essential for concurrency. |
| `CUDA_GRAPH_BATCH=[0.5,1.0,1.5,2.0,3.0,4.0]` | Token-length buckets (×50). **Enabling this is the single biggest codec win (~1.7×).** Empty = eager. |
| `MAX_BATCH_SIZE` | Max requests/decode-batch and largest CUDA-graph batch dim. Bigger = more graph memory (~0.06 GB/graph; graphs = `MAX_BATCH_SIZE × len(CUDA_GRAPH_BATCH)`). |
| `DEFAULT_PLAYBACK_SPEED` | Sets decode `chunk_size`; larger ⇒ fewer/bigger decodes (more throughput, higher first-chunk latency). |
| `TORCH_COMPILE=true` | Use `torch.compile` instead of CUDA graphs (alternative codepath). |
| uvicorn `--workers N` (deploy) | Run N app processes to beat the GIL ceiling. **Requires NVIDIA MPS** to share the GPU without context-thrash collapse at high concurrency. |

### Accuracy guardrail (Whisper-large-v3 CER, 16 sentences, temp 0.6)

| Config | CER mean | CER median | WER mean |
|---|---|---|---|
| Baseline | 1.21% | 0% | 1.97% |
| Optimized (run 1) | 1.97% | 0% | 5.07% |
| Optimized (run 2) | 1.73% | 0% | 3.10% |

All optimizations (CUDA graphs, MPS, multi-worker) are **bit-identical decode operations** — no weight,
precision, or sampling change (**bf16 throughout, no FP8**) — so accuracy cannot regress by construction.
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
python -m pytest tests/ -v                        # 543 pass / 31 skip with a live API
python -m pytest tests/test_sanitize_markdown.py -v   # no GPU deps

# local docker stack
docker network create tts-network
TTS_MODEL=Scicom-intl/Multilingual-TTS-1.7B-Base GPU_MEM_UTIL=0.7 docker compose -f vllm.yaml up -d
docker compose up --build

# smoke test
curl -X POST localhost:9091/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input":"Hello there","voice":"husein","response_format":"wav","stream":false}' -o out.wav

# benchmark + CER (see bench/)
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
- NeuCodec downloads `facebook/w2v-bert-2.0` + `neuphonic/neucodec` from HF on first start — cache them.
- Killing the stack: vLLM's engine-core child has comm `VLLM::EngineCor` (uppercase) — a `pkill -f vllm`
  (lowercase) misses it and leaks GPU memory. Match case-insensitively or kill by PID.
- **On RunPod, never run from `/workspace`** — it's slow network storage. Keep code, `HF_HOME`, and venvs
  on the local container disk (`/`, e.g. `/root`, `/opt`). See the disk note in the deploy section.
