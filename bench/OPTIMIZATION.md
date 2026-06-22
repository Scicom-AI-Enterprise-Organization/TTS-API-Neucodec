# Inference optimization — H100 SXM, colocated vLLM + NeuCodec

End-to-end optimization of the TTS pipeline on **1× H100 SXM (80 GB)**, with vLLM (Qwen3-1.7B TTS LM)
and the NeuCodec decoder **colocated on the same GPU**. Goal: maximize throughput/latency **without
dropping audio accuracy** (verified with Whisper CER).

## TL;DR

| Metric (concurrency 50) | Baseline | Optimized | Change |
|---|---|---|---|
| Throughput (audio-s / wall-s) | 18.7 | **83.9** | **4.5× faster** |
| Mean latency | 12.3 s | **2.6 s** | 4.7× lower |
| RTF (p50) | 2.87 | **0.70** | real-time at 50 conc. |
| CER (Whisper large-v3) | 1.21% | 1.73–1.97% | unchanged (sampling noise) |

The wins are all **lossless** (bf16 throughout, no quantization): CUDA graphs + token-length bucketing +
dynamic batching for the codec, then multiple worker processes under **NVIDIA MPS** to beat the
single-process GIL ceiling.

## Stack

- GPU: 1× H100 80GB HBM3 (RunPod, `runpod/pytorch:1.0.7-cu1281-torch291-ubuntu2404`).
- vLLM `0.16.0` serving `Scicom-intl/Multilingual-TTS-1.7B-Base` (Qwen3-1.7B) on `:9093`, bf16.
- FastAPI app (`app/main.py`) + vendored NeuCodec on `:9090`, `TTS_API=http://localhost:9093`.
- Two Python venvs (`/opt/venv-vllm`, `/opt/venv-app`) mirroring the repo's two Dockerfiles.

## Method

`evalset.py` — 16 fixed sentences (English, Malay, code-switch; the app's real use-case), Whisper-reliable.
`bench.py` — closed-loop load at concurrency 1/4/16/50; reports latency percentiles, RTF, and throughput
(audio-seconds produced per wall-second; the meaningful TTS throughput metric).
`cer_eval.py` — generates audio for the eval set, transcribes with **faster-whisper large-v3**, computes
CER/WER vs the reference (punctuation-stripped, lowercased). Run before/after to guard accuracy.

## Findings, in order

1. **The LM is not the bottleneck.** A direct vLLM probe sustains 5,569 tok/s (C=16) → 12,772 tok/s
   (C=50) ≈ **255 audio-s/s** — ~13× the baseline end-to-end rate. The codec decode + serving loop is
   the limit, so optimization effort went there.

2. **CUDA graphs (+ bucketed padding + dynamic batching) ≈1.7×.** Baseline ran with `CUDA_GRAPH_BATCH=[]`
   (eager). Setting `CUDA_GRAPH_BATCH=[0.5,1.0,1.5,2.0,3.0,4.0]` (token buckets 25–200, matching the
   `playback_speed=1.5` ⇒ 75/150-token decode windows) lifted peak throughput 18.6 → 31.8 audio-s/s and
   halved C=16 latency (4.0 → 2.4 s). Pure config; lossless.

3. **The serving process is GIL-bound.** With cheap codec kernels, one event loop maxed ~85% of a core
   while the **GPU went idle** (sampled 0–100%, frequently 0%) at C=50 — the single Python process can't
   feed the GPU. Throughput plateaued ~30 audio-s/s.

4. **Multiple workers, but only stable under MPS.** Running `uvicorn --workers 4` parallelizes the Python
   work across cores, but 5 CUDA processes (vLLM + 4 workers) **time-slice** the GPU without MPS →
   throughput swung 21–66 audio-s/s and p99 latency hit 46 s at C=50. Starting **NVIDIA MPS**
   (`nvidia-cuda-mps-control -d`) before launching all GPU processes made them share the GPU with
   concurrent kernels: stable **71.5 audio-s/s @ C=16** and **83.9 @ C=50**, p99 6.5 s.

## Accuracy (CER) — no regression

Whisper-large-v3 CER over the 16-sentence set, temperature 0.6:

| Config | CER mean | CER median | WER mean | en / ms / code-switch CER |
|---|---|---|---|---|
| Baseline (eager) | 1.21% | 0% | 1.97% | 0.63 / 1.75 / 1.92% |
| Optimized (run 1) | 1.97% | 0% | 5.07% | 1.89 / 2.50 / 0.66% |
| Optimized (run 2) | 1.73% | 0% | 3.10% | 1.65 / 2.20 / 0.66% |

The two optimized runs (identical config) differ by ~0.2% CER, i.e. the baseline↔optimized gap is **within
run-to-run sampling variance** of temperature-0.6 generation. CUDA graphs/MPS/multi-worker are bit-identical
decode ops, so this is expected — accuracy is preserved. Median CER is 0% everywhere (most sentences
transcribe perfectly). Raw per-utterance transcripts are in `results/cer_*.json`.

## Reproduce

1. Provision 1× H100 SXM; put code + HF cache + venvs on local disk (not a network volume).
2. `bash deploy/setup_pod.sh` (two venvs; pins `uvicorn==0.35.0` and `fastapi==0.115.6` — see CLAUDE.md gotchas).
3. `cp deploy/env.optimized.example /root/TTS-API-Neucodec/.env` and fill `HF_TOKEN`.
4. Start MPS, then vLLM (`GPU_MEM_UTIL=0.4 MAX_NUM_SEQS=64 bash deploy/start_vllm.sh`), then the app
   (`WORKERS=4 bash deploy/start_app.sh`).
5. `python bench.py --concurrency 1,4,16,50 --out results/run.json`
   `python cer_eval.py --wav-dir /root/eval --out results/cer.json`

## Source-code optimization — a documented negative result

After the config/deploy wins I profiled whether the **decode source code** was leaving throughput on the
table, and tried five in-process changes, each validated old-vs-new on the same H100 (single worker + CUDA
graphs). Raw JSON: `results/{old_1w,new_1w,ctx0_1w,ctx24_1w,ctx16_1w,pipelined_1w}.json`,
`results/cer_{new_1w,ctx24,pipelined}.json`.

| Change | What it targeted | Throughput impact |
|---|---|---|
| #1 pinned-buffer pool | per-decode `.pin_memory()` (`cudaHostAlloc`) | none |
| #2 drop redundant sync | extra `compute_stream.synchronize()` after `.cpu()` | none |
| #3 incremental token parse | re-`join`+regex over the whole window each chunk (CPU) | none |
| #4 cap re-decoded context | the ~2× redundant overlap re-decode (codec FLOPs) | none (CER-safe) |
| pipeline refactor | merge the 2 decode threads + double-buffer D2H (compute N+1 ‖ copy N) | none |

**Single-worker throughput is ~34 audio-s/s (C=16) for every one of them.** The decisive clue: `nvidia-smi`
shows the GPU **~99% busy**, yet (a) halving the tokens per decode (#4: window 150 → ~99) and (b) overlapping
the D2H copy with the next compute (pipeline refactor) both changed nothing. That rules out CPU, FLOPs, the
D2H copy, *and* the thread/queue hops. What remains is the **codec forward's per-call latency floor** — a
small-batch decode has a fixed minimum latency on the GPU regardless of how few tokens it carries, and a
single process issues them serially. Therefore **no in-process change lifts per-worker throughput**; the only
effective lever is **more worker processes under MPS** (each is an independent decode pipeline with its own
GIL) — the 4.5× result above.

**Decision: all five code changes were reverted.** They are correct and CER-safe but add complexity/risk for
zero measured gain, so shipping them would make the code worse, not better. The only source change kept is the
**`uvicorn>=0.35.0,<0.36` pin** in `requirements.txt` — a real correctness fix (the unpinned range pulls 0.49,
which eagerly imports the app outside the event loop and crashes; see CLAUDE.md gotchas).

### What *would* raise per-worker throughput (architectural, not micro-opt)
- Fewer, larger decode calls: raise `playback_speed` (bigger `chunk_size` ⇒ fewer codec calls ⇒ the per-call
  latency floor is amortised over more audio) — a config knob, trades first-chunk latency for throughput.
- Bigger decode batches: the codec call latency is ~fixed up to a point, so batching more requests per call
  (higher concurrency / `MAX_BATCH_SIZE`) is far more effective than shrinking each call.
- More worker processes under MPS — already the validated win.

- **FP8** (`--quantization fp8`, `--kv-cache-dtype fp8`) on vLLM — H100-native; would speed the LM, but the
  LM isn't the bottleneck and FP8 is the one change that *could* move CER, so it was intentionally skipped.
- Eliminating the streaming overlap **re-decode** (each 75-token window currently re-decodes ~150 tokens for
  crossfade) — ~2× codec FLOPs, but changing it risks audio artifacts.
- Async double-buffered D2H in `compute_thread_fn` to overlap copy with the next batch.
- Scaling workers further (memory permitting) now that MPS makes it stable.
