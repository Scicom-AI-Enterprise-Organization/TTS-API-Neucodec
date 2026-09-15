# `bench/synth` — render a sentence file through N TTS checkpoints

Offline batch synthesis for **checkpoint comparison**: same text, same speaker, same sampling,
same decoder, one directory of wavs per model. Not a benchmark and not the serving path — there
is no FastAPI app, no normalizer, no streaming stitcher, nothing to make two models differ except
the weights.

Two stages, because the LM and the codec are separable and the expensive half should only run once:

| Stage | Script | Does |
|---|---|---|
| 1 | `gen_tokens.py` | text → `<\|s_N\|>` ids with **vLLM offline**, one process per checkpoint, writes `tokens/<slug>.json` |
| 2 | `decode_tokens.py` | ids → 24 kHz wav with the vendored NeuCodec, **one-shot decode** of the whole utterance |

`run_models.sh` is the two of them over a list of checkpoints, LM on one GPU and the codec on
another.

```bash
VENV=/path/to/venv LM_GPU=6 CODEC_GPU=7 \
  bash bench/synth/run_models.sh sentences.txt /out <slug>:<hf repo> <slug>:<hf repo>
```

`<slug>` names the output directory, `<hf repo>` is the checkpoint. The candidates are usually
**private** model repos, so they are named in the run record, not here — and `HF_TOKEN` has to be
in the environment (from an env file, not inline in the command, where it lands in `ps`).
`LM_GPU`/`CODEC_GPU` are indices on a shared box: check `nvidia-smi` and pick idle ones.

Three things this gets right, each of which is a way a checkpoint comparison silently stops
comparing checkpoints:

1. **One-shot decode, not the streaming stitcher.** The NeuCodec decoder is non-causal, so the
   growing-window + crossfade path in `app/main.py` leaves a ~0.7–1.5 dB envelope tilt that has
   nothing to do with the checkpoint. One window over the whole token stream has none of it.
2. **Prod's loudness treatment, applied identically to every model.** LM-sampled speech tokens
   carry loudness: raw peaks run past full scale on ~half of utterances (measured 1.25–1.40), so a
   naive PCM_16 write clips and the louder checkpoint "wins" a listening test on volume alone.
   `decode_tokens.py` re-applies `normalize_chunk` from `app/main.py` in one-shot form (active-RMS
   → `-16 dBFS`, ±12 dB clamp, peak-headroom cap, tanh soft-knee limiter) and also keeps the
   untouched decoder output as float32 under `raw/`.
3. **Tokens are kept.** `tokens/<slug>.json` holds the ids, finish reason and token counts, so a
   decode change can be re-run — and audited — without paying for generation again.

`gen_tokens.py` builds the serving app's single-turn prompt,
`<|im_start|>{speaker}: {text}<|speech_start|>` (`app/interleave.py:build_prompt` with no history),
and stops on `[151643, 151645]`. Rows whose `finish_reason` is not `stop`, or that came back under
10 speech tokens, are printed as `suspect rows` — check that line before trusting a set.

Environment on a fresh box (vLLM brings its own torch; the vendored codec needs a few extras):

```bash
uv venv --python 3.12 venv && source venv/bin/activate
uv pip install "vllm==0.10.2" "transformers==4.56.2" \
  vector-quantize-pytorch==1.17.8 local-attention einops soundfile torchaudio "numpy>=2.1,<2.3"
```

Both pins matter: vLLM 0.10.2 dies on transformers 5.x (`Qwen2Tokenizer has no attribute
all_special_tokens_extended`), and the repo's `numpy==1.26.4` breaks the scipy that ships with it
(`np.long`). See the CLAUDE.md section for the rest of the traps.

Result sets live outside this repo (`ucc_ai_research/evaluation/tts/synthetic-audio/<date>/`);
the skill `tts-synth-checkpoints` drives the whole thing on a GPU box.
