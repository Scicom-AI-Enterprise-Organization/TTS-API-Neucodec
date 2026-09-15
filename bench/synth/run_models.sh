#!/bin/bash
# Render one sentence file through N checkpoints: LM on $LM_GPU, codec on $CODEC_GPU.
#
#   bash run_models.sh <sentences.txt> <outdir> <slug:hf_repo> [<slug:hf_repo> ...]
#
# Env: VENV (a venv to activate; default: use whatever python is already active),
#      SPEAKER, TEMPERATURE, TOP_P, REPETITION_PENALTY, SEED, GPU_MEM, LM_GPU, CODEC_GPU.
# LM_GPU/CODEC_GPU are indices on a SHARED box -- check nvidia-smi and pick idle ones.
# Each model is a fresh vLLM process (one engine cannot hold three checkpoints);
# a model that fails is reported and skipped, the rest still run.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SENTENCES="$1"; OUTDIR="$2"; shift 2

VENV="${VENV:-}"
SPEAKER="${SPEAKER:-TM_English_Normal}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.15}"
SEED="${SEED:-1234}"
GPU_MEM="${GPU_MEM:-0.25}"
LM_GPU="${LM_GPU:-6}"
CODEC_GPU="${CODEC_GPU:-7}"

[ -n "$VENV" ] && source "$VENV/bin/activate"
export VLLM_LOGGING_LEVEL=WARNING TOKENIZERS_PARALLELISM=false
mkdir -p "$OUTDIR/tokens" "$OUTDIR/wav"

for entry in "$@"; do
  slug="${entry%%:*}"; repo="${entry#*:}"
  echo "=============== $slug  ($repo)"
  CUDA_VISIBLE_DEVICES="$LM_GPU" python "$HERE/gen_tokens.py" \
    --model "$repo" --sentences "$SENTENCES" --out "$OUTDIR/tokens/$slug.json" \
    --speaker "$SPEAKER" --temperature "$TEMPERATURE" --top-p "$TOP_P" \
    --repetition-penalty "$REPETITION_PENALTY" --seed "$SEED" --gpu-mem "$GPU_MEM" \
    || echo "!!! FAILED $slug"
done

echo "=============== decode"
CUDA_VISIBLE_DEVICES="$CODEC_GPU" python "$HERE/decode_tokens.py" \
  --tokens "$OUTDIR"/tokens/*.json --outdir "$OUTDIR/wav" --device cuda \
  || echo "!!! DECODE FAILED"
echo ALL_DONE
