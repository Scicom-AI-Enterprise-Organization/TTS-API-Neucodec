#!/usr/bin/env bash
# Launch vLLM serving the Qwen3-1.7B TTS model. Config via env vars (defaults = baseline).
set -u
export HF_HOME=/root/hf
export VLLM_LOGGING_LEVEL=INFO
MODEL=${MODEL:-Scicom-intl/Multilingual-TTS-1.7B-Base}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.7}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
EXTRA_ARGS=${EXTRA_ARGS:-}
LOG=${LOG:-/root/vllm.log}

pkill -f "vllm serve" 2>/dev/null; sleep 2
echo "Launching vLLM: MODEL=$MODEL GPU_MEM_UTIL=$GPU_MEM_UTIL MAX_NUM_SEQS=$MAX_NUM_SEQS EXTRA_ARGS='$EXTRA_ARGS'"
nohup /opt/venv-vllm/bin/vllm serve "$MODEL" \
  --dtype bfloat16 \
  --port 9093 \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --served-model-name TTS-model \
  $EXTRA_ARGS \
  > "$LOG" 2>&1 &
echo "vLLM PID $!  (log: $LOG)"
