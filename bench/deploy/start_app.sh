#!/usr/bin/env bash
# Launch the FastAPI NeuCodec decoder app, colocated on the same GPU as vLLM.
# Config comes from /root/TTS-API-Neucodec/.env (written per-experiment).
set -u
export HF_HOME=/root/hf
export PYTHONUNBUFFERED=1
cd /root/TTS-API-Neucodec
LOG=${LOG:-/root/app.log}
APP_PORT=${APP_PORT:-9090}   # 9091 is taken by RunPod nginx; 9090 is its proxy upstream
WORKERS=${WORKERS:-1}
pkill -9 -f "uvicorn app.main:app" 2>/dev/null; sleep 2
# load .env verbatim (handles values with spaces/brackets like CUDA_GRAPH_BATCH=[0.5, 1.0])
while IFS='=' read -r k v; do
  [ -z "$k" ] && continue
  case "$k" in \#*) continue;; esac
  export "$k=$v"
done < /root/TTS-API-Neucodec/.env
echo "Launching app on :$APP_PORT workers=$WORKERS  DYNAMIC_BATCHING=$DYNAMIC_BATCHING CUDA_GRAPH_BATCH=$CUDA_GRAPH_BATCH MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-} TORCH_COMPILE=${TORCH_COMPILE:-}"
nohup /opt/venv-app/bin/python -m uvicorn app.main:app \
  --host 0.0.0.0 --port "$APP_PORT" --workers "$WORKERS" \
  > "$LOG" 2>&1 &
echo "app PID $!  (log: $LOG)"
