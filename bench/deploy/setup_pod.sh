#!/usr/bin/env bash
# Pod environment setup: two venvs (vllm + app) mirroring the repo's two Dockerfiles.
set -uo pipefail
cd /root/TTS-API-Neucodec
export DEBIAN_FRONTEND=noninteractive
export HF_HOME=/root/hf
mkdir -p /root/hf

echo "===STEP apt==="
apt-get update -qq && apt-get install -y -qq ffmpeg build-essential python3-dev git curl >/dev/null 2>&1
echo "apt done rc=$?"

echo "===STEP venv-vllm==="
uv venv /opt/venv-vllm --python 3.12 --system-site-packages
VLLM_PIP="uv pip install --python /opt/venv-vllm/bin/python"
$VLLM_PIP vllm==0.16.0
$VLLM_PIP "fastapi==0.115.6"   # starlette 1.x breaks vLLM prometheus middleware
echo "vllm install rc=$?"
/opt/venv-vllm/bin/python -c "import vllm, torch; print('VLLM', vllm.__version__, 'TORCH', torch.__version__, 'CUDA', torch.version.cuda)"

echo "===STEP venv-app==="
uv venv /opt/venv-app --python 3.12 --system-site-packages
APP_PIP="uv pip install --python /opt/venv-app/bin/python"
$APP_PIP -r requirements.txt
echo "app requirements rc=$?"
# eval deps for the CER harness (faster-whisper avoids transformers conflicts; jiwer for CER)
$APP_PIP faster-whisper jiwer
echo "eval deps rc=$?"
/opt/venv-app/bin/python -c "import torch, numpy, librosa, soundfile, transformers, fastapi, fasttext; from app.neucodec import NeuCodec; print('APP imports OK torch', torch.__version__, 'np', numpy.__version__, 'tf', transformers.__version__)"

echo "===ALL DONE==="
