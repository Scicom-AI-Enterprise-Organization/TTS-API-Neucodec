"""Serve the TTS API with the LM in-process (LM_BACKEND=inprocess) on chosen GPUs and a free port.

One process per model: no vLLM server, no second port. Use it to put a checkpoint through the
serving path (interleave, stitcher, loudness, fade-in) the way production serves it, e.g. one
candidate per GPU side by side:

    python bench/serve_inprocess.py --model /mnt/data/ckpt/a --gpus 3                # random port
    python bench/serve_inprocess.py --model /mnt/data/ckpt/b --gpus 4,5 --tp 2 --port 9191
    python bench/serve_inprocess.py --model ... --gpus 6 --env-file bench/inprocess.env.example \\
        --url-file /tmp/b.url

--gpus becomes CUDA_VISIBLE_DEVICES: the LM takes the first --tp of them and the codec runs on the
first one too, unless --codec-device names another (cuda:1 = the second of --gpus). --port 0 (the default) binds a free port; the
URL is printed (and written to --url-file) once the API answers, i.e. after the engine loaded.
Runs uvicorn with ONE worker on purpose: every worker would load its own engine. Ctrl-C / SIGTERM
stops it. Needs a venv with vllm next to this repo's requirements (same torch: 2.9.1).
"""
import argparse
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def read_env_file(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip().removeprefix("export ").strip()] = v.strip().strip('"').strip("'")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="checkpoint dir or HF repo id (LM_MODEL)")
    ap.add_argument("--gpus", required=True, help="CUDA_VISIBLE_DEVICES, e.g. 3 or 4,5")
    ap.add_argument("--tp", type=int, default=1, help="LM tensor parallel size (<= number of --gpus)")
    ap.add_argument("--port", type=int, default=0, help="0 = a free random port")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--gpu-mem", type=float, default=0.3, help="LM_GPU_MEMORY_UTILIZATION")
    ap.add_argument("--codec-device", default="",
                    help="DEVICE for NeuCodec, e.g. cuda:1 for the 2nd of --gpus; default unset like "
                         "production = cuda, the first of --gpus")
    ap.add_argument("--env-file", help="extra settings (production's), e.g. bench/inprocess.env.example")
    ap.add_argument("--url-file", help="write the URL here once the API answers")
    ap.add_argument("--python", default=sys.executable, help="the venv python to run uvicorn with")
    ap.add_argument("--ready-timeout", type=int, default=1200, help="seconds to wait for the API")
    a = ap.parse_args()
    if a.tp > len(a.gpus.split(",")):
        sys.exit(f"--tp {a.tp} needs at least {a.tp} GPUs in --gpus ({a.gpus})")
    port = a.port or free_port()
    env = dict(os.environ)
    if a.env_file:
        env.update(read_env_file(a.env_file))
    env.update({
        "LM_BACKEND": "inprocess", "LM_MODEL": a.model, "LM_TENSOR_PARALLEL_SIZE": str(a.tp),
        "LM_GPU_MEMORY_UTILIZATION": str(a.gpu_mem), "CUDA_VISIBLE_DEVICES": a.gpus,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        # own interleave store: the default /dev/shm dir is shared with any other instance
        "INTERLEAVE_STORE_DIR": env.get("INTERLEAVE_STORE_DIR") or f"/dev/shm/tts-interleave-inproc-{port}",
        "PYTHONUNBUFFERED": "1",
    })
    if a.codec_device:
        env["DEVICE"] = a.codec_device
    else:
        env.pop("DEVICE", None)
    cmd = [a.python, "-m", "uvicorn", "app.main:app", "--host", a.host, "--port", str(port), "--workers", "1"]
    print(f"starting {a.model} on GPU(s) {a.gpus} (tp={a.tp}, codec {a.codec_device or 'cuda'}) at port {port}", flush=True)
    proc = subprocess.Popen(cmd, cwd=REPO, env=env)

    def stop(*_):
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    url = f"http://{a.host}:{port}"
    t0 = time.time()
    while time.time() - t0 < a.ready_timeout:
        if proc.poll() is not None:
            sys.exit(f"the API exited during startup (code {proc.returncode})")
        try:
            urllib.request.urlopen(url + "/docs", timeout=5)
            break
        except Exception:  # noqa: BLE001 -- not up yet
            time.sleep(5)
    else:
        proc.terminate()
        sys.exit(f"the API did not answer within {a.ready_timeout} s")
    print(f"URL {url}  (ready in {time.time() - t0:.0f} s)", flush=True)
    if a.url_file:
        with open(a.url_file, "w") as f:
            f.write(url + "\n")
    sys.exit(proc.wait())


if __name__ == "__main__":
    main()
