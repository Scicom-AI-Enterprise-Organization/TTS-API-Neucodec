# LiveKit agent stress test

Stress-tests the TTS API through a **real LiveKit agent path** — text in, WebRTC audio out —
so measurements include everything a live caller hears (agent session, TTS plugin streaming,
Opus encode/decode). No STT, no LLM: text sent on the `tts-input` text-stream topic goes
straight to `session.say()` → `openai.TTS` plugin → this repo's `/v1/audio/speech`.

- `stress_agent.py` — minimal `livekit-agents` worker (auto-dispatches to every room).
- `load_client.py` — joins N rooms concurrently, sends text lines, captures the agent's
  audio track, measures per utterance: TTFB (text → first audible frame), audio duration,
  active-speech RMS, peak. Emits JSONL rows + a summary.

## Setup

```bash
# livekit-server binary (or see ucc_ai_research/evaluation/stt/setup_livekit.sh)
curl -sSL https://get.livekit.io | bash

uv venv /root/lk-venv --python 3.12
uv pip install --python /root/lk-venv/bin/python \
  "livekit-agents==1.6.8" "livekit-plugins-openai==1.6.8" python-dotenv numpy soundfile

cat > .env <<EOF
LIVEKIT_URL=ws://127.0.0.1:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
TTS_BASE_URL=http://127.0.0.1:9099/v1
TTS_VOICE=husein
TTS_MODEL=TTS-model
EOF
```

## Run

```bash
livekit-server --dev --bind 127.0.0.1 &          # dev keys: devkey/secret
/root/lk-venv/bin/python stress_agent.py start &
/root/lk-venv/bin/python load_client.py --concurrency 8 --utterances 4 --out c8.json
```

## Results (2026-08-07, H20-3e, colocated vLLM TTS engine, `mode=llm` normalizer in path)

Serving: **0 errors** at concurrency 1/4/8; TTFB p50 1.1–1.3 s (includes the LLM-normalizer
hop), p95 ≈ 2.5 s at C=8.

Same-sentence loudness across 8 rooms, before/after `STREAM_NORMALIZE` (now default-on):

| | `STREAM_NORMALIZE=false` | `STREAM_NORMALIZE=true` |
|---|---|---|
| Per-sentence active-RMS spread | 2.3 – 10.1 dB | **0.5 – 1.7 dB** |
| Clipped at full scale | 17/32 | 4/32 (brief transients) |
| TTFB p50 | 1.20 s | 1.34 s (noise) |

The spread comes from the LM's sampled tokens (temperature 0.6–0.7), not LiveKit or the
decoder: temp-0 runs are byte-identical, and the same spread reproduces over plain curl.

## Gotchas

- On a box with high CPU load (e.g. colocated training), the worker reports its load to
  livekit-server and the server **silently refuses to dispatch** ("no servers available").
  `stress_agent.py` sets `load_fnc=lambda *_: 0.0` to opt out — fine for a test rig, wrong
  for production.
- The dev-mode HMAC key triggers a PyJWT `InsecureKeyLengthWarning` — harmless here.
