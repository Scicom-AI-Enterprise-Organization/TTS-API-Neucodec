# LiveKit agent stress test

Stress-tests the TTS API through a **real LiveKit agent path** — text in, WebRTC audio out —
so measurements include everything a live caller hears (agent session, TTS plugin streaming,
Opus encode/decode). No STT, no LLM: text sent on the `tts-input` text-stream topic goes
straight to `session.say()` → `openai.TTS` plugin → this repo's `/v1/audio/speech`.

- `stress_agent.py` — minimal `livekit-agents` worker (auto-dispatches to every room).
- `load_client.py` — joins N rooms concurrently, sends text lines, captures the agent's
  audio track, measures per utterance: TTFB (text → first audible frame), audio duration,
  active-speech RMS, peak. Emits JSONL rows + a summary. **Shards across processes**
  (`--procs`, default one per 8 rooms) — one process cannot drive more than that without
  becoming the bottleneck itself (see Gotchas).

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
TTS_INTERLEAVE=false AGENT_IDLE_PROCESSES=8 /root/lk-venv/bin/python stress_agent.py start &
/root/lk-venv/bin/python load_client.py --concurrency 8 --utterances 4 --out c8.json

# a full sweep; --procs shards the client, which matters above 8 rooms
for c in 1 4 8 16; do
  /root/lk-venv/bin/python load_client.py --concurrency $c --utterances 8 \
    --texts ../pitch_stress_texts.txt --out lk_c$c.json
done
```

Agent env: `TTS_BASE_URL` `TTS_VOICE` `TTS_MODEL`, plus `TTS_INTERLEAVE` (send
`X-Interleave-Id`, default true) and `AGENT_IDLE_PROCESSES` (warm job-process pool,
default 4 — raise it for bursts).

## Results (2026-09-23) — full write-up in [`../LIVEKIT.md`](../LIVEKIT.md)

![livekit bench](../../docs/img/livekit_bench.png)

8 utterances per room, `TM_English_Normal`, patched app on one H20 + TP=4 LM elsewhere.

| rooms | LiveKit p50 | p95 | +`interleave_id` p50 | HTTP direct p50 | agent tax | errors |
|---|---|---|---|---|---|---|
| 1 | 0.232 | 0.589 | 0.310 | 0.102 | +130 ms | 0 |
| 4 | 0.246 | 0.613 | 0.316 | 0.111 | +135 ms | 0 |
| 8 | 0.256 | 0.698 | 0.316 | 0.120 | +136 ms | 0 |
| 16 | 0.258 | 0.736 | 0.307 | 0.147 | +111 ms | 0 |

- **The agent + WebRTC tax is a constant** (~130 ms), not a slope. A LiveKit TTFB
  regression is almost never LiveKit.
- **Flat to 16 rooms**, 0 errors in 688 utterances. The API itself is clean to 64
  (TTFB p50 0.920 s, 179 audio-s/s, 0 errors).
- **`interleave_id` costs ~64 ms** and shrinks under load (78 ms at 1 room, 49 at 16).
  At 16 rooms it tightens loudness sd 1.05 → 0.92 dB and the TTFB p95 0.736 → 0.594 s.
- **32 rooms is a ceiling of this rig**, not the API — ICE setup for 64 peers through one
  dev-config server. `LIVEKIT.md` has what was tried.

### Loudness (2026-08-07, the run that made `STREAM_NORMALIZE` default-on)

Same-sentence loudness across 8 rooms:

| | `STREAM_NORMALIZE=false` | `STREAM_NORMALIZE=true` |
|---|---|---|
| Per-sentence active-RMS spread | 2.3 – 10.1 dB | **0.5 – 1.7 dB** |
| Clipped at full scale | 17/32 | 4/32 (brief transients) |
| TTFB p50 | 1.20 s | 1.34 s (noise) |

It still holds through the whole transport: per-utterance sd stayed **0.92 – 1.26 dB** at
every concurrency in the 2026-09-23 sweep. The spread comes from the LM's sampled tokens
(temperature 0.6–0.7), not LiveKit or the decoder: temp-0 runs are byte-identical, and
the same spread reproduces over plain curl.

## Gotchas

- **One client process cannot drive more than ~8 rooms.** Each room's coroutine scans its
  frame buffer and runs numpy RMS on the shared event loop, so past that the *client's*
  scheduling delay is charged to the server as TTFB. Measured: 16 rooms from one process
  reported TTFB p50 **14.08 s**; the same 16 rooms over two processes reported **0.307 s**,
  nothing server-side changed, and the TTS app averaged 19% of one core throughout.
  `--procs` shards automatically — do not turn it off to "keep the run simple".
- **Raise the fd limit.** Each room opens several WebRTC sockets; on the default 1024 the
  *client* fails at 16 rooms with `Too many open files (os error 24)`, and the rooms that
  survive report inflated TTFB that reads exactly like server saturation. `load_client.py`
  raises `RLIMIT_NOFILE` when it shards.
- **Before believing a latency cliff, check what the service under test was doing.** Both
  traps above were recorded as server results first. If the app is idle, the cliff is yours.
- **`await ctx.connect()` before `session.start()`**, not after. The wrong order works at
  low concurrency and drops jobs under a burst — livekit-server kills a job whose room is
  not connected within 10 s of `job_entry`.
- **`num_idle_processes` is an admission limit.** livekit-agents runs each room in its own
  subprocess and keeps only a few spare; a burst larger than the pool has to boot new
  interpreters and rooms time out. `AGENT_IDLE_PROCESSES` sets it here.
- On a box with high CPU load (e.g. colocated training), the worker reports its load to
  livekit-server and the server **silently refuses to dispatch** ("no servers available").
  `stress_agent.py` sets `load_fnc=lambda *_: 0.0` to opt out — fine for a test rig, wrong
  for production.
- The dev-mode HMAC key triggers a PyJWT `InsecureKeyLengthWarning` — harmless here.
