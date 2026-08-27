# Streamable TTS API

Streaming Text-to-Speech and Voice Conversion API with dynamic batching, CUDA Graphs, and torch.compile support.

## Setup

### 1. Start vLLM backend

```bash
TTS_MODEL=Scicom-intl/Multilingual-Expressive-TTS-1.7B GPU_MEM_UTIL=0.7 \
docker compose -f vllm.yaml up --detach
```

Set `TTS_API=http://tts-engine:9093` in your [.env](.env) to point to the vLLM backend.

### 2. Configure environment

Copy [.env_example](.env_example) to `.env` and adjust as needed. See [app/env.py](app/env.py) for all available variables:

| Variable | Default | Description |
|---|---|---|
| `TTS_API` | `http://tts-engine:9093` | vLLM backend URL |
| `TTS_API_KEY` | ` ` | Bearer token sent to the vLLM backend when set |
| `MODEL_NAME` | `TTS-model` | Model identifier |
| `DEVICE` | ` ` (auto) | Codec decode device: empty = cuda→npu→cpu autodetect, or force `npu`/`cpu` |
| `DEFAULT_SPEAKER` | see [app/env.py](app/env.py) | Default voice |
| `SPEAKERS` | see [app/env.py](app/env.py) | Available voices (comma-separated) |
| `DEFAULT_TEMPERATURE` | `0.6` | Sampling temperature |
| `DEFAULT_REPETITION_PENALTY` | `1.15` | Repetition penalty |
| `DEFAULT_MAX_TOKENS` | `3072` | Max output tokens |
| `DEFAULT_PLAYBACK_SPEED` | `2.0` | First decode window in seconds ×50 tokens (2.0 ⇒ 100 tokens = 2 s) |
| `STREAM_CHUNK_GROWTH` | `2.0` | Each later decode window grows by this factor (1.0 = fixed windows) |
| `STREAM_MAX_CHUNK_S` | `10.0` | Cap on grown decode windows (seconds) |
| `STREAM_PAST_CONTEXT_S` | `3.0` | Past tokens included in every decode window then sliced off (no latency cost; pulls windowed decode toward one-shot) |
| `DEFAULT_PLAYBACK_OVERLAP_SPEED` | `0.2` | Overlap speed for crossfading |
| `DEFAULT_SPEAKING_RATE` | `1.0` | Default speaking rate (`speaking_rate` request field): 1.3 = 30% faster, 0.8 = slower, pitch preserved (WSOLA time stretch on the decoded audio, see below) |
| `DEFAULT_NORMALIZE_MALAYSIAN` | `false` | Default for the `normalize_malaysian` request field |
| `DEFAULT_NORMALIZER_MODE` | `rule` | Default for the `mode` request field (`rule` or `llm`) |
| `STREAM_CROSSFADE` | `true` | Context-primed windows + raised-cosine crossfade at chunk boundaries (removes the boundary click; `false` = legacy hard cut) |
| `CROSSFADE_MS` | `12.0` | Crossfade blend width in ms |
| `STREAM_NORMALIZE` | `true` | Loudness-normalize streamed audio toward `TARGET_RMS_DB` (running per-utterance estimate; kills the 3–10 dB run-to-run LM loudness variance and full-scale clipping) |
| `TARGET_RMS_DB` | `-16.0` | Target active-speech RMS (dBFS) when `STREAM_NORMALIZE` is on |
| `MAX_GAIN_DB` / `GAIN_SLEW_DB` | `12` / `1` | Gain clamp and max gain change per chunk |
| `DYNAMIC_BATCHING` | `true` | Batch concurrent decode calls (free at concurrency 1) |
| `MICROSLEEP` | `1e-4` | Batch collection interval (seconds) |
| `MAX_BATCH_SIZE` | `16` | Max requests per batch |
| `CUDA_GRAPH_BATCH` | `[]` (eager) | CUDA graph token-length buckets (seconds ×50). Must cover grown windows (`STREAM_MAX_CHUNK_S`+`STREAM_PAST_CONTEXT_S` ≈ 13.5 s) or oversize decodes fall back to eager, e.g. `[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 13.5]` |
| `TORCH_COMPILE` | `false` | Use torch.compile instead of CUDA Graphs |
| `DEBUG_AUDIO` | `false` | Save intermediate audio chunks to disk |
| `SENTRY_DSN` | ` ` | Sentry DSN for error tracking |
| `ENABLE_TRACING_SPANS` | `true` | Hot-path spans: dynamic-batch wait, vLLM wait, codec decode. **Also needs an exporter configured** (`OTLP_ENDPOINT` etc.) or spans are not built at all. See [Tracing](#tracing-loki--tempo) |
| `TRACING_SPANS_REQUIRE_EXPORTER` | `true` | The gate above. Set `false` only when a span processor is installed in code rather than via environment |
| `TRACE_ASGI_MESSAGE_SPANS` | `false` | Keep the OTel ASGI `http receive`/`http send` span per ASGI message. Off because streaming made it ~500 empty spans per request |
| `DISCONNECT_POLL_S` | `0.25` | How often the LM reader may ask whether the client disconnected (each check is a real ASGI receive) |
| `OTLP_ENDPOINT` | ` ` | Tempo OTLP endpoint for traces (e.g. `http://localhost:4317`). Handled by `wan` |
| `SERVICE_NAME` | `fastapi` | Service name on spans and logs. Handled by `wan` |
| `TRACING_SAMPLE` | `1.0` | Head sampling ratio; drop below 1.0 before enabling spans under load |
| `OPENAI_BASE_URL` | ` ` | OpenAI-compatible endpoint for the LLM normalizer (`mode: "llm"`). Empty = llm mode disabled |
| `OPENAI_API_KEY` | ` ` | API key for `OPENAI_BASE_URL` |
| `OPENAI_MODEL_NAME` | ` ` | Model to use on `OPENAI_BASE_URL` (e.g. `google/gemma-4-31b-it`). **Not** `MODEL_NAME`, which is the TTS model |
| `OPENAI_TIMEOUT` | `10` | LLM normalizer request timeout (seconds); bounds the TTS stall before rule-based fallback |
| `CONTEXT_STORE` | `file` | Where the cross-request speech context (`request_id`) lives: `file` = one small JSON per id in `CONTEXT_STORE_DIR`, shared by every uvicorn worker on the host; `memory` = this process only; `off` = ignore `request_id`. See [Speech context](#speech-context-request_id) |
| `CONTEXT_STORE_DIR` | ` ` | Directory for `CONTEXT_STORE=file`. Empty = `/dev/shm/tts-context` (RAM) when `/dev/shm` exists, else `$TMPDIR/tts-context`. All workers must see the same directory |
| `CONTEXT_MAX_S` | `20` | Seconds of previous speech tokens (×50) kept per id and prepended to the prompt, left-trimmed oldest-first |
| `CONTEXT_TTL_S` | `600` | Idle seconds after which an id's history is forgotten |
| `CONTEXT_MIN_GEN_TOKENS` | `1000` | Generation room (LM tokens, = 20 s of speech) the context may never squeeze below; beyond that the request's `max_tokens` is clamped instead |
| `LM_MAX_MODEL_LEN` | `4096` | The LM server's `--max-model-len` ([vllm.yaml](vllm.yaml)). vLLM rejects prompt + `max_tokens` beyond it with a 400, so context requests are sized against it |

### 3. Run the API

**GPU:**

```bash
docker compose up --build
```

**CPU:**

```bash
docker compose -f docker-compose-cpu.yaml up --build
```

## Tracing (Loki + Tempo)

The app calls [`wan.patch()`](https://github.com/Scicom-AI-Enterprise-Organization/wan)
at startup, which always gives it JSON logs carrying the active trace id, a request log
line per request, Prometheus metrics at `/metrics`, health probes and Scalar docs at
`/scalar`. That library owns `SERVICE_NAME`, `OTLP_ENDPOINT`, `TRACING_SAMPLE` and the
rest of the OTLP/log config.

`ENABLE_TRACING_SPANS` (on by default) adds this repo's own spans on the serving hot path
([app/tracing.py](app/tracing.py)), which answer where a request's time actually went:

```
POST /v1/audio/speech                 (fastapi instrumentation)
├── tts.normalize                     normalizer.mode=rule|llm, chars in/out
│   ├── normalize.llm                 the LLM normalizer call (mode="llm")
│   └── normalize.rule                the rule-based pipeline
└── tts.stream                        tts.ttfb_s, tts.lm_wait_s, tts.decode_wait_s,
    │                                 tts.decodes, tts.chunks, tts.audio_bytes
    ├── lm.generate                   the vLLM SSE stream, lm.deltas
    │   ├── lm.connect                POST → response headers
    │   └── lm.first_token            headers → first speech token (prefill)
    ├── tts.chunk (index=0)           one emitted audio chunk
    │   └── codec.decode              codec.tokens
    │       ├── codec.batch_wait      queued → picked up by the batch collector
    │       ├── codec.batch_prep      batch formed → H2D copy issued (batch thread)
    │       ├── codec.compute_wait    H2D issued → compute thread starts
    │       └── codec.gpu_decode      graph replay + D2H, batch.size, codec.cuda_graph
    └── tts.chunk (index=1) ...
```

So: **dynamic batching** = `codec.batch_wait` + `codec.batch_prep` + `codec.compute_wait`,
**waiting on vLLM** = `lm.connect` + `lm.first_token` and the `tts.lm_wait_s` aggregate
(how long the stitcher had no tokens left to decode), **decoding speech tokens** =
`codec.gpu_decode`, or `tts.decode_wait_s` for the whole per-request wait. Voice
conversion adds `vc.load_audio`, `codec.encode`, `codec.encode_wait` and
`codec.encode_gpu`.

Two things make this safe to leave in the code path:

- **Removable, not just cheap, and off unless someone is listening.** Spans are built
  only when `ENABLE_TRACING_SPANS` is on *and* an exporter is configured *and*
  opentelemetry is importable. Fail any of the three and every helper is a shared
  `contextlib.nullcontext()` or a function returning `None` before doing anything — no
  tracer lookup, no `time_ns()`, no per-decode dict. The decode loop is GIL-bound, so
  tracing has to be genuinely absent when off.
- **Explicit parents.** A decode batch is built from N different requests, so the
  batching threads cannot use the ambient span context; each queued item carries its
  request's context plus the timestamp of the previous hop, and each stage is recorded
  after the fact with those timestamps.

Under real load, sample: `TRACING_SAMPLE=0.05` keeps the trace volume (and the ~5 extra
spans per decode) sane.

### `OTLP_ENDPOINT` is part of the switch

`ENABLE_TRACING_SPANS` on its own does nothing: with no exporter configured the hot-path
spans are **not built at all**, and the app logs why at startup rather than leaving you to
wonder where the spans went:

```
hot-path spans requested but no span exporter is configured, so they are disabled rather
than built and dropped. Set OTLP_ENDPOINT (or ENABLE_CONSOLE_SPAN_EXPORTER=true) to
collect them, or TRACING_SPANS_REQUIRE_EXPORTER=false if a processor is installed in code.
```

The reason is that an OpenTelemetry SDK with no span processor attached still *builds*
every span, stores its attributes, and then drops it. Measured over 3000 iterations of one
request's worth of spans (24 spans with the real attribute sets, Apple M-series —
indicative, not H100 numbers):

| `ENABLE_TRACING_SPANS` | exporter | per request |
|---|---|---|
| `false`, or on with no exporter | — | **2.6 µs** (the nullcontext path) |
| `true` | none attached — what this gate prevents | 292 µs, all wasted |
| `true` | `BatchSpanProcessor` | 685 µs |

Any one of these opens the gate, so a console-exporter debug session or an
auto-instrumented deployment is not silently un-traced: `OTLP_ENDPOINT`,
`OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`, `JAEGER_HOST`,
`ENABLE_CONSOLE_SPAN_EXPORTER=true`. If you attach a processor in code instead, set
`TRACING_SPANS_REQUIRE_EXPORTER=false`.

What you give up when no exporter is configured is the *stage-level* `spanID` on log
lines: `traceID` and the request's own `spanID` still come from the FastAPI
instrumentation (as does the `X-Trace-Id` response header), but with no hot-path spans
every line of a request carries the same span id again.

One upstream default is turned off here. The OpenTelemetry ASGI instrumentation opens a
span per ASGI *message*, and a streaming response polls the receive channel as it reads
the LM stream — measured on an H20, one request produced **505 empty
`POST /v1/audio/speech http receive` spans** next to 19 real ones, which is both useless
in the flame graph and a large multiple on Tempo's ingest. Two fixes, and the same shape
of request now emits 25 spans and **zero** ASGI-message spans:

- `DISCONNECT_POLL_S` (0.25 s) bounds how often the LM reader asks whether the client
  disconnected. Each check is a real ASGI receive, and aiohttp yields two lines per SSE
  event, so the old per-line poll cost ~2 receives per speech token — event-loop time
  spent whether or not tracing is on. This alone took 505 noise spans to 9.
- `_suppress_asgi_message_spans()` in [app/main.py](app/main.py) removes the remainder,
  including the per-chunk `http send` spans that scale with utterance length. Set
  `TRACE_ASGI_MESSAGE_SPANS=true` to get the upstream behaviour back.

```bash
# a full Tempo + Loki + Alloy + Prometheus + Grafana stack to point it at
git clone https://github.com/Scicom-AI-Enterprise-Organization/wan
docker compose -f wan/grafana/docker-compose.yaml up -d \
  tempo loki alloy prometheus grafana

# then, in .env
ENABLE_TRACING_SPANS=true
OTLP_ENDPOINT=http://localhost:4327
SERVICE_NAME=tts-api
```

`ENABLE_CONSOLE_SPAN_EXPORTER=true` prints spans to stdout instead, which is enough to
check the tree without a backend.

## API Endpoints

### `GET /v1/audio/speaker`

Returns the list of available speaker voices.

### `POST /v1/audio/normalize` — Text Normalization

Normalizes text for TTS input. Strips markdown/HTML, then normalizes with one of two engines
selected by `mode`:

- **`rule`** (default) — the built-in rule-based pipeline; optionally applies Malaysian text
  normalization (email, URL, phone, IC, money, time, units, etc.) when `normalize_malaysian` is set.
- **`llm`** — sends the text to an OpenAI-compatible LLM (`OPENAI_BASE_URL` / `OPENAI_MODEL_NAME`)
  with a multilingual few-shot prompt ([app/prompt.py](app/prompt.py)). The reply is constrained to
  `{"normalized": "..."}` via `response_format` json_schema (vLLM guided decoding), so the model can
  only return the normalized text. Returns `400` if `OPENAI_*` is not configured, `502` if the LLM
  call fails. `normalize_malaysian` is ignored in this mode.

**Parameters:**

| Field | Type | Default | Description |
|---|---|---|---|
| `input` | string | required | Text to normalize |
| `normalize_malaysian` | bool | `DEFAULT_NORMALIZE_MALAYSIAN` (`false`) | Apply Malaysian text normalization (`rule` mode only) |
| `mode` | `rule` \| `llm` | `DEFAULT_NORMALIZER_MODE` (`rule`) | Normalization engine |

**Example:**

```bash
curl -X POST 'http://localhost:9091/v1/audio/normalize' \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "**Harga** rumah RM500,000. Hubungi 012-1234567 atau email husein.zol05@gmail.com",
    "normalize_malaysian": true
  }'

# LLM-based normalization
curl -X POST 'http://localhost:9091/v1/audio/normalize' \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "baki saya tinggal RM1,250.50 dan IC saya 960314875079",
    "mode": "llm"
  }'
# {"output":"baki saya tinggal seribu dua ratus lima puluh ringgit lima puluh sen dan IC saya
#   sembilan enam kosong tiga satu empat lapan tujuh lima kosong tujuh sembilan.","mode":"llm"}
```

### `POST /v1/audio/speech` — Text-to-Speech

Accepts JSON body.

**Parameters:**

| Field | Type | Default | Description |
|---|---|---|---|
| `input` | string | required | Text to synthesize |
| `voice` | string | `DEFAULT_SPEAKER` | Speaker voice (list via `GET /v1/audio/speaker`) |
| `model` | string | `TTS-model` | Model name |
| `response_format` | `pcm` \| `wav` | `pcm` | Output audio format |
| `temperature` | float | `0.6` | Sampling temperature |
| `repetition_penalty` | float | `1.15` | Repetition penalty |
| `max_tokens` | int | `3072` | Max output tokens |
| `stream` | bool | `true` | Stream audio response |
| `playback_speed` | float | `2.0` | First decode window (×50 tokens; later windows grow per `STREAM_CHUNK_GROWTH`) |
| `playback_overlap_speed` | float | `0.2` | Overlap for crossfading |
| `normalize_malaysian` | bool | `DEFAULT_NORMALIZE_MALAYSIAN` (`false`) | Apply Malaysian text normalization |
| `mode` | `rule` \| `llm` | `DEFAULT_NORMALIZER_MODE` (`rule`) | Normalization engine (see `/v1/audio/normalize`); `llm` falls back to `rule` if the LLM call fails, so speech is still produced |
| `stream_normalize` | bool | `STREAM_NORMALIZE` (`true`) | Per-utterance loudness normalization toward `TARGET_RMS_DB` |
| `speaking_rate` | float | `DEFAULT_SPEAKING_RATE` (`1.0`) | Speaking rate, `0.5`–`2.0`: `1.3` speaks 30% faster, `0.8` slower, **pitch unchanged**. Also accepted as `speed` (OpenAI-compatible clients). See [Speaking rate](#speaking-rate) |
| `request_id` | string | none | Speech-context id: requests sharing it are generated **in the context of the previous ones** (their text + speech tokens go into the prompt), so an agent that splits one reply into several TTS calls gets one continuous prosody. Alias `context_id`; also accepted as the `X-Context-Id` request header. See [Speech context](#speech-context-request_id) |

**Example:**

```bash
curl -X POST 'http://localhost:9091/v1/audio/speech' \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Hello! How can I help you? can I get your passport number sir.",
    "voice": "husein",
    "model": "TTS-model",
    "response_format": "wav",
    "temperature": 0.7,
    "stream": true,
    "playback_speed": 0.5,
    "playback_overlap_speed": 0.1,
    "speaking_rate": 1.2,
    "normalize_malaysian": true
  }' \
  --output output.wav
```

#### Speaking rate

`speaking_rate` (alias `speed`) changes how fast the voice talks **without changing its pitch**.
The LM has no rate control and speech tokens are a fixed 50 Hz, so this is done after NeuCodec
decode, on the PCM stream, with WSOLA (waveform-similarity overlap-add — the SoundTouch-style
tempo change, `app/timestretch.py`): the audio is copied out in ~40 ms blocks whose input hop is
`rate`× the output hop, so whole pitch periods are dropped (faster) or repeated (slower) while the
samples inside each block are untouched; each block's start is searched ±7.5 ms for the best
waveform match to the previous block's tail and the two are crossfaded over 8 ms. Plain
resampling would shorten every period too (chipmunk effect). It is a stage on the stitched stream
(after crossfade + loudness normalization, before the pcm/wav/SSE layers), stateful across
chunks with ~55 ms of lookahead — so it works for streaming, adds no measurable latency to the
2 s first chunk, and leaves the token/decode/batching pipeline untouched. `1.0` bypasses it
entirely. Range `0.5`–`2.0` (422 outside): beyond that WSOLA on speech starts to buzz / drop
consonants.

```bash
curl -X POST 'http://localhost:9091/v1/audio/speech' -H 'Content-Type: application/json' \
  -d '{"input": "Hello there, how can I help you?", "voice": "husein", "speaking_rate": 1.3,
       "response_format": "wav", "stream": false}' --output fast.wav
```

#### Speech context (`request_id`)

Streaming agents do not send one utterance per TTS call. LiveKit's `StreamAdapter`, for
example, cuts the LLM's reply into sentence-ish chunks (≥ 20 chars, so often 5–6 words) and
synthesizes each with its **own** request, strictly one after the other. Every chunk therefore
starts the LM cold — it has no memory of how the previous chunk sounded — so pitch register,
pace and energy reset at each join and the reply sounds stitched together, even though the model
handles the same paragraph fine in a single request.

`request_id` gives the LM that memory. Send the same id on consecutive chunks and each one is
prompted with the previous turns in the model's own multi-turn format (the one `/v1/audio/vc`
uses to prime a reference voice) — the text that was spoken **and the speech tokens the LM
produced for it** — followed by the new text:

```
<|im_start|>husein: hello my name is husein,<|speech_start|><|s_…|>…<|s_…|><|im_end|>
<|im_start|>husein: i like to eat chicken rice.<|speech_start|>          ← generated
```

Only the new turn's tokens are generated, decoded and streamed; the LM continues in the
prosodic state it left off in. When a turn finishes cleanly its (text, tokens) pair is appended
to the id's history for the next chunk (a turn cut off by `max_tokens` — `finish_reason:
length` — is not stored: misaligned text/audio is worse context than none). A `request_id` with
no history behaves exactly like a request without one, so the first chunk pays nothing.

```bash
# chunk 1 (no history yet -> plain prompt) …
curl -s -X POST localhost:9091/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input":"hello my name is husein,","voice":"husein","request_id":"room-42","response_format":"wav","stream":false}' -o c1.wav
# … chunk 2 is generated in the context of chunk 1
curl -s -D - -X POST localhost:9091/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input":"i like to eat chicken rice.","voice":"husein","request_id":"room-42","response_format":"wav","stream":false}' -o c2.wav
#   X-Context-Id: room-42
#   X-Context-Turns: 1          <- turns in the prompt
#   X-Context-Tokens: 137       <- their speech tokens (2.7 s)
curl -s localhost:9091/v1/audio/context/room-42        # inspect: stored turns (text + token counts)
curl -s -X DELETE localhost:9091/v1/audio/context/room-42   # forget (else idle ids expire after CONTEXT_TTL_S)
```

The id is whatever identifies one continuous speaker session on the client — a LiveKit room /
agent session, a call id. Clients that cannot add body fields (the livekit `openai.TTS` plugin
drives the stock `openai` client) send it as the **`X-Context-Id` header** instead:
`openai.TTS(client=openai.AsyncClient(base_url=..., api_key=..., default_headers={"X-Context-Id": ctx.room.name}))`
— see [`bench/livekit/stress_agent.py`](bench/livekit/stress_agent.py). The full design
walkthrough (prompt format, trim rules, worker sharing, why not a leader worker) is in
[`SPEECH_CONTEXT.md`](SPEECH_CONTEXT.md).

**Left trim.** A call can run for minutes while the LM window is fixed (`LM_MAX_MODEL_LEN`, 4096
tokens ≈ 80 s of speech end to end), so the history is capped at `CONTEXT_MAX_S` (20 s = 1000
speech tokens) per id: whole oldest turns are dropped first, then the oldest kept turn is cut to
its last N tokens with its text shortened in proportion, so one long previous utterance still
contributes its tail — the most recent seconds are what continuity needs. `max_tokens` is then
clamped so prompt + generation fit the window (vLLM 400s instead of truncating); the context
only shrinks further if that would leave under `CONTEXT_MIN_GEN_TOKENS` of generation room. A
voice switch on the same id starts cold (priming with another voice's tokens is voice
conversion, not continuity).

**Multi-worker.** With `uvicorn --workers N` consecutive chunks land on arbitrary workers, so the
history lives outside the process: `CONTEXT_STORE=file` (default) keeps one ~1–6 KB JSON per id
under `CONTEXT_STORE_DIR` (`/dev/shm`, i.e. RAM, by default; docker's default 64 MB `/dev/shm` holds ~10k ids) — atomic replace-on-write, one
`flock` around each append, TTL sweep — which every worker on the host sees with no leader
election, extra port, or new dependency. A turn is committed **before** the LM reader signals
end-of-stream to the audio stitcher, so by the time a client has received chunk N in full, chunk
N is in the store for chunk N+1 whichever worker that one hits (and LiveKit does wait for chunk
N before requesting N+1). Overlapping requests on one id are kept in request-arrival order.
Note `repetition_penalty` in vLLM also covers prompt tokens, so a context request penalizes the
previous turns' codes exactly as `/v1/audio/vc` does with its reference tokens.

### `POST /v1/audio/vc` — Voice Conversion

Accepts multipart form data. Clones the reference voice and generates speech for new text.

**Parameters:**

| Field | Type | Default | Description |
|---|---|---|---|
| `reference_audio` | file | required | Reference audio file |
| `reference_text` | string | required | Transcript of the reference audio |
| `generate_text` | string | required | Text to generate in the cloned voice |
| `model` | string | `TTS-model` | Model name |
| `response_format` | `pcm` \| `wav` | `pcm` | Output audio format |
| `temperature` | float | `0.6` | Sampling temperature |
| `repetition_penalty` | float | `1.15` | Repetition penalty |
| `max_tokens` | int | `3072` | Max output tokens |
| `stream` | bool | `true` | Stream audio response |
| `playback_speed` | float | `2.0` | First decode window (×50 tokens) |
| `playback_overlap_speed` | float | `0.2` | Overlap for crossfading |
| `speaking_rate` | float | `DEFAULT_SPEAKING_RATE` (`1.0`) | Speaking rate `0.5`–`2.0`, pitch preserved (see [Speaking rate](#speaking-rate)) |

**Example:**

```bash
curl -X POST 'http://localhost:9091/v1/audio/vc' \
  -H 'Content-Type: multipart/form-data' \
  --form 'reference_audio=@jenny.wav' \
  --form 'reference_text=I wonder if I shall ever be happy enough to have real lace on my clothes and bows on my caps.' \
  --form 'generate_text=Ye encik, apa yang saya boleh tolong?' \
  --form 'model=TTS-model' \
  --form 'response_format=wav' \
  --form 'temperature=0.7' \
  --form 'repetition_penalty=1.15' \
  --form 'max_tokens=3072' \
  --form 'stream=true' \
  --form 'playback_speed=1.5' \
  --form 'playback_overlap_speed=0.2' \
  --form 'speaking_rate=1.0' \
  --output vc.wav
```

## Gradio Demo

`gradio_app.py` is a simple UI for poking at a running API instance (local or remote) without curl —
tabs for text-to-speech, text normalization, and voice conversion.

```bash
pip install gradio requests
TTS_API_BASE_URL=http://localhost:9091 python gradio_app.py   # defaults to localhost:9091
```

Open the printed local URL, set the API base URL if needed, and click **Check connection** to list
available speakers.

## Unit Tests

### Running tests

```bash
pip install pytest requests

# or with uv (no install needed), e.g. the LLM normalizer tests:
uv run --with aiohttp --with pytest -- pytest tests/test_llm_normalizer.py -v
# include the live LLM tests (hits OPENAI_BASE_URL):
set -a; source .env; set +a
uv run --with aiohttp --with pytest -- pytest tests/test_llm_normalizer.py -v

# run all tests (unit + integration against live API)
python -m pytest tests/ -v

# run only markdown sanitization tests (no dependencies beyond app/rules.py)
python -m pytest tests/test_sanitize_markdown.py -v

# run the speech-context tests (no GPU deps)
uv run --with pytest -- pytest tests/test_context.py -v

# run the tracing helper tests (no GPU; add opentelemetry-sdk for the enabled-path tests)
uv run --with pytest --with opentelemetry-sdk -- pytest tests/test_tracing.py -v

# run normalizer tests (requires app/normalizer dependencies: dateparser, unidecode, numpy)
python -m pytest tests/test_normalizer.py -v

# run Malaysian rules stress tests
python -m pytest tests/test_malaysian_rules.py -v

# run multilingual tests
python -m pytest tests/test_multilingual.py -v

# run TTS & VC integration tests against a live API (default: http://localhost:9091)
python -m pytest tests/test_tts_vc_api.py -v

# run TTS & VC tests against a different URL
TTS_TEST_URL=http://localhost:8080 python -m pytest tests/test_tts_vc_api.py -v

# run normalize API integration tests (requires in-process app import)
python -m pytest tests/test_normalize_api.py -v
```

### Test files

**623 passed, 35 skipped** on a full run with a live API and `OPENAI_*` configured.

| File | Tests | Dependencies | Description |
|---|---|---|---|
| `tests/test_sanitize_markdown.py` | 66 | None (only `app/rules.py`) | Markdown and HTML stripping: bold, italic, headings, links, images, code blocks, blockquotes, lists, horizontal rules, HTML tags. Edge cases for IC numbers, phone numbers, URLs, underscore variables, unicode. |
| `tests/test_normalizer.py` | 150 | `app/normalizer`, `app/rules` | Text normalization with exact input/output checks in both Malay and English: email (`husein.zol05@gmail.com` -> `HUSEIN dot ZOL kosong lima di GMAIL dot COM`), URL, phone, IC number, money (RM/USD), time, percentage, units (kg, km, celsius, liter, MB), dates, cardinals, ordinals, fractions, multipliers, hingga/range, contractions, alpha-num splitting, replace mappings, and combined markdown+normalizer pipeline. |
| `tests/test_malaysian_rules.py` | 208 | `app/normalizer`, `app/rules` | Stress tests for Malaysian normalization rules. Exhaustive coverage of: money (RM whole/sen/zero/sentence, USD with K/M suffixes), IC numbers (standard/young/zeros/multiple), phone numbers (mobile 012/011, landline 03, multiple), email (basic/subdomain/sentence/multiple), URL (https/www/path/IP), time (AM/PM/midnight/morning/late night), percentages (decimal/100/small), units (celsius/kg/g/km/liter/ml/mb/gb), dates, zero-prefix numbers, passports, year normalization (tahun 2024/1999/2000/1945), pada hari bulan, ordinals (ke-1/ke-100/Roman), cardinals, fractions, multiplier (x kali), hingga, Hijri year, elongated words, tak prefix, all 51 pronunciation replacements (dr/mr/mrs/Sdn Bhd/LRT/MRT/KL/PDRM/CCTV/UMNO/5G/US), pattern ranges (100-200 ringgit), all contractions, alpha-num splitting, replace mappings, and 12 complex multi-type sentence tests simulating the full pipeline. |
| `tests/test_multilingual.py` | 76 | `app/normalizer`, `app/rules` | Multilingual passthrough tests for Chinese (Simplified/Traditional), Korean, Tamil, Arabic, Japanese (Hiragana/Katakana/Kanji), Thai, Hindi/Devanagari, and emoji. Verifies non-Latin scripts pass through untouched while ASCII content (RM, phone, email, URL, IC, time, %) is still normalized. Tests mixed-script sentences, markdown stripping with multilingual text, and the non-ASCII-attached-to-ASCII edge case (e.g. `价格是RM500` passes through raw vs `价格是 RM500` normalizes). |
| `tests/test_tts_vc_api.py` | 57 | Live API (`TTS_TEST_URL`, default `http://localhost:9091`) | Integration tests for TTS (`POST /v1/audio/speech`) and VC (`POST /v1/audio/vc`) endpoints. **TTS tests** (29): WAV/PCM format validation (sample rate 24000, mono, 16-bit), streaming vs buffered, all speakers (husein/jenny/idayu), speaker list endpoint, markdown/HTML/link sanitization, Malaysian normalization with numbers, temperature/speed/max_tokens parameters, `speaking_rate` (valid wav, SSE streaming, 2.0 shorter than 0.5, `speed` alias, 422 out of range), short/long/English/multilingual text. **Context tests** (7): `request_id` speech context — no headers without an id, second chunk reports the first as `X-Context-Turns: 1` with a token count matching its duration, `GET`/`DELETE /v1/audio/context/{id}`, `X-Context-Id` header and `context_id` alias, voice switch starts cold, headers on pcm/SSE streaming; skipped when the server runs `CONTEXT_STORE=off`. **VC tests** (21): uses `jenny.wav` with reference text, WAV/PCM format validation, streaming vs buffered, Malay/English/long/short generate text, markdown/HTML/link/code sanitization in both reference_text and generate_text, temperature/speed/max_tokens/speaking_rate parameters. Auto-skipped when the API is not reachable. |
| `tests/test_normalize_api.py` | 35 | In-process app import (GPU/models) | Integration tests for `POST /v1/audio/normalize` endpoint. Auto-skipped when the app cannot be imported. Tests `normalize_malaysian=false` (sanitize only), `normalize_malaysian=true` (full normalization), and the `mode` enum (`rule` default, `llm`, invalid → 422). |
| `tests/test_tracing.py` | 12 | `pytest` only (no GPU/torch; 8 need `opentelemetry-sdk`) | Unit tests for the hot-path spans (`app/tracing.py`): disabled by default, disabled helpers are no-ops returning a single shared `nullcontext`, and — with the OTel SDK installed — spans nest, an explicitly passed parent beats the ambient context (what the batching threads rely on), `record_span` honours the given timestamps and drops a stage with no start time, attributes are cleaned, exceptions mark the span. |
| `tests/test_timestretch.py` | 33 | `numpy` + `pytest` only (no GPU/torch) | Unit tests for the speaking-rate stretcher (`app/timestretch.py`): rate 1.0 is an exact identity, duration scales by the rate (0.5–2.0) to within one block, a steady tone keeps its frequency (pitch preserved), joins are continuous, output is bit-identical regardless of how the input is chunked (streaming), first chunk emitted promptly, empty/sub-block/silence/full-scale edge cases, PCM16 helpers. |
| `tests/test_context.py` | 34 | `pytest` only (no GPU/torch) | Unit tests for the speech context (`app/context.py`): left trim (whole turns first, proportional tail cut of the oldest kept turn, min-partial threshold), voice filtering, the multi-turn prompt format, `fit_context` (prompt + `max_tokens` never exceed the LM window, generation floor wins over context), memory and file stores (shared between instances, trim on append, TTL + sweep, corrupt-file recovery, hostile keys, 8 concurrent writers with no lost updates / torn reads), `X-Context-*` headers latin-1 safe. |
| `tests/test_llm_normalizer.py` | 28 | `aiohttp` + `pytest` only (no GPU/torch) | Unit tests for the LLM-based normalizer (`app/llm_normalizer.py`): few-shot message building from `app/prompt.py`, strict JSON schema, reply parsing (clean/fenced/bare/plain-text/unusable), and full HTTP behaviour against a local fake OpenAI server (auth header, payload shape, 400 retry without `response_format`, 5xx/unreachable/unconfigured errors). 4 live tests hit the real `OPENAI_BASE_URL` (Malay money, English IC, Chinese money, passthrough) and are skipped unless `OPENAI_*` is set. |

## Benchmark — H100 SXM vs H200 SXM

End-to-end throughput/latency of the **optimized stack** on a single GPU, measured on RunPod secure cloud
(June 2026). Both pods run the same colocated layout — vLLM `0.16.0` serving the Qwen3-1.7B TTS LM plus the
vendored NeuCodec decoder on one GPU — with the optimized configuration: **dynamic batching + CUDA graphs +
4 `uvicorn` workers under NVIDIA MPS**, bf16 throughout (no quantization), torch `2.9.1` / CUDA `12.8`.
vLLM `--gpu-memory-utilization 0.4 --max-num-seqs 64`. Driver 580.126.09 (H100) / 550.127.05 (H200).

Load generator: [`bench/bench.py`](bench/bench.py), **non-streaming**, concurrency 1/4/16/50, `per-conc-mult 5`,
`temperature 0.6`, `max_tokens 1024`, over the fixed 16-sentence eval set (~4.5–4.9 s audio/request; English,
Malay, code-switch). **Median of 5 runs per level; 0 request errors at every level.** Throughput is the
meaningful TTS metric: **audio-seconds produced per wall-second** (values > the eval clip length mean the
server emits audio faster than real time in aggregate).

### Throughput (audio-seconds produced per wall-second)

| Concurrency | H100 SXM 80GB | H200 SXM 141GB | H200 advantage |
|---|---|---|---|
| 1  | 6.7   | 8.4   | 1.25× |
| 4  | 25.2  | 29.2  | 1.16× |
| 16 | 44.0  | 81.9  | **1.86×** |
| 50 | 101.9 | 143.8 | **1.41×** |

### Latency & real-time factor (RTF)

| Metric | H100 @1 | H200 @1 | H100 @50 | H200 @50 |
|---|---|---|---|---|
| Mean latency (s) | 0.69 | 0.55 | 2.11 | 1.49 |
| p99 latency (s)  | 1.25 | 1.01 | 6.44 | 3.44 |
| RTF p50          | 0.15 | 0.12 | 0.37 | 0.32 |

**Takeaways**

- **H200 wins at every concurrency**, with the largest margins under load — **1.86× at C=16** and **1.41×
  at C=50** — where the bandwidth-bound NeuCodec decode dominates. The H200's higher HBM bandwidth (≈4.8 TB/s
  vs ≈3.35 TB/s) and 141 GB capacity give the colocated vLLM + 4 codec workers more headroom.
- **Both run faster than real time at all concurrencies** (RTF p50 < 1). Single-request latency is ~0.6 s
  (H100) / ~0.5 s (H200) for ~4.5 s of audio (RTF ≈ 0.15 / 0.12).
- **H200 is also far more stable.** Across the 5 runs the H200 throughput was tight (C=16: 80.2–83.5; C=50:
  141–145), while the H100 showed large run-to-run swings at high concurrency (C=16: 32–60; C=50: 95–116) —
  transient MPS/codec contention stalls that the H200's extra bandwidth and memory absorb. The table reports
  medians; raw per-run JSON is in [`bench/results/runpod_h100_h200.json`](bench/results/runpod_h100_h200.json).

The full optimization writeup (baseline → CUDA graphs → multi-worker + MPS, and a documented negative result
on source-level micro-optimizations) is in [`bench/OPTIMIZATION.md`](bench/OPTIMIZATION.md). To reproduce, see
the ready scripts in [`bench/deploy/`](bench/deploy) (`setup_pod.sh`, `start_vllm.sh`, `start_app.sh`).

## LiveKit agent stress test & loudness consistency

[`bench/livekit/`](bench/livekit) drives the API through a **real LiveKit agent** (text in → agent
`session.say()` → TTS → WebRTC audio out) at concurrency 1/4/8: 0 errors, TTFB p50 ≈ 1.2 s with the
LLM normalizer in the path. It also quantified utterance-to-utterance loudness variance — the LM's
sampled speech tokens carry loudness, so identical text at temperature 0.6–0.7 spans **3–10 dB**
active-RMS (and different voices sit ~5 dB apart in natural level), and roughly half
of hot utterances clip at full scale. `STREAM_NORMALIZE=true` (default; per-request
`stream_normalize`) collapses the spread to **< 2 dB** with no added latency: one static gain per
utterance (locked after the first ~1 s of voiced audio — no mid-utterance drift), boost capped by
running-peak headroom, and a tanh soft-knee limiter instead of a hard clip. Details and
before/after numbers in [`bench/livekit/README.md`](bench/livekit/README.md).

LiveKit also **chunks** each reply into sentence-sized TTS requests, which is why consecutive
chunks used to sound disconnected: each was generated cold. The agent now passes the room name as
`X-Context-Id` so every chunk is generated in the context of the previous ones — see
[Speech context](#speech-context-request_id).

Streaming decode quality itself is near one-shot: the stitcher decodes **growing windows** (first =
`playback_speed`×50 tokens, ×`STREAM_CHUNK_GROWTH` per step up to `STREAM_MAX_CHUNK_S`) with
`STREAM_PAST_CONTEXT_S` of already-generated past tokens included in every window and sliced off
after decode (free, unlike future context). Measured streamed-vs-one-shot envelope gap on identical
tokens: **0.03–0.04 dB median** (worst 0.3–0.5 dB, first window only) vs 0.25–0.55 dB median with
the old fixed 1.5 s windows. For offline generation (`stream: false`), `"playback_speed": 10`
decodes the whole utterance in one window.

## Huawei Ascend 910B3 NPU — support & the precision quality gap

The full stack runs on Huawei Ascend 910B3 (CANN 8.5.2): the **vLLM LM via
[`vllm-ascend`](https://github.com/vllm-project/vllm-ascend) on NPU 0**, and the **NeuCodec decoder via
`torch_npu` on NPU 1**. The app is unchanged apart from device selection — set `DEVICE=npu` and it uses a
`torch.npu` stream shim, skipping CUDA graphs/streams automatically (see [app/main.py](app/main.py)). Pin
each stage to a chip with `ASCEND_RT_VISIBLE_DEVICES` (0 for vLLM, 1 for the codec).

**Working install** (Python **3.11** — vLLM 0.11 uses py3.10 syntax that breaks on 3.9):

```bash
# codec app venv (py3.9 ok): torch 2.8.0 + torch_npu 2.8.0.post5  → see requirements-npu.txt
# vLLM venv (py3.11): install vllm first, then vllm-ascend (it pins torch back to 2.7.1)
uv venv /root/vllm-venv311 --python 3.11 && source /root/vllm-venv311/bin/activate
uv pip install vllm==0.11.0
uv pip install vllm-ascend==0.11.0 "setuptools<81"   # torch 2.7.1 + torch-npu 2.7.1; <81 keeps pkg_resources
source /usr/local/Ascend/ascend-toolkit/set_env.sh && source /usr/local/Ascend/nnal/atb/set_env.sh
```

The NeuCodec **decode runs cleanly on the NPU** (bit-identical to CUDA — the vendored RoPE in
[`app/neucodec/_rope.py`](app/neucodec/_rope.py) removed the torchtune/torchao dependency, which is what
made py3.9 + NPU viable). **Encode** currently must run on CPU (the encoder's alias-free resample mis-shapes
on NPU); this only affects VC/token-extraction, not TTS decode.

### 910B3 vs H100 — measured (single-stream, 16-sentence eval set, bf16, temp 0.6)

| Metric | H100 SXM | Ascend 910B3 | Notes |
|---|---|---|---|
| LM rate (vLLM) | ~380 tok/s | ~81 tok/s | **~4.8× slower** |
| End-to-end RTF | ~0.13 | ~0.68 | LM-bound; codec decode is fast on both |
| Intelligibility (Whisper large-v3 CER) | 2.2% | 3.4% | comparable — content not garbled |
| **Naturalness (UTMOSv2 MOS)** | **3.2** | **2.6** | **−0.6 MOS — audibly less natural** |

> ⚠️ **Open issue — the bf16 quality gap.** With identical decode and identical sampling settings, the
> Ascend LM produces **measurably less-natural speech tokens** (−0.6 MOS) than the H100, despite comparable
> intelligibility. The gap is in the `vllm-ascend` LM path, i.e. the 910B3's bf16 compute kernels — **not**
> the codec. Mitigations tested and their MOS:
>
> | Config | MOS | Result |
> |---|---|---|
> | Ascend bf16 + ACL graphs (default) | 2.60 | baseline |
> | Ascend bf16 + `--enforce-eager` | 2.45 | **no help** — ACL graph capture is not the cause |
> | Ascend `--dtype float16` + eager | 1.73 | **much worse** — keep bf16 |
> | H100 bf16 | 3.21 | target |
>
> Neither disabling graph capture nor switching precision closes it, so **for this TTS model the 910B3 is
> both slower and less natural than H100 at bf16.** MOS measured with
> [faster-UTMOSv2](https://github.com/Scicom-AI-Enterprise-Organization/faster-UTMOSv2).
>
> **Root cause — isolated (not fixable via config):** the gap is entirely in the **LM tokens the
> `vllm-ascend` bf16 forward produces**, not the codec:
> - **Codec decode is bit-identical on NPU vs CPU** — decoding the *same* tokens on both gives
>   `MAE = 0.00000` on all 16 clips. Rules out the decoder.
> - **Sampling is already fp32** (AscendSampler softmax `dtype=float32`) and matmul HF32 is **off**
>   (`torch.npu.matmul.allow_hf32 == False`) — both full precision. Rules out sampler + matmul.
> - **fp32 is impossible on this stack**: `--dtype float32` crashes the engine — the Ascend paged-KV
>   attention op (`ReshapeCacheOperation`, `ERR00100`) only supports bf16/fp16. So the fused
>   **attention kernel is locked to bf16**, and its internal accumulation is what diverges from the
>   H100's bf16 attention, shifting sampled tokens toward less-natural speech.
>
> There is **no serving-flag fix** today. Real fixes require a higher-precision (fp32-accumulate)
> attention kernel from Huawei/`vllm-ascend`, a different attention backend, or fine-tuning the model to
> the 910B3 bf16 numerics. Tracked for upstream.