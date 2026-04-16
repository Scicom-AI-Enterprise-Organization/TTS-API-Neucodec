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
| `MODEL_NAME` | `TTS-model` | Model identifier |
| `DEFAULT_SPEAKER` | `husein` | Default voice |
| `SPEAKERS` | `husein,idayu,jenny` | Available voices |
| `DEFAULT_TEMPERATURE` | `0.6` | Sampling temperature |
| `DEFAULT_REPETITION_PENALTY` | `1.15` | Repetition penalty |
| `DEFAULT_MAX_TOKENS` | `3072` | Max output tokens |
| `DEFAULT_PLAYBACK_SPEED` | `1.5` | Playback speed multiplier |
| `DEFAULT_PLAYBACK_OVERLAP_SPEED` | `0.2` | Overlap speed for crossfading |
| `DYNAMIC_BATCHING` | `false` | Enable dynamic batching |
| `MICROSLEEP` | `1e-4` | Batch collection interval (seconds) |
| `MAX_BATCH_SIZE` | `16` | Max requests per batch |
| `CUDA_GRAPH_BATCH` | `[]` | CUDA graph bucket sizes, e.g. `[0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 10.0]` |
| `TORCH_COMPILE` | `false` | Use torch.compile instead of CUDA Graphs |
| `DEBUG_AUDIO` | `false` | Save intermediate audio chunks to disk |
| `SENTRY_DSN` | ` ` | Sentry DSN for error tracking |

### 3. Run the API

**GPU:**

```bash
docker compose up --build
```

**CPU:**

```bash
docker compose -f docker-compose-cpu.yaml up --build
```

## API Endpoints

### `GET /v1/audio/speaker`

Returns the list of available speaker voices.

### `POST /v1/audio/normalize` — Text Normalization

Normalizes text for TTS input. Strips markdown/HTML and optionally applies Malaysian text normalization (email, URL, phone, IC, money, time, units, etc.).

**Parameters:**

| Field | Type | Default | Description |
|---|---|---|---|
| `input` | string | required | Text to normalize |
| `normalize_malaysian` | bool | `false` | Apply Malaysian text normalization |

**Example:**

```bash
curl -X POST 'http://localhost:9091/v1/audio/normalize' \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "**Harga** rumah RM500,000. Hubungi 012-1234567 atau email husein.zol05@gmail.com",
    "normalize_malaysian": true
  }'
```

### `POST /v1/audio/speech` — Text-to-Speech

Accepts JSON body.

**Parameters:**

| Field | Type | Default | Description |
|---|---|---|---|
| `input` | string | required | Text to synthesize |
| `voice` | string | `husein` | Speaker voice |
| `model` | string | `TTS-model` | Model name |
| `response_format` | `pcm` \| `wav` | `pcm` | Output audio format |
| `temperature` | float | `0.6` | Sampling temperature |
| `repetition_penalty` | float | `1.15` | Repetition penalty |
| `max_tokens` | int | `3072` | Max output tokens |
| `stream` | bool | `true` | Stream audio response |
| `playback_speed` | float | `1.5` | Playback speed |
| `playback_overlap_speed` | float | `0.2` | Overlap for crossfading |
| `normalize_malaysian` | bool | `true` | Apply Malaysian text normalization |

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
    "normalize_malaysian": true
  }' \
  --output output.wav
```

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
| `playback_speed` | float | `1.5` | Playback speed |
| `playback_overlap_speed` | float | `0.2` | Overlap for crossfading |

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
  --output vc.wav
```

## Unit Tests

### Running tests

```bash
pip install pytest requests

# run all tests (unit + integration against live API)
python -m pytest tests/ -v

# run only markdown sanitization tests (no dependencies beyond app/rules.py)
python -m pytest tests/test_sanitize_markdown.py -v

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

**543 passed, 31 skipped** on a full run with a live API.

| File | Tests | Dependencies | Description |
|---|---|---|---|
| `tests/test_sanitize_markdown.py` | 66 | None (only `app/rules.py`) | Markdown and HTML stripping: bold, italic, headings, links, images, code blocks, blockquotes, lists, horizontal rules, HTML tags. Edge cases for IC numbers, phone numbers, URLs, underscore variables, unicode. |
| `tests/test_normalizer.py` | 150 | `app/normalizer`, `app/rules` | Text normalization with exact input/output checks in both Malay and English: email (`husein.zol05@gmail.com` -> `HUSEIN dot ZOL kosong lima di GMAIL dot COM`), URL, phone, IC number, money (RM/USD), time, percentage, units (kg, km, celsius, liter, MB), dates, cardinals, ordinals, fractions, multipliers, hingga/range, contractions, alpha-num splitting, replace mappings, and combined markdown+normalizer pipeline. |
| `tests/test_malaysian_rules.py` | 208 | `app/normalizer`, `app/rules` | Stress tests for Malaysian normalization rules. Exhaustive coverage of: money (RM whole/sen/zero/sentence, USD with K/M suffixes), IC numbers (standard/young/zeros/multiple), phone numbers (mobile 012/011, landline 03, multiple), email (basic/subdomain/sentence/multiple), URL (https/www/path/IP), time (AM/PM/midnight/morning/late night), percentages (decimal/100/small), units (celsius/kg/g/km/liter/ml/mb/gb), dates, zero-prefix numbers, passports, year normalization (tahun 2024/1999/2000/1945), pada hari bulan, ordinals (ke-1/ke-100/Roman), cardinals, fractions, multiplier (x kali), hingga, Hijri year, elongated words, tak prefix, all 51 pronunciation replacements (dr/mr/mrs/Sdn Bhd/LRT/MRT/KL/PDRM/CCTV/UMNO/5G/US), pattern ranges (100-200 ringgit), all contractions, alpha-num splitting, replace mappings, and 12 complex multi-type sentence tests simulating the full pipeline. |
| `tests/test_multilingual.py` | 76 | `app/normalizer`, `app/rules` | Multilingual passthrough tests for Chinese (Simplified/Traditional), Korean, Tamil, Arabic, Japanese (Hiragana/Katakana/Kanji), Thai, Hindi/Devanagari, and emoji. Verifies non-Latin scripts pass through untouched while ASCII content (RM, phone, email, URL, IC, time, %) is still normalized. Tests mixed-script sentences, markdown stripping with multilingual text, and the non-ASCII-attached-to-ASCII edge case (e.g. `价格是RM500` passes through raw vs `价格是 RM500` normalizes). |
| `tests/test_tts_vc_api.py` | 43 | Live API (`TTS_TEST_URL`, default `http://localhost:9091`) | Integration tests for TTS (`POST /v1/audio/speech`) and VC (`POST /v1/audio/vc`) endpoints. **TTS tests** (24): WAV/PCM format validation (sample rate 24000, mono, 16-bit), streaming vs buffered, all speakers (husein/jenny/idayu), speaker list endpoint, markdown/HTML/link sanitization, Malaysian normalization with numbers, temperature/speed/max_tokens parameters, short/long/English/multilingual text. **VC tests** (19): uses `jenny.wav` with reference text, WAV/PCM format validation, streaming vs buffered, Malay/English/long/short generate text, markdown/HTML/link/code sanitization in both reference_text and generate_text, temperature/speed/max_tokens parameters. Auto-skipped when the API is not reachable. |
| `tests/test_normalize_api.py` | 31 | In-process app import (GPU/models) | Integration tests for `POST /v1/audio/normalize` endpoint. Auto-skipped when the app cannot be imported. Tests both `normalize_malaysian=false` (sanitize only) and `normalize_malaysian=true` (full normalization). |

## Stress Test

Run the stress test against the current worker configuration defined in [docker-compose.yaml](docker-compose.yaml):

```bash
docker compose -f stress-test.yaml up --build
```

Uses default parameters from [app/env.py](app/env.py).

### H100 NVL

This one based on H100 NVL in Azure Standard_NC40ads_H100_v5

#### dynamic batching v1, older private commit

```
stress-test  | === SLO LATENCY REPORT ===
stress-test  | Total Requests: 50
stress-test  | Min Time: 1.606s
stress-test  | Max Time: 3.539s
stress-test  | Avg Time: 2.676s
stress-test  | P50 (Median): 2.568s
stress-test  | P90: 3.431s
stress-test  | P95: 3.474s
stress-test  | P99: 3.510s
stress-test  |
stress-test  | === AUDIO & REAL-TIME FACTOR (RTF) REPORT ===
stress-test  | Avg Audio Duration: 5.702s
stress-test  |
stress-test  | --- RTF Percentiles ---
stress-test  | Min RTF: 0.294s
stress-test  | Max RTF: 0.591s
stress-test  | Avg RTF: 0.470
stress-test  | P50 RTF: 0.434
stress-test  | P90 RTF: 0.581
stress-test  | P95 RTF: 0.587
stress-test  | P99 RTF: 0.589
stress-test  | ==========================
```

#### dynamic batching v2

```
stress-test  | === SLO LATENCY REPORT ===
stress-test  | Total Requests: 50
stress-test  | Min Time: 1.840s
stress-test  | Max Time: 2.296s
stress-test  | Avg Time: 2.164s
stress-test  | P50 (Median): 2.198s
stress-test  | P90: 2.286s
stress-test  | P95: 2.290s
stress-test  | P99: 2.295s
stress-test  |
stress-test  | === AUDIO & REAL-TIME FACTOR (RTF) REPORT ===
stress-test  | Avg Audio Duration: 5.573s
stress-test  |
stress-test  | --- RTF Percentiles ---
stress-test  | Min RTF: 0.329s
stress-test  | Max RTF: 0.554s
stress-test  | Avg RTF: 0.392
stress-test  | P50 RTF: 0.379
stress-test  | P90 RTF: 0.427
stress-test  | P95 RTF: 0.503
stress-test  | P99 RTF: 0.529
stress-test  | ==========================
```

### GPU RTX 3090 Ti

#### dynamic batching

Use [.env_dynamicbatching](.env_dynamicbatching).

```
stress-test  | === SLO LATENCY REPORT ===
stress-test  | Total Requests: 50
stress-test  | Min Time: 6.229s
stress-test  | Max Time: 6.545s
stress-test  | Avg Time: 6.429s
stress-test  | P50 (Median): 6.424s
stress-test  | P90: 6.530s
stress-test  | P95: 6.543s
stress-test  | P99: 6.544s
stress-test  |
stress-test  | === AUDIO & REAL-TIME FACTOR (RTF) REPORT ===
stress-test  | Avg Audio Duration: 5.759s
stress-test  |
stress-test  | --- RTF Percentiles ---
stress-test  | Min RTF: 0.966s
stress-test  | Max RTF: 1.702s
stress-test  | Avg RTF: 1.124
stress-test  | P50 RTF: 1.099
stress-test  | P90 RTF: 1.194
stress-test  | P95 RTF: 1.203
stress-test  | P99 RTF: 1.405
stress-test  | ==========================
```

#### dynamic batching with CUDA Graphs

Use [.env_dynamicbatching_cudagraph](.env_dynamicbatching_cudagraph).

```
stress-test  | === SLO LATENCY REPORT ===
stress-test  | Total Requests: 50
stress-test  | Min Time: 6.125s
stress-test  | Max Time: 6.323s
stress-test  | Avg Time: 6.262s
stress-test  | P50 (Median): 6.273s
stress-test  | P90: 6.313s
stress-test  | P95: 6.322s
stress-test  | P99: 6.322s
stress-test  |
stress-test  | === AUDIO & REAL-TIME FACTOR (RTF) REPORT ===
stress-test  | Avg Audio Duration: 5.731s
stress-test  |
stress-test  | --- RTF Percentiles ---
stress-test  | Min RTF: 0.983s
stress-test  | Max RTF: 1.701s
stress-test  | Avg RTF: 1.105
stress-test  | P50 RTF: 1.067
stress-test  | P90 RTF: 1.160
stress-test  | P95 RTF: 1.278
stress-test  | P99 RTF: 1.540
stress-test  | ==========================
```

#### Without dynamic batching

Use [.env_nodynamicbatching](.env_nodynamicbatching).

```
stress-test  | === SLO LATENCY REPORT ===
stress-test  | Total Requests: 50
stress-test  | Min Time: 4.847s
stress-test  | Max Time: 7.734s
stress-test  | Avg Time: 7.136s
stress-test  | P50 (Median): 7.291s
stress-test  | P90: 7.637s
stress-test  | P95: 7.667s
stress-test  | P99: 7.710s
stress-test  |
stress-test  | === AUDIO & REAL-TIME FACTOR (RTF) REPORT ===
stress-test  | Avg Audio Duration: 5.630s
stress-test  |
stress-test  | --- RTF Percentiles ---
stress-test  | Min RTF: 1.228s
stress-test  | Max RTF: 1.282s
stress-test  | Avg RTF: 1.267
stress-test  | P50 RTF: 1.269
stress-test  | P90 RTF: 1.278
stress-test  | P95 RTF: 1.281
stress-test  | P99 RTF: 1.281
stress-test  | ==========================
```
