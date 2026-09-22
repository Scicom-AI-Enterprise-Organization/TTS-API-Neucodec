# Tests

How to run them, and what each file covers.

## Running tests

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

## What each file covers

**623 passed, 35 skipped** on a full run with a live API and `OPENAI_*` configured.

| File | Tests | Dependencies | Description |
|---|---|---|---|
| `tests/test_sanitize_markdown.py` | 66 | None (only `app/rules.py`) | Markdown and HTML stripping: bold, italic, headings, links, images, code blocks, blockquotes, lists, horizontal rules, HTML tags. Edge cases for IC numbers, phone numbers, URLs, underscore variables, unicode. |
| `tests/test_normalizer.py` | 150 | `app/normalizer`, `app/rules` | Text normalization with exact input/output checks in both Malay and English: email (`husein.zol05@gmail.com` -> `HUSEIN dot ZOL kosong lima di GMAIL dot COM`), URL, phone, IC number, money (RM/USD), time, percentage, units (kg, km, celsius, liter, MB), dates, cardinals, ordinals, fractions, multipliers, hingga/range, contractions, alpha-num splitting, replace mappings, and combined markdown+normalizer pipeline. |
| `tests/test_malaysian_rules.py` | 208 | `app/normalizer`, `app/rules` | Stress tests for Malaysian normalization rules. Exhaustive coverage of: money (RM whole/sen/zero/sentence, USD with K/M suffixes), IC numbers (standard/young/zeros/multiple), phone numbers (mobile 012/011, landline 03, multiple), email (basic/subdomain/sentence/multiple), URL (https/www/path/IP), time (AM/PM/midnight/morning/late night), percentages (decimal/100/small), units (celsius/kg/g/km/liter/ml/mb/gb), dates, zero-prefix numbers, passports, year normalization (tahun 2024/1999/2000/1945), pada hari bulan, ordinals (ke-1/ke-100/Roman), cardinals, fractions, multiplier (x kali), hingga, Hijri year, elongated words, tak prefix, all 51 pronunciation replacements (dr/mr/mrs/Sdn Bhd/LRT/MRT/KL/PDRM/CCTV/UMNO/5G/US), pattern ranges (100-200 ringgit), all contractions, alpha-num splitting, replace mappings, and 12 complex multi-type sentence tests simulating the full pipeline. |
| `tests/test_multilingual.py` | 76 | `app/normalizer`, `app/rules` | Multilingual passthrough tests for Chinese (Simplified/Traditional), Korean, Tamil, Arabic, Japanese (Hiragana/Katakana/Kanji), Thai, Hindi/Devanagari, and emoji. Verifies non-Latin scripts pass through untouched while ASCII content (RM, phone, email, URL, IC, time, %) is still normalized. Tests mixed-script sentences, markdown stripping with multilingual text, and the non-ASCII-attached-to-ASCII edge case (e.g. `价格是RM500` passes through raw vs `价格是 RM500` normalizes). |
| `tests/test_tts_vc_api.py` | 50 | Live API (`TTS_TEST_URL`, default `http://localhost:9091`) | Integration tests for TTS (`POST /v1/audio/speech`) and VC (`POST /v1/audio/vc`) endpoints. **TTS tests** (29): WAV/PCM format validation (sample rate 24000, mono, 16-bit), streaming vs buffered, all speakers (husein/jenny/idayu), speaker list endpoint, markdown/HTML/link sanitization, Malaysian normalization with numbers, temperature/speed/max_tokens parameters, `speaking_rate` (valid wav, SSE streaming, 2.0 shorter than 0.5, `speed` alias, 422 out of range), short/long/English/multilingual text. **VC tests** (21): uses `jenny.wav` with reference text, WAV/PCM format validation, streaming vs buffered, Malay/English/long/short generate text, markdown/HTML/link/code sanitization in both reference_text and generate_text, temperature/speed/max_tokens/speaking_rate parameters. Auto-skipped when the API is not reachable. |
| `tests/test_normalize_api.py` | 35 | In-process app import (GPU/models) | Integration tests for `POST /v1/audio/normalize` endpoint. Auto-skipped when the app cannot be imported. Tests `normalize_malaysian=false` (sanitize only), `normalize_malaysian=true` (full normalization), and the `mode` enum (`rule` default, `llm`, invalid → 422). |
| `tests/test_tracing.py` | 12 | `pytest` only (no GPU/torch; 8 need `opentelemetry-sdk`) | Unit tests for the hot-path spans (`app/tracing.py`): disabled by default, disabled helpers are no-ops returning a single shared `nullcontext`, and — with the OTel SDK installed — spans nest, an explicitly passed parent beats the ambient context (what the batching threads rely on), `record_span` honours the given timestamps and drops a stage with no start time, attributes are cleaned, exceptions mark the span. |
| `tests/test_timestretch.py` | 33 | `numpy` + `pytest` only (no GPU/torch) | Unit tests for the speaking-rate stretcher (`app/timestretch.py`): rate 1.0 is an exact identity, duration scales by the rate (0.5–2.0) to within one block, a steady tone keeps its frequency (pitch preserved), joins are continuous, output is bit-identical regardless of how the input is chunked (streaming), first chunk emitted promptly, empty/sub-block/silence/full-scale edge cases, PCM16 helpers. |
| `tests/test_llm_normalizer.py` | 28 | `aiohttp` + `pytest` only (no GPU/torch) | Unit tests for the LLM-based normalizer (`app/llm_normalizer.py`): few-shot message building from `app/prompt.py`, strict JSON schema, reply parsing (clean/fenced/bare/plain-text/unusable), and full HTTP behaviour against a local fake OpenAI server (auth header, payload shape, 400 retry without `response_format`, 5xx/unreachable/unconfigured errors). 4 live tests hit the real `OPENAI_BASE_URL` (Malay money, English IC, Chinese money, passthrough) and are skipped unless `OPENAI_*` is set. |
