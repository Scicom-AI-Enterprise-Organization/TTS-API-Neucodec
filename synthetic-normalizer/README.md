---
language:
- en
- ms
- id
- zh
- ta
- si
- tl
- ar
- fr
- es
- de
- it
- pt
- nl
- pl
task_categories:
- text-generation
tags:
- text-normalization
- tts
- inverse-text-normalization
- code-switching
- malaysia
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: val.jsonl
  - split: test
    path: test.jsonl
- config_name: sft
  data_files:
  - split: train
    path: train_sft.jsonl
  - split: validation
    path: val_sft.jsonl
  - split: test
    path: test_sft.jsonl
---

# Multilingual TTS text normalizer (written → spoken)

Training pairs for fine-tuning a small LLM as a **text-to-speech normalizer**: `text` is a sentence
the way people type it (digits, currency symbols, dates, phone numbers, …) and `normalized` is the
exact spoken form, in the same language, with nothing left that a TTS model cannot say.

**52,698 rows** — 16 monolingual locales and **6 Malaysian code-switched pairs**. Every row is
digit-free on the spoken side.

```python
from datasets import load_dataset

ds = load_dataset("Scicom-intl/Multilingual-Normalizer")               # text / normalized
sft = load_dataset("Scicom-intl/Multilingual-Normalizer", "sft")       # chat messages, ready to train
```

```json
{"id": "ms-en-t-000015", "lang": "ms-en", "language": "Malay-English code-switching (Malaysia)",
 "source": "template", "template_id": "7cc461db78", "slots": ["money:ms", "date:ms"],
 "text": "Encik, bil bulan ini RM 66.50 dan due date pada 25-12-2022.",
 "normalized": "Encik, bil bulan ini enam puluh enam ringgit lima puluh sen dan due date pada dua puluh lima Disember dua ribu dua puluh dua.",
 "split": "train"}
```

## Languages

| lang | language | number words from | grammar caveat |
|---|---|---|---|
| en | English (Malaysian context, RM/USD) | `app.spoken_normalizer` | — |
| ms | Malay | `app.spoken_normalizer` | — |
| id | Indonesian | num2words | — |
| zh | Mandarin (Malaysian context) | `app.spoken_normalizer` | — |
| ta | Tamil (Malaysia, RM) | `app.spoken_normalizer` | — |
| ta-LK | Tamil (Sri Lanka, Rs/சதம்) | `app.spoken_normalizer` | — |
| si | Sinhala | own tables (`verbalize.py`) | **needs native review**: -යි and case suffixes; thousands 11–19 and ≥100,000 left to LLM rows |
| tl | Filipino | own tables | **needs native review**: linker (-ng/na) applied heuristically; Spanish-derived time/date words only in LLM rows |
| ar | Arabic (MSA) | num2words + own counted-noun forms | **needs native review**: gender agreement of bare counts not modelled; dates/times/units only in LLM rows |
| fr, es, de, it, pt, nl | French, Spanish, German, Italian, Portuguese, Dutch | num2words + locale conventions | dates in the running-text form (`am fünfzehnten März`, `le quinze mars`); bare counts avoid 1 (un/une) |
| pl | Polish | num2words | **needs native review**: only int/money/percent/decimal/phone/codes deterministic; dates, ordinals, units, years only in LLM rows |
| **ms-en, en-ms, zh-en, zh-ms, ta-en, ta-ms** | **Malaysian code-switching** | `app.spoken_normalizer`, one language **per number** | **ta-en / ta-ms need native review** |

Arabic rows use Eastern Arabic digits (٠-٩) in ~30% of the written side.

## Code-switching (9,000 rows)

Malaysian speech is not one language per sentence. A sentence carries a matrix language and drops
words, phrases and often the number itself into another, and **the reading of the digits follows the
fragment they sit in**, not the sentence:

```
Encik, bil bulan ini RM250.50 dan due date pada 12/3/2024.
        ↓ Malay clause              ↓ Malay clause
"… dua ratus lima puluh ringgit lima puluh sen … dua belas Mac dua ribu dua puluh empat."

உங்கள் bill RM66, due date 13 March 2027.
        ↓ Tamil clause    ↓ English clause
"உங்கள் bill அறுபத்தாறு ரிங்கிட், due date the thirteenth of March twenty twenty-seven."
```

That decision is the label. So these rows are **not** LLM-written: a frame is hand-written with the
read-language tagged on every slot (`{money:ms}`, `{date:en}`), the value is filled and formatted in
that language, and the spoken form comes from the rule verbalizer with the language **forced** —
digit-correct and language-correct by construction. (The normalizer LLM was tried first and is not a
usable teacher here: asked for the spoken form of `bil anda RM250` it answered *"bil anda ringgit
malaysia dua ratus lima puluh"* — the currency before the amount, which no Malay speaker says.)

Each slot is read together with the carrier words next to it, because the cue is what fixes the
reading — `704251` alone is a quantity, `nombor rujukan anda 704251` is read digit by digit; `9.50`
alone is a decimal, `9.50 மணிக்கு` is a time.

| pair | matrix + embedded | rows |
|---|---|---|
| `ms-en` | Malay with English (Bahasa rojak) | 1,500 |
| `en-ms` | Malaysian English with Malay | 1,500 |
| `zh-en` | Mandarin with English | 1,500 |
| `zh-ms` | Mandarin with Malay | 1,500 |
| `ta-en` | Tamil with English | 1,500 |
| `ta-ms` | Tamil with Malay | 1,500 |

## The two sources, tagged per row

- `source: "template"` (49,000) — a sentence frame with typed slots (`{money}`, `{date}`, `{phone}`,
  …) filled with random locale-formatted values; the spoken side is produced **deterministically**.
  Monolingual frames are 10 hand-written seeds per locale plus LLM-written ones; code-switched frames
  are all hand-written. Digit-correct by construction; grammar risk only where the caveat column says
  so, because slots that are not safe in a locale are never filled deterministically there.
- `source: "llm"` (3,698) — natural sentences written by an LLM (gemma-4-31b) per category, then
  normalized by the same LLM with a per-locale prompt and two deterministic few-shot pairs. Kept only
  if **no digit survives**, the output is in the locale's script, ≥80% of the non-numeric words are
  preserved, and the length ratio is sane (the `checks` field records this). Expect a few percent
  residual LLM errors. `category: "plain"` rows are identity pairs (nothing to normalize).

Splits are 90/5/5, **by template** for template rows (no frame is shared between train and val/test)
and by text hash for LLM rows.

## Files

- `train.jsonl`, `val.jsonl`, `test.jsonl` — `id, lang, language, source, text, normalized, split`
  plus `template_id, slots` (template rows) or `category, checks` (LLM rows).
- `*_sft.jsonl` — the same rows as `{"messages": [system, user, assistant]}`, ready for chat
  fine-tuning. The system message is the normalizer prompt for that locale; for a code-switched pair
  it says the sentence is mixed and that each number is read in the language of the words around it.
- `raw/` — the intermediate caches the release was built from: `template_pairs.jsonl` (all filled
  frames), `templates_llm.jsonl` (LLM-written monolingual frames), `llm_sentences.jsonl` and
  `llm_pairs.jsonl` (the LLM rows with their check results, including the ones that were dropped).

## Known limits

- Template rows repeat sentence frames; the LLM rows are there for lexical diversity. Do not train on
  template rows alone.
- The code-switched rows are all template rows: ~30 hand-written frames per pair. They teach the
  *number-reading decision* across languages, not open-domain rojak vocabulary.
- Deterministic Sinhala, Filipino, Arabic, Polish and the Tamil code-switched output has not been
  checked by native speakers.
- Malaysian-context bias throughout: RM amounts, Malaysian phone and IC formats, local service
  domains (telco, e-wallet, clinic, parcel, ride-hailing).

## Provenance

Generated with the `synthetic-normalizer` pipeline of the Scicom TTS API repo
(`synthetic_normalizer.{templates_llm,generate,llm_pairs,build}`); the deterministic verbalizer
for English, Malay, Mandarin and Tamil is that repo's rule normalizer (`app.spoken_normalizer`).
Rebuild or scale:

```bash
set -a; source .env; set +a                                   # OPENAI_* for the LLM stages only
uv run --with num2words --with aiohttp python -m synthetic_normalizer.templates_llm --per-locale 60
uv run --with num2words python -m synthetic_normalizer.generate --per-locale 2500 --cs-per-locale 1500
uv run --with num2words --with aiohttp python -m synthetic_normalizer.llm_pairs --per-category 12
uv run --with num2words python -m synthetic_normalizer.build --sft
```

`generate` is free (no LLM) and the LLM stages are cached and incremental. Adding a code-switched
pair means adding frames to `codeswitch.py`; adding a locale means extending `verbalize.py` (number
words, currencies, months, units, safe slots) and `locales.py` (formats, seeds).
