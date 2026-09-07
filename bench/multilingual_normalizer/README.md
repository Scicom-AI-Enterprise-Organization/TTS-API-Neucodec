# Multilingual normalizer dataset (written → spoken)

Published: **[Scicom-intl/Multilingual-Normalizer](https://huggingface.co/datasets/Scicom-intl/Multilingual-Normalizer)**
(the standalone copy of this pipeline lives in `synthetic-normalizer/`).

Training pairs for fine-tuning a small LLM as a TTS text normalizer: `text` is a sentence the way
people type it (digits, currency symbols, dates, phone numbers, …), `normalized` is the exact spoken
form in the same language. Sixteen locales:

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

## Code-switching (9,000 rows, `codeswitch.py`)

Malaysian speech is not one language per sentence, and **the reading of a number follows the
fragment it sits in**, not the sentence: `உங்கள் bill RM66, due date 13 March 2027.` is
`உங்கள் bill அறுபத்தாறு ரிங்கிட், due date the thirteenth of March twenty twenty-seven.`

That decision is the label, so these rows are hand-written, not LLM-written: a frame tags the
read-language on every slot (`{money:ms}`, `{date:en}`), the value is filled and formatted in that
language, and the spoken side comes from `app.spoken_normalizer` with the language **forced** per
slot. Each slot is normalized together with the carrier words next to it and they are stripped off
again, because the cue is what fixes the reading (`704251` is a quantity, `nombor rujukan anda
704251` is digit by digit; `9.50` is a decimal, `9.50 மணிக்கு` is a time). Asking the LLM instead
was tried and is not usable: for `bil anda RM250` it answered "bil anda ringgit malaysia dua ratus
lima puluh", the currency before the amount.

1,500 rows per pair, ~24-36 frames each: `ms-en` `en-ms` `zh-en` `zh-ms` `ta-en` `ta-ms`.
Traps that cost a rebuild, all guarded now: a Tamil suffix starting with an independent vowel
(`ஆம்`) **merges into** the number word by sandhi, so a frame that glues it after a slot cannot be
reassembled (`validate()` rejects it); a number glued to a Chinese classifier (`{range}个`) is read
in Chinese, not English; and the generic `{unit}` picks any unit, which turns a data plan into
kilograms (hence `unit_data` / `unit_temp` / `int_small` / `time_plain` in `CS_FILLERS`).

## Two sources, tagged per row

- `source: "template"` — a sentence template (10 hand-written seeds per locale + LLM-written ones,
  `templates_llm.jsonl`) with typed slots (`{money}`, `{date}`, `{phone}`, …) filled with random
  locale-formatted values; the spoken side is produced **deterministically** (`verbalize.py`, or the
  validated `app.spoken_normalizer` for en/ms/zh/ta). Digit-correct by construction; grammar risk only
  where the caveat column says so, because slots that are not safe in a locale (`verbalize.SAFE_SLOTS`)
  are never filled deterministically there.
- `source: "llm"` — natural sentences written by the normalizer LLM (gemma-4-31b via the `OPENAI_*`
  proxy) per category, then normalized by the same LLM with a multilingual prompt and two few-shot
  pairs of the locale. Kept only if **no digit survives**, the output is in the locale's script, ≥80% of
  the non-numeric words are preserved, and the length ratio is sane (`checks` field). Expect a few
  percent residual LLM errors (on the Malaysian corpus the same LLM was wrong on 28/497 hard sentences).
  `category: "plain"` rows are identity pairs (nothing to normalize; output == input).

Arabic rows use Eastern Arabic digits (٠-٩) in ~30% of the written side.

## Current build (2026-09-06)

52,698 rows: 49,000 template rows (2,500 per locale + 1,500 per code-switched pair) + 3,698 LLM rows
(~230 per locale, 20 categories × 12 sentences, 2.5% dropped by the checks). Per-locale counts in `stats.md`. Every row is
digit-free on the normalized side. Scale with the flags below; the LLM stages cost about one HTTP call per
sentence (~25 minutes for this build at 6 concurrent requests).

## Files (`bench/results/multilingual_normalizer/`)

- `train.jsonl`, `val.jsonl`, `test.jsonl` — rows: `id, lang, language, source, text, normalized, split`
  plus `template_id, slots` (template rows) or `category, checks` (LLM rows). Split is by template for
  template rows (no template shared across splits) and by text hash for LLM rows, 90/5/5.
- `*_sft.jsonl` — the same rows as `{"messages": [system, user, assistant]}` (system = the per-locale
  normalizer prompt from `llm_pairs.system_prompt`), ready for chat fine-tuning.
- `stats.md` — counts per locale / source / split and per slot.
- Intermediate caches: `template_pairs.jsonl`, `templates_llm.jsonl`, `llm_sentences.jsonl`, `llm_pairs.jsonl`.

## Rebuild / scale

```bash
set -a; source .env; set +a                                   # OPENAI_* for the LLM stages
uv run --with num2words --with aiohttp python -m bench.multilingual_normalizer.templates_llm --per-locale 60
uv run --with num2words python -m bench.multilingual_normalizer.generate --per-locale 2500 --cs-per-locale 1500
uv run --with num2words --with aiohttp python -m bench.multilingual_normalizer.llm_pairs --per-category 12 --batches 1
uv run --with num2words python -m bench.multilingual_normalizer.build --sft
```

All LLM stages are cached and incremental: raise `--per-locale` / `--batches` and rerun to grow the set.
`--per-locale` on `generate` is free (no LLM). Add a locale by extending `verbalize.py` (number words,
currencies, months, units, `SAFE_SLOTS`) and `locales.py` (formats, seeds).

## Known limits

- Template rows repeat sentence frames; the LLM rows are there for lexical diversity. Do not train on
  template rows alone.
- The code-switched rows are template rows only (~30 hand-written frames per pair): they teach the
  number-reading decision across languages, not open-domain rojak vocabulary.
- Deterministic Sinhala, Filipino, Arabic, Polish and Tamil code-switched output has not been checked by native speakers; the
  `checks`-passing LLM rows are the more natural reference for those languages until it has.
