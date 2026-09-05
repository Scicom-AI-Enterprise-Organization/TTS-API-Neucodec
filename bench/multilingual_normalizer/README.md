# Multilingual normalizer dataset (written → spoken)

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

## Current build (2026-09-05)

43,698 rows: 40,000 template rows (2,500 per locale, ~100 templates each) + 3,698 LLM rows (~230 per locale,
20 categories × 12 sentences, 2.5% dropped by the checks). Per-locale counts in `stats.md`. Every row is
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
uv run --with num2words python -m bench.multilingual_normalizer.generate --per-locale 2500
uv run --with num2words --with aiohttp python -m bench.multilingual_normalizer.llm_pairs --per-category 12 --batches 1
uv run --with num2words python -m bench.multilingual_normalizer.build --sft
```

All LLM stages are cached and incremental: raise `--per-locale` / `--batches` and rerun to grow the set.
`--per-locale` on `generate` is free (no LLM). Add a locale by extending `verbalize.py` (number words,
currencies, months, units, `SAFE_SLOTS`) and `locales.py` (formats, seeds).

## Known limits

- Template rows repeat sentence frames; the LLM rows are there for lexical diversity. Do not train on
  template rows alone.
- Sentence-level language only: a row is one language (plus Latin names, acronyms, emails). Malay/English
  code-switching is covered by the Malaysian corpus in `bench/normalizer_corpus.py`, not here.
- Deterministic Sinhala, Filipino, Arabic and Polish output has not been checked by native speakers; the
  `checks`-passing LLM rows are the more natural reference for those languages until it has.
