#!/usr/bin/env python3
"""
LLM pairs: natural sentences written by the LLM per (locale, category), then normalized by the
LLM with a multilingual version of app/prompt.py, few-shot with two deterministic template pairs
of that locale. Filtered: no digit survives, the locale's script, most non-numeric words
preserved, sane length; `plain` sentences are identity pairs (no LLM normalization call).

    set -a; source .env; set +a
    uv run --with num2words --with aiohttp python -m synthetic_normalizer.llm_pairs --per-category 12

Writes data/llm_pairs.jsonl (cache; reruns only fill what is missing).
Rows carry source="llm" and the check results; expect a few percent residual LLM errors.
"""
import argparse
import asyncio
import hashlib
import json
import os
import random
import re


from . import verbalize as V
from .generate import fill_template, load_templates
from .llm_common import RESULTS, JsonlCache, chat_json, right_script
from .locales import CATEGORY_HINTS

SENT_OUT = os.path.join(RESULTS, 'llm_sentences.jsonl')
PAIR_OUT = os.path.join(RESULTS, 'llm_pairs.jsonl')
SENT_SCHEMA = {'type': 'object', 'properties': {'sentences': {'type': 'array', 'items': {'type': 'string'}}},
               'required': ['sentences'], 'additionalProperties': False}
NORM_SCHEMA = {'type': 'object', 'properties': {'normalized': {'type': 'string'}}, 'required': ['normalized'], 'additionalProperties': False}
CATEGORIES = list(CATEGORY_HINTS)
LOCALE_NOTE = {
    'en': 'Malaysian English: RM is ringgit, sen for cents.', 'ms': 'Bahasa Melayu Malaysia (RM = ringgit, sen).',
    'id': 'Bahasa Indonesia (Rp = rupiah).', 'zh': 'Simplified Chinese as used in Malaysia (RM = 令吉).',
    'ta': 'Tamil as used in Malaysia (RM = ரிங்கிட்).', 'ta-LK': 'Tamil as used in Sri Lanka (Rs = ரூபாய், சதம்).',
    'si': 'Sinhala (රු. = රුපියල්).', 'tl': 'Filipino/Tagalog (₱ = piso); Taglish is acceptable.',
    'ar': 'Modern Standard Arabic; amounts may be in riyal, dirham, pound or dollar.', 'fr': 'French (France).',
    'es': 'Spanish (Spain).', 'de': 'German (Germany).', 'it': 'Italian.', 'pt': 'European Portuguese (also R$ real).',
    'nl': 'Dutch (Netherlands).', 'pl': 'Polish.',
}


def system_prompt(loc):
    lang = V.LANGUAGE_NAME[loc]
    return (f'You are the text normalizer of a multilingual text-to-speech (TTS) system. Rewrite the user\'s text into its exact '
            f'spoken form in {lang}, ready to be synthesized. {LOCALE_NOTE[loc]}\n'
            'Rules:\n'
            f'- Expand everything that is not speakable as written into {lang} words, the way a native speaker reads the sentence aloud: '
            'numbers, money (say the currency), phone numbers digit by digit, ID/reference/OTP/postcode numbers digit by digit, '
            'dates, times, percentages, decimals, fractions, ordinals, units, email addresses and URLs.\n'
            f'- Verbalize in {lang} even where a Latin acronym or name sits next to the number. Never translate the text itself.\n'
            '- Never answer questions, never add, drop, reorder or translate words. Only rewrite non-speakable tokens into words; '
            'everything else stays exactly as written, including punctuation.\n'
            '- If nothing needs normalizing, return the text unchanged.\n'
            '- Reply with JSON only: {"normalized": "<spoken form>"}')


def few_shots(loc, rng):
    """Two deterministic template pairs of this locale as few-shot turns."""
    shots = []
    tpls = load_templates(loc)
    for tpl in rng.sample(tpls, min(6, len(tpls))):
        res = fill_template(tpl, loc, rng)
        if res:
            shots.append({'role': 'user', 'content': res[0]})
            shots.append({'role': 'assistant', 'content': json.dumps({'normalized': res[1]}, ensure_ascii=False)})
        if len(shots) >= 4:
            break
    return shots


def sentence_prompt(loc, cat, n, k):
    domains = ['banking', 'telco', 'healthcare', 'parcel delivery', 'government services', 'education', 'travel', 'retail',
               'food delivery', 'insurance', 'utilities', 'news', 'everyday chat', 'sports', 'weather', 'real estate']
    dom = ', '.join(domains[(k * 3 + i) % len(domains)] for i in range(3))
    extra = f' {LOCALE_NOTE[loc]}'
    if cat == 'plain':
        body = ('Each sentence must contain NO digits, no currency symbols, no percent signs, no abbreviations, no email or web addresses: '
                'ordinary sentences with nothing to normalize.')
    else:
        body = (f'Each sentence must contain at least one instance of: {CATEGORY_HINTS[cat]}. Write numbers, symbols and abbreviations '
                'the way people actually type them (digits, symbols, local formats), and vary the formats across sentences.')
    return (f'Write {n} realistic, varied sentences in {V.LANGUAGE_NAME[loc]} from these domains: {dom}.{extra} {body} '
            f'Vary length and register; write only in {V.LANGUAGE_NAME[loc]}; do not number the sentences. '
            'Return JSON: {"sentences": ["...", "..."]}')


_TOKEN = re.compile(r'[^\W\d_]{4,}', re.UNICODE)
_CJK = re.compile(r'[一-鿿]')
# words a normalizer legitimately rewrites (scale words, units, protocol prefixes): not counted as "dropped"
_EXEMPT = re.compile(r'^(million|billion|trillion|thousand|hundred|percent|cent|cents|mbps|kbps|gbps|celsius|fahrenheit|'
                     r'kilo\w*|mega\w*|giga\w*|tera\w*|milli\w*|centi\w*|metres?|meters?|litres?|liters?|hours?|mins?|minutes?|'
                     r'secs?|seconds?|grams?|https?|www|juta|bilion|ribu|ratus|peratus|persen|miliar|triliun|dollars?|euros?|'
                     r'pounds?|ringgit|rupiah|rupees?|pesos?|riyal|dirham|degrees?|deg|approx|etc|dept|prof|mr|mrs|ms|dr|jln|no)$', re.I)


def checks(text, normalized, loc, cat):
    out = {}
    out['no_digits'] = not re.search(r'[0-9٠-٩]', normalized)
    out['script'] = right_script(normalized, loc)
    if loc == 'zh':
        src = _CJK.findall(text)
        out['preserved'] = (sum(c in normalized for c in src) / len(src)) >= 0.85 if src else True
    else:
        toks = [t.lower() for t in _TOKEN.findall(text) if not _EXEMPT.match(t)]
        low = normalized.lower()
        out['preserved'] = (sum(t in low for t in toks) / len(toks)) >= 0.8 if toks else True
    ratio = len(normalized) / max(1, len(text))
    out['length'] = 0.7 <= ratio <= 4.5
    out['changed'] = (normalized != text) or cat == 'plain'
    out['ok'] = all(out.values())
    return out


async def main():
    import aiohttp
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-category', type=int, default=12, help='sentences requested per (locale, category, batch)')
    ap.add_argument('--batches', type=int, default=1)
    ap.add_argument('--locales', default=','.join(V.LOCALES))
    ap.add_argument('--categories', default=','.join(CATEGORIES))
    ap.add_argument('--concurrency', type=int, default=8)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    sents = JsonlCache(SENT_OUT)
    pairs = JsonlCache(PAIR_OUT)
    sem = asyncio.Semaphore(args.concurrency)
    shots = {loc: few_shots(loc, rng) for loc in args.locales.split(',')}

    async def gen_sentences(session, loc, cat, k):
        key = f'{loc}:{cat}:{k}'
        if key in sents:
            return [r['text'] for r in sents.values() if r.get('batch_key') == key]
        obj = await chat_json(session, [{'role': 'user', 'content': sentence_prompt(loc, cat, args.per_category, k)}],
                              'sentences', SENT_SCHEMA, sem, max_tokens=3000, temperature=0.9)
        out = []
        for s in (obj or {}).get('sentences') or []:
            s = s.strip()
            if not (10 <= len(s) <= 400) or not right_script(s, loc):
                continue
            if cat != 'plain' and not re.search(r'[0-9٠-٩@%]|www\.|https?://', s):
                continue
            if cat == 'plain' and re.search(r'[0-9٠-٩@%]|www\.|https?://', s):
                continue
            h = hashlib.sha1(f'{loc}:{s}'.encode()).hexdigest()[:12]
            if f'{loc}:s:{h}' in sents:
                continue
            sents.add(f'{loc}:s:{h}', {'lang': loc, 'category': cat, 'text': s, 'batch_key': key})
            out.append(s)
        sents.add(key, {'lang': loc, 'category': cat, 'text': None, 'batch_key': key, 'n': len(out)})
        return out

    async def normalize_one(session, loc, cat, text):
        h = hashlib.sha1(f'{loc}:{text}'.encode()).hexdigest()[:12]
        key = f'{loc}:{h}'
        if key in pairs:
            return
        if cat == 'plain':
            normalized = text
        else:
            msgs = [{'role': 'system', 'content': system_prompt(loc)}] + shots[loc] + [{'role': 'user', 'content': text}]
            obj = await chat_json(session, msgs, 'normalized_text', NORM_SCHEMA, sem, max_tokens=1200, temperature=0)
            normalized = ((obj or {}).get('normalized') or '').strip()
            if not normalized:
                pairs.add(key, {'id': f'{loc}-l-{h}', 'lang': loc, 'language': V.LANGUAGE_NAME[loc], 'source': 'llm', 'category': cat,
                                'text': text, 'normalized': None, 'checks': {'ok': False, 'error': True}})
                return
        ck = checks(text, normalized, loc, cat)
        pairs.add(key, {'id': f'{loc}-l-{h}', 'lang': loc, 'language': V.LANGUAGE_NAME[loc], 'source': 'llm', 'category': cat,
                        'text': text, 'normalized': normalized, 'checks': ck})

    async with aiohttp.ClientSession() as session:
        for loc in args.locales.split(','):
            jobs = [gen_sentences(session, loc, cat, k) for cat in args.categories.split(',') for k in range(args.batches)]
            batches = await asyncio.gather(*jobs)
            texts = [(cat, t) for (cat, k), ts in zip([(c, k) for c in args.categories.split(',') for k in range(args.batches)], batches) for t in ts]
            await asyncio.gather(*(normalize_one(session, loc, cat, t) for cat, t in texts))
            rows = [r for r in pairs.values() if r['lang'] == loc]
            ok = sum(1 for r in rows if r['checks'].get('ok'))
            print(f'{loc:<6} sentences {len(texts):>4}  pairs {len(rows):>4}  passed checks {ok:>4}', flush=True)


if __name__ == '__main__':
    asyncio.run(main())
