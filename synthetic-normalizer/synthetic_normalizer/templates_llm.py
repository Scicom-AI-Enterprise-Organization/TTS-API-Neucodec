#!/usr/bin/env python3
"""
Ask the LLM for more sentence templates per locale (typed slots in braces, no digits), so the
template pairs are not limited to the ten hand-written seeds. Validated (only that locale's
safe slots, at least one slot, no digits, right script) and cached in
data/templates_llm.jsonl; generate.py picks them up automatically.

    set -a; source .env; set +a
    uv run --with num2words --with aiohttp python -m synthetic_normalizer.templates_llm --per-locale 60
"""
import argparse
import asyncio
import hashlib
import json
import os
import re


from . import verbalize as V
from .llm_common import RESULTS, JsonlCache, chat_json, right_script
from .locales import CATEGORY_HINTS

OUT = os.path.join(RESULTS, 'templates_llm.jsonl')
SCHEMA = {'type': 'object', 'properties': {'templates': {'type': 'array', 'items': {'type': 'string'}}},
          'required': ['templates'], 'additionalProperties': False}
DOMAINS = ['a bank or e-wallet', 'a telco or internet provider', 'a clinic or hospital', 'a parcel delivery company',
           'a government office or utility', 'a school or university', 'an airline or travel agency', 'an online shop',
           'a restaurant or food delivery app', 'an insurance company', 'a ride-hailing app', 'a hotel']


def prompt(loc, n, domain):
    slots = ', '.join('{' + s + '}' for s in V.SAFE_SLOTS[loc])
    hints = '\n'.join(f'  {{{s}}}: {CATEGORY_HINTS[s]}' for s in V.SAFE_SLOTS[loc])
    extra = ''
    if loc == 'tl':
        extra = ' Write natural Filipino (Taglish is fine).'
    if loc == 'ta-LK':
        extra = ' Use Sri Lankan Tamil vocabulary and context (rupees, Colombo, Jaffna).'
    if loc == 'ar':
        extra = ' Use Modern Standard Arabic.'
    return (f'Write {n} varied, natural sentences in {V.LANGUAGE_NAME[loc]} that {domain} might say or send to a customer '
            f'(voice assistant, SMS, chat, announcement).{extra} Each sentence must contain one to three placeholders from this list, '
            f'written exactly like this: {slots}. A placeholder stands for a value that will be written with digits:\n{hints}\n'
            f'Rules: never write digits or numbers yourself, use the placeholders; vary sentence structure, length and register '
            f'(formal and casual, statements and questions); write only in {V.LANGUAGE_NAME[loc]}, no translations or romanization; '
            f'do not number the sentences. Return JSON: {{"templates": ["...", "..."]}}')


SLOT_RE = re.compile(r'\{([a-z_]+)\}')


_SI_SUFFIX_AFTER_SLOT = re.compile(r'\{[a-z]+\}\s?(?:ක්|ක|කි|කට|කින්|ට|දී|වත්|ෙන්)(?![඀-෿])')


def clean(t):
    return re.sub(r'^[\s,;:،、]+', '', t.strip())


def valid(t, loc):
    t = clean(t)
    slots = SLOT_RE.findall(t)
    if not slots or len(slots) > 4 or set(slots) - set(V.SAFE_SLOTS[loc]):
        return False
    if re.search(r'[0-9٠-٩]', t) or re.search(r'\{[^}]*\}', SLOT_RE.sub('', t)):
        return False
    if not right_script(SLOT_RE.sub('', t), loc):
        return False
    if loc == 'si' and _SI_SUFFIX_AFTER_SLOT.search(t):       # රු. 250ක: the suffix changes the number word
        return False
    return (8 if loc == 'zh' else 20) <= len(t) <= 260


async def main():
    import aiohttp
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-locale', type=int, default=60, help='target number of valid LLM templates per locale')
    ap.add_argument('--batch', type=int, default=15, help='templates requested per call')
    ap.add_argument('--locales', default=','.join(V.LOCALES))
    ap.add_argument('--concurrency', type=int, default=6)
    args = ap.parse_args()
    cache = JsonlCache(OUT)
    have = {loc: sum(1 for r in cache.values() if r['lang'] == loc and r['ok']) for loc in V.LOCALES}
    sem = asyncio.Semaphore(args.concurrency)

    async def fetch(session, loc, k):
        key = f'{loc}:{k}'
        if key in cache:
            return
        obj = await chat_json(session, [{'role': 'user', 'content': prompt(loc, args.batch, DOMAINS[k % len(DOMAINS)])}],
                              'templates', SCHEMA, sem, max_tokens=2500, temperature=0.9)
        tpls = (obj or {}).get('templates') or []
        n_ok = 0
        for t in tpls:
            t = clean(t)
            ok = valid(t, loc)
            n_ok += ok
            cache.add(f'{loc}:{hashlib.sha1(t.encode()).hexdigest()[:12]}', {'lang': loc, 'template': t, 'ok': ok, 'batch': k})
        cache.add(key, {'lang': loc, 'template': None, 'ok': False, 'batch': k, 'n_ok': n_ok})
        print(f'  {loc:<6} batch {k:>2}: {n_ok}/{len(tpls)} valid', flush=True)

    async with aiohttp.ClientSession() as session:
        for loc in args.locales.split(','):
            need = args.per_locale - have[loc]
            if need <= 0:
                print(f'{loc}: {have[loc]} cached, skip'); continue
            calls = -(-need // max(1, int(args.batch * 0.7)))
            done = sum(1 for r in cache.values() if r['lang'] == loc and r['template'] is None)
            await asyncio.gather(*(fetch(session, loc, k) for k in range(done, done + calls)))
    total = {loc: sum(1 for r in cache.values() if r['lang'] == loc and r['ok']) for loc in V.LOCALES}
    print('valid LLM templates per locale:', total)


if __name__ == '__main__':
    asyncio.run(main())
