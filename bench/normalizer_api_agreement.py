#!/usr/bin/env python3
"""
End-to-end check of mode="spoken" against mode="llm" through a running app: every corpus
sentence goes through /v1/audio/normalize twice and the outputs are compared after the app's
own pre/post cleanup (markdown sanitizing, replace mappings, trailing period). This is the
agreement a TTS request actually sees, and it also times both modes.

    python bench/normalizer_api_agreement.py --url http://127.0.0.1:9088
"""
import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalizer_corpus import CORPUS  # noqa: E402


def canon(s):
    s = (s or '').lower().replace('-', ' ')
    return re.sub(r'[\s,.;:!?。，！？、]+', ' ', s).strip()


async def normalize(session, url, text, mode):
    t0 = time.perf_counter()
    async with session.post(url + '/v1/audio/normalize', json={'input': text, 'mode': mode}) as r:
        d = await r.json()
    return (d.get('output') if isinstance(d, dict) else json.dumps(d)), time.perf_counter() - t0


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:9091')
    ap.add_argument('--out', default='')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    rows, by_lang = [], defaultdict(list)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as s:
        for cid, lang, cat, text in CORPUS:
            spoken, t_s = await normalize(s, args.url, text, 'spoken')
            llm, t_l = await normalize(s, args.url, text, 'llm')
            agree = canon(spoken) == canon(llm)
            row = {'id': cid, 'lang': lang, 'category': cat, 'text': text, 'spoken': spoken, 'llm': llm,
                   'agree': agree, 't_spoken': t_s, 't_llm': t_l}
            rows.append(row)
            by_lang[lang].append(row)
            if not agree and not args.quiet:
                print(f'DIFF {cid:<13} in    : {text}\n                  llm   : {llm}\n                  spoken: {spoken}')

    print('\n== agreement through the API (spoken vs llm) ==')
    for lang in ('en', 'ms', 'cs', 'zh', 'ta'):
        rs = by_lang.get(lang, [])
        if rs:
            a = sum(r['agree'] for r in rs)
            print(f'{lang:<4} n={len(rs):>3}  agree {a:>3}/{len(rs)} ({100 * a / len(rs):5.1f}%)')
    a = sum(r['agree'] for r in rows)
    print(f'ALL  n={len(rows):>3}  agree {a:>3}/{len(rows)} ({100 * a / len(rows):5.1f}%)')
    print(f"latency: spoken mean {statistics.mean(r['t_spoken'] for r in rows) * 1000:.1f} ms, "
          f"llm mean {statistics.mean(r['t_llm'] for r in rows) * 1000:.0f} ms "
          f"(p90 {sorted(r['t_llm'] for r in rows)[int(0.9 * len(rows))] * 1000:.0f} ms)")
    if args.out:
        json.dump(rows, open(args.out, 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    asyncio.run(main())
