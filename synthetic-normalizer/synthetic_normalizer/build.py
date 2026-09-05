#!/usr/bin/env python3
"""
Merge template and LLM pairs into the release files: dedupe, split by template (so no template
seen in training appears in val/test) and by text hash for LLM rows, write
data/{train,val,test}.jsonl and stats.md. With --sft also writes
*_sft.jsonl in chat-messages form for fine-tuning.

    uv run --with num2words python -m synthetic_normalizer.build --sft
"""
import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict

from . import verbalize as V
from .llm_common import RESULTS
from .llm_pairs import system_prompt, checks

TEMPLATE_PAIRS = os.path.join(RESULTS, 'template_pairs.jsonl')
LLM_PAIRS = os.path.join(RESULTS, 'llm_pairs.jsonl')


def bucket(key):
    h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) % 100
    return 'train' if h < 90 else 'val' if h < 95 else 'test'


def template_splits(rows):
    """Per locale, order the template ids by hash and take every 20th as test and every 20th (offset 10)
    as val: an even 90/5/5 over templates even with only ~100 templates per locale."""
    out = {}
    per_loc = defaultdict(set)
    for r in rows:
        per_loc[r['lang']].add(r['template_id'])
    for loc, tids in per_loc.items():
        ordered = sorted(tids, key=lambda t: hashlib.sha1((loc + t).encode()).hexdigest())
        for i, t in enumerate(ordered):
            out[(loc, t)] = 'test' if i % 20 == 0 else 'val' if i % 20 == 10 else 'train'
    return out


_LEAD_JUNK = re.compile(r'^[\s,;:،、]+')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sft', action='store_true')
    ap.add_argument('--include-failed-llm', action='store_true', help='keep LLM rows that failed a check (flagged)')
    args = ap.parse_args()
    rows = []
    if os.path.exists(TEMPLATE_PAIRS):
        trows = [json.loads(line) for line in open(TEMPLATE_PAIRS) if line.strip()]
        splits = template_splits(trows)
        for r in trows:
            r['split'] = splits[(r['lang'], r['template_id'])]
            rows.append(r)
    if os.path.exists(LLM_PAIRS):
        for line in open(LLM_PAIRS):
            if line.strip():
                r = json.loads(line)
                if r.get('text') is None or r.get('normalized') is None:
                    continue
                r['text'], r['normalized'] = _LEAD_JUNK.sub('', r['text']), _LEAD_JUNK.sub('', r['normalized'])
                r['checks'] = checks(r['text'], r['normalized'], r['lang'], r['category'])   # current rules, no LLM call
                if not r['checks']['ok'] and not args.include_failed_llm:
                    continue
                r.pop('_key', None)
                r['split'] = bucket(r['lang'] + ':' + r['text'])
                rows.append(r)
    seen, out = set(), []
    for r in rows:
        k = (r['lang'], r['text'])
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    by_split = defaultdict(list)
    for r in out:
        by_split[r['split']].append(r)
    for split, rs in by_split.items():
        with open(os.path.join(RESULTS, f'{split}.jsonl'), 'w') as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        if args.sft:
            with open(os.path.join(RESULTS, f'{split}_sft.jsonl'), 'w') as f:
                for r in rs:
                    f.write(json.dumps({'messages': [{'role': 'system', 'content': system_prompt(r['lang'])},
                                                     {'role': 'user', 'content': r['text']},
                                                     {'role': 'assistant', 'content': r['normalized']}],
                                        'lang': r['lang'], 'source': r['source']}, ensure_ascii=False) + '\n')
    # stats
    c = Counter((r['lang'], r['source'], r['split']) for r in out)
    langs = [l for l in V.LOCALES if any(k[0] == l for k in c)]
    lines = ['# Multilingual normalizer dataset: stats', '', f'Total rows: {len(out)}', '',
             '| lang | language | template train/val/test | llm train/val/test | total |', '|---|---|---|---|---|']
    for l in langs:
        t = [c[(l, 'template', s)] for s in ('train', 'val', 'test')]
        m = [c[(l, 'llm', s)] for s in ('train', 'val', 'test')]
        lines.append(f'| {l} | {V.LANGUAGE_NAME[l]} | {t[0]}/{t[1]}/{t[2]} | {m[0]}/{m[1]}/{m[2]} | {sum(t) + sum(m)} |')
    cat = Counter()
    for r in out:
        for s in (r.get('slots') or [r.get('category')]):
            cat[s] += 1
    lines += ['', 'Slot / category counts (a template row counts once per slot):', '',
              ', '.join(f'{k} {v}' for k, v in cat.most_common())]
    open(os.path.join(RESULTS, 'stats.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
