#!/usr/bin/env python3
"""
Ground truth for the rule normalizer: run bench/normalizer_corpus.py through the live LLM
normalizer (app.llm_normalizer.llm_normalize, temperature 0) and store the pairs.

Output: bench/results/normalizer_truth.jsonl, one {"id","lang","category","text","llm"} per
line. Incremental: ids already present are skipped, so extending the corpus only costs the new
sentences. --redo <ids> re-queries specific ids; --repeat N stores N samples per new id
(as "llm_samples") to see how stable the LLM is on that input.

    set -a; source .env; set +a
    uv run --with aiohttp python bench/normalizer_truth.py
"""
import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from normalizer_corpus import CORPUS  # noqa: E402
from app.llm_normalizer import llm_normalize, LLMNormalizerError  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'normalizer_truth.jsonl')


def load(path):
    rows = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows[r['id']] = r
    return rows


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--redo', default='', help='comma-separated ids to re-query')
    ap.add_argument('--repeat', type=int, default=1)
    ap.add_argument('--concurrency', type=int, default=4)
    args = ap.parse_args()

    have = load(args.out)
    redo = set(filter(None, args.redo.split(',')))
    todo = [c for c in CORPUS if c[0] not in have or c[0] in redo]
    print(f'{len(have)} cached, {len(todo)} to query')
    sem = asyncio.Semaphore(args.concurrency)

    async def one(c):
        cid, lang, cat, text = c
        async with sem:
            samples = []
            for _ in range(args.repeat):
                try:
                    samples.append(await llm_normalize(text))
                except LLMNormalizerError as e:
                    samples.append(f'<<ERROR {e}>>')
        row = {'id': cid, 'lang': lang, 'category': cat, 'text': text, 'llm': samples[0]}
        if args.repeat > 1:
            row['llm_samples'] = samples
        flag = '' if samples[0] != text else '  (unchanged)'
        print(f'{cid:<14} {samples[0]}{flag}', flush=True)
        return row

    results = await asyncio.gather(*(one(c) for c in todo))
    for r in results:
        have[r['id']] = r
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    order = {c[0]: i for i, c in enumerate(CORPUS)}
    with open(args.out, 'w') as f:
        for cid in sorted(have, key=lambda k: order.get(k, 10 ** 6)):
            f.write(json.dumps(have[cid], ensure_ascii=False) + '\n')
    print(f'wrote {len(have)} rows to {args.out}')


if __name__ == '__main__':
    asyncio.run(main())
