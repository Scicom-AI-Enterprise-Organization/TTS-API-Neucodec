#!/usr/bin/env python3
"""
Normalizer cost probe: latency of /v1/audio/normalize in rule vs llm mode per sentence,
and whether the LLM actually changed anything (whitespace-insensitive compare). The LLM
normalizer runs BEFORE the LM prompt is built, so its latency is TTFB, one-for-one.

    python bench/normalizer_probe.py --url http://127.0.0.1:9087
"""
import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evalset import EVAL_SET  # noqa: E402

EXTRA = [
    ('num_1', 'Your total is RM1,250.50 and the meeting is at 3pm on 12/9/2026.'),
    ('abbr_1', 'Dr. Lim from Scicom Sdn Bhd will call you re: invoice no. 4471.'),
    ('plain_2', 'Sure, I can help with that. Could you tell me a bit more about the problem?'),
    ('plain_3', 'Baik, saya faham. Boleh encik ceritakan sedikit lagi tentang masalah tersebut?'),
    ('plain_4', 'Of course. Let me check that for you and I will get back to you in a moment.'),
]


def squash(s):
    return re.sub(r'\s+', ' ', s.strip())


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:9091')
    ap.add_argument('--reps', type=int, default=1)
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    items = [(i, t) for i, _v, t in EVAL_SET] + EXTRA
    rows = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as s:
        for _ in range(args.reps):
            for tid, text in items:
                out = {}
                for mode in ('rule', 'llm'):
                    t0 = time.perf_counter()
                    async with s.post(args.url + '/v1/audio/normalize',
                                      json={'input': text, 'mode': mode, 'normalize_malaysian': True}) as r:
                        d = await r.json()
                    dt = time.perf_counter() - t0
                    res = d.get('normalized') if isinstance(d, dict) and 'normalized' in d else json.dumps(d)
                    out[mode] = (dt, res)
                same_llm = squash(out['llm'][1]) == squash(text)
                same_rule = squash(out['rule'][1]) == squash(text)
                rows.append({'id': tid, 'text': text, 'rule_s': out['rule'][0], 'llm_s': out['llm'][0],
                             'rule': out['rule'][1], 'llm': out['llm'][1],
                             'llm_changed': not same_llm, 'rule_changed': not same_rule})
                line = (f"{tid:<10} rule {out['rule'][0] * 1000:5.0f}ms changed={str(not same_rule):<5} | "
                        f"llm {out['llm'][0] * 1000:5.0f}ms changed={str(not same_llm):<5}")
                if not same_llm:
                    line += f"\n           in : {text}\n           llm: {out['llm'][1]}"
                if not same_rule:
                    line += f"\n           rule: {out['rule'][1]}"
                print(line, flush=True)
    llm = [r['llm_s'] for r in rows]
    print(f"\nllm normalize latency: mean {statistics.mean(llm):.3f}s  p50 {statistics.median(llm):.3f}s  "
          f"max {max(llm):.3f}s   | changed text in {sum(r['llm_changed'] for r in rows)}/{len(rows)}")
    if args.out:
        json.dump(rows, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    asyncio.run(main())
