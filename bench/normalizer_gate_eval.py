#!/usr/bin/env python3
"""
Does the LLM-skip gate change any output? End-to-end check of `needs_normalization`.

Runs every corpus sentence through /v1/audio/normalize (mode=llm) on two instances:
  --old-url  an instance WITHOUT the gate (LLM_NORMALIZER_SKIP_PLAIN=false, or old code)
  --new-url  an instance WITH it
and reports, per sentence, whether the gate skipped the LLM, whether both outputs are
identical, and the latency of each. The pass criterion is simple: identical output for
every sentence. A sentence the gate skipped whose outputs differ is a gate MISS (the real
LLM would have rewritten it) -- the corpus below is built to hunt for those, so add any new
suspicious pattern here first. `changed~` is only an approximation of "the LLM changed
something" (old output vs raw input; both paths also apply the app's pre/post replace
mappings, e.g. '!' -> ',', so it over-reports) and is printed for the LLM-called rows only.

Result on tm-h20, 2026-09-04 (gemma-4-31b): identical 51/51, skipped 42/51, 545 ms -> 1 ms.

    PYTHONPATH=. python bench/normalizer_gate_eval.py --old-url http://127.0.0.1:9087 --new-url http://127.0.0.1:9088
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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from evalset import EVAL_SET  # noqa: E402
from app.llm_normalizer import needs_normalization  # noqa: E402

# Plain-looking sentences that might still tempt the LLM into a rewrite, and clear
# normalization cases, so both gate directions are exercised.
EXTRA = [
    ('plain_2', 'Sure, I can help with that. Could you tell me a bit more about the problem?'),
    ('plain_3', 'Baik, saya faham. Boleh encik ceritakan sedikit lagi tentang masalah tersebut?'),
    ('plain_4', 'Of course. Let me check that for you and I will get back to you in a moment.'),
    ('plain_5', "Don't worry, we'll sort it out. It's no problem at all."),
    ('plain_6', 'Well... I am not sure about that, to be honest.'),
    ('plain_7', 'She said “thank you” and left quietly.'),
    ('plain_8', 'Encik Ahmad akan datang esok pagi bersama Puan Siti.'),
    ('plain_9', "I'll send it to you via WhatsApp later tonight."),
    ('plain_10', 'Your order has been shipped and should arrive next Tuesday.'),
    ('plain_11', 'It costs around two hundred ringgit, more or less.'),
    ('plain_12', 'Meet me at half past three in the afternoon, okay?'),
    ('plain_13', 'The first, second and third items are ready for pickup.'),
    ('plain_14', 'Selamat Hari Raya Aidilfitri, maaf zahir dan batin!'),
    ('plain_15', 'The Wi-Fi at the hotel is free for all guests.'),
    ('plain_16', 'Tan Sri will join the meeting later, together with Datuk Ali.'),
    ('plain_17', 'Boleh saya dapatkan nama penuh encik untuk semakan?'),
    ('plain_18', 'Kami akan hubungi encik semula dalam masa dua hari bekerja.'),
    ('plain_19', 'Please visit our website for more details about the promotion.'),
    ('plain_20', 'Okay lah, nanti saya check dan inform you balik ya.'),
    ('plain_21', 'Twenty-four seven support is available for premium members.'),
    ('plain_22', 'Our co-workers in Kuala Lumpur and Johor Bahru will assist you.'),
    ('plain_23', 'Terima kasih, jumpa lagi!'),
    ('zh_plain_1', '你好，请问有什么可以帮您？'),
    ('zh_plain_2', '总共是一千二百五十令吉，谢谢。'),
    ('zh_plain_3', '我们会在两个工作日内回复您。'),
    ('brand_1', 'You can pay with Touch n Go eWallet or the MySejahtera app.'),
    ('num_1', 'Your total is RM1,250.50 and the meeting is at 3pm on 12/9/2026.'),
    ('num_2', 'Call me at 012-3456789 before 10:45 AM.'),
    ('abbr_1', 'Dr. Lim from Scicom Sdn Bhd will call you re: invoice no. 4471.'),
    ('abbr_2', 'Mr Tan and Mrs Lee live on Jln Ampang.'),
    ('acro_1', 'Please bring your IC and enter the OTP sent by SMS.'),
    ('sym_1', 'Discount up to 25% for Q&A participants.'),
    ('url_1', 'Email us at support@scicom.com.my or visit www.scicom.com.my.'),
    ('zh_num_1', '会议在2024年3月15日下午2点30分。'),
    ('md_1', '**Important:** your _account_ is ready.'),
]


def squash(s):
    return re.sub(r'\s+', ' ', (s or '').strip()).rstrip('.').strip()


async def normalize(session, url, text):
    t0 = time.perf_counter()
    async with session.post(url + '/v1/audio/normalize',
                            json={'input': text, 'mode': 'llm', 'normalize_malaysian': True}) as r:
        d = await r.json()
    return d.get('output') if isinstance(d, dict) else json.dumps(d), time.perf_counter() - t0


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old-url', required=True, help='instance without the gate')
    ap.add_argument('--new-url', required=True, help='instance with the gate')
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    corpus = [(i, t) for i, _v, t in EVAL_SET] + EXTRA
    rows = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as s:
        for tid, text in corpus:
            old, t_old = await normalize(s, args.old_url, text)
            new, t_new = await normalize(s, args.new_url, text)
            gate = needs_normalization(text)
            llm_changed = squash(old) != squash(text)
            row = {'id': tid, 'text': text, 'gate_calls_llm': gate, 'identical': old == new,
                   'llm_changed': llm_changed, 't_old': t_old, 't_new': t_new, 'old': old, 'new': new}
            rows.append(row)
            flag = 'MISS' if (not gate and old != new) else ('WASTE~' if (gate and not llm_changed) else 'ok')
            print(f"{tid:<11} gate={'llm ' if gate else 'skip'} identical={str(old == new):<5} "
                  f"changed~={(str(llm_changed) if gate else '-'):<5} old {t_old * 1000:5.0f}ms new {t_new * 1000:5.0f}ms  {flag}")
            if old != new:
                print(f"            old: {old}\n            new: {new}")

    n = len(rows)
    ident = sum(r['identical'] for r in rows)
    skipped = [r for r in rows if not r['gate_calls_llm']]
    misses = [r for r in skipped if not r['identical']]
    waste = [r for r in rows if r['gate_calls_llm'] and not r['llm_changed']]
    print(f"\nidentical output old vs new: {ident}/{n}")
    print(f"gate skipped the LLM on {len(skipped)}/{n} sentences; misses (skipped, outputs differ): {len(misses)}; "
          f"wasted calls (LLM called, nothing changed~): {len(waste)}")
    if skipped:
        print(f"latency on skipped sentences: old(LLM) mean {statistics.mean(r['t_old'] for r in skipped) * 1000:.0f}ms "
              f"-> new mean {statistics.mean(r['t_new'] for r in skipped) * 1000:.0f}ms")
    for r in misses:
        print(f"  MISS {r['id']}: {r['text']}\n       llm: {r['old']}")
    if args.out:
        json.dump(rows, open(args.out, 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    asyncio.run(main())
