#!/usr/bin/env python3
"""
How close is the rule normalizer to the LLM normalizer? Scores app.spoken_normalizer.normalize
against bench/results/normalizer_truth.jsonl (the live LLM's outputs on bench/normalizer_corpus.py).

Two texts "agree" when they are identical after canonicalization: lower-case, hyphens and
runs of whitespace/punctuation collapsed to one space, trailing period dropped -- the app's
_post_normalize() strips hyphens and pads spaces anyway, and none of that is audible. CER is the
character error rate of the rule output against the LLM output on those canonical forms, so a
sentence that differs in one word still gets partial credit.

    PYTHONPATH=. python bench/normalizer_agreement.py            # summary + every mismatch
    PYTHONPATH=. python bench/normalizer_agreement.py --lang ta  # one language
    PYTHONPATH=. python bench/normalizer_agreement.py --quiet    # summary only
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from app.spoken_normalizer import normalize, detect_lang  # noqa: E402

TRUTH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'normalizer_truth.jsonl')


def canon(s):
    s = (s or '').lower().replace('-', ' ').replace('–', ' ')
    s = re.sub(r'[\s,.;:!?。，！？、]+', ' ', s)
    return s.strip()


def cer(hyp, ref):
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--truth', default=TRUTH)
    ap.add_argument('--lang', default='')
    ap.add_argument('--category', default='')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--show-agree', action='store_true')
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.truth) if l.strip()]
    rows = [r for r in rows if not r['llm'].startswith('<<ERROR')]
    if args.lang:
        rows = [r for r in rows if r['lang'] == args.lang]
    if args.category:
        rows = [r for r in rows if r['category'] == args.category]

    by_lang, by_cat = defaultdict(list), defaultdict(list)
    lang_wrong = 0
    for r in rows:
        exp_lang = 'ms' if r['lang'] == 'cs' else r['lang']
        det = detect_lang(r['text'])
        if r['lang'] != 'cs' and det != exp_lang:
            lang_wrong += 1
        out = normalize(r['text'])
        agree = canon(out) == canon(r['llm'])
        c = cer(canon(out), canon(r['llm']))
        r.update({'rule': out, 'agree': agree, 'cer': c, 'det': det})
        by_lang[r['lang']].append(r)
        by_cat[r['category']].append(r)
        if (not agree and not args.quiet) or args.show_agree:
            flag = 'ok  ' if agree else 'DIFF'
            print(f"{flag} {r['id']:<13} [{det}] in : {r['text']}\n                     llm : {r['llm']}\n                     rule: {out}")

    def line(name, rs):
        n = len(rs)
        a = sum(x['agree'] for x in rs)
        mc = sum(x['cer'] for x in rs) / n
        print(f'{name:<10} n={n:>3}  agree {a:>3}/{n} ({100 * a / n:5.1f}%)  mean CER vs LLM {mc:.3f}')

    print('\n== by language ==')
    for lang in ('en', 'ms', 'cs', 'zh', 'ta'):
        if by_lang.get(lang):
            line(lang, by_lang[lang])
    print('== by category ==')
    for cat, rs in sorted(by_cat.items()):
        line(cat, rs)
    line('ALL', rows)
    print(f'language detection wrong on {lang_wrong} non-code-switch rows')


if __name__ == '__main__':
    main()
