#!/usr/bin/env python3
"""
Which language detector should the spoken normalizer trust? Compares, on bench/normalizer_corpus.py
plus 40 extra code-switched sentences with a designed majority language:

  markers     app.spoken_normalizer.lang.detect_lang (script, then Malay/English marker words)
  lid176      fastText lid.176.ftz (1 MB, 176 languages; id -> ms, zh/yue/wuu -> zh)
  mesolitica  mesolitica/fasttext-language-detection-bahasa-en (331 MB; labels bahasa/english/other)
  llm         the normalizer's LLM asked for the dominant language (cached in results/langid_llm.jsonl)

Ground truth: the corpus label for monolingual rows; the designed majority for the 40 extra
code-switched rows; the LLM's majority label for the 22 corpus code-switched rows (they were
written without one).

    set -a; source .env; set +a
    uv run --with fasttext-wheel --with "numpy<2" --with huggingface_hub --with aiohttp \
        python bench/langid_compare.py --llm
"""
import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..'))
from normalizer_corpus import CORPUS  # noqa: E402
from app.spoken_normalizer.lang import detect_lang  # noqa: E402

LID176_URL = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz'
LLM_CACHE = os.path.join(HERE, 'results', 'langid_llm.jsonl')

# (id, designed majority, text). Malay-majority with English insertions, then the reverse.
EXTRA_CS = [
    ('csx_1', 'ms', 'Encik, saya dah check, payment RM120 tu belum masuk lagi dalam system kami.'),
    ('csx_2', 'ms', 'Boleh tak awak forward email tu kepada saya sebelum pukul 5 petang?'),
    ('csx_3', 'ms', 'Kalau nak cancel booking, sila call kami sekurang-kurangnya 24 jam awal.'),
    ('csx_4', 'ms', 'Saya akan update status order anda dalam masa 2 hari bekerja, okay?'),
    ('csx_5', 'ms', 'Nanti technician kami akan datang antara pukul 10 pagi hingga 12 tengah hari.'),
    ('csx_6', 'ms', 'Untuk claim insurance, encik perlu submit borang ini bersama resit asal.'),
    ('csx_7', 'ms', 'Sistem kami tengah maintenance, cuba lagi dalam 30 minit ya.'),
    ('csx_8', 'ms', 'Puan boleh reschedule appointment tu melalui apps kami bila-bila masa.'),
    ('csx_9', 'ms', 'Jangan risau, refund akan masuk ke akaun puan dalam 5 hingga 7 hari.'),
    ('csx_10', 'ms', 'Saya minta maaf, agent kami semua busy sekarang, boleh tunggu sekejap?'),
    ('csx_11', 'ms', 'Kalau ada sebarang issue lain, encik boleh WhatsApp kami di 011-2345 6789.'),
    ('csx_12', 'ms', 'Baki anda sekarang RM250, dan next bill akan keluar pada 1 Oktober.'),
    ('csx_13', 'ms', 'Untuk verify identity, saya perlukan 4 digit terakhir nombor IC encik.'),
    ('csx_14', 'ms', 'Delivery pesanan anda dijangka sampai esok sebelum pukul 6 petang.'),
    ('csx_15', 'ms', 'Pakej internet ni include 50GB data dan unlimited call, harga RM68 sebulan.'),
    ('csx_16', 'ms', 'Maaf ya, slot pada 15 Mac dah penuh, yang available pukul 3 petang 16 Mac.'),
    ('csx_17', 'ms', 'Saya akan escalate case ni kepada supervisor saya dan follow up dengan encik esok.'),
    ('csx_18', 'ms', 'Kad debit puan akan expire pada 12/2026, kad baru akan dipos secara automatik.'),
    ('csx_19', 'ms', 'Discount 20% ni valid untuk pelajar sahaja, perlu tunjuk kad matrik.'),
    ('csx_20', 'ms', 'Tolong confirm alamat penghantaran: No. 12, Jalan Tun Razak, poskod 50450.'),
    ('csx_21', 'en', 'Okay boss, your parcel is out for delivery, sampai around 3pm today.'),
    ('csx_22', 'en', 'Sorry ya, the system is down since 10.30am, we are working on it.'),
    ('csx_23', 'en', 'Your balance is RM85.50 lah, you can settle it before 15/3/2024.'),
    ('csx_24', 'en', 'Please hold on sekejap, I will check the status of order 4471.'),
    ('csx_25', 'en', 'The technician will come tomorrow between 10am and 12pm, boleh?'),
    ('csx_26', 'en', 'Your appointment with Dr. Lim is confirmed for 5 June at 2.30pm, jangan lupa ya.'),
    ('csx_27', 'en', 'I have updated your address to No. 12, Jalan Ampang, thank you encik.'),
    ('csx_28', 'en', 'The refund of RM250 will reach your account within 5-7 working days, okay?'),
    ('csx_29', 'en', 'Can you send the invoice to support@scicom.com.my by 5pm? Terima kasih.'),
    ('csx_30', 'en', "Your OTP is 482913, valid for 10 minutes, please don't share it with anyone ya."),
    ('csx_31', 'en', 'We open 9am-6pm on weekdays, and until 1pm on Saturday, tutup on Sunday.'),
    ('csx_32', 'en', 'The package includes 50GB data and unlimited calls for RM68 per month, murah kan?'),
    ('csx_33', 'en', 'Your car plate WXY 1234 is registered under Encik Ahmad since 2019.'),
    ('csx_34', 'en', 'Press 1 for English, tekan 2 untuk Bahasa Melayu, or hold for an operator.'),
    ('csx_35', 'en', "I'm sorry, the slot on 15 March is full, the next available one is 16 March at 3pm, boleh?"),
    ('csx_36', 'en', 'Your ticket number is 305, estimated waiting time 15 minutes, sila tunggu.'),
    ('csx_37', 'en', 'Please bring your IC and the original receipt to counter 3, okay encik?'),
    ('csx_38', 'en', "The 30-day trial ends on 15/10, after that it's RM29.90 per month lah."),
    ('csx_39', 'en', 'We received your payment of RM1,250.50 on 12/9/2026, terima kasih.'),
    ('csx_40', 'en', 'Your flight MH370 departs at 0730 from gate A12, selamat jalan!'),
]

LANG_WORD = {'en': 'english', 'ms': 'malay', 'zh': 'mandarin', 'ta': 'tamil'}
WORD_LANG = {v: k for k, v in LANG_WORD.items()}


def rows():
    out = [{'id': c[0], 'set': c[1], 'truth': None if c[1] == 'cs' else c[1], 'text': c[3]} for c in CORPUS]
    out += [{'id': i, 'set': 'csx', 'truth': lang, 'text': t} for i, lang, t in EXTRA_CS]
    return out


# --------------------------------------------------------------------------- detectors
def load_fasttext():
    import fasttext
    from huggingface_hub import hf_hub_download
    lid_path = os.path.join(HERE, 'results', 'lid.176.ftz')
    if not os.path.exists(lid_path):
        import urllib.request
        urllib.request.urlretrieve(LID176_URL, lid_path)
    lid = fasttext.load_model(lid_path)
    meso = fasttext.load_model(hf_hub_download('mesolitica/fasttext-language-detection-bahasa-en', 'fasttext.ftz'))
    return lid, meso


def det_lid176(model, text):
    labels, probs = model.predict(text.replace('\n', ' '), k=1)
    lab = labels[0].replace('__label__', '')
    return {'id': 'ms', 'ms': 'ms', 'en': 'en', 'zh': 'zh', 'yue': 'zh', 'wuu': 'zh', 'ta': 'ta'}.get(lab, 'other:' + lab), float(probs[0])


def det_meso(model, text):
    labels, probs = model.predict(text.replace('\n', ' '), k=1)
    lab = labels[0].replace('__label__', '')
    return {'bahasa': 'ms', 'english': 'en'}.get(lab, 'other'), float(probs[0])


async def llm_labels(texts, concurrency=6):
    import aiohttp
    from app.env import OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL_NAME
    cache = {}
    if os.path.exists(LLM_CACHE):
        for line in open(LLM_CACHE):
            if line.strip():
                r = json.loads(line)
                cache[r['text']] = r['llm']
    todo = [t for t in texts if t not in cache]
    print(f'llm: {len(cache)} cached, {len(todo)} to query', flush=True)
    sem = asyncio.Semaphore(concurrency)
    prompt = ('What is the dominant language of this text? Answer with exactly one word from: '
              'english, malay, mandarin, tamil.\n\nText: ')

    async def one(session, text):
        body = {'model': OPENAI_MODEL_NAME, 'temperature': 0, 'max_tokens': 5,
                'messages': [{'role': 'user', 'content': prompt + text}]}
        async with sem:
            async with session.post(f'{OPENAI_BASE_URL.rstrip("/")}/chat/completions', json=body,
                                    headers={'Authorization': f'Bearer {OPENAI_API_KEY}'}) as r:
                data = await r.json()
        ans = data['choices'][0]['message']['content'].strip().lower().strip('.').split()[0] if data.get('choices') else 'error'
        return text, WORD_LANG.get(ans, 'other:' + ans)

    if todo:
        async with aiohttp.ClientSession() as session:
            for text, lab in await asyncio.gather(*(one(session, t) for t in todo)):
                cache[text] = lab
        with open(LLM_CACHE, 'w') as f:
            for t, lab in cache.items():
                f.write(json.dumps({'text': t, 'llm': lab}, ensure_ascii=False) + '\n')
    return cache


# --------------------------------------------------------------------------- report
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--llm', action='store_true', help='also run (and cache) the LLM as a detector')
    ap.add_argument('--errors', action='store_true', help='print every wrong prediction')
    args = ap.parse_args()

    data = rows()
    lid, meso = load_fasttext()
    llm = {}
    if args.llm or os.path.exists(LLM_CACHE):
        llm = asyncio.run(llm_labels([r['text'] for r in data])) if args.llm else \
            {json.loads(l)['text']: json.loads(l)['llm'] for l in open(LLM_CACHE) if l.strip()}
    for r in data:
        if r['truth'] is None:                       # corpus code-switch rows: the LLM's majority is the truth
            r['truth'] = llm.get(r['text'], 'cs?')

    detectors = {'markers': lambda t: (detect_lang(t), None),
                 'lid176': lambda t: det_lid176(lid, t),
                 'mesolitica': lambda t: det_meso(meso, t)}
    if llm:
        detectors['llm'] = lambda t: (llm.get(t, '?'), None)

    lat = {}
    for name, fn in detectors.items():
        if name == 'llm':
            continue
        t0 = time.perf_counter()
        for _ in range(5):
            for r in data:
                fn(r['text'])
        lat[name] = (time.perf_counter() - t0) / (5 * len(data)) * 1e6
    for name, fn in detectors.items():
        for r in data:
            r[name] = fn(r['text'])[0]

    sets = [('en', 'English'), ('ms', 'Malay'), ('zh', 'Mandarin'), ('ta', 'Tamil'),
            ('cs', 'code-switch (corpus, LLM majority)'), ('csx', 'code-switch (designed majority)')]
    print(f"\n{'set':<40}" + ''.join(f'{n:>14}' for n in detectors))
    for key, label in sets:
        rs = [r for r in data if r['set'] == key and not r['truth'].startswith('cs?')]
        line = f'{label + f" (n={len(rs)})":<40}'
        for name in detectors:
            ok = sum(r[name] == r['truth'] for r in rs)
            line += f'{ok:>5}/{len(rs):<3} {100 * ok / len(rs):4.0f}%'
        print(line)
    rs = [r for r in data if not r['truth'].startswith('cs?')]
    line = f'{"ALL (n=%d)" % len(rs):<40}'
    for name in detectors:
        ok = sum(r[name] == r['truth'] for r in rs)
        line += f'{ok:>5}/{len(rs):<3} {100 * ok / len(rs):4.0f}%'
    print(line)
    print('\nlatency us/sentence: ' + ', '.join(f'{k} {v:.0f}' for k, v in lat.items()) + ' (llm: one HTTP round trip, ~500 ms)')

    print('\nwhat the wrong predictions were (predicted:count), per detector and set:')
    for name in detectors:
        for key, label in sets:
            c = Counter(r[name] for r in data if r['set'] == key and r[name] != r['truth'] and not r['truth'].startswith('cs?'))
            if c:
                print(f'  {name:<11} {key:<4} ' + ', '.join(f'{k}:{v}' for k, v in c.most_common()))
    if llm:
        agree = [(r['id'], r['truth'], r['llm'], r['text']) for r in data if r['set'] == 'csx' and r['llm'] != r['truth']]
        print(f'\nLLM vs designed majority on the 40 extra code-switched rows: {40 - len(agree)}/40 agree')
        for a in agree:
            print('   ', *a[:3], a[3][:70])
    if args.errors:
        print('\nerrors:')
        for name in detectors:
            for r in data:
                if r[name] != r['truth'] and not r['truth'].startswith('cs?'):
                    print(f'  {name:<11} {r["id"]:<13} truth={r["truth"]:<3} got={r[name]:<10} {r["text"][:70]}')


if __name__ == '__main__':
    main()
