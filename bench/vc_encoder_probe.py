#!/usr/bin/env python3
"""
/v1/audio/vc through two instances that differ only in the reference encoder (VC_ENCODER):

    python bench/vc_encoder_probe.py --urls neucodec=http://127.0.0.1:9087,s4=http://127.0.0.1:9088 \
        --reps 8 --save-dir /root/ttfb-test/vc_probe [--stt]

Per request, streamed as pcm so the first byte is audio: ttfb (first audio byte), total, seconds of audio. The requests
alternate between the instances, so a busy minute on the shared LM lands on both. With --stt every take is transcribed
(STT_URL, STT_URL_API, STT_MODEL from the environment) and scored against the sentence asked for.

The reference clip is jenny.wav of this repository. A request's encode time is the part of ttfb that differs: the
prompt, the LM and the decoder are the same on both sides.
"""
import argparse
import io
import json
import os
import re
import statistics
import time
import wave

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
REF_TEXT = 'I wonder if I shall ever be happy enough to have real lace on my clothes and bows on my caps.'
TEXTS = [
    ('en', 'Thank you for calling. Your appointment is confirmed for Tuesday morning, and we will send you a reminder one day before.'),
    ('ms', 'Terima kasih kerana menghubungi kami. Temu janji anda telah disahkan pada pagi Selasa.'),
    ('en', 'Please hold for a moment while I check the status of your order.'),
]
SR = 24000


def words(s):
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split()


def wer(ref, hyp):
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)], len(r)


def one(url, ref, text, model):
    t0 = time.perf_counter()
    r = requests.post(f'{url}/v1/audio/vc', files={'reference_audio': ('ref.wav', ref, 'audio/wav')},
                      data={'reference_text': REF_TEXT, 'generate_text': text, 'model': model, 'response_format': 'pcm',
                            'stream': 'true'}, stream=True, timeout=120)
    r.raise_for_status()
    ttfb, pcm = None, bytearray()
    for chunk in r.iter_content(chunk_size=None):
        if chunk and ttfb is None:
            ttfb = time.perf_counter() - t0
        pcm += chunk
    return ttfb, time.perf_counter() - t0, bytes(pcm)


def to_wav(pcm):
    b = io.BytesIO()
    with wave.open(b, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm)
    return b.getvalue()


def transcribe(wav, lang):
    url = os.environ['STT_URL'].rstrip('/')
    url += '/audio/transcriptions' if url.endswith('/v1') else '/v1/audio/transcriptions'
    r = requests.post(url, headers={'Authorization': f"Bearer {os.environ['STT_URL_API']}"},
                      files={'file': ('a.wav', wav, 'audio/wav')},
                      data={'model': os.environ['STT_MODEL'], 'response_format': 'json', 'language': lang}, timeout=120)
    r.raise_for_status()
    return r.json().get('text', '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--urls', required=True, help='name=url,name=url')
    ap.add_argument('--reps', type=int, default=8)
    ap.add_argument('--model', default='TTS-model')
    ap.add_argument('--ref', default=os.path.join(HERE, '..', 'jenny.wav'))
    ap.add_argument('--save-dir', default='')
    ap.add_argument('--stt', action='store_true')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    arms = [x.split('=', 1) for x in a.urls.split(',')]
    ref = open(a.ref, 'rb').read()
    for name, url in arms:                                  # one warm-up each, not counted
        one(url, ref, TEXTS[0][1], a.model)
    rows = []
    for rep in range(a.reps):
        for ti, (lang, text) in enumerate(TEXTS):
            for name, url in (arms if rep % 2 == 0 else arms[::-1]):
                ttfb, total, pcm = one(url, ref, text, a.model)
                row = {'arm': name, 'rep': rep, 'text': ti, 'lang': lang, 'ttfb': ttfb, 'total': total, 'audio_s': len(pcm) / (2 * SR)}
                wav = to_wav(pcm)
                if a.save_dir:
                    os.makedirs(f'{a.save_dir}/{name}', exist_ok=True)
                    open(f'{a.save_dir}/{name}/t{ti}-r{rep}.wav', 'wb').write(wav)
                if a.stt:
                    e, n = wer(text, transcribe(wav, lang))
                    row.update(errors=e, ref_words=n)
                rows.append(row)
    if a.out:
        json.dump(rows, open(a.out, 'w'), indent=1)
    for name, _ in arms:
        rs = [r for r in rows if r['arm'] == name]
        t = sorted(r['ttfb'] for r in rs)
        line = (f"{name:10s} requests {len(rs):3d} | ttfb ms p50 {1000 * statistics.median(t):6.1f} p90 {1000 * t[int(0.9 * (len(t) - 1))]:6.1f} "
                f"min {1000 * t[0]:6.1f} | audio s mean {statistics.mean(r['audio_s'] for r in rs):.2f} | empty {sum(r['audio_s'] < 0.5 for r in rs)}")
        if a.stt:
            line += f" | WER {100 * sum(min(r['errors'], r['ref_words']) for r in rs) / sum(r['ref_words'] for r in rs):.1f}%"
            line += f" | takes with half the words wrong {sum(r['errors'] >= 0.5 * r['ref_words'] for r in rs)}"
        print(line)
    if len(arms) == 2:
        d = sorted(x['ttfb'] - y['ttfb'] for x in rows if x['arm'] == arms[1][0]
                   for y in rows if y['arm'] == arms[0][0] and (y['rep'], y['text']) == (x['rep'], x['text']))
        print(f"ttfb {arms[1][0]} minus {arms[0][0]}, paired: median {1000 * statistics.median(d):+.1f} ms, "
              f"{sum(v < 0 for v in d)} of {len(d)} pairs faster")


if __name__ == '__main__':
    main()
