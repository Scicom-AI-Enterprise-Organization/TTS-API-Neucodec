"""A/B listen for the `request_id` speech context (SPEECH_CONTEXT.md).

Emulates what a chunking agent (LiveKit StreamAdapter) does to one reply: the text is cut
into short chunks and each chunk is a separate /v1/audio/speech request, played back to
back. Three conditions per (text, voice):

  A   nocontext  each chunk generated cold (today's behaviour)
  B1  context    each chunk with the same request_id, context_mode=turns -> the previous
                 chunks as closed VC-style turns, the new chunk as a new turn
  B2  continue   same request_id, context_mode=continue -> one turn, all the text, the
                 previous chunks' tokens as prefix: the LM resumes mid-utterance
  C   oneshot    the whole text in a single request (reference: what the model does when
                 nobody chunks the text)

Writes per-chunk wavs, the back-to-back concatenation per condition, and a results.json
with durations and the X-Context-Turns/Tokens the server reported per chunk (the turns
count climbing across chunks that hit different uvicorn workers is the cross-worker
store working).

    python bench/context_ab.py --url http://127.0.0.1:9097 --out audio/context_ab --takes 2
"""

import argparse
import io
import json
import os
import time
import uuid
import wave

import requests

CASES = [
    {
        'name': 'en_support',
        'voice': 'husein',
        'chunks': [
            "Hello, my name is Husein,",
            "and I'll be helping you today.",
            "I see that you're calling about your recent order,",
            "which was placed last Tuesday.",
            "Let me pull up the details for you right now.",
            "It looks like the package left our warehouse yesterday,",
            "so it should arrive within two to three working days.",
            "Is there anything else I can help you with?",
        ],
    },
    {
        'name': 'ms_support',
        'voice': 'idayu',
        'chunks': [
            "Selamat pagi, nama saya Idayu,",
            "dan saya akan membantu anda hari ini.",
            "Saya lihat anda menghubungi kami",
            "tentang pesanan anda minggu lepas.",
            "Sila tunggu sebentar sementara saya semak butirannya.",
            "Bungkusan anda telah keluar dari gudang semalam,",
            "jadi ia sepatutnya sampai dalam masa dua hingga tiga hari bekerja.",
            "Ada apa-apa lagi yang boleh saya bantu?",
        ],
    },
    {
        'name': 'en_normal_voice',
        'voice': 'TM_English_Normal',
        'chunks': [
            "Thank you for waiting.",
            "I have checked your account,",
            "and the refund was approved this morning.",
            "You should see the amount back on your card",
            "within five to seven working days.",
            "I'm sorry again for the inconvenience.",
        ],
    },
    {
        'name': 'chicken_rice',
        'voice': 'husein',
        'chunks': [
            "hello my name is husein,",
            "i like to eat chicken rice.",
        ],
    },
]

SR = 24000


def speak(url, text, voice, request_id=None, **fields):
    body = {
        'input': text, 'voice': voice, 'response_format': 'wav', 'stream': False,
        # rule-based normalizer: deterministic and fast, so both conditions get the
        # exact same text and the A/B is about prosody only
        'mode': 'rule',
    }
    if request_id:
        body['request_id'] = request_id
    body.update(fields)
    t0 = time.perf_counter()
    r = requests.post(f'{url}/v1/audio/speech', json=body, timeout=300)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    with wave.open(io.BytesIO(r.content)) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2
        pcm = w.readframes(w.getnframes())
    return pcm, {
        'text': text,
        'latency_s': round(dt, 3),
        'audio_s': round(len(pcm) / 2 / SR, 3),
        'context_mode': r.headers.get('X-Context-Mode'),
        'context_turns': r.headers.get('X-Context-Turns'),
        'context_tokens': r.headers.get('X-Context-Tokens'),
    }


def write_wav(path, pcm):
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)


def run_case(url, out, case, take):
    name = f"{case['name']}_take{take}"
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    res = {'case': case['name'], 'voice': case['voice'], 'take': take, 'conditions': {}}

    for cond in ('nocontext', 'context', 'continue'):
        rid = f"ab-{case['name']}-{take}-{cond}-{uuid.uuid4().hex[:8]}" if cond != 'nocontext' else None
        fields = {'context_mode': {'context': 'turns', 'continue': 'continue'}[cond]} if cond != 'nocontext' else {}
        parts, infos = [], []
        for i, text in enumerate(case['chunks']):
            pcm, info = speak(url, text, case['voice'], request_id=rid, **fields)
            write_wav(os.path.join(d, f'{cond}_chunk{i}.wav'), pcm)
            parts.append(pcm)
            infos.append(info)
            print(f"  {name} {cond:9s} chunk{i}: {info['audio_s']:5.2f}s  "
                  f"turns={info['context_turns']} tokens={info['context_tokens']}  {text}")
        write_wav(os.path.join(d, f'{cond}.wav'), b''.join(parts))
        res['conditions'][cond] = {
            'request_id': rid,
            'audio_s': round(sum(len(p) for p in parts) / 2 / SR, 3),
            'chunks': infos,
        }
        if rid:
            requests.delete(f'{url}/v1/audio/context/{rid}', timeout=30)

    pcm, info = speak(url, ' '.join(case['chunks']), case['voice'])
    write_wav(os.path.join(d, 'oneshot.wav'), pcm)
    res['conditions']['oneshot'] = {'audio_s': info['audio_s'], 'chunks': [info]}
    print(f"  {name} oneshot          : {info['audio_s']:5.2f}s")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:9097')
    ap.add_argument('--out', default='audio/context_ab')
    ap.add_argument('--takes', type=int, default=1)
    ap.add_argument('--cases', default='', help='comma-separated case names (default all)')
    args = ap.parse_args()

    want = set(filter(None, args.cases.split(',')))
    cases = [c for c in CASES if not want or c['name'] in want]
    os.makedirs(args.out, exist_ok=True)
    results = []
    for case in cases:
        for take in range(1, args.takes + 1):
            results.append(run_case(args.url, args.out, case, take))
    with open(os.path.join(args.out, 'results.json'), 'w') as f:
        json.dump({'url': args.url, 'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
                   'results': results}, f, indent=2, ensure_ascii=False)
    print(f'wrote {args.out}/results.json')


if __name__ == '__main__':
    main()
