#!/usr/bin/env python3
"""Generate speech tokens ONCE from the TTS LM, so both codecs decode identical tokens.

Isolating the codec is the whole point: any difference in the scored audio then comes
from the decoder, not from LM sampling noise (temp 0.6 spreads prosody/loudness a lot).
"""
import argparse, json, os, re, sys, time
import urllib.request

TOKEN_RE = re.compile(r'<\|s_(\d+)\|>')


def one(api, key, model, speaker, text, max_tokens, temperature, rep_pen, timeout=300):
    prompt = f'<|im_start|>{speaker}: {text}<|speech_start|>'
    body = json.dumps({
        'model': model, 'prompt': prompt, 'max_tokens': max_tokens,
        'temperature': temperature, 'repetition_penalty': rep_pen, 'stream': False,
    }).encode()
    req = urllib.request.Request(
        api, data=body,
        headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    ch = d['choices'][0]
    return [int(x) for x in TOKEN_RE.findall(ch['text'])], ch.get('finish_reason')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--texts', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--api', default='http://127.0.0.1:9093/v1/completions')
    ap.add_argument('--model', default='TTS-model')
    ap.add_argument('--max-tokens', type=int, default=1200)
    ap.add_argument('--temperature', type=float, default=0.6)
    ap.add_argument('--repetition-penalty', type=float, default=1.15)
    a = ap.parse_args()

    key = os.environ.get('TTS_API_KEY', '')
    rows = [json.loads(l) for l in open(a.texts)]
    t0 = time.time()
    with open(a.out, 'w') as f:
        for i, r in enumerate(rows):
            try:
                ids, fr = one(a.api, key, a.model, r['speaker'], r['text'],
                              a.max_tokens, a.temperature, a.repetition_penalty)
            except Exception as e:
                print(f'[gen] {r["id"]} FAILED: {e}', flush=True)
                continue
            r = dict(r, token_ids=ids, n_tokens=len(ids), finish_reason=fr,
                     audio_s=round(len(ids) / 50.0, 3))
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            f.flush()
            if i % 20 == 0:
                print(f'[gen] {i}/{len(rows)} {r["id"]} n={len(ids)} '
                      f'({r["audio_s"]}s) fr={fr} elapsed={time.time()-t0:.0f}s', flush=True)
    print(f'[gen] done {len(rows)} in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
