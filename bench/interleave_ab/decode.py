#!/usr/bin/env python3
"""Decode each condition's speech tokens to 24 kHz audio with the production NeuCodec.

One decode per condition per paragraph: for `interleave` and `cold` the chunks' token
lists are concatenated first and decoded in a SINGLE window, exactly like `single`.
That is deliberate -- the streaming stitcher's growing windows, crossfade and
per-request loudness normalization are already characterized (CLAUDE.md, bench/TTFB.md)
and would otherwise sit on top of the effect being measured. What is left in the audio
is the LM's own prosody: identical decoder, identical window, identical everything but
the token stream.

Also writes `<id>_<cond>.joins.json`: the sample offset of every chunk boundary, which
`score.py` needs to look at the seams.

    CUDA_VISIBLE_DEVICES=6 python bench/interleave_ab/decode.py --tokens ... --out-dir ...
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SR = 24_000
SAMPLES_PER_TOKEN = 480          # 24 kHz / 50 tokens-per-second


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokens', default='bench/results/interleave_ab/tokens.jsonl')
    ap.add_argument('--out-dir', default='bench/results/interleave_ab/wav')
    ap.add_argument('--conds', default='single,interleave,cold')
    a = ap.parse_args()

    from app.neucodec import NeuCodec
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    codec = NeuCodec.from_pretrained('neuphonic/neucodec').eval().to(device)
    assert codec.sample_rate == SR, f'unexpected codec rate {codec.sample_rate}'
    print(f'[decode] neucodec sample_rate={codec.sample_rate} device={device}', flush=True)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(a.tokens)]
    conds = a.conds.split(',')
    t0 = time.time()
    index = []

    for i, r in enumerate(rows):
        for cond in conds:
            parts = [x['tokens'] for x in r[cond]]
            ids = [t for p in parts for t in p]
            if not ids:
                print(f'[decode] {r["id"]}/{cond}: no tokens, skipped', flush=True)
                continue
            with torch.no_grad():
                y = codec.decode_code(torch.tensor(ids, device=device)[None, None])
            wav = y[0, 0].float().cpu().numpy()
            path = out / f'{r["id"]}_{cond}.wav'
            sf.write(path, wav, SR, subtype='FLOAT')
            # cumulative chunk ends, in samples; the last one is the end of the clip
            ends, n = [], 0
            for p in parts:
                n += len(p)
                ends.append(n * SAMPLES_PER_TOKEN)
            index.append({'id': r['id'], 'lang': r['lang'], 'cond': cond,
                          'wav': path.name, 'text': r['text'], 'chunks': r['chunks'],
                          'n_tokens': len(ids), 'dur_s': round(len(wav) / SR, 4),
                          'chunk_ends': ends, 'joins': ends[:-1]})
        if (i + 1) % 20 == 0:
            print(f'[decode] {i+1}/{len(rows)} {time.time()-t0:.0f}s', flush=True)

    (out / 'index.json').write_text(json.dumps(index, ensure_ascii=False))
    print(f'[decode] wrote {len(index)} wavs to {out} in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
