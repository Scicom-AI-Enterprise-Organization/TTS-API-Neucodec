#!/usr/bin/env python3
"""Encode REAL audio to FSQ codes with the production 24 kHz NeuCodec encoder.

The FSQ codebook and encoder are frozen and SHARED by both codecs (WideCodec is a
decoder-only finetune), so encoding once and decoding twice is the same trick used for
the TTS arm -- one code stream, two decoders.

Purpose: decide whether WideCodec is simply worse, or merely mismatched to this TTS LM.
On codes from real audio, the model card says WideCodec should WIN. If it wins here and
loses on LM-sampled tokens, the TTS gap is a pairing/distribution effect, not codec quality.
"""
import argparse, glob, json, os, sys
import librosa
import numpy as np
import soundfile as sf
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    sys.path.insert(0, a.work)
    from nc24 import NeuCodec
    m = NeuCodec.from_pretrained('neuphonic/neucodec').eval().to('cuda')
    print(f'[enc] sample_rate={m.sample_rate} hop={m.hop_length}', flush=True)

    files = sorted(glob.glob(os.path.join(a.ref_dir, '*.wav')))
    if a.limit:
        files = files[:a.limit]
    print(f'[enc] {len(files)} refs', flush=True)

    with open(a.out, 'w') as f:
        for i, p in enumerate(files):
            try:
                x, sr = sf.read(p, dtype='float32', always_2d=False)
                if x.ndim > 1:
                    x = x.mean(axis=1)
                if sr != 16_000:
                    x = librosa.resample(x, orig_sr=sr, target_sr=16_000)
                y = torch.from_numpy(np.ascontiguousarray(x)).float().view(1, 1, -1)
                with torch.no_grad():
                    # Keep y on CPU: encode_code feeds it to the HF feature extractor
                    # (numpy-only) and moves things to the device itself.
                    codes = m.encode_code(y)   # [1,1,T] @16k; path form needs TorchCodec
                ids = codes[0, 0].cpu().tolist()
            except Exception as e:
                print(f'[enc] FAILED {p}: {e}', flush=True)
                continue
            f.write(json.dumps({
                'id': f'r{i:04d}', 'ref': os.path.basename(p),
                'token_ids': ids, 'n_tokens': len(ids), 'speaker': 'real',
            }) + '\n')
            if i % 25 == 0:
                print(f'[enc] {i}/{len(files)} n={len(ids)}', flush=True)
    print('[enc] done', flush=True)


if __name__ == '__main__':
    main()
