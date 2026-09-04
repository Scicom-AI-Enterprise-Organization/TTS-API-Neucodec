#!/usr/bin/env python3
"""Decode one pre-generated token stream through ONE codec arm.

Both arms consume byte-identical `token_ids`, so the wavs differ only by decoder.
Decoding is one-shot (whole utterance in a single window) -- no streaming windows,
so the crossfade stitcher's own envelope/seam artifacts cannot confound the codec
comparison.

arm=neucodec  : repo-vendored NeuCodec, neuphonic/neucodec weights -> 24 kHz  (what prod serves)
arm=widecodec : WideCodec's own bundled package, decoder_depth=20   -> 44.1 kHz
"""
import argparse, json, os, sys, time

import numpy as np
import soundfile as sf
import torch


def load_codec(arm, work, device):
    if arm == 'neucodec':
        sys.path.insert(0, work)                 # nc24/ = copy of repo app/neucodec
        from nc24 import NeuCodec
        m = NeuCodec.from_pretrained('neuphonic/neucodec')
        expect_sr = 24_000
    elif arm == 'widecodec':
        wc = os.path.join(work, 'WideCodec')     # bundled `neucodec/` package lives here
        sys.path.insert(0, wc)
        from neucodec import NeuCodec
        m = NeuCodec._from_pretrained(model_id='Scicom-intl/WideCodec', decoder_depth=20,
                                      token=os.environ.get('HF_TOKEN'))
        expect_sr = 44_100
    else:
        raise SystemExit(f'unknown arm {arm}')

    # The WideCodec loader filters shape-mismatched keys and loads with strict=False, so a
    # wrong-shape checkpoint would load "fine" and emit noise. Verify the decoder really
    # got its weights instead of trusting the silent path.
    assert m.sample_rate == expect_sr, f'{arm}: sample_rate {m.sample_rate} != {expect_sr}'
    n_dec = sum(p.numel() for n, p in m.generator.named_parameters())
    print(f'[{arm}] sample_rate={m.sample_rate} hop={m.hop_length} '
          f'decoder_params={n_dec/1e6:.1f}M', flush=True)
    return m.eval().to(device), m.sample_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['neucodec', 'widecodec'])
    ap.add_argument('--tokens', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--gpu', required=True)
    a = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu     # set before any CUDA init
    device = 'cuda'
    os.makedirs(a.out_dir, exist_ok=True)

    codec, sr = load_codec(a.arm, a.work, device)
    rows = [json.loads(l) for l in open(a.tokens)]

    meta_p = os.path.join(a.out_dir, '_decode_meta.jsonl')
    t0 = time.time()
    with open(meta_p, 'w') as mf:
        for i, r in enumerate(rows):
            ids = r['token_ids']
            if not ids:
                continue
            codes = torch.tensor(ids, dtype=torch.long)[None, None].to(device)
            torch.cuda.synchronize(); t1 = time.time()
            with torch.no_grad():
                y = codec.decode_code(codes)
            torch.cuda.synchronize(); dt = time.time() - t1
            wav = y[0, 0].float().cpu().numpy()
            p = os.path.join(a.out_dir, f'{r["id"]}.wav')
            sf.write(p, wav, sr, subtype='FLOAT')   # float32: no 16-bit quantization
            mf.write(json.dumps({
                'id': r['id'], 'arm': a.arm, 'sr': sr, 'n_tokens': len(ids),
                'wav_s': round(len(wav) / sr, 4), 'decode_s': round(dt, 4),
                'peak': round(float(np.abs(wav).max()), 5),
            }) + '\n')
            if i % 25 == 0:
                print(f'[{a.arm}] {i}/{len(rows)} {r["id"]} '
                      f'{len(wav)/sr:.2f}s decode={dt*1000:.0f}ms', flush=True)
    print(f'[{a.arm}] done {len(rows)} in {time.time()-t0:.0f}s -> {a.out_dir}', flush=True)


if __name__ == '__main__':
    main()
