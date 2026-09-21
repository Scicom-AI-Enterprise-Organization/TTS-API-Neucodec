"""Stress the served TTS for the thing the demo session reported: a voice that is calm
and even and then, part-way through, turns loud and excited.

No recording of that demo exists, so this reproduces it from the description instead of
from the artefact. "Mid-utterance" is the load-bearing word, because it separates three
different mechanisms that all sound like one bug:

M1  A step at a CHUNK JOIN. LiveKit's StreamAdapter cuts a reply into several
    `/v1/audio/speech` calls; each one starts the LM cold, so it picks a fresh register
    and a fresh energy (measured cold: +0.53 st up at 56% of joins, bench/INTERLEAVE_AB.md).
    Only audible mid-sentence if the splitter cuts mid-sentence -- hence `--chunk-words`.
M2  The stitcher's own loudness gain, INSIDE one request. STREAM_NORMALIZE estimates the
    gain from the audio it has emitted so far and slews it GAIN_SLEW_DB per window until
    it locks at ~1 s of voiced audio (app/main.py normalize_chunk). For a long utterance
    that is a brief settle at the start; for a short chunk the gain can still be moving
    over most of it. A swell, not a step -- and it is the API's, not LiveKit's.
M3  The LM itself. Sampling at temperature 0.6-0.8 from an *Expressive* checkpoint can
    simply change register part-way through an utterance. Nothing downstream can fix it.

The arms separate them: `oneshot_raw` has no gain and no joins, so anything it shows is
M3; `oneshot_norm` minus `oneshot_raw` is M2; `chunked_norm` minus `oneshot_norm` is M1;
`chunked_interleave` says how much of M1 interleaving takes back.

Audio is written as one wav per utterance with the exact join offsets recorded, so the
scorer never has to infer a boundary. Score it with bench/pitch_stress_score.py.

    uv run --with aiohttp python bench/pitch_stress.py \
      --url https://tts-api-... --texts bench/pitch_stress_texts.txt \
      --concurrency 1,4,8 --reps 2 --out bench/results/pitch_stress
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
import wave

SR = 24_000


def chunk_words(text: str, n: int) -> list[str]:
    """Split into ~n-word pieces, the way a streaming agent hands text to a
    non-streaming TTS. n<=0 means "do not split"."""
    if n <= 0:
        return [text]
    words = text.split()
    if len(words) <= n:
        return [text]
    return [' '.join(words[i:i + n]) for i in range(0, len(words), n)]


def chunk_sentences(text: str) -> list[str]:
    """Sentence-ish split, closer to what LiveKit's basic SentenceTokenizer does."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in (s.strip() for s in parts) if p] or [text]


ARMS = {
    # name:               (split, stream_normalize, interleave, carry)
    'oneshot_raw':        ('none',  False, False, False),
    'oneshot_norm':       ('none',  True,  False, False),
    'chunked_norm':       ('words', True,  False, False),
    'chunked_raw':        ('words', False, False, False),
    'chunked_interleave': ('words', True,  True,  False),
    # the loudness estimate carried across the chunks of one reply (STREAM_NORMALIZE_CARRY)
    'chunked_carry':      ('words', True,  True,  True),
}


async def say(session, url, text, voice, stream_normalize, interleave_id, timeout, carry=False):
    body = {
        'input': text,
        'voice': voice,
        'response_format': 'pcm',
        'stream': True,
        'stream_normalize': stream_normalize,
        'stream_normalize_carry': carry,
    }
    if interleave_id:
        body['interleave_id'] = interleave_id
    t0 = time.perf_counter()
    ttfb = None
    buf = bytearray()
    async with session.post(f'{url}/v1/audio/speech', json=body, timeout=timeout) as r:
        r.raise_for_status()
        async for block in r.content.iter_chunked(8192):
            if block:
                if ttfb is None:
                    ttfb = time.perf_counter() - t0
                buf.extend(block)
    return bytes(buf), (ttfb if ttfb is not None else float('nan')), time.perf_counter() - t0


async def one_utterance(session, args, arm, text_id, text, rep, conc, outdir):
    split, norm, inter, carry = ARMS[arm]
    if split == 'none':
        pieces = [text]
    elif split == 'sentences':
        pieces = chunk_sentences(text)
    else:
        pieces = chunk_words(text, args.chunk_words)

    key = f'{arm}_t{text_id:02d}_r{rep}_c{conc}'
    iid = f'stress-{key}' if inter else None

    pcm = bytearray()
    joins, parts = [], []
    for i, piece in enumerate(pieces):
        try:
            b, ttfb, wall = await say(session, args.url, piece, args.voice, norm, iid,
                                      args.timeout, carry)
        except Exception as e:
            return {'arm': arm, 'text_id': text_id, 'rep': rep, 'concurrency': conc,
                    'error': f'{type(e).__name__}: {e}', 'piece': i}
        if i:                                   # a join sits before every piece but the first
            joins.append(len(pcm) // 2)
        pcm.extend(b)
        parts.append({'text': piece, 'samples': len(b) // 2,
                      'ttfb_s': round(ttfb, 4), 'wall_s': round(wall, 4)})

    wav = f'{key}.wav'
    with wave.open(os.path.join(outdir, wav), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(bytes(pcm))
    return {'arm': arm, 'text_id': text_id, 'rep': rep, 'concurrency': conc,
            'text': text, 'wav': wav, 'sr': SR, 'n_pieces': len(pieces),
            'joins': joins, 'parts': parts, 'samples': len(pcm) // 2,
            'duration_s': round(len(pcm) / 2 / SR, 3),
            'stream_normalize': norm, 'interleave': inter, 'carry': carry}


async def main_async(args):
    import aiohttp

    texts = [l.strip() for l in open(args.texts, encoding='utf-8') if l.strip()]
    if args.limit:
        texts = texts[:args.limit]
    arms = [a.strip() for a in args.arms.split(',') if a.strip()]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f'unknown arm {a!r}; pick from {", ".join(ARMS)}')
    concs = [int(c) for c in args.concurrency.split(',')]

    os.makedirs(args.out, exist_ok=True)
    recs_path = os.path.join(args.out, 'records.jsonl')
    done = set()
    if os.path.exists(recs_path) and not args.fresh:     # resume
        with open(recs_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r['arm'], r['text_id'], r['rep'], r['concurrency']))
                except Exception:
                    pass
        print(f'resuming: {len(done)} rows already done')

    jobs = [(arm, ti, rep, c)
            for c in concs for rep in range(args.reps)
            for arm in arms for ti in range(len(texts))
            if (arm, ti, rep, c) not in done]
    print(f'{len(jobs)} utterances to render '
          f'({len(arms)} arms x {len(texts)} texts x {args.reps} reps x {len(concs)} conc)')

    conn = aiohttp.TCPConnector(limit=max(concs) * 2 or 8)
    out = open(recs_path, 'a')
    async with aiohttp.ClientSession(connector=conn) as session:
        # Group by concurrency so each block really runs at that level, and utterances
        # inside a block overlap the way callers do.
        for c in concs:
            block = [j for j in jobs if j[3] == c]
            if not block:
                continue
            print(f'--- concurrency {c}: {len(block)} utterances')
            sem = asyncio.Semaphore(c)

            async def run(job):
                arm, ti, rep, cc = job
                async with sem:
                    return await one_utterance(session, args, arm, ti, texts[ti], rep, cc, args.out)

            t0 = time.time()
            for i in range(0, len(block), max(1, c * 4)):
                batch = block[i:i + max(1, c * 4)]
                for r in await asyncio.gather(*(run(j) for j in batch)):
                    out.write(json.dumps(r) + '\n')
                out.flush()
                print(f'    {min(i + len(batch), len(block))}/{len(block)}'
                      f'  {time.time() - t0:.0f}s', flush=True)
    out.close()

    with open(recs_path) as f:
        rows = [json.loads(l) for l in f]
    bad = [r for r in rows if r.get('error')]
    print(f'\n{len(rows)} rows, {len(bad)} errors -> {recs_path}')
    for r in bad[:5]:
        print('  ERROR', r['arm'], r['text_id'], r['error'])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--url', required=True, help='TTS API base, e.g. https://host (no /v1)')
    p.add_argument('--texts', required=True)
    p.add_argument('--voice', default='TM_English_Normal')
    p.add_argument('--arms', default=','.join(ARMS))
    p.add_argument('--chunk-words', type=int, default=5,
                   help='words per piece in the chunked arms (LiveKit-ish); 0 = no split')
    p.add_argument('--concurrency', default='1,4,8')
    p.add_argument('--reps', type=int, default=2)
    p.add_argument('--limit', type=int, default=0, help='first N texts only')
    p.add_argument('--timeout', type=float, default=180.0)
    p.add_argument('--out', default='bench/results/pitch_stress')
    p.add_argument('--fresh', action='store_true', help='ignore existing records.jsonl')
    args = p.parse_args()
    args.url = args.url.rstrip('/')
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
