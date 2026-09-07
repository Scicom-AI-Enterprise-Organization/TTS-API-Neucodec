"""Synthetic long-form Malay/English texts for the interleave A/B (`bench/INTERLEAVE_AB.md`).

The A/B needs paragraphs that a streaming agent would *chunk*: long enough that one
`say()` becomes several TTS requests, with real commas and full stops so the chunk
boundaries fall where LiveKit's `StreamAdapter` would put them. So each text is a
40-90 word paragraph on an everyday topic, written by the OPENAI_* LLM (the same one
`app/llm_normalizer.py` talks to) and cached to
`bench/results/interleave_ab/corpus.json` -- generation is a one-off, the A/B must
re-run on the identical text every time.

Digits and abbreviations are kept OUT on purpose: the text goes to the LM verbatim
(the A/B bypasses the API's normalizer), so anything unspeakable would be read as
garbage in all three conditions and only add noise.

    set -a; source .env; set +a
    uv run --with aiohttp python bench/interleave_ab/corpus.py --per-lang 40

`chunk_text()` is imported by `run.py`; it is the chunker, and it is deliberately a
copy of what livekit-agents' StreamAdapter does to a reply -- split at sentence ends,
then at commas, merging pieces until each is at least `min_chars` (20, its default).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path

OUT = Path('bench/results/interleave_ab/corpus.json')

# ---------------------------------------------------------------- chunking

_SPLIT_RE = re.compile(r'(?<=[.!?])\s+|(?<=[,;:])\s+')


def chunk_text(text: str, min_chars: int = 20) -> list[str]:
    """Split `text` the way livekit-agents' StreamAdapter splits an LLM reply.

    Pieces shorter than `min_chars` are merged forward (and a trailing runt merged
    back), so every chunk is a speakable fragment ending on its own punctuation --
    which is exactly the shape each `/v1/audio/speech` request gets in production.
    """
    pieces = [p.strip() for p in _SPLIT_RE.split(text.strip()) if p.strip()]
    out: list[str] = []
    for p in pieces:
        if out and len(out[-1]) < min_chars:
            out[-1] = f'{out[-1]} {p}'
        else:
            out.append(p)
    if len(out) > 1 and len(out[-1]) < min_chars:
        out[-2] = f'{out[-2]} {out.pop()}'
    return out


# ---------------------------------------------------------------- generation

PROMPTS = {
    'en': (
        'Write {n} different natural English paragraphs that a friendly voice assistant '
        'would speak out loud. Each paragraph must be {lo}-{hi} words, one paragraph per '
        'topic, and must contain at least two commas and at least two full stops so it '
        'breaks into four to six short spoken fragments. Conversational and flowing, not '
        'a list. Absolutely no digits, no numbers written as figures, no abbreviations, no '
        'acronyms, no emoji, no markdown, no quotation marks. Topics: {topics}. '
        'Return strictly a JSON array of {n} strings and nothing else.'
    ),
    'ms': (
        'Tulis {n} perenggan bahasa Melayu Malaysia yang natural, seperti yang akan '
        'dituturkan oleh pembantu suara mesra. Setiap perenggan mesti {lo}-{hi} patah '
        'perkataan, satu perenggan untuk satu topik, dan mesti mengandungi sekurang-kurangnya '
        'dua koma dan dua noktah supaya ia boleh dipecahkan kepada empat hingga enam serpihan '
        'pertuturan pendek. Gaya berbual dan mengalir, bukan senarai. Jangan sekali-kali guna '
        'angka, nombor dalam bentuk digit, singkatan, akronim, emoji, markdown atau tanda '
        'petikan. Guna bahasa Melayu Malaysia (bukan Indonesia). Topik: {topics}. '
        'Kembalikan hanya array JSON yang mengandungi {n} string, tiada apa-apa lagi.'
    ),
}

TOPICS = {
    'en': [
        'planning a weekend trip', 'a slow morning at home', 'helping a customer with a bill',
        'the weather turning', 'learning to cook a family recipe', 'a delayed flight',
        'looking after a new plant', 'catching up with an old friend', 'moving to a new house',
        'a quiet evening walk', 'choosing a phone plan', 'the first day at a new job',
        'a rainy afternoon at the market', 'teaching a child to swim', 'fixing a leaking tap',
        'waiting at a clinic', 'a neighbourhood football match', 'sorting out an online order',
        'a long drive up north', 'preparing for a wedding',
    ],
    'ms': [
        'merancang percutian hujung minggu', 'pagi yang santai di rumah', 'membantu pelanggan dengan bil',
        'cuaca yang berubah', 'belajar memasak resipi keluarga', 'penerbangan yang tertangguh',
        'menjaga pokok baharu', 'berbual dengan kawan lama', 'berpindah ke rumah baharu',
        'berjalan petang yang tenang', 'memilih pelan telefon', 'hari pertama di tempat kerja baharu',
        'petang hujan di pasar', 'mengajar anak berenang', 'membaiki paip bocor',
        'menunggu di klinik', 'perlawanan bola sepak kampung', 'menguruskan pesanan dalam talian',
        'pemanduan jauh ke utara', 'persiapan majlis perkahwinan',
    ],
}


async def _ask(session, prompt: str) -> list[str]:
    base = os.environ['OPENAI_BASE_URL'].rstrip('/')
    payload = {
        'model': os.environ['OPENAI_MODEL_NAME'],
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.9,
        'max_tokens': 4096,
    }
    headers = {'Authorization': f"Bearer {os.environ['OPENAI_API_KEY']}"}
    async with session.post(f'{base}/chat/completions', json=payload, headers=headers) as r:
        r.raise_for_status()
        body = await r.json()
    txt = body['choices'][0]['message']['content']
    m = re.search(r'\[.*\]', txt, re.S)
    if not m:
        raise ValueError(f'no JSON array in reply: {txt[:200]}')
    return [s.strip() for s in json.loads(m.group(0)) if isinstance(s, str) and s.strip()]


def _clean(s: str) -> str:
    s = re.sub(r'\s+', ' ', s).strip().strip('"').strip()
    if not s.endswith(('.', '!', '?')):
        s += '.'
    return s


def _ok(s: str, lo: int, hi: int) -> bool:
    """Reject anything the LM would read badly or that will not chunk."""
    if re.search(r'\d', s):
        return False
    if len(s.split()) < lo - 8 or len(s.split()) > hi + 25:
        return False
    return len(chunk_text(s)) >= 3


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-lang', type=int, default=40)
    ap.add_argument('--lo', type=int, default=45)
    ap.add_argument('--hi', type=int, default=85)
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()

    out = Path(a.out)
    if out.exists() and not a.force:
        print(f'{out} exists ({len(json.loads(out.read_text())["texts"])} texts); --force to regenerate')
        return 0

    import aiohttp
    rng = random.Random(1234)
    texts: list[dict] = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        for lang in ('en', 'ms'):
            kept: list[str] = []
            batch, attempt = 8, 0
            while len(kept) < a.per_lang and attempt < 20:
                attempt += 1
                topics = ', '.join(rng.sample(TOPICS[lang], k=min(batch, len(TOPICS[lang]))))
                prompt = PROMPTS[lang].format(n=batch, lo=a.lo, hi=a.hi, topics=topics)
                try:
                    got = await _ask(session, prompt)
                except Exception as e:                       # a flaky proxy must not lose the run
                    print(f'  {lang} attempt {attempt}: {type(e).__name__}: {e}', file=sys.stderr)
                    continue
                for s in got:
                    s = _clean(s)
                    if _ok(s, a.lo, a.hi) and s not in kept:
                        kept.append(s)
                print(f'  {lang}: {len(kept)}/{a.per_lang} after attempt {attempt}')
            for i, s in enumerate(kept[:a.per_lang]):
                texts.append({'id': f'{lang}{i:03d}', 'lang': lang, 'text': s,
                              'chunks': chunk_text(s)})

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'model': os.environ.get('OPENAI_MODEL_NAME'), 'texts': texts},
                              ensure_ascii=False, indent=1))
    nch = [len(t['chunks']) for t in texts]
    nw = [len(t['text'].split()) for t in texts]
    print(f'wrote {out}: {len(texts)} texts, '
          f'words {min(nw)}-{max(nw)} (mean {sum(nw)/len(nw):.0f}), '
          f'chunks {min(nch)}-{max(nch)} (mean {sum(nch)/len(nch):.1f})')
    return 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
