"""Benchmark for app/fade.py: how often a response starts mid-waveform, and whether it clicks.

About 1 request in 20 the LM's first speech tokens are already voiced, so the first decoded
window begins mid-waveform. This sends texts exactly like livekit's openai.TTS plugin (SSE,
speed 1.0, server defaults) at several closed-loop concurrencies and records per response:

    hot     first 10 ms within 25 dB of the response's own speech level -- the model started
            voiced. A 10 ms fade does not change this one (the ramp reaches full level by
            the end of those 10 ms); it measures the cause.
    click   the FIRST MILLISECOND already above 0.05 full scale -- the audible edge, and what
            the fade removes.

Run it against a build without the fade and one with it (FADE_IN_MS=0 vs 10):

    python bench/fade_bench.py --url http://127.0.0.1:9091 --texts bench/fade_texts.txt \
        --concurrency 1,8,16 --per-level 240 --label nofade --out /tmp/fade
"""
import argparse, asyncio, base64, json, os, time
import numpy as np

SR = 24000


def onset_db(x):
    fr = 240
    e = 20 * np.log10(np.array([np.sqrt(np.mean(x[i:i + fr] ** 2)) for i in range(0, len(x) - fr, fr)]) + 1e-9)
    return float(e[0] - np.percentile(e, 95)) if len(e) >= 20 else None


async def say(s, url, text, voice):
    body = {'input': text, 'model': 'TTS-model', 'voice': voice, 'speed': 1.0, 'stream_format': 'sse'}
    ch = []
    async with s.post(f'{url}/v1/audio/speech', json=body, timeout=180) as r:
        r.raise_for_status()
        async for raw in r.content:
            l = raw.decode('utf-8', 'ignore').strip()
            if l.startswith('data: ') and l[6:] != '[DONE]':
                try:
                    ev = json.loads(l[6:])
                except json.JSONDecodeError:
                    continue
                if ev.get('type') == 'speech.audio.delta':
                    ch.append(base64.b64decode(ev.get('delta') or ev.get('audio') or ''))
    return np.frombuffer(b''.join(ch), dtype=np.int16).astype(np.float32) / 32768


async def level(a, texts, conc):
    import aiohttp
    import soundfile as sf
    q = asyncio.Queue()
    for i in range(a.per_level):
        q.put_nowait((i, texts[i % len(texts)]))
    rows = []
    d = os.path.join(a.out, f'{a.label}_c{conc}')
    os.makedirs(d, exist_ok=True)

    async def worker(s):
        while True:
            try:
                i, t = q.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                x = await say(s, a.url, t, a.voice)
            except Exception as e:                            # noqa: BLE001
                rows.append({'i': i, 'error': str(e)[:120]})
                continue
            o = onset_db(x)
            hot = o is not None and o > -25
            if hot:
                sf.write(os.path.join(d, f'hot_{i:04d}.wav'), x, SR, subtype='PCM_16')
            f1 = float(abs(x[:24]).max()) if len(x) else None
            rows.append({'i': i, 'text': t, 'onset_db': o, 'first5ms': float(abs(x[:120]).max()) if len(x) else None,
                         'first1ms': f1, 'first_sample': float(abs(x[0])) if len(x) else None,
                         'hot': hot, 'click': bool(f1 is not None and f1 > 0.05)})

    t0 = time.time()
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=conc * 2)) as s:
        await asyncio.gather(*(worker(s) for _ in range(conc)))
    ok = [r for r in rows if 'error' not in r and r['onset_db'] is not None]
    hot = sum(r['hot'] for r in ok)
    clk = sum(r['click'] for r in ok)
    f1 = np.array([r['first1ms'] for r in ok])
    print(f'{a.label} c={conc:3d}: hot {hot:3d}/{len(ok)} = {100 * hot / max(1, len(ok)):5.1f}%  '
          f'click {clk:3d} = {100 * clk / max(1, len(ok)):5.1f}%  first-1ms max p99 {np.percentile(f1, 99):.3f} max {f1.max():.3f}  '
          f'unscored {len(rows) - len(ok)}  wall {time.time() - t0:5.0f}s', flush=True)
    with open(os.path.join(a.out, f'{a.label}_c{conc}.jsonl'), 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return {'concurrency': conc, 'n': len(ok), 'hot': hot, 'rate': hot / max(1, len(ok)),
            'click': clk, 'click_rate': clk / max(1, len(ok)),
            'first1ms_p99': float(np.percentile(f1, 99)), 'first1ms_max': float(f1.max())}


async def main(a):
    texts = [l.strip() for l in open(a.texts, encoding='utf-8') if l.strip()]
    os.makedirs(a.out, exist_ok=True)
    res = [await level(a, texts, int(c)) for c in a.concurrency.split(',')]
    json.dump({'url': a.url, 'label': a.label, 'levels': res}, open(os.path.join(a.out, f'{a.label}_summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--url', default='http://127.0.0.1:9091')
    p.add_argument('--voice', default='TM_English_Normal')
    p.add_argument('--texts', required=True)
    p.add_argument('--concurrency', default='1,8,16')
    p.add_argument('--per-level', type=int, default=240)
    p.add_argument('--label', default='prod')
    p.add_argument('--out', required=True)
    asyncio.run(main(p.parse_args()))
