#!/usr/bin/env python3
"""
End-to-end TTS benchmark for the colocated vLLM + NeuCodec stack.
Measures per-request latency, time-to-first-audio (TTFB, streaming), real-time
factor (RTF), and aggregate throughput under closed-loop concurrency.
Runs on the pod; talks to the app over localhost.
"""
import os, sys, time, json, argparse, asyncio, statistics, io, wave
import aiohttp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evalset import EVAL_SET

def pct(xs, p):
    if not xs: return None
    xs = sorted(xs); k = max(0, min(len(xs)-1, int(round((p/100.0)*len(xs)+0.5))-1))
    return xs[k]

def wav_dur(b):
    try:
        with wave.open(io.BytesIO(b), 'rb') as w:
            return w.getnframes()/float(w.getframerate())
    except Exception:
        # raw pcm fallback (16-bit mono 24k)
        return (len(b)/2)/24000.0 if b else 0.0

async def one_request(session, url, text, voice, max_tokens, temperature, stream):
    payload = {"input": text, "voice": voice, "model": "TTS-model",
               "response_format": "wav", "temperature": temperature,
               "max_tokens": max_tokens, "stream": stream,
               "normalize_malaysian": False}
    t0 = time.perf_counter(); ttfb = None; buf = bytearray()
    async with session.post(url + "/v1/audio/speech", json=payload) as resp:
        if resp.status != 200:
            txt = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {txt[:200]}")
        async for chunk in resp.content.iter_chunked(8192):
            if ttfb is None and chunk:
                ttfb = time.perf_counter() - t0
            buf.extend(chunk)
    total = time.perf_counter() - t0
    dur = wav_dur(bytes(buf))
    return {"latency": total, "ttfb": ttfb, "audio_dur": dur,
            "bytes": len(buf), "rtf": (total/dur if dur > 0 else None)}

async def worker(name, session, url, args, jobs, results, errors):
    while True:
        try:
            idx = jobs.pop()
        except IndexError:
            return
        _id, voice, text = EVAL_SET[idx % len(EVAL_SET)]
        try:
            r = await one_request(session, url, text, voice, args.max_tokens, args.temperature, args.stream)
            results.append(r)
        except Exception as e:
            errors.append(str(e))

async def run_level(url, concurrency, total, args):
    jobs = list(range(total))
    results, errors = [], []
    conn = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        t0 = time.perf_counter()
        await asyncio.gather(*[worker(i, session, url, args, jobs, results, errors)
                               for i in range(concurrency)])
        wall = time.perf_counter() - t0
    lat = [r["latency"] for r in results]
    rtf = [r["rtf"] for r in results if r["rtf"] is not None]
    ttfb = [r["ttfb"] for r in results if r["ttfb"] is not None]
    audio_total = sum(r["audio_dur"] for r in results)
    n = len(results)
    return {
        "concurrency": concurrency, "requests": n, "errors": len(errors),
        "error_sample": errors[:3],
        "wall_s": round(wall, 3),
        "throughput_req_s": round(n/wall, 3) if wall > 0 else None,
        "throughput_audio_s_per_s": round(audio_total/wall, 3) if wall > 0 else None,
        "latency_s": {"avg": round(statistics.mean(lat),3) if lat else None,
                      "p50": round(pct(lat,50),3) if lat else None,
                      "p90": round(pct(lat,90),3) if lat else None,
                      "p95": round(pct(lat,95),3) if lat else None,
                      "p99": round(pct(lat,99),3) if lat else None,
                      "max": round(max(lat),3) if lat else None},
        "ttfb_s": {"avg": round(statistics.mean(ttfb),3) if ttfb else None,
                   "p50": round(pct(ttfb,50),3) if ttfb else None,
                   "p90": round(pct(ttfb,90),3) if ttfb else None} if args.stream else None,
        "rtf": {"avg": round(statistics.mean(rtf),3) if rtf else None,
                "p50": round(pct(rtf,50),3) if rtf else None,
                "p90": round(pct(rtf,90),3) if rtf else None},
        "avg_audio_dur_s": round(audio_total/n, 3) if n else None,
    }

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:9090")
    ap.add_argument("--concurrency", default="1,4,16,50")
    ap.add_argument("--per-conc-mult", type=int, default=4, help="requests = concurrency * mult (min 24)")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # warmup
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=300)) as s:
        for i in range(args.warmup):
            _id, voice, text = EVAL_SET[i % len(EVAL_SET)]
            try: await one_request(s, args.url, text, voice, args.max_tokens, args.temperature, args.stream)
            except Exception as e: print("warmup err", e)
    print(f"warmup done ({args.warmup})", flush=True)

    levels = [int(x) for x in args.concurrency.split(",")]
    report = {"label": args.label, "stream": args.stream, "max_tokens": args.max_tokens,
              "temperature": args.temperature, "levels": []}
    for c in levels:
        total = max(24, c*args.per_conc_mult)
        res = await run_level(args.url, c, total, args)
        report["levels"].append(res)
        print(json.dumps(res), flush=True)
    if args.out:
        with open(args.out, "w") as f: json.dump(report, f, indent=2)
        print("wrote", args.out, flush=True)

if __name__ == "__main__":
    asyncio.run(main())
