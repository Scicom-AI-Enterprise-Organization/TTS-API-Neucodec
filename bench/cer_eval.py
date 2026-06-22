#!/usr/bin/env python3
"""
Accuracy guardrail: generate audio for the fixed eval set, transcribe with
Whisper large-v3, and compute CER/WER vs the reference text. Used to confirm
inference optimizations (esp. FP8) don't degrade intelligibility.
Runs on the pod.
"""
import os, sys, json, time, argparse, re, asyncio
import aiohttp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evalset import EVAL_SET

def norm_text(s):
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)   # drop punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s

def lang_of(_id):
    if _id.startswith("en"): return "en"
    if _id.startswith("ms"): return "ms"
    return None  # code-switch -> auto-detect

async def gen_one(session, url, text, voice, max_tokens, temperature, out_path):
    payload = {"input": text, "voice": voice, "model": "TTS-model",
               "response_format": "wav", "temperature": temperature,
               "max_tokens": max_tokens, "stream": False, "normalize_malaysian": False}
    async with session.post(url + "/v1/audio/speech", json=payload) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:200]}")
        data = await resp.read()
    with open(out_path, "wb") as f: f.write(data)
    return len(data)

async def gen_all(url, wav_dir, samples, max_tokens, temperature):
    os.makedirs(wav_dir, exist_ok=True)
    manifest = []
    sem = asyncio.Semaphore(4)
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=300)) as s:
        async def task(_id, voice, text, k):
            out = os.path.join(wav_dir, f"{_id}_{k}.wav")
            async with sem:
                await gen_one(s, url, text, voice, max_tokens, temperature, out)
            return {"id": _id, "k": k, "voice": voice, "text": text,
                    "lang": lang_of(_id), "wav": out}
        tasks = [task(_id, voice, text, k)
                 for (_id, voice, text) in EVAL_SET for k in range(samples)]
        for fut in asyncio.as_completed(tasks):
            m = await fut; manifest.append(m)
            print(f"  generated {m['id']}_{m['k']}", flush=True)
    manifest.sort(key=lambda m: (m["id"], m["k"]))
    return manifest

def transcribe(manifest, model_size):
    from faster_whisper import WhisperModel
    import jiwer
    print(f"loading whisper {model_size} ...", flush=True)
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    rows = []
    for m in manifest:
        segs, info = model.transcribe(m["wav"], language=m["lang"], beam_size=5)
        hyp = " ".join(s.text for s in segs).strip()
        ref_n, hyp_n = norm_text(m["text"]), norm_text(hyp)
        cer = jiwer.cer(ref_n, hyp_n) if ref_n else None
        wer = jiwer.wer(ref_n, hyp_n) if ref_n else None
        rows.append({**m, "hyp": hyp, "ref_norm": ref_n, "hyp_norm": hyp_n,
                     "detected_lang": info.language, "cer": cer, "wer": wer})
        print(f"  {m['id']}_{m['k']} cer={cer:.3f} wer={wer:.3f} | hyp: {hyp[:70]}", flush=True)
    return rows

def summarize(rows):
    cers = [r["cer"] for r in rows if r["cer"] is not None]
    wers = [r["wer"] for r in rows if r["wer"] is not None]
    def mean(x): return sum(x)/len(x) if x else None
    def median(x):
        x = sorted(x); n = len(x)
        return None if n == 0 else (x[n//2] if n % 2 else (x[n//2-1]+x[n//2])/2)
    # per-language breakdown
    bylang = {}
    for r in rows:
        g = (r["id"].split("_")[0])
        bylang.setdefault(g, []).append(r["cer"])
    return {"n": len(rows),
            "cer_mean": round(mean(cers),4) if cers else None,
            "cer_median": round(median(cers),4) if cers else None,
            "wer_mean": round(mean(wers),4) if wers else None,
            "cer_by_group": {k: round(sum(v)/len(v),4) for k,v in bylang.items()}}

async def amain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:9090")
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--whisper", default="large-v3")
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-gen", action="store_true")
    args = ap.parse_args()

    man_path = os.path.join(args.wav_dir, "manifest.json")
    if args.skip_gen and os.path.exists(man_path):
        manifest = json.load(open(man_path))
    else:
        print("generating eval audio ...", flush=True)
        manifest = await gen_all(args.url, args.wav_dir, args.samples, args.max_tokens, args.temperature)
        json.dump(manifest, open(man_path, "w"), indent=2)
    rows = transcribe(manifest, args.whisper)
    summary = summarize(rows)
    out = {"label": args.label, "summary": summary, "rows": rows}
    json.dump(out, open(args.out, "w"), indent=2)
    print("\n=== CER SUMMARY (%s) ===" % args.label, flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print("wrote", args.out, flush=True)

if __name__ == "__main__":
    asyncio.run(amain())
