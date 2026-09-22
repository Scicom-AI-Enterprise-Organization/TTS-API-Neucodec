"""LiveKit TTS stress-test load client.

Joins N rooms concurrently (auto-dispatching one stress agent per room), sends
text lines on the "tts-input" topic, captures the agent's published audio and
measures per utterance: TTFB (text sent -> first audible frame), audio duration,
wall time, active-speech RMS. Writes JSONL rows + a summary.

Shard across processes with --procs (default: one process per 8 rooms). ONE python
process cannot drive many rooms: each room's coroutine scans its frame buffer and runs
numpy RMS on the shared event loop, so past ~8 rooms the CLIENT becomes the bottleneck
and its own scheduling delay is charged to the server as TTFB. Measured: 16 rooms from
one process reported ttfb p50 14.08 s; the same 16 rooms split over two processes
reported 0.307 s, with nothing server-side changed.

With --wav-dir it also writes one wav per utterance plus a records.jsonl in the
shape bench/pitch_stress_score.py reads, so the audio a caller actually hears can be
scored for pitch/tone/volume steps with the same metric as the API-direct run.

  python load_client.py --concurrency 4 --utterances 3 --out c4.json
  python load_client.py --texts ../pitch_stress_texts.txt --wav-dir lk/ --arm lk_interleave
"""

import argparse
import asyncio
import json
import math
import multiprocessing
import os
import shutil
import tempfile
import time

import numpy as np
from livekit import api, rtc

SENTENCES = [
    "Terima kasih kerana menghubungi kami, ada apa-apa lagi yang saya boleh bantu?",
    "Baki anda RM1,250.50 dan nombor telefon 012-3456789.",
    "Your appointment is on 15/3/2026 at 10:45 AM, please arrive early sir.",
    "Okay boleh, saya akan hantar invois melalui emel sebelum pukul 5 petang ini.",
]

FRAME_DB_THRESH = -45.0


def frame_db(a):
    if not len(a):
        return -120.0
    x = a.astype(np.float32) / 32768.0
    r = np.sqrt(np.mean(x * x))
    return 20 * np.log10(max(r, 1e-6))


async def measure_utterance(frames, cut, t_send, timeout, silence_end, want_samples=False):
    """Poll captured frames after index `cut` for speech onset then trailing silence."""
    t_first = None
    t_last_loud = None
    while True:
        now = time.monotonic()
        if t_first is None and now - t_send > timeout:
            return {"ok": False, "error": "timeout waiting for audio"}
        if t_first is not None and now - t_last_loud > silence_end:
            break
        for k in range(cut, len(frames)):
            t_recv, a = frames[k]
            if frame_db(a) > FRAME_DB_THRESH:
                if t_first is None:
                    t_first = t_recv
                t_last_loud = t_recv
            cut = k + 1
        await asyncio.sleep(0.05)

    samples = np.concatenate([a for t, a in frames if t_first <= t <= t_last_loud])
    x = samples.astype(np.float32) / 32768.0
    frame = 1200  # 50 ms @ 24k
    rms = np.array([np.sqrt(np.mean(x[i:i + frame] ** 2)) for i in range(0, max(1, len(x) - frame), frame // 2)])
    db = 20 * np.log10(np.maximum(rms, 1e-8))
    active = db[db > -50]
    lvl = 20 * np.log10(np.sqrt(np.mean((10 ** (active / 20)) ** 2))) if len(active) else -120.0
    out_samples = samples if want_samples else None
    return {
        "ok": True,
        "samples": out_samples,
        "ttfb": round(t_first - t_send, 3),
        "audio_s": round(len(samples) / 24000, 2),
        "wall_s": round(t_last_loud - t_send, 2),
        "rms_db": round(float(lvl), 2),
        "peak": round(float(np.abs(x).max()), 3),
    }


async def run_room(i, args, rows):
    room_name = f"{args.prefix}-{i}"
    token = (
        api.AccessToken(args.api_key, args.api_secret)
        .with_identity(f"loader-{i}")
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )
    room = rtc.Room()
    frames = []
    track_ready = asyncio.Event()

    @room.on("track_subscribed")
    def on_track(track, pub, participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            try:
                stream = rtc.AudioStream(track, sample_rate=24000, num_channels=1)
            except TypeError:
                stream = rtc.AudioStream(track)

            async def pump():
                async for ev in stream:
                    frames.append((time.monotonic(), np.frombuffer(ev.frame.data, dtype=np.int16).copy()))

            asyncio.create_task(pump())
            track_ready.set()

    try:
        await room.connect(args.url, token)
    except Exception as e:
        rows.append({"room": i, "ok": False, "error": f"connect: {e}"})
        return
    try:
        await asyncio.wait_for(track_ready.wait(), timeout=30)
    except asyncio.TimeoutError:
        rows.append({"room": i, "ok": False, "error": "agent audio track never arrived"})
        await room.disconnect()
        return
    await asyncio.sleep(0.5)

    texts = args.text_list
    for j in range(args.utterances):
        text = texts[j % len(texts)]
        cut = len(frames)
        t_send = time.monotonic()
        try:
            await room.local_participant.send_text(text, topic="tts-input")
            row = await measure_utterance(frames, cut, t_send, args.timeout,
                                          args.silence_end, want_samples=bool(args.wav_dir))
        except Exception as e:
            row = {"ok": False, "error": repr(e)}
        samples = row.pop("samples", None)
        if args.wav_dir and samples is not None and len(samples):
            import soundfile as sf

            wav = f"{args.arm}_t{j:02d}_r{i}_c{args.concurrency}.wav"
            sf.write(f"{args.wav_dir}/{wav}", samples, 24000, subtype="PCM_16")
            # the scorer's record shape; joins are unknown through LiveKit (the agent
            # chops the text itself), so it falls back to seams detected from pauses.
            row["record"] = {"arm": args.arm, "text_id": j, "rep": i,
                             "concurrency": args.concurrency, "text": text,
                             "wav": wav, "sr": 24000, "joins": [], "n_pieces": 0,
                             "duration_s": round(len(samples) / 24000, 3)}
        row.update({"room": i, "utt": j, "text": text[:36]})
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "record"},
                         ensure_ascii=False), flush=True)

    if args.save_wav and i == 0 and frames:
        import soundfile as sf

        sf.write(args.save_wav, np.concatenate([a for _, a in frames]), 24000)
    await room.disconnect()


def summarize(rows, conc, wall):
    ok = [r for r in rows if r.get("ok")]
    bad = [r for r in rows if not r.get("ok")]
    summary = {"concurrency": conc, "utterances": len(rows), "ok": len(ok),
               "errors": len(bad), "wall_s": round(wall, 1)}
    if ok:
        ttfb = sorted(r["ttfb"] for r in ok)
        rms = [r["rms_db"] for r in ok]
        summary.update({
            "ttfb_p50": ttfb[len(ttfb) // 2],
            "ttfb_p95": ttfb[min(len(ttfb) - 1, int(len(ttfb) * 0.95))],
            "ttfb_max": ttfb[-1],
            "audio_s_total": round(sum(r["audio_s"] for r in ok), 1),
            "rms_db_mean": round(float(np.mean(rms)), 2),
            "rms_db_spread": round(max(rms) - min(rms), 2),
            "rms_db_std": round(float(np.std(rms)), 2),
            "peak_max": max(r["peak"] for r in ok),
        })
    if bad:
        summary["error_samples"] = [r.get("error") for r in bad[:3]]
    return summary


async def run_shard(args, room_ids):
    rows = []
    await asyncio.gather(*(run_room(i, args, rows) for i in room_ids))
    return rows


def shard_entry(args, room_ids, out_path):
    """Child process: drive `room_ids`, dump raw rows for the parent to merge."""
    rows = asyncio.run(run_shard(args, room_ids))
    with open(out_path, "w") as f:
        json.dump(rows, f, ensure_ascii=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://127.0.0.1:7880")
    p.add_argument("--api-key", default="devkey")
    p.add_argument("--api-secret", default="secret")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--utterances", type=int, default=3)
    p.add_argument("--timeout", type=float, default=90)
    p.add_argument("--silence-end", type=float, default=1.5)
    p.add_argument("--prefix", default=f"stress-{int(time.time())}")
    p.add_argument("--save-wav", default="")
    p.add_argument("--out", default="")
    p.add_argument("--texts", default="", help="one utterance per line; default: SENTENCES")
    p.add_argument("--wav-dir", default="", help="save one wav per utterance + records.jsonl")
    p.add_argument("--arm", default="livekit", help="label for the scorer records")
    p.add_argument("--procs", type=int, default=0,
                   help="client processes to shard rooms over (0 = auto, see --rooms-per-proc)")
    p.add_argument("--rooms-per-proc", type=int, default=8,
                   help="rooms one process can drive before it becomes the bottleneck")
    args = p.parse_args()
    args.text_list = SENTENCES
    if args.texts:
        args.text_list = [l.strip() for l in open(args.texts, encoding="utf-8") if l.strip()]
    if args.wav_dir:
        os.makedirs(args.wav_dir, exist_ok=True)

    nproc = args.procs or max(1, math.ceil(args.concurrency / args.rooms_per_proc))
    nproc = min(nproc, args.concurrency)
    shards = [list(range(k, args.concurrency, nproc)) for k in range(nproc)]

    t0 = time.monotonic()
    if nproc == 1:
        rows = asyncio.run(run_shard(args, shards[0]))
    else:
        # fds: each room opens several WebRTC sockets, and the default 1024 makes the
        # client -- not the server -- fail at c>=16 with "Too many open files".
        try:
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
        except Exception:                                   # noqa: BLE001 — best effort
            pass
        ctx = multiprocessing.get_context("spawn")
        tmp = tempfile.mkdtemp(prefix="lkload-")
        procs = []
        for k, ids in enumerate(shards):
            out = os.path.join(tmp, f"shard{k}.json")
            pr = ctx.Process(target=shard_entry, args=(args, ids, out))
            pr.start()
            procs.append((pr, out))
        rows = []
        for pr, out in procs:
            pr.join()
            if os.path.exists(out):
                rows.extend(json.load(open(out)))
            else:
                rows.append({"ok": False, "error": f"shard died rc={pr.exitcode}"})
        shutil.rmtree(tmp, ignore_errors=True)
    wall = time.monotonic() - t0

    summary = summarize(rows, args.concurrency, wall)
    summary["client_procs"] = nproc
    print("SUMMARY " + json.dumps(summary))
    if args.wav_dir:
        with open(f"{args.wav_dir}/records.jsonl", "a") as f:
            for r in rows:
                if r.get("record"):
                    f.write(json.dumps(r["record"], ensure_ascii=False) + "\n")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary,
                       "rows": [{k: v for k, v in r.items() if k != "record"} for r in rows]},
                      f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
