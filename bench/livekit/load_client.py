"""LiveKit TTS stress-test load client.

Joins N rooms concurrently (auto-dispatching one stress agent per room), sends
text lines on the "tts-input" topic, captures the agent's published audio and
measures per utterance: TTFB (text sent -> first audible frame), audio duration,
wall time, active-speech RMS. Writes JSONL rows + a summary.

  python load_client.py --concurrency 4 --utterances 3 --out c4.json
"""

import argparse
import asyncio
import json
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


async def measure_utterance(frames, cut, t_send, timeout, silence_end):
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
    return {
        "ok": True,
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

    for j in range(args.utterances):
        text = SENTENCES[j % len(SENTENCES)]
        cut = len(frames)
        t_send = time.monotonic()
        try:
            await room.local_participant.send_text(text, topic="tts-input")
            row = await measure_utterance(frames, cut, t_send, args.timeout, args.silence_end)
        except Exception as e:
            row = {"ok": False, "error": repr(e)}
        row.update({"room": i, "utt": j, "text": text[:36]})
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    if args.save_wav and i == 0 and frames:
        import soundfile as sf

        sf.write(args.save_wav, np.concatenate([a for _, a in frames]), 24000)
    await room.disconnect()


async def main():
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
    args = p.parse_args()

    rows = []
    t0 = time.monotonic()
    await asyncio.gather(*(run_room(i, args, rows) for i in range(args.concurrency)))
    wall = time.monotonic() - t0

    ok = [r for r in rows if r.get("ok")]
    bad = [r for r in rows if not r.get("ok")]
    summary = {"concurrency": args.concurrency, "utterances": len(rows), "ok": len(ok), "errors": len(bad), "wall_s": round(wall, 1)}
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
    print("SUMMARY " + json.dumps(summary))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    asyncio.run(main())
