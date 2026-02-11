import os
import asyncio
import time
import aiohttp
from tqdm import tqdm
import statistics
import wave
import io

WARMUP_COUNT = int(os.environ.get('WARMUP_COUNT', '3'))
CONCURRENCY = int(os.environ.get('CONCURRENCY', '50'))
LOCAL_TTS_API = os.environ.get('LOCAL_TTS_API', 'http://tts-api:9091')

async def stress_test():
    json_data = {
        "input": "Hello! How can I help you? Apa yang saya boleh bantu encik? Encik nak makan ayam gepuk ke?",
        "voice": "husein",
        "model": "Scicom-intl/Multilingual-TTS-1.7B-v0.1",
        "response_format": "wav",
        "temperature": 0.6,
        "repetition_penalty": 1.15,
        "max_tokens": 3072,
        "stream": False,
        "playback_speed": 1.5,
        "playback_overlap_speed": 0.5,
        "normalize": True
    }

    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        async with session.post(LOCAL_TTS_API + "/v1/audio/speech", json=json_data) as response:
            audio_bytes = await response.read()

    total_time = time.time() - start_time

    audio_duration = 0.0
    try:
        with wave.open(io.BytesIO(audio_bytes), 'rb') as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            audio_duration = frames / float(rate)
    except Exception as e:
        audio_duration = 0.0

    rtf = total_time / audio_duration if audio_duration > 0 else None

    return total_time, audio_duration, rtf


def print_slo_report(results):
    times = [r[0] for r in results]
    durations = [r[1] for r in results]
    rtfs = [r[2] for r in results if r[2] is not None]

    times_sorted = sorted(times)
    count = len(times_sorted)
    avg = sum(times_sorted) / count
    p50 = statistics.median(times_sorted)
    p90 = times_sorted[int(0.90 * count) - 1]
    p95 = times_sorted[int(0.95 * count) - 1]
    p99 = times_sorted[int(0.99 * count) - 1]

    rtfs_sorted = sorted(rtfs)
    rtf_count = len(rtfs_sorted)
    avg_rtf = sum(rtfs_sorted) / rtf_count
    rtf_p50 = statistics.median(rtfs_sorted)
    rtf_p90 = rtfs_sorted[int(0.90 * rtf_count) - 1]
    rtf_p95 = rtfs_sorted[int(0.95 * rtf_count) - 1]
    rtf_p99 = rtfs_sorted[int(0.99 * rtf_count) - 1]

    print("=== SLO LATENCY REPORT ===")
    print(f"Total Requests: {count}")
    print(f"Min Time: {min(times_sorted):.3f}s")
    print(f"Max Time: {max(times_sorted):.3f}s")
    print(f"Avg Time: {avg:.3f}s")
    print(f"P50 (Median): {p50:.3f}s")
    print(f"P90: {p90:.3f}s")
    print(f"P95: {p95:.3f}s")
    print(f"P99: {p99:.3f}s")

    print("\n=== AUDIO & REAL-TIME FACTOR (RTF) REPORT ===")
    print(f"Avg Audio Duration: {sum(durations)/count:.3f}s")

    print("\n--- RTF Percentiles ---")
    print(f"Min RTF: {min(rtfs_sorted):.3f}s")
    print(f"Max RTF: {max(rtfs_sorted):.3f}s")
    print(f"Avg RTF: {avg_rtf:.3f}")
    print(f"P50 RTF: {rtf_p50:.3f}")
    print(f"P90 RTF: {rtf_p90:.3f}")
    print(f"P95 RTF: {rtf_p95:.3f}")
    print(f"P99 RTF: {rtf_p99:.3f}")
    print("==========================\n")


async def run_stress_test(concurrency):
    tasks = [stress_test() for i in range(concurrency)]
    results = await asyncio.gather(*tasks)
    return results


if __name__ == '__main__':
    print(f"Warmup runs: {WARMUP_COUNT}")
    for _ in range(WARMUP_COUNT):
        asyncio.run(run_stress_test(1))
    print(f"Done Warmup runs!\n")

    results = asyncio.run(run_stress_test(CONCURRENCY))
    print_slo_report(results)
