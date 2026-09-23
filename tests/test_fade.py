"""app/fade.py -- first-samples fade-in. Pure numpy, runs anywhere."""
import asyncio
import random

import numpy as np

from app.fade import FadeIn, fade_in_pcm16

SR = 24_000
N = 240   # 10 ms


def _hot_start(n=4000, seed=0):
    """Speech-level audio from sample 0 -- the defect this exists for."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    return (np.sin(2 * np.pi * 220 * t) * 0.5 * 32767 + rng.normal(0, 200, n)).astype('<i2')


def _run(x, splits):
    f = FadeIn(N)
    b = x.tobytes()
    out, pos = [], 0
    for s in splits + [len(b)]:
        out.append(f.process(b[pos:s]))
        pos = s
    return np.frombuffer(b''.join(out), dtype='<i2')


def test_first_sample_is_silent_and_ramp_is_monotone_gain():
    x = _hot_start()
    y = _run(x, [])
    assert y[0] == 0
    g = y[:N].astype(float) / np.where(x[:N] == 0, 1, x[:N])
    ok = x[:N] != 0
    assert np.all(g[ok] >= -1e-3) and np.all(g[ok] <= 1 + 1e-3)


def test_samples_after_the_ramp_are_untouched():
    x = _hot_start()
    y = _run(x, [])
    assert np.array_equal(y[N:], x[N:])
    assert len(y) == len(x)


def test_output_is_identical_however_the_stream_is_chunked():
    x = _hot_start()
    ref = _run(x, [])
    rnd = random.Random(1)
    for _ in range(50):
        cuts = sorted(rnd.sample(range(1, len(x) * 2), rnd.randint(1, 30)))
        assert np.array_equal(_run(x, cuts), ref)      # includes odd byte boundaries


def test_one_byte_chunks():
    x = _hot_start(600)
    assert np.array_equal(_run(x, list(range(1, len(x) * 2))), _run(x, []))


def test_stream_shorter_than_the_ramp():
    x = _hot_start(100)
    y = _run(x, [])
    assert len(y) == 100 and y[0] == 0


def test_silent_start_is_unchanged_audibly():
    x = np.concatenate([np.zeros(N, dtype='<i2'), _hot_start(1000)])
    y = _run(x, [])
    assert np.array_equal(y, x)                        # the ramp lies over silence


def test_zero_length_ramp_is_a_passthrough():
    x = _hot_start()
    f = FadeIn(0)
    assert f.process(x.tobytes()) == x.tobytes()


def test_async_wrapper():
    x = _hot_start()
    b = x.tobytes()

    async def gen():
        for i in range(0, len(b), 333):
            yield b[i:i + 333]

    async def collect():
        return b''.join([c async for c in fade_in_pcm16(gen(), N)])

    y = np.frombuffer(asyncio.run(collect()), dtype='<i2')
    assert np.array_equal(y, _run(x, []))
