"""
Unit tests for the speaking-rate time stretcher (app/timestretch.py).

Pure numpy -- no torch/GPU/live API needed:
    python -m pytest tests/test_timestretch.py -v
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from app.timestretch import WSOLA, float_to_pcm16, pcm16_to_float, MIN_RATE, MAX_RATE

SR = 24000


def speechlike(seconds=3.0, seed=0):
    """A harmonic tone with vibrato and an amplitude envelope, plus a little noise:
    periodic enough to expose join clicks, non-stationary enough to defeat a
    degenerate 'all candidates equal' search."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    f0 = 180 + 8 * np.sin(2 * np.pi * 3 * t)
    ph = 2 * np.pi * np.cumsum(f0) / SR
    x = (0.5 * np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.12 * np.sin(3 * ph))
    x *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.7 * t)
    x += 0.01 * rng.standard_normal(len(t))
    return x.astype(np.float32)


def stretch(x, rate, chunks=None, **kw):
    ts = WSOLA(rate, SR, **kw)
    parts = [ts.process(c) for c in (chunks if chunks is not None else [x])]
    parts.append(ts.flush())
    return np.concatenate(parts)


def peak_hz(y):
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    return np.fft.rfftfreq(len(y), 1 / SR)[np.argmax(spec)]


class TestIdentity:
    def test_rate_one_is_exact_identity(self):
        x = speechlike()
        y = stretch(x, 1.0)
        assert len(y) == len(x)
        np.testing.assert_allclose(y, x, atol=1e-6)

    def test_rate_one_identity_when_chunked(self):
        x = speechlike()
        y = stretch(x, 1.0, chunks=np.split(x, [1000, 1001, 30000, 60000]))
        np.testing.assert_allclose(y, x, atol=1e-6)

    def test_pcm16_round_trip_is_byte_identical(self):
        pcm = float_to_pcm16(speechlike(1.0))
        assert float_to_pcm16(pcm16_to_float(pcm)) == pcm


class TestRatio:
    @pytest.mark.parametrize('rate', [0.5, 0.8, 1.3, 1.5, 2.0])
    def test_duration_scales_by_rate(self, rate):
        x = speechlike()
        y = stretch(x, rate)
        expect = len(x) / rate
        # exact to within one block (40 ms) -- the tail is cut at block granularity
        assert abs(len(y) - expect) <= int(SR * 0.040), (len(y), expect)

    @pytest.mark.parametrize('rate', [0.7, 1.3, 1.6])
    def test_faster_is_shorter_slower_is_longer(self, rate):
        x = speechlike(2.0)
        y = stretch(x, rate)
        assert (len(y) < len(x)) == (rate > 1.0)


class TestPitchPreserved:
    @pytest.mark.parametrize('rate', [0.5, 0.75, 1.3, 2.0])
    def test_steady_tone_keeps_its_frequency(self, rate):
        t = np.arange(SR * 2) / SR
        x = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        y = stretch(x, rate)
        # resampling would have moved this to 200 * rate
        assert abs(peak_hz(y) - 200.0) < 3.0, peak_hz(y)

    @pytest.mark.parametrize('rate', [0.8, 1.3])
    def test_joins_are_continuous(self, rate):
        # a bad join is a sample-to-sample jump far above the signal's own slope
        x = speechlike()
        y = stretch(x, rate)
        assert np.max(np.abs(np.diff(y))) <= np.max(np.abs(np.diff(x))) * 1.05


class TestStreaming:
    @pytest.mark.parametrize('rate', [0.5, 0.8, 1.0, 1.3, 2.0])
    def test_output_independent_of_chunking(self, rate):
        x = speechlike()
        rng = np.random.default_rng(1)
        one_shot = stretch(x, rate)
        for _ in range(3):
            cuts = np.sort(rng.integers(1, len(x), 50))
            chunked = stretch(x, rate, chunks=np.split(x, cuts))
            assert len(chunked) == len(one_shot)
            assert np.array_equal(chunked, one_shot)

    def test_tiny_chunks(self):
        x = speechlike(1.0)
        one_shot = stretch(x, 1.3)
        tiny = stretch(x, 1.3, chunks=np.split(x, np.arange(1, len(x), 97)))
        assert np.array_equal(tiny, one_shot)

    def test_first_chunk_emits_promptly(self):
        # a 2 s first chunk must come out (almost) whole, not be held for lookahead
        ts = WSOLA(1.3, SR)
        y = ts.process(speechlike(2.0))
        assert len(y) >= int(SR * 2.0 / 1.3) - int(SR * 0.060)

    def test_empty_chunks_are_harmless(self):
        ts = WSOLA(1.3, SR)
        assert len(ts.process(np.zeros(0, np.float32))) == 0
        x = speechlike(1.0)
        a = ts.process(x)
        b = ts.process(np.zeros(0, np.float32))
        c = ts.flush()
        assert len(b) == 0
        assert abs(len(a) + len(c) - len(x) / 1.3) <= int(SR * 0.040)


class TestEdges:
    def test_empty_stream(self):
        ts = WSOLA(1.3, SR)
        assert len(ts.flush()) == 0

    def test_shorter_than_one_block_passes_through(self):
        x = speechlike(1.0)[:500]
        ts = WSOLA(1.3, SR)
        y = np.concatenate([ts.process(x), ts.flush()])
        np.testing.assert_array_equal(y, x)

    def test_silence_stays_silent(self):
        y = stretch(np.zeros(SR, np.float32), 1.3)
        assert np.all(y == 0)
        assert abs(len(y) - SR / 1.3) <= int(SR * 0.040)

    def test_output_stays_in_range(self):
        # a crossfade of two full-scale segments must not exceed full scale
        x = np.sign(speechlike(1.0)).astype(np.float32)
        y = stretch(x, 0.8)
        assert np.max(np.abs(y)) <= 1.0 + 1e-6

    def test_invalid_rate(self):
        with pytest.raises(ValueError):
            WSOLA(0.0, SR)
        with pytest.raises(ValueError):
            WSOLA(-1.0, SR)

    def test_api_bounds(self):
        assert 0 < MIN_RATE < 1.0 < MAX_RATE


class TestPcmHelpers:
    def test_empty(self):
        assert float_to_pcm16(np.zeros(0)) == b''
        assert len(pcm16_to_float(b'')) == 0

    def test_clips(self):
        b = float_to_pcm16(np.array([2.0, -2.0, 0.0], np.float32))
        assert np.frombuffer(b, np.int16).tolist() == [32767, -32767, 0]
