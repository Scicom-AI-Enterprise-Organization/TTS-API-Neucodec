"""Fade in the first few milliseconds of every response.

Why: about 1 request in 20 the LM's first speech tokens are already voiced, so the first
decoded window starts mid-waveform. Emitted as is, playback opens at full level on its very
first sample -- a click. Present at concurrency 1 and with or without the decode-batcher fix,
so it is the model's output, not load (bench/FADE_IN.md).

A raised-cosine ramp over the first FADE_IN_MS removes the click. On the other ~95% the
ramp sits over near-silence and changes nothing audible.

Pure numpy and stateful across chunks, so the output is identical however the int16 byte
stream is split -- including a chunk boundary inside one sample (odd byte counts).
"""
from __future__ import annotations

import numpy as np


class FadeIn:
    """Raised-cosine fade-in over the first `n` samples of an int16 PCM byte stream."""

    def __init__(self, n: int):
        self.n = max(0, int(n))
        self.done = 0            # samples already emitted
        self._carry = b''        # a trailing odd byte, completed by the next chunk
        k = np.arange(self.n, dtype=np.float64)
        self._ramp = 0.5 - 0.5 * np.cos(np.pi * k / self.n) if self.n else k

    def process(self, chunk: bytes) -> bytes:
        if self.done >= self.n and not self._carry:
            return chunk                          # past the ramp: pass through untouched
        b = self._carry + chunk
        whole = len(b) - (len(b) % 2)
        self._carry = b[whole:]
        if not whole:
            return b''
        x = np.frombuffer(b[:whole], dtype='<i2')
        if self.done < self.n:
            m = min(self.n - self.done, len(x))
            x = x.copy()
            g = self._ramp[self.done:self.done + m]
            x[:m] = np.round(x[:m].astype(np.float64) * g).astype('<i2')
        self.done += len(x)
        return x.tobytes()

    def flush(self) -> bytes:
        """A dangling half sample cannot be played; drop it (the stream was malformed)."""
        self._carry = b''
        return b''


async def fade_in_pcm16(gen, n):
    """Async wrapper: fade in the first `n` samples of a stream of int16 PCM chunks."""
    f = FadeIn(n)
    async for chunk in gen:
        out = f.process(chunk)
        if out:
            yield out
