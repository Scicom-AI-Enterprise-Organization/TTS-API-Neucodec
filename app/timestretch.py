"""Streaming pitch-preserving time stretch (speaking rate) for the PCM output stream.

The LM has no speaking-rate control and the codec's speech tokens are a fixed 50 Hz, so
rate is changed after decode, in the audio domain. Plain resampling would shorten every
pitch period along with the utterance (chipmunk / slowed-tape). Instead this is WSOLA
(waveform-similarity overlap-add, the SoundTouch-style tempo change): the waveform is
copied to the output in blocks of ``seq_ms``; each block advances ``rate`` times further
through the input than through the output, so whole pitch periods are dropped (rate > 1,
faster) or repeated (rate < 1, slower) while the samples inside a block are untouched --
pitch and timbre stay put, only duration changes. At every block joint the next block's
start is searched within +/-``seek_ms/2`` of its nominal position for the offset whose
waveform best matches the tail of the previous block (normalised cross-correlation, with
a mild preference for the nominal position), and the two are crossfaded over ``overlap_ms``
so the periods line up and the join is continuous.

Stateful and streaming: feed ``process(chunk)`` any chunk sizes, ``flush()`` at the end.
Output for a given input is identical regardless of how it was chunked (each block only
looks at input inside its own search window), and needs only ``seq_ms + seek_ms`` (~55 ms)
of lookahead. Rate 1.0 is an exact identity (the nominal position always wins the search),
so callers may bypass it or not.

Pure numpy, no torch: it runs on the event-loop side of the serving process on already
decoded audio. Cost is one (seek x overlap) matvec per block, ~1-2 ms per 2 s chunk.
"""

import numpy as np

# Defaults are SoundTouch's speech settings: a 40 ms block keeps several pitch periods so
# a period can be dropped/repeated without a change of timbre, the +/-7.5 ms search covers
# at least one period down to ~65 Hz, and an 8 ms crossfade hides the join.
DEFAULT_SEQ_MS = 40.0
DEFAULT_SEEK_MS = 15.0
DEFAULT_OVERLAP_MS = 8.0

# Practical WSOLA range for speech: below 0.5 the repeated periods start to buzz, above 2.0
# consonants are dropped wholesale. Enforced on the request models in app/main.py.
MIN_RATE = 0.5
MAX_RATE = 2.0


class WSOLA:
    """Streaming WSOLA time stretcher. ``rate`` > 1 shortens (speaks faster)."""

    def __init__(self, rate, sr, seq_ms=DEFAULT_SEQ_MS, seek_ms=DEFAULT_SEEK_MS,
                 overlap_ms=DEFAULT_OVERLAP_MS):
        rate = float(rate)
        if not rate > 0:
            raise ValueError(f'rate must be > 0, got {rate}')
        self.rate = rate
        self.sr = int(sr)
        self.ov = max(2, int(self.sr * overlap_ms / 1000.0))            # crossfade length
        self.seq = max(2 * self.ov + 1, int(self.sr * seq_ms / 1000.0))  # block length
        self.seek = max(1, int(self.sr * seek_ms / 1000.0))             # search span
        self.half = self.seek // 2
        # every block appends hop_out samples to the output (its first `ov` are blended
        # with the previous block's held-back tail) and moves the nominal input position
        # by hop_in = hop_out * rate. The search offset is relative to the nominal
        # position each time, so the ratio never drifts.
        self.hop_out = self.seq - self.ov
        self.hop_in = self.hop_out * self.rate
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, self.ov)))
        self.up = ramp.astype(np.float32)
        self.down = (1.0 - ramp).astype(np.float32)

        self.buf = np.zeros(0, dtype=np.float32)   # unconsumed input
        self.buf_start = 0                          # absolute input index of buf[0]
        self.n_in = 0                               # input samples received
        self.n_out = 0                              # output samples returned
        self.k = 0                                  # blocks placed
        self.mid = None                             # held-back tail (ov samples) of the last block
        self.eps = 1e-12

    def process(self, x):
        """Feed a chunk of float samples; returns the stretched samples now available."""
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if len(x):
            self.n_in += len(x)
            self.buf = np.concatenate([self.buf, x]) if len(self.buf) else x.copy()
        y, _ = self._run(limit=None)
        self.n_out += len(y)
        return y

    def flush(self):
        """End of stream: stretch the buffered tail. Call once; the instance is spent."""
        limit = self.n_in
        if limit == 0:
            return np.zeros(0, dtype=np.float32)
        # zero-pad so every block whose search window touches real audio can be placed,
        # then cut the result exactly where the last real input sample lands.
        pad = np.zeros(self.seq + self.seek, dtype=np.float32)
        self.buf = np.concatenate([self.buf, pad]) if len(self.buf) else pad
        y, cut = self._run(limit=limit)
        if self.mid is not None:
            # the last block's held-back tail is its natural continuation
            y = np.concatenate([y, self.mid])
            self.mid = None
        if cut is not None:
            y = y[:cut]
        self.buf = np.zeros(0, dtype=np.float32)
        self.n_out += len(y)
        return y

    # -- internals ----------------------------------------------------------------------

    def _run(self, limit):
        """Place every block whose search window is fully buffered; returns the
        concatenated new output.

        ``limit`` (flush only) is the absolute index one past the last real input sample:
        blocks are then restricted to start inside the real audio, and the returned ``cut``
        is where the real audio ends in (this output + the held-back tail).
        """
        out = []
        n = 0
        cut = None
        buf_end = self.buf_start + len(self.buf)
        while True:
            c = int(round(self.k * self.hop_in))   # nominal start of this block
            if self.mid is None:
                lo = hi = c                          # first block: nothing to match yet
            else:
                lo = max(0, c - self.half)
                hi = max(lo, c + (self.seek - self.half) - 1)
            if limit is not None:
                if lo >= limit:
                    break
                hi = min(hi, limit - 1)
            if hi + self.seq > buf_end:
                break                                # need more input
            s = self._seek(lo, hi, c) if hi > lo else lo
            i = s - self.buf_start
            blk = self.buf[i: i + self.seq]
            if self.mid is None:
                out.append(blk[: self.hop_out])
            else:
                out.append(self.mid * self.down + blk[: self.ov] * self.up)
                out.append(blk[self.ov: self.hop_out])
            self.mid = blk[self.hop_out:].copy()
            if limit is not None:
                cut = n + (limit - s)                # input s+j lands at output n+j
            n += self.hop_out
            self.k += 1
            # drop input no later search window can reach. Clamped to what has actually
            # arrived: at high rates the next window starts beyond the buffered data, and
            # advancing buf_start past n_in would misplace the next appended chunk.
            keep_from = min(max(0, int(round(self.k * self.hop_in)) - self.half), buf_end)
            drop = keep_from - self.buf_start
            if drop > 0:
                self.buf = self.buf[drop:]
                self.buf_start = keep_from
                buf_end = self.buf_start + len(self.buf)
        if not out:
            return np.zeros(0, dtype=np.float32), cut
        return np.concatenate(out), cut

    def _seek(self, lo, hi, c):
        """Best block start in [lo, hi] (absolute, nominal ``c``): the candidate whose first
        ``ov`` samples best match the held-back tail of the previous block."""
        i0 = lo - self.buf_start
        n = hi - lo + 1
        win = np.lib.stride_tricks.sliding_window_view(
            self.buf[i0: i0 + n + self.ov - 1], self.ov
        )                                            # (n, ov) candidate heads
        corr = win @ self.mid
        norm = np.sqrt(np.einsum('ij,ij->i', win, win) + self.eps)
        # centre preference (SoundTouch's heuristic): (corr + 0.1) * w with w = 1 at the
        # nominal position and 0.75 at the window edges. Breaks ties between equally good
        # periodic matches toward the nominal position and keeps silence from wandering.
        t = (2.0 * (np.arange(lo, hi + 1) - c)) / self.seek
        score = (corr / norm + 0.1) * (1.0 - 0.25 * t * t)
        return lo + int(np.argmax(score))


def float_to_pcm16(y):
    if len(y) == 0:
        return b''
    return np.rint(np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


def pcm16_to_float(chunk):
    return np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767.0
