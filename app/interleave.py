"""Interleaved generation for `/v1/audio/speech` (the `interleave_id` request field).

Why this exists
---------------
Streaming agents do not hand the TTS one utterance at a time. LiveKit's `openai.TTS`
plugin is a non-streaming TTS, so `livekit-agents` wraps it in a `StreamAdapter` that
cuts the LLM's reply into sentence-ish chunks (often 5-6 words) and synthesizes each
with its **own** `/v1/audio/speech` request, strictly one after the other. The request
at T+1 has no idea it continues T: it starts the LM cold, so pitch register, pace and
energy reset at every join and the reply sounds stitched together -- even though the
same model renders the whole paragraph naturally when it gets it in one prompt.

The LM was trained for exactly this -- but only OUR LMs are. `--interleave_style full`
is an in-house GPUPlatform packing; no open-source TTS LM has it, and the checkpoints
that do are private model repos. So this module's benefit is a property of the served
model, not of the prompt shape: verified on a private interleave-trained checkpoint
(bench/INTERLEAVE_AB.md -- chunk N->N+1 register/level jump down ~25%, no latency cost),
while on a model packed without it the same prompt bought nothing and collapsed 7.5% of
chunks. Serving such a model, `INTERLEAVE_STORE=off` is the right setting.

Packed with *interleaved* documents (GPUPlatform `pack_stage1.py --interleave_style
full`), consecutive chunks of one recording are written as consecutive turns inside a
single attention document::

    <|im_start|>{spk}: {text_0}<|speech_start|>{audio_0}<|im_end|>
    <|im_start|>{spk}: {text_1}<|speech_start|>{audio_1}<|im_end|> ...

That is what teaches long-form style, and it is a prompt shape, not just a training
one: chunk 1 is generated *with chunk 0's text and speech tokens in the prompt*. So a
request carrying an `interleave_id` that has previous turns behind it is prompted::

    <|im_start|>husein: hello my name is husein,<|speech_start|><|s_1834|>...<|im_end|>
    <|im_start|>husein: i like to eat chicken rice.<|speech_start|>
                                                                  ^ the LM generates here

- History turns carry the **normalized text that was actually prompted** and the
  **speech tokens the LM actually produced** -- its own previous output, not re-encoded
  audio.
- Only the new turn is generated, decoded and streamed; the history is prefill only.
  ~1000 tokens of prefill on a ~12.8k tok/s LM is a few tens of milliseconds, and the
  codec path (the real bottleneck) is untouched.
- A request with no id, or an id with no history yet, builds byte-for-byte the plain
  single-turn prompt this API has always sent.

How much is retained
--------------------
`MAX_RETAIN_INTERLEAVE` (default 5) turns, and `INTERLEAVE_MAX_S` seconds of speech
tokens, whichever binds first. Five sentence-sized turns is ~10-15 s of speech: enough
for the LM to hold a register, far short of the 4096-token window. The turn cap is
applied at both ends (the store never keeps more, a request may prompt with fewer via
`max_retain_interleave`); the second cap is what keeps a *long* previous `say()` from
eating the window on its own -- it is trimmed from the left, whole oldest turns first,
then the oldest surviving turn cut to its tail with its text shortened in proportion (a
rough alignment, but speech rate inside one utterance is uniform enough that the tail
text matches the tail audio to within a second). `fit_interleave()` then makes sure
prompt + generation fit the LM window, clamping the request's `max_tokens` before ever
cutting history below `INTERLEAVE_MIN_GEN_TOKENS` of generation room.

Sharing across workers
----------------------
The API runs as N uvicorn worker processes on one box (CLAUDE.md: `--workers N` + MPS),
and consecutive chunks of one conversation land on arbitrary workers. Rather than
electing a leader worker to hold the history (an extra server, an election, and a single
point of loss when uvicorn recycles that worker), `FileInterleaveStore` keeps one small
JSON file per id in a directory every worker can see -- `/dev/shm` by default on Linux,
so it is RAM -- with atomic replace-on-write and one directory-wide flock around the
read-modify-write of `append`. A file is ~1-6 KB, a read or write is tens of
microseconds, and there is nothing to elect or restart. Entries expire
`INTERLEAVE_TTL_S` after their last update and are swept opportunistically.
`MemoryInterleaveStore` is the same thing in-process, for a single worker or tests.

Ordering: a turn is committed *before* the LM reader hands the stitcher its
end-of-stream marker (see `stream_speech` in app/main.py), so by the time a client has
received the whole response for chunk N, chunk N is in the store for chunk N+1 -- no
matter which worker either one hits. If a client does overlap requests on one id, turns
are kept sorted by request arrival time, so the prompt still lists them in text order.

This module imports no torch / app code so `tests/test_interleave.py` runs anywhere.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - windows
    fcntl = None

TOKENS_PER_SECOND = 50            # NeuCodec frame rate; one speech token = 20 ms of audio
# Prompt-size estimate for the LM window check. Every <|s_N|> and every special token is
# exactly one LM token; plain text is estimated at one token per character, which is
# right for CJK and a 3-4x overestimate for Latin script. Overestimating only clamps
# `max_tokens` a little harder (from a default of 3072 no request ever reaches), while
# underestimating would make vLLM reject the request with a 400 -- so err high.
TEXT_TOKENS_PER_CHAR = 1.0
TURN_SPECIAL_TOKENS = 3           # <|im_start|> <|speech_start|> <|im_end|>
NEW_TURN_SPECIAL_TOKENS = 2       # <|im_start|> <|speech_start|>
# Never keep a partial (left-cut) turn shorter than this: below ~1 s the text/audio
# proportion heuristic is meaningless and the fragment gives the LM nothing.
MIN_PARTIAL_TOKENS = TOKENS_PER_SECOND
LM_WINDOW_MARGIN = 16             # slack on the LM-window estimate (BOS etc.)
# Fallback guard (RequestInterleave.fallback_hold_tokens): with history in the prompt the
# LM can decide the utterance is already over and emit end-of-speech after 0-8 tokens --
# the previous turn ended in a sentence-final fall and silence, because it was generated
# as a complete utterance. Measured on tm-h20 before the interleave finetune: 3/40 chunks.
# The LM reader holds the first tokens back until at least this many have arrived -- 4 per
# word of text, floored at 0.2 s and capped at 1 s, i.e. about what the first decode window
# needs anyway, so the wait it can add is a few ms at 530 LM tok/s -- and if the LM stops
# before that many, the chunk is regenerated with no history. A real rendering runs ~15-20
# tokens per word, so this only trips on the collapse.
FALLBACK_TOKENS_PER_WORD = 4
FALLBACK_MIN_TOKENS = 10
FALLBACK_MAX_TOKENS = TOKENS_PER_SECOND


@dataclass
class Turn:
    """One finished utterance: the normalized text that was prompted and the speech
    tokens the LM produced for it."""
    text: str
    tokens: list[int]
    voice: str
    ts: float = 0.0               # request arrival time; keeps turns in text order

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "Turn":
        return cls(
            text=str(d.get('text', '')),
            tokens=[int(t) for t in d.get('tokens', [])],
            voice=str(d.get('voice', '')),
            ts=float(d.get('ts', 0.0)),
        )


def seconds_to_tokens(seconds: float) -> int:
    return max(0, int(round(seconds * TOKENS_PER_SECOND)))


def total_tokens(turns: list[Turn]) -> int:
    return sum(len(t.tokens) for t in turns)


def retain_last(turns: list[Turn], max_retain: int) -> list[Turn]:
    """The last `max_retain` turns (<=0 means "no limit", not "nothing").

    This is the `max_retain_interleave` cap, applied to the store on every append and
    again per request. Newest turns win: continuity comes from the seconds just before
    the new chunk, and the interleaved documents the LM trained on are consecutive.
    """
    if max_retain is None or max_retain <= 0:
        return list(turns)
    return list(turns[-max_retain:])


def trim_turn_tail(turn: Turn, keep: int) -> Turn:
    """Keep the last `keep` speech tokens of a turn and the matching tail of its text.

    Text is cut in the same proportion as the tokens, on a word boundary when the text
    has words (Latin/Malay), by character otherwise (CJK without spaces). It is a
    heuristic alignment, good to roughly a second; see the module docstring.
    """
    n = len(turn.tokens)
    if keep >= n:
        return turn
    frac = keep / n
    words = turn.text.split()
    if len(words) > 1:
        k = max(1, int(round(len(words) * frac)))
        text = ' '.join(words[-k:])
    else:
        k = max(1, int(round(len(turn.text) * frac)))
        text = turn.text[-k:]
    return Turn(text=text, tokens=list(turn.tokens[-keep:]), voice=turn.voice, ts=turn.ts)


def trim_turns(turns: list[Turn], max_tokens: int) -> list[Turn]:
    """Left-trim to at most `max_tokens` speech tokens.

    Newest turns win. Whole turns are dropped from the left; the oldest turn that still
    partly fits is cut to its tail (if >= MIN_PARTIAL_TOKENS of room remain) so a
    single long utterance still contributes its ending. Empty turns are dropped.
    """
    if max_tokens <= 0:
        return []
    kept: list[Turn] = []
    total = 0
    for turn in reversed(turns):
        n = len(turn.tokens)
        if n == 0:
            continue
        if total + n <= max_tokens:
            kept.append(turn)
            total += n
            continue
        room = max_tokens - total
        if room >= MIN_PARTIAL_TOKENS:
            kept.append(trim_turn_tail(turn, room))
        break
    kept.reverse()
    return kept


def select_voice(turns: list[Turn], voice: str) -> list[Turn]:
    """The trailing run of turns spoken by `voice`.

    Interleaving is meant to carry one speaker's prosody forward; priming the LM with
    another voice's tokens is voice conversion, not continuity. A voice switch on the
    same id therefore starts cold and the older voice's turns age out of the store.
    """
    out: list[Turn] = []
    for turn in reversed(turns):
        if turn.voice != voice:
            break
        out.append(turn)
    out.reverse()
    return out


def estimate_text_tokens(text: str) -> int:
    return int(len(text) * TEXT_TOKENS_PER_CHAR) + 1


def estimate_turn_tokens(turn: Turn) -> int:
    return len(turn.tokens) + estimate_text_tokens(turn.text) + TURN_SPECIAL_TOKENS


def estimate_prompt_tokens(turns: list[Turn], text: str) -> int:
    return (
        sum(estimate_turn_tokens(t) for t in turns)
        + estimate_text_tokens(text)
        + NEW_TURN_SPECIAL_TOKENS
    )


def fit_interleave(
    turns: list[Turn],
    text: str,
    max_tokens: int,
    max_model_len: int,
    retain_max_tokens: int,
    min_gen_tokens: int,
    max_retain: int = 0,
) -> tuple[list[Turn], int]:
    """Choose the history that fits the LM window and the `max_tokens` to send with it.

    Returns (turns, max_tokens). At most `max_retain` turns are considered; their speech
    tokens are then budgeted to `retain_max_tokens`, reduced only as far as needed to
    leave `min(max_tokens, min_gen_tokens)` tokens of generation room. `max_tokens` is
    finally clamped to whatever room the (over-)estimated prompt leaves, so vLLM never
    sees prompt + max_tokens beyond `max_model_len` -- which it rejects with a 400
    rather than truncating.
    """
    max_tokens = max(1, int(max_tokens))
    turns = retain_last(turns, max_retain)
    gen_floor = min(max_tokens, max(1, int(min_gen_tokens)))
    new_turn = estimate_text_tokens(text) + NEW_TURN_SPECIAL_TOKENS
    budget = min(retain_max_tokens, max_model_len - new_turn - gen_floor - LM_WINDOW_MARGIN)
    turns = trim_turns(turns, budget)
    # the trim budgets speech tokens; the turns' text costs LM tokens too, so drop
    # oldest turns until the full prompt estimate leaves the generation floor.
    while turns and estimate_prompt_tokens(turns, text) + gen_floor + LM_WINDOW_MARGIN > max_model_len:
        turns = turns[1:]
    room = max_model_len - estimate_prompt_tokens(turns, text) - LM_WINDOW_MARGIN
    return turns, max(1, min(max_tokens, room))


def tokens_to_str(tokens: list[int]) -> str:
    return ''.join(f'<|s_{i}|>' for i in tokens)


def build_prompt(turns: list[Turn], voice: str, text: str) -> str:
    """The interleaved prompt for a request with `turns` of history.

    Each history turn is a closed `<|im_start|>...<|im_end|>` block carrying its text and
    the speech tokens generated for it, then the new turn is opened at `<|speech_start|>`
    for the LM to fill -- the exact document shape the model was packed with
    (`--interleave_style full`), and the same multi-turn format `/v1/audio/vc` uses to
    prime a reference voice.

    With no turns this is the plain single-turn TTS prompt, character for character.
    """
    if not turns:
        return f'<|im_start|>{voice}: {text}<|speech_start|>'
    parts = [
        f'<|im_start|>{t.voice}: {t.text}<|speech_start|>{tokens_to_str(t.tokens)}<|im_end|>'
        for t in turns
    ]
    parts.append(f'<|im_start|>{voice}: {text}<|speech_start|>')
    return ''.join(parts)


# --------------------------------------------------------------------------- stores

class InterleaveStore:
    """Per-id history of finished turns, capped to `max_retain` turns and `max_tokens`
    speech tokens on every append, expiring `ttl_s` after the last append. All methods
    are synchronous and cheap (microseconds) by design: `append` runs inside the LM
    reader's finally-block, ahead of the end-of-stream marker, so the store is
    consistent before the response can complete."""

    def __init__(self, ttl_s: float, max_tokens: int, max_retain: int = 0):
        self.ttl_s = float(ttl_s)
        self.max_tokens = int(max_tokens)
        self.max_retain = int(max_retain)

    def get(self, key: str) -> list[Turn]:
        raise NotImplementedError

    def append(self, key: str, turn: Turn) -> list[Turn]:
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        raise NotImplementedError

    def _merge(self, turns: list[Turn], turn: Turn) -> list[Turn]:
        turns = turns + [turn]
        turns.sort(key=lambda t: t.ts)        # overlapping requests: keep text order
        return trim_turns(retain_last(turns, self.max_retain), self.max_tokens)


class MemoryInterleaveStore(InterleaveStore):
    """In-process store: one worker, or tests."""

    def __init__(self, ttl_s: float, max_tokens: int, max_retain: int = 0):
        super().__init__(ttl_s, max_tokens, max_retain)
        self._data: dict[str, tuple[float, list[Turn]]] = {}
        self._last_sweep = 0.0

    def get(self, key: str) -> list[Turn]:
        item = self._data.get(key)
        if item is None:
            return []
        updated, turns = item
        if time.time() - updated > self.ttl_s:
            self._data.pop(key, None)
            return []
        return list(turns)

    def append(self, key: str, turn: Turn) -> list[Turn]:
        turns = self._merge(self.get(key), turn)
        self._data[key] = (time.time(), turns)
        self._sweep()
        return turns

    def delete(self, key: str) -> bool:
        return self._data.pop(key, None) is not None

    def _sweep(self):
        now = time.time()
        if now - self._last_sweep < self.ttl_s / 4:
            return
        self._last_sweep = now
        for k in [k for k, (updated, _) in self._data.items() if now - updated > self.ttl_s]:
            self._data.pop(k, None)


class FileInterleaveStore(InterleaveStore):
    """Shared across worker processes on one host through a directory of JSON files.

    One file per id (name = sha1 of the id, so any id string is safe), written to a temp
    name and `os.replace`d into place so readers never see a torn file, and one
    directory-wide `flock` (on a lock file that is never deleted, so every process locks
    the same inode) around the read-modify-write of `append`. Reads take no lock. Stale
    files (mtime older than the TTL) are swept at most every TTL/4 per process.
    """

    LOCK_NAME = '.lock'

    def __init__(self, directory: str, ttl_s: float, max_tokens: int, max_retain: int = 0):
        super().__init__(ttl_s, max_tokens, max_retain)
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self._lock_path = os.path.join(directory, self.LOCK_NAME)
        self._last_sweep = 0.0

    def _path(self, key: str) -> str:
        return os.path.join(self.directory, hashlib.sha1(key.encode('utf-8')).hexdigest() + '.json')

    def _read(self, path: str) -> list[Turn]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                doc = json.load(f)
        except FileNotFoundError:
            return []
        except (OSError, ValueError) as e:
            logging.warning(f'interleave store: unreadable {path}: {e}')
            return []
        if time.time() - float(doc.get('updated', 0.0)) > self.ttl_s:
            return []
        return [Turn.from_json(t) for t in doc.get('turns', [])]

    def _write(self, path: str, turns: list[Turn]):
        tmp = f'{path}.{os.getpid()}.tmp'
        doc = {'updated': time.time(), 'turns': [t.to_json() for t in turns]}
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(doc, f, separators=(',', ':'))
        os.replace(tmp, path)

    class _Locked:
        def __init__(self, lock_path: str):
            self.lock_path = lock_path
            self.fh = None

        def __enter__(self):
            self.fh = open(self.lock_path, 'a')
            if fcntl is not None:
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX)
            return self

        def __exit__(self, *exc):
            try:
                if fcntl is not None:
                    fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
            finally:
                self.fh.close()
            return False

    def get(self, key: str) -> list[Turn]:
        return self._read(self._path(key))

    def append(self, key: str, turn: Turn) -> list[Turn]:
        path = self._path(key)
        os.makedirs(self.directory, exist_ok=True)   # survives a tmp cleaner removing it
        with self._Locked(self._lock_path):
            turns = self._merge(self._read(path), turn)
            self._write(path, turns)
        self._sweep()
        return turns

    def delete(self, key: str) -> bool:
        try:
            os.remove(self._path(key))
            return True
        except FileNotFoundError:
            return False

    def _sweep(self):
        now = time.time()
        if now - self._last_sweep < self.ttl_s / 4:
            return
        self._last_sweep = now
        try:
            with os.scandir(self.directory) as it:
                for entry in it:
                    if entry.name.startswith('.'):
                        continue          # the lock file
                    try:
                        if entry.is_file() and now - entry.stat().st_mtime > self.ttl_s:
                            os.remove(entry.path)
                    except OSError:
                        pass              # raced another worker's sweep
        except OSError as e:
            logging.warning(f'interleave store: sweep failed: {e}')


def default_store_dir() -> str:
    """RAM-backed and shared by every process on the host when available."""
    if os.path.isdir('/dev/shm'):
        return '/dev/shm/tts-interleave'
    import tempfile
    return os.path.join(tempfile.gettempdir(), 'tts-interleave')


def make_store(kind: str, directory: str, ttl_s: float, max_tokens: int,
               max_retain: int = 0) -> Optional[InterleaveStore]:
    """`file` (default; shared across workers), `memory` (single process), `off`."""
    kind = (kind or 'file').strip().lower()
    if kind in ('off', 'none', 'false', ''):
        return None
    if kind == 'memory':
        return MemoryInterleaveStore(ttl_s, max_tokens, max_retain)
    if kind == 'file':
        return FileInterleaveStore(directory or default_store_dir(), ttl_s, max_tokens, max_retain)
    raise ValueError(f'INTERLEAVE_STORE={kind!r}: expected file, memory or off')


@dataclass
class RequestInterleave:
    """What one TTS request needs to read and later commit its interleave history.

    Built in the endpoint (prompt side), carried into `stream_speech`, and consumed by
    the LM reader when generation finishes cleanly: `commit()` appends the new turn.
    `turns` is the history that went into the prompt (already filtered/capped/trimmed);
    `tokens` its speech-token count, for headers/spans/logs.
    """
    store: InterleaveStore
    key: str
    voice: str
    text: str
    turns: list[Turn] = field(default_factory=list)
    ts: float = field(default_factory=time.time)
    max_retain: int = 0

    @property
    def tokens(self) -> int:
        return total_tokens(self.turns)

    def fallback_hold_tokens(self) -> int:
        """Speech tokens the LM must produce for this text before its output is trusted
        (see FALLBACK_TOKENS_PER_WORD). 0 when there is no history in the prompt."""
        if not self.turns:
            return 0
        words = max(1, len(self.text.split()))
        return int(min(FALLBACK_MAX_TOKENS, max(FALLBACK_MIN_TOKENS, FALLBACK_TOKENS_PER_WORD * words)))

    def plain_prompt(self) -> str:
        """This request without any history -- what it would have been with no id."""
        return build_prompt([], self.voice, self.text)

    def commit(self, token_ids: list[int]) -> Optional[list[Turn]]:
        if not token_ids:
            return None
        try:
            return self.store.append(
                self.key, Turn(text=self.text, tokens=list(token_ids), voice=self.voice, ts=self.ts)
            )
        except Exception as e:  # a lost history must never fail the request
            logging.warning(f'interleave store: commit for {self.key!r} failed: {e}')
            return None

    def headers(self) -> dict:
        # header values must be latin-1 clean (Starlette would 500 encoding them);
        # ids are arbitrary client strings, so anything else is percent-encoded.
        key = self.key
        if not (key.isascii() and key.isprintable()):
            from urllib.parse import quote
            key = quote(key, safe='')
        return {
            'X-Interleave-Id': key,
            'X-Interleave-Turns': str(len(self.turns)),
            'X-Interleave-Tokens': str(self.tokens),
            'X-Interleave-Max-Retain': str(self.max_retain),
        }
