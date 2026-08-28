"""Cross-request speech context for `/v1/audio/speech` (the `request_id` field).

Why this exists
---------------
Streaming agents (LiveKit's TTS StreamAdapter, for one) do not hand the TTS one
utterance at a time: they cut the LLM's text into sentence-ish chunks of a few words
and synthesize each with its own request. Every chunk therefore starts the LM cold --
no memory of how the previous one sounded -- so pitch register, pace and energy reset at
each join and the result sounds stitched, even though the model itself handles long
passages fine when it gets them in one prompt.

The fix is to give the LM that memory. A request carrying a `request_id` that has
previous turns behind it is prompted with those turns *in the model's own multi-turn
format* -- the same one `/v1/audio/vc` uses to prime a reference voice::

    <|im_start|>husein: hello my name is husein,<|speech_start|><|s_..|>...<|im_end|>
    <|im_start|>husein: i like to eat chicken rice.<|speech_start|>

i.e. text + the speech tokens the LM actually produced for it, then the new text. The LM
continues in the prosodic state it left off in instead of restarting; only the new
turn's tokens are decoded and streamed. When a turn finishes generating cleanly, its
(text, tokens) pair is appended to the context for the next chunk.

Left trim
---------
A conversation can run for minutes while the LM's window is fixed (`LM_MAX_MODEL_LEN`,
4096 tokens; speech tokens are 50/s, so ~80 s end to end). The history is kept to at
most `CONTEXT_MAX_S` seconds of speech tokens, trimmed from the left: whole oldest
turns are dropped first, then -- so a single long previous utterance still contributes
its tail -- the oldest kept turn may be cut to its last N tokens with its text shortened
in proportion (a rough alignment, but speech rate within one utterance is close enough
to uniform that the tail text matches the tail audio to within a second or so). What
matters for continuity is the most recent few seconds, which is exactly what survives.
`fit_context()` then makes sure prompt + generation fit in the LM window, clamping the
request's `max_tokens` down before ever cutting context below `CONTEXT_MIN_GEN_TOKENS`
of generation room.

Sharing across workers
----------------------
The API runs as N uvicorn worker processes on one box (CLAUDE.md: `--workers N` + MPS),
and consecutive chunks of one conversation land on arbitrary workers. Rather than
electing a leader worker to hold the history (an extra server, an election, a
single point of loss when uvicorn recycles that worker), `FileContextStore` keeps one
small JSON file per context id in a directory every worker can see -- `/dev/shm` by
default on Linux, so it is RAM -- with atomic replace-on-write and one directory-wide
flock around the read-modify-write append. A file is ~1-6 KB, a read or write is tens of
microseconds, and there is nothing to elect or restart. Entries expire `CONTEXT_TTL_S`
after their last update and are swept opportunistically. `MemoryContextStore` is the
same thing in-process, for a single worker or tests.

Ordering: a turn is committed *before* the LM reader hands the stitcher its end-of-stream
marker (see `stream_speech` in app/main.py), so by the time a client has received the
whole response for chunk N, chunk N is in the store for chunk N+1 -- no matter which
worker either one hit. If a client does overlap requests on one id, turns are kept
sorted by request arrival time, so the order in the prompt still matches the text order.

This module imports no torch / app code so `tests/test_context.py` runs anywhere.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
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
# Fallback guard (RequestContext.fallback_hold_tokens): with history in the prompt the LM
# sometimes decides the utterance is already over and emits end-of-speech after 0-8
# tokens (H20 A/B: 13/40 `continue` chunks, 3/40 `turns` chunks; 0/40 without context).
# The LM reader holds the first tokens back until at least this many have arrived --
# 4 per word of text, floored at 0.2 s and capped at 1 s so it always stays under the
# first decode window (no added latency) -- and if the LM stops before that, the chunk
# is regenerated without context. A real rendering runs ~15-20 tokens per word.
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

    Context is meant to carry one speaker's prosody forward; priming the LM with another
    voice's tokens is voice conversion, not continuity. A voice switch on the same id
    therefore starts cold and the older voice's turns age out of the store.
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


def fit_context(
    turns: list[Turn],
    text: str,
    max_tokens: int,
    max_model_len: int,
    context_max_tokens: int,
    min_gen_tokens: int,
) -> tuple[list[Turn], int]:
    """Choose the context that fits the LM window and the `max_tokens` to send with it.

    Returns (turns, max_tokens). The context budget is `context_max_tokens` speech
    tokens, reduced only as far as needed to leave `min(max_tokens, min_gen_tokens)`
    tokens of generation room; `max_tokens` is then clamped to whatever room is left
    after the (over-)estimated prompt, so vLLM never sees prompt + max_tokens beyond
    `max_model_len` (which it rejects with a 400 rather than truncating).
    """
    max_tokens = max(1, int(max_tokens))
    gen_floor = min(max_tokens, max(1, int(min_gen_tokens)))
    new_turn = estimate_text_tokens(text) + NEW_TURN_SPECIAL_TOKENS
    budget = min(context_max_tokens, max_model_len - new_turn - gen_floor - LM_WINDOW_MARGIN)
    turns = trim_turns(turns, budget)
    # the trim budgets speech tokens; the turns' text costs LM tokens too, so drop
    # oldest turns until the full prompt estimate leaves the generation floor.
    while turns and estimate_prompt_tokens(turns, text) + gen_floor + LM_WINDOW_MARGIN > max_model_len:
        turns = turns[1:]
    room = max_model_len - estimate_prompt_tokens(turns, text) - LM_WINDOW_MARGIN
    return turns, max(1, min(max_tokens, room))


def tokens_to_str(tokens: list[int]) -> str:
    return ''.join(f'<|s_{i}|>' for i in tokens)


CONTEXT_MODES = ('turns', 'continue')


def join_texts(texts: list[str]) -> str:
    """Previous chunk texts as one running utterance (continue mode).

    The rule normalizer closes every chunk with '.', so a chunk that ended mid-sentence
    ("hello my name is husein,") was prompted as "hello my name is husein,." -- fine as
    a standalone turn, but inside one utterance that reads as a full stop. Drop a '.'
    that directly follows other punctuation.
    """
    s = ' '.join(t.strip() for t in texts if t and t.strip())
    return re.sub(r'([,;:!?\u2026])\.(?=\s|$)', r'\1', s)


def build_prompt(turns: list[Turn], voice: str, text: str, mode: str = 'turns') -> str:
    """The prompt for a request with `turns` of context.

    `turns` (default): each context turn is a closed <|im_start|>...<|im_end|> block --
    the format `/v1/audio/vc` primes a reference voice with -- then the new turn is
    opened at <|speech_start|> for the LM to fill. The LM gets the previous speech as
    conditioning but starts a *new utterance*.

    `continue`: ONE turn whose text is every previous chunk plus the new text, with the
    previous chunks' speech tokens already in place after <|speech_start|>. The LM is
    resumed mid-utterance -- exactly the state it is in while generating a long text --
    and emits the tokens for the remaining text. The previous tokens are the prefix, so
    only the new speech is generated and decoded, as in turns mode.

    With no turns both modes are exactly the plain single-turn TTS prompt.
    """
    if mode not in CONTEXT_MODES:
        raise ValueError(f'context mode {mode!r}: expected one of {CONTEXT_MODES}')
    if not turns:
        return f'<|im_start|>{voice}: {text}<|speech_start|>'
    if mode == 'continue':
        prev_text = join_texts([t.text for t in turns])
        prev_tokens = ''.join(tokens_to_str(t.tokens) for t in turns)
        return f'<|im_start|>{voice}: {prev_text} {text}<|speech_start|>{prev_tokens}'
    parts = [
        f'<|im_start|>{t.voice}: {t.text}<|speech_start|>{tokens_to_str(t.tokens)}<|im_end|>'
        for t in turns
    ]
    parts.append(f'<|im_start|>{voice}: {text}<|speech_start|>')
    return ''.join(parts)


# --------------------------------------------------------------------------- stores

class ContextStore:
    """Per-context-id history of finished turns, trimmed to `max_tokens` speech tokens
    on every append and expiring `ttl_s` after the last append. All methods are
    synchronous and cheap (microseconds) by design: `append` runs inside the LM
    reader's finally-block, ahead of the end-of-stream marker, so the store is
    consistent before the response can complete."""

    def __init__(self, ttl_s: float, max_tokens: int):
        self.ttl_s = float(ttl_s)
        self.max_tokens = int(max_tokens)

    def get(self, key: str) -> list[Turn]:
        raise NotImplementedError

    def append(self, key: str, turn: Turn) -> list[Turn]:
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        raise NotImplementedError

    def _merge(self, turns: list[Turn], turn: Turn) -> list[Turn]:
        turns = turns + [turn]
        turns.sort(key=lambda t: t.ts)        # overlapping requests: keep text order
        return trim_turns(turns, self.max_tokens)


class MemoryContextStore(ContextStore):
    """In-process store: one worker, or tests."""

    def __init__(self, ttl_s: float, max_tokens: int):
        super().__init__(ttl_s, max_tokens)
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


class FileContextStore(ContextStore):
    """Shared across worker processes on one host through a directory of JSON files.

    One file per context id (name = sha1 of the id, so any id string is safe), written
    to a temp name and `os.replace`d into place so readers never see a torn file, and
    one directory-wide `flock` (on a lock file that is never deleted, so every process
    locks the same inode) around the read-modify-write of `append`. Reads take no lock.
    Stale files (mtime older than the TTL) are swept at most every TTL/4 per process.
    """

    LOCK_NAME = '.lock'

    def __init__(self, directory: str, ttl_s: float, max_tokens: int):
        super().__init__(ttl_s, max_tokens)
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
            logging.warning(f'context store: unreadable {path}: {e}')
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
            logging.warning(f'context store: sweep failed: {e}')


def default_store_dir() -> str:
    """RAM-backed and shared by every process on the host when available."""
    if os.path.isdir('/dev/shm'):
        return '/dev/shm/tts-context'
    import tempfile
    return os.path.join(tempfile.gettempdir(), 'tts-context')


def make_store(kind: str, directory: str, ttl_s: float, max_tokens: int) -> Optional[ContextStore]:
    """`file` (default; shared across workers), `memory` (single process), `off`."""
    kind = (kind or 'file').strip().lower()
    if kind in ('off', 'none', 'false', ''):
        return None
    if kind == 'memory':
        return MemoryContextStore(ttl_s, max_tokens)
    if kind == 'file':
        return FileContextStore(directory or default_store_dir(), ttl_s, max_tokens)
    raise ValueError(f'CONTEXT_STORE={kind!r}: expected file, memory or off')


@dataclass
class RequestContext:
    """What one TTS request needs to read and later commit its context.

    Built in the endpoint (prompt side), carried into `stream_speech`, and consumed
    by the LM reader when generation finishes cleanly: `commit()` appends the new
    turn. `turns` is the history that went into the prompt (already trimmed/filtered);
    `tokens` its speech-token count, for headers/spans/logs.
    """
    store: ContextStore
    key: str
    voice: str
    text: str
    turns: list[Turn] = field(default_factory=list)
    ts: float = field(default_factory=time.time)
    mode: str = 'turns'

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
        """This request without any history -- what it would have been with no request_id."""
        return build_prompt([], self.voice, self.text)

    def commit(self, token_ids: list[int]) -> Optional[list[Turn]]:
        if not token_ids:
            return None
        try:
            return self.store.append(
                self.key, Turn(text=self.text, tokens=list(token_ids), voice=self.voice, ts=self.ts)
            )
        except Exception as e:  # a lost context must never fail the request
            logging.warning(f'context store: commit for {self.key!r} failed: {e}')
            return None

    def headers(self) -> dict:
        # header values must be latin-1 clean (Starlette would 500 encoding them);
        # ids are arbitrary client strings, so anything else is percent-encoded.
        key = self.key
        if not (key.isascii() and key.isprintable()):
            from urllib.parse import quote
            key = quote(key, safe='')
        return {
            'X-Context-Id': key,
            'X-Context-Mode': self.mode,
            'X-Context-Turns': str(len(self.turns)),
            'X-Context-Tokens': str(self.tokens),
        }
