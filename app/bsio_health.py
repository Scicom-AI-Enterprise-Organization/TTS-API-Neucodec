"""
bettersentryio instrumentation: liveness and stall detection for the decode pipeline.

Why this exists, in one sentence: the process can stay up and keep answering /docs while
the decode path has stopped working, and nothing in this service can currently tell you
that. The comment in `compute_thread_fn` records the exact incident —

    "An unhandled exception here used to kill the thread outright. The process stayed up,
     compute_queue was never drained again, and every subsequent decode hung forever on
     its future with nothing logged."

That case is now caught by a try/except. The one that is still open is a **hang** rather
than a raise: `compute_stream.synchronize()` or the model call blocking forever. No
exception is raised, the thread is alive, the process is healthy, and every request times
out. This module detects that.

Entirely opt-in: with BSIO_KEY unset, nothing here starts and nothing changes.

--------------------------------------------------------------------------------
The one non-obvious design decision
--------------------------------------------------------------------------------

A naive `progress = batches_completed` does not work for this service, because the
pipeline is **demand driven**: at 3am with no traffic, `compute_queue.get()` blocks, no
batches complete, the counter sits still — and bettersentryio would report STALLED on a
perfectly healthy service.

So progress counts *the pipeline functioning*, which is either of two things:

    progress = batches_completed + idle_confirmations

An "idle confirmation" is only recorded when the pipeline is genuinely quiescent: every
queue empty **and** nothing in flight. That distinction is what makes the signal
unambiguous:

    healthy + busy   -> batches_completed advances
    healthy + idle   -> idle_confirmations advances (nothing is pending)
    WEDGED           -> in flight > 0, so completed cannot advance and idle is blocked
                        from advancing -> progress frozen -> STALLED

A CUDA hang is precisely the third row: an item taken off the queue, futures waiting, and
`synchronize()` never returning.

--------------------------------------------------------------------------------
Why one monitor per worker
--------------------------------------------------------------------------------

uvicorn runs this with `--workers 4`; each worker is a separate process with its own
queues and its own decode threads. If all four beat the same monitor name, each sends its
own progress value, the value changes on almost every beat regardless of whether anything
is wedged, and stall detection stops working entirely. Worse, one healthy worker would
keep the monitor green while the other three are dead.

So each worker claims a stable slot with flock() and beats its own monitor. flock is
released by the kernel when the process dies, so a restarted worker reclaims the same slot
instead of creating a new monitor on every deploy.
"""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import socket
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# One stdlib-only file at the repo root, next to app/ — same place Python already finds
# `app` itself from, so a plain import works with no path juggling. It lives in the repo
# rather than being installed because there is no published package yet; when there is,
# this becomes a requirements.txt line and the file goes away.
try:
    import bettersentryio
except Exception as exc:  # noqa: BLE001 - never let instrumentation break the import
    bettersentryio = None  # type: ignore[assignment]
    logger.warning("bettersentryio not importable, monitoring disabled: %s", exc)


# ---------------------------------------------------------------- configuration

BSIO_URL = os.environ.get("BSIO_URL", "")
BSIO_KEY = os.environ.get("BSIO_KEY", "")
BSIO_ENV = os.environ.get("BSIO_ENV", "production")
BSIO_MONITOR = os.environ.get("BSIO_MONITOR", "tts-api-decode")

# Beat interval, and how long the pipeline may hold work without completing any of it
# before we call it stalled. A single batch decodes in seconds, so three minutes of
# "something pending, nothing finishing" is not a slow batch — it is a hang.
BSIO_EVERY = int(os.environ.get("BSIO_EVERY", "30"))
BSIO_GRACE = int(os.environ.get("BSIO_GRACE", "30"))
BSIO_STALL_WINDOW = int(os.environ.get("BSIO_STALL_WINDOW", "180"))

# Must match --workers. Only used to size the slot range.
BSIO_WORKERS = int(os.environ.get("BSIO_WORKERS", "4"))
BSIO_SLOT_DIR = os.environ.get("BSIO_SLOT_DIR", "/tmp/bsio-slots")

enabled = bool(BSIO_KEY) and bettersentryio is not None


# ------------------------------------------------------------------- counters

class _Pipeline:
    """
    Counters for one worker process.

    A plain `+= 1` on a module global is not atomic across the decode threads, and this is
    read by the event loop while two threads write it, so it takes a lock. The lock is
    held for a few nanoseconds against work that takes seconds — it costs nothing here.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.started = 0        # batches taken off a queue
        self.completed = 0      # batches finished, successfully or not
        self.idle_confirmed = 0 # times we observed a genuinely quiescent pipeline
        self.failed = 0

    def start(self) -> None:
        with self._lock:
            self.started += 1

    def done(self, failed: bool = False) -> None:
        with self._lock:
            self.completed += 1
            if failed:
                self.failed += 1

    def confirm_idle(self) -> None:
        with self._lock:
            self.idle_confirmed += 1

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self.started - self.completed

    @property
    def progress(self) -> int:
        """What bettersentryio watches. See the module docstring for why idle counts."""
        with self._lock:
            return self.completed + self.idle_confirmed

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "started": self.started,
                "completed": self.completed,
                "failed": self.failed,
                "idle_confirmed": self.idle_confirmed,
                "in_flight": self.started - self.completed,
                "progress": self.completed + self.idle_confirmed,
            }


tts = _Pipeline()
vc = _Pipeline()


# ------------------------------------------------------------------ worker slot

_slot_handle = None  # kept for the process lifetime; closing it releases the lock


def claim_worker_slot(workers: int = BSIO_WORKERS) -> int:
    """
    Take the lowest free slot, held by flock for as long as this process lives.

    Returns -1 if no slot could be claimed, in which case we fall back to the pid so the
    worker is still visible rather than silently unmonitored.
    """
    global _slot_handle
    try:
        os.makedirs(BSIO_SLOT_DIR, exist_ok=True)
        for slot in range(max(1, workers)):
            path = os.path.join(BSIO_SLOT_DIR, f"{BSIO_MONITOR}.{slot}.lock")
            handle = open(path, "w")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                handle.close()
                continue
            handle.write(str(os.getpid()))
            handle.flush()
            _slot_handle = handle
            return slot
    except Exception as exc:  # noqa: BLE001
        logger.warning("bsio: could not claim a worker slot: %s", exc)
    return -1


# ------------------------------------------------------------------- heartbeat

_beat: Optional["bettersentryio.Beat"] = None
_monitor_names: tuple[str, str] = ("", "")


async def _heartbeat(quiescent: Callable[[], bool], vc_quiescent: Callable[[], bool]) -> None:
    """
    Beats from the event loop, not from the decode threads.

    The decode threads block in `queue.get()` when idle, so they cannot beat while the
    service is quiet — which is exactly when we still need to know the process is alive.
    The event loop is always running.
    """
    tts_monitor, vc_monitor = _monitor_names
    while True:
        try:
            # Only credit idleness when nothing is pending anywhere. This is the check
            # that keeps "quiet" from looking like "wedged".
            if tts.in_flight == 0 and quiescent():
                tts.confirm_idle()
            if vc.in_flight == 0 and vc_quiescent():
                vc.confirm_idle()

            _beat.beat(
                tts_monitor,
                progress=tts.progress,
                every=BSIO_EVERY,
                grace=BSIO_GRACE,
                stall_window=BSIO_STALL_WINDOW,
            )
            _beat.beat(
                vc_monitor,
                progress=vc.progress,
                every=BSIO_EVERY,
                grace=BSIO_GRACE,
                stall_window=BSIO_STALL_WINDOW,
            )
        except Exception:  # noqa: BLE001 - a heartbeat must never kill its own task
            logger.exception("bsio: heartbeat iteration failed")

        await asyncio.sleep(BSIO_EVERY)


def start(quiescent: Callable[[], bool], vc_quiescent: Callable[[], bool]):
    """
    Install error capture and start the heartbeat. Returns the asyncio task, or None.

    `quiescent` and `vc_quiescent` are supplied by main.py because only it can see the
    queues; passing them in keeps this module from importing main and creating a cycle.
    """
    global _beat, _monitor_names

    if not enabled:
        logger.info("bsio: BSIO_KEY not set, monitoring disabled")
        return None

    # Checked before anything else, and before the coroutine object exists. main.py runs at
    # module scope, which is inside the loop on uvicorn 0.35 and outside it on >=0.36 (see
    # the pin in requirements.txt). If someone bumps uvicorn, monitoring should print one
    # clear line — not a traceback plus "coroutine was never awaited", which reads like a
    # real fault and is a false lead during an incident.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        logger.warning(
            "bsio: no running event loop at import, monitoring disabled "
            "(uvicorn >=0.36 imports the app outside the loop; requirements.txt pins 0.35.x)"
        )
        return None

    try:
        # Error capture: patches sys/threading/asyncio/logging so a raise anywhere is
        # reported with a stacktrace. Independent of the heartbeat below — one catches
        # code that raises, the other catches work that stops without raising.
        bettersentryio.init(
            base_url=BSIO_URL,
            key=BSIO_KEY,
            environment=BSIO_ENV,
            in_app_include=("/app/",),
        )

        slot = claim_worker_slot()
        suffix = f"w{slot}" if slot >= 0 else f"pid{os.getpid()}"
        _monitor_names = (f"{BSIO_MONITOR}-{suffix}", f"{BSIO_MONITOR}-vc-{suffix}")

        _beat = bettersentryio.Beat(base_url=BSIO_URL, key=BSIO_KEY, environment=BSIO_ENV)

        logger.info(
            "bsio: monitoring as %s / %s on %s (host %s)",
            _monitor_names[0], _monitor_names[1], BSIO_URL, socket.gethostname(),
        )
        return asyncio.create_task(_heartbeat(quiescent, vc_quiescent))
    except Exception:  # noqa: BLE001 - instrumentation must never stop the service booting
        logger.exception("bsio: failed to start monitoring; continuing without it")
        return None


def health() -> dict:
    """Reported by /health/deep so the pipeline state is visible without the UI."""
    out = {
        "enabled": enabled,
        "monitors": list(_monitor_names) if enabled else [],
        "tts": tts.snapshot(),
        "vc": vc.snapshot(),
    }
    if enabled and _beat is not None:
        out["delivery"] = _beat.stats()
    return out
