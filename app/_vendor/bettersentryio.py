"""
bettersentryio client — stdlib only, fire-and-forget.

Drop this file next to your service and import it. There is nothing to pip install.

Two halves, and you want both. Heartbeats catch a loop that stopped *working*; error
capture catches code that *raised*. Neither sees the other's failure — that is measured,
not assumed: see PLAN.md §7a.

    from bettersentryio import Beat, init

    init()                            # installs the exception hooks, once, at startup
    bsio = Beat()                     # reads BSIO_URL and BSIO_KEY from the environment
    bsio.beat("tts-batcher", progress=batches_done, every=30)

`init()` monkey-patches the places Python drops exceptions on the floor, so you do not
have to wrap anything in try/except:

    sys.excepthook          uncaught in the main thread
    threading.excepthook    uncaught in a thread
    asyncio task factory    a task that dies — see the note on that below
    logging                 logger.exception(...) and error(..., exc_info=True)

Three properties matter more than features here, because this code runs inside the
loop you are trying to protect:

1. **It never raises.** A monitoring call that can throw turns an observability
   problem into an outage. Every failure is swallowed (and counted).
2. **It never blocks the caller.** Beats are handed to a single background worker
   thread. The caller returns immediately, whatever the network is doing.
3. **It cannot pile up.** The worker holds one pending beat per monitor, newest
   wins. A loop beating faster than the network can keep up drops stale beats
   instead of growing a queue or spawning threads.
"""

from __future__ import annotations

import asyncio
import atexit
import collections
import json
import linecache
import logging
import os
import socket
import sys
import threading
import traceback
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

__all__ = [
    "Beat",
    "BsioLogHandler",
    "BsioMiddleware",
    "Client",
    "capture_exception",
    "capture_message",
    "client",
    "init",
]

_SHUTDOWN = object()


class Beat:
    """A heartbeat sender. Create one per process and share it."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        key: Optional[str] = None,
        environment: Optional[str] = None,
        timeout: float = 2.0,
    ) -> None:
        self.base_url = (base_url or os.environ.get("BSIO_URL", "")).rstrip("/")
        self.key = key or os.environ.get("BSIO_KEY", "")
        self.environment = environment or os.environ.get("BSIO_ENV") or None
        self.timeout = timeout

        self.sent = 0
        self.failed = 0
        self.dropped = 0
        self.last_error: Optional[str] = None

        # One pending beat per monitor, newest wins.
        self._pending: Dict[Tuple[str, Optional[str]], str] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._closed = False
        self._worker = threading.Thread(
            target=self._run, name="bettersentryio", daemon=True
        )
        self._worker.start()
        atexit.register(self.close)

    # ------------------------------------------------------------------ public

    def beat(
        self,
        monitor: str,
        progress: Optional[int] = None,
        every: Optional[int] = None,
        grace: Optional[int] = None,
        stall_window: Optional[int] = None,
        environment: Optional[str] = None,
    ) -> None:
        """
        Record one heartbeat. Returns immediately; never raises.

        monitor      slug, created on the first beat. e.g. "tts-batcher"
        progress     a counter that only goes up (batches, tokens, rows). Supplying
                     it is what enables stall detection: beats arriving while this
                     stands still means the loop is alive but doing no work.
        every        seconds you expect between beats
        grace        extra seconds before the monitor is called MISSING
        stall_window seconds of frozen progress before it is called STALLED
        """
        if not self.base_url or not self.key:
            # Misconfiguration must not crash the host service; it is visible in stats().
            self.dropped += 1
            self.last_error = "BSIO_URL or BSIO_KEY is not set"
            return

        env = environment or self.environment
        params = {"key": self.key}
        if progress is not None:
            params["progress"] = str(int(progress))
        if every is not None:
            params["every"] = str(int(every))
        if grace is not None:
            params["grace"] = str(int(grace))
        if stall_window is not None:
            params["stall_window"] = str(int(stall_window))
        if env:
            params["env"] = env

        url = "{}/api/0/beat/{}?{}".format(
            self.base_url,
            urllib.parse.quote(monitor, safe=""),
            urllib.parse.urlencode(params),
        )

        with self._lock:
            if self._closed:
                return
            key = (monitor, env)
            if key in self._pending:
                self.dropped += 1  # superseded before it went out
            self._pending[key] = url
        self._wake.set()

    def stats(self) -> Dict[str, object]:
        """Counters, for exposing on your own health endpoint."""
        return {
            "sent": self.sent,
            "failed": self.failed,
            "dropped": self.dropped,
            "last_error": self.last_error,
        }

    def close(self, timeout: float = 3.0) -> None:
        """Flush pending beats and stop the worker. Safe to call twice."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._wake.set()
        self._worker.join(timeout=timeout)

    # ------------------------------------------------------------------ worker

    def _run(self) -> None:
        while True:
            self._wake.wait(timeout=1.0)
            self._wake.clear()

            with self._lock:
                batch = list(self._pending.values())
                self._pending.clear()
                closed = self._closed

            for url in batch:
                self._send(url)

            if closed:
                return

    def _send(self, url: str) -> None:
        try:
            req = urllib.request.Request(url, method="POST")
            req.add_header("User-Agent", "bettersentryio-python")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if 200 <= resp.status < 300:
                    self.sent += 1
                else:
                    self.failed += 1
                    self.last_error = "HTTP {}".format(resp.status)
        except Exception as exc:  # noqa: BLE001 - a beat must never escape
            self.failed += 1
            self.last_error = "{}: {}".format(type(exc).__name__, exc)


# ======================================================================== errors
#
# Everything below is error capture. It keeps the same three promises as Beat: never
# raise, never block, never pile up.


class _Reporter:
    """
    Posts events to /api/0/errors on a background thread.

    A separate worker from Beat's on purpose. Beat's queue is newest-wins per monitor,
    which is right for heartbeats and wrong for errors — every distinct crash matters, so
    this one is a bounded FIFO that drops the *oldest* when full. Under a crash storm you
    keep the beginning of the storm, which is the part that explains it.
    """

    def __init__(self, base_url: str, key: str, timeout: float = 3.0, capacity: int = 100) -> None:
        self.base_url = base_url.rstrip("/")
        self.key = key
        self.timeout = timeout
        self.capacity = capacity

        self.sent = 0
        self.failed = 0
        self.dropped = 0
        self.last_error: Optional[str] = None

        self._queue: "collections.deque[bytes]" = collections.deque()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._closed = False
        self._worker = threading.Thread(
            target=self._run, name="bettersentryio-errors", daemon=True
        )
        self._worker.start()
        atexit.register(self.close)

    def submit(self, event: Dict[str, object]) -> None:
        if not self.base_url or not self.key:
            self.dropped += 1
            self.last_error = "BSIO_URL or BSIO_KEY is not set"
            return
        try:
            body = json.dumps(event, default=str).encode("utf-8")
        except Exception as exc:  # noqa: BLE001 - serialising must not crash the host
            self.dropped += 1
            self.last_error = "encode: {}".format(type(exc).__name__)
            return

        with self._lock:
            if self._closed:
                return
            while len(self._queue) >= self.capacity:
                self._queue.popleft()
                self.dropped += 1
            self._queue.append(body)
        self._wake.set()

    def stats(self) -> Dict[str, object]:
        return {
            "sent": self.sent,
            "failed": self.failed,
            "dropped": self.dropped,
            "queued": len(self._queue),
            "last_error": self.last_error,
        }

    def close(self, timeout: float = 3.0) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._wake.set()
        self._worker.join(timeout=timeout)

    def _run(self) -> None:
        while True:
            self._wake.wait(timeout=1.0)
            self._wake.clear()
            while True:
                with self._lock:
                    if not self._queue:
                        closed = self._closed
                        break
                    body = self._queue.popleft()
                self._post(body)
            if closed:
                return

    def _post(self, body: bytes) -> None:
        url = "{}/api/0/errors".format(self.base_url)
        try:
            req = urllib.request.Request(url, data=body, method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("X-BSIO-Key", self.key)
            req.add_header("User-Agent", "bettersentryio-python")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if 200 <= resp.status < 300:
                    self.sent += 1
                else:
                    self.failed += 1
                    self.last_error = "HTTP {}".format(resp.status)
        except Exception as exc:  # noqa: BLE001 - reporting must never escape
            self.failed += 1
            self.last_error = "{}: {}".format(type(exc).__name__, exc)


# This file, so our own frames can be excluded from "your code". Without this the ASGI
# middleware appears as the top in-app frame of every request error it captures — noise in
# the trace, and it feeds the grouping fingerprint.
_SELF_FILE = os.path.abspath(__file__)


def _is_in_app(filename: str, in_app_include: Tuple[str, ...]) -> bool:
    """
    Your code or somebody else's. Grouping leans on this, so a wrong answer here merges
    unrelated bugs or splits one.
    """
    if not filename:
        return False
    try:
        if os.path.abspath(filename) == _SELF_FILE:
            return False
    except Exception:  # noqa: BLE001 - odd filenames must not break capture
        pass
    lowered = filename.replace("\\", "/").lower()
    for marker in ("/site-packages/", "/dist-packages/", "/lib/python", "<frozen ",
                   "/.venv/", "/venv/"):
        if marker in lowered:
            return False
    if in_app_include:
        return any(part in lowered for part in in_app_include)
    return True


def _frames(tb: object, in_app_include: Tuple[str, ...]) -> List[Dict[str, object]]:
    """Oldest frame first, which is the order a stacktrace is read in."""
    out: List[Dict[str, object]] = []
    for frame, lineno in traceback.walk_tb(tb):  # type: ignore[arg-type]
        code = frame.f_code
        filename = code.co_filename
        module = frame.f_globals.get("__name__", "")
        entry: Dict[str, object] = {
            "filename": filename,
            "function": code.co_name,
            "module": module,
            "lineno": lineno,
            "in_app": _is_in_app(filename, in_app_include),
        }
        line = linecache.getline(filename, lineno).strip()
        if line:
            entry["context_line"] = [line]
        out.append(entry)
    return out


class Client:
    """Holds the config and the reporter. Create it with init()."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        key: Optional[str] = None,
        environment: Optional[str] = None,
        release: Optional[str] = None,
        server_name: Optional[str] = None,
        in_app_include: Optional[Tuple[str, ...]] = None,
        timeout: float = 3.0,
    ) -> None:
        self.base_url = (base_url or os.environ.get("BSIO_URL", "")).rstrip("/")
        self.key = key or os.environ.get("BSIO_KEY", "")
        self.environment = environment or os.environ.get("BSIO_ENV") or "production"
        self.release = release or os.environ.get("BSIO_RELEASE") or ""
        self.server_name = server_name or os.environ.get("BSIO_SERVER_NAME") or socket.gethostname()
        self.in_app_include = tuple(p.lower() for p in (in_app_include or ()))
        self.reporter = _Reporter(self.base_url, self.key, timeout=timeout)

    # ------------------------------------------------------------------ capture

    def capture_exception(
        self,
        exc: Optional[BaseException] = None,
        mechanism: str = "manual",
        handled: bool = False,
        extra: Optional[Dict[str, object]] = None,
        request: Optional[Dict[str, object]] = None,
        transaction: str = "",
    ) -> None:
        """Report an exception. Never raises, whatever is wrong with the argument."""
        try:
            if exc is None:
                exc = sys.exc_info()[1]
            if exc is None:
                return

            values: List[Dict[str, object]] = []
            seen = set()
            current: Optional[BaseException] = exc
            # Walk the cause chain oldest-first so the raised exception ends up last,
            # which is where the engine looks for the one that matters.
            chain: List[BaseException] = []
            while current is not None and id(current) not in seen:
                seen.add(id(current))
                chain.append(current)
                current = current.__cause__ or current.__context__
            for item in reversed(chain):
                values.append({
                    "type": type(item).__name__,
                    "value": str(item),
                    "module": getattr(type(item), "__module__", ""),
                    "stacktrace": {
                        "frames": _frames(item.__traceback__, self.in_app_include)
                    },
                    "mechanism": {"type": mechanism, "handled": handled},
                })

            self._submit({
                "level": "error",
                "message": "",
                "transaction": transaction,
                "exception": {"values": values},
                "extra": extra or {},
                "request": request,
            })
        except Exception:  # noqa: BLE001 - capture must never become the outage
            self.reporter.dropped += 1

    def capture_message(
        self,
        message: str,
        level: str = "error",
        logger: str = "",
        extra: Optional[Dict[str, object]] = None,
    ) -> None:
        try:
            self._submit({
                "level": level,
                "logger": logger,
                "message": message,
                "extra": extra or {},
            })
        except Exception:  # noqa: BLE001
            self.reporter.dropped += 1

    def _submit(self, event: Dict[str, object]) -> None:
        event.setdefault("event_id", uuid.uuid4().hex)
        event.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        event["environment"] = self.environment
        if self.release:
            event["release"] = self.release
        event["server_name"] = self.server_name
        self.reporter.submit(event)

    def stats(self) -> Dict[str, object]:
        return self.reporter.stats()


_client: Optional[Client] = None
_installed = False


def client() -> Optional[Client]:
    """The client init() created, or None if init() was never called."""
    return _client


def capture_exception(exc: Optional[BaseException] = None, **kw: object) -> None:
    """Module-level convenience. A no-op if init() was not called."""
    if _client is not None:
        _client.capture_exception(exc, **kw)  # type: ignore[arg-type]


def capture_message(message: str, **kw: object) -> None:
    if _client is not None:
        _client.capture_message(message, **kw)  # type: ignore[arg-type]


class BsioLogHandler(logging.Handler):
    """Turns logger.exception(...) and error(..., exc_info=True) into events."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if _client is None:
                return
            if record.name.startswith("bettersentryio"):
                return  # never report our own failures back to ourselves
            if record.exc_info and record.exc_info[1] is not None:
                _client.capture_exception(
                    record.exc_info[1],
                    mechanism="logging",
                    handled=True,
                    transaction=record.name,
                )
            else:
                _client.capture_message(
                    record.getMessage(),
                    level=record.levelname.lower(),
                    logger=record.name,
                )
        except Exception:  # noqa: BLE001 - a log call must never raise
            pass


def _install_hooks(capture_logging: bool, logging_level: int) -> None:
    global _installed
    if _installed:
        return
    _installed = True

    # --- uncaught in the main thread ---------------------------------------
    previous_excepthook = sys.excepthook

    def excepthook(exc_type, exc_value, tb):  # type: ignore[no-untyped-def]
        try:
            if _client is not None and not issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
                _client.capture_exception(exc_value, mechanism="excepthook", handled=False)
                # The process is going away; flush before it does or the event is lost.
                _client.reporter.close(timeout=2.0)
        finally:
            previous_excepthook(exc_type, exc_value, tb)

    sys.excepthook = excepthook

    # --- uncaught in a thread ---------------------------------------------
    previous_threadhook = getattr(threading, "excepthook", None)

    def threadhook(args):  # type: ignore[no-untyped-def]
        try:
            if _client is not None and not issubclass(args.exc_type, SystemExit):
                _client.capture_exception(
                    args.exc_value, mechanism="threading.excepthook", handled=False,
                    transaction=getattr(args.thread, "name", ""),
                )
        finally:
            if previous_threadhook is not None:
                previous_threadhook(args)

    if previous_threadhook is not None:
        threading.excepthook = threadhook  # type: ignore[assignment]

    # --- asyncio ----------------------------------------------------------
    #
    # Two hooks, and the first is the one that matters.
    #
    # A task factory adds a done-callback to every task, so a task that dies is reported
    # the moment it finishes. The loop's exception handler is NOT enough on its own:
    # asyncio only reports an unretrieved task exception when the Task is garbage
    # collected, and code that keeps a reference (`self.task = create_task(...)`, which is
    # the documented way to avoid tasks being collected early) never triggers it. We
    # measured exactly that — a loop died and its RuntimeError appeared nowhere at all.
    def _task_factory(loop, coro, **kwargs):  # type: ignore[no-untyped-def]
        try:
            task = asyncio.Task(coro, loop=loop, **kwargs)
        except TypeError:
            task = asyncio.Task(coro, loop=loop)  # older signatures

        def _done(t):  # type: ignore[no-untyped-def]
            try:
                if t.cancelled():
                    return
                exc = t.exception()
                if exc is not None and _client is not None:
                    _client.capture_exception(
                        exc, mechanism="asyncio.task", handled=False,
                        transaction=getattr(getattr(t, "get_coro", lambda: None)(), "__name__", ""),
                    )
            except Exception:  # noqa: BLE001
                pass

        task.add_done_callback(_done)
        return task

    def _exception_handler(loop, context):  # type: ignore[no-untyped-def]
        try:
            exc = context.get("exception")
            if _client is not None:
                if exc is not None:
                    _client.capture_exception(exc, mechanism="asyncio.handler", handled=False)
                else:
                    _client.capture_message(
                        str(context.get("message", "asyncio error")), logger="asyncio"
                    )
        except Exception:  # noqa: BLE001
            pass
        loop.default_exception_handler(context)

    def _attach(loop) -> None:  # type: ignore[no-untyped-def]
        try:
            if getattr(loop, "_bsio_attached", False):
                return
            loop.set_task_factory(_task_factory)
            loop.set_exception_handler(_exception_handler)
            loop._bsio_attached = True  # noqa: SLF001
        except Exception:  # noqa: BLE001
            pass

    # Attach to a loop that already exists...
    try:
        _attach(asyncio.get_event_loop_policy().get_event_loop())
    except Exception:  # noqa: BLE001
        pass
    try:
        _attach(asyncio.get_running_loop())
    except Exception:  # noqa: BLE001
        pass

    # ...and to any loop started later, whoever creates it. uvicorn makes its own.
    for method in ("run_forever", "run_until_complete"):
        original = getattr(asyncio.base_events.BaseEventLoop, method, None)
        if original is None or getattr(original, "_bsio_wrapped", False):
            continue

        def wrap(original=original):  # bind per iteration
            def wrapper(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                _attach(self)
                return original(self, *args, **kwargs)
            wrapper._bsio_wrapped = True  # noqa: SLF001
            return wrapper

        setattr(asyncio.base_events.BaseEventLoop, method, wrap())

    # --- logging ----------------------------------------------------------
    if capture_logging:
        handler = BsioLogHandler(level=logging_level)
        logging.getLogger().addHandler(handler)


def init(
    base_url: Optional[str] = None,
    key: Optional[str] = None,
    environment: Optional[str] = None,
    release: Optional[str] = None,
    server_name: Optional[str] = None,
    in_app_include: Optional[Tuple[str, ...]] = None,
    capture_logging: bool = True,
    logging_level: int = logging.ERROR,
    timeout: float = 3.0,
) -> Client:
    """
    Install error capture. Call once, as early as possible.

    Idempotent: calling it again replaces the config but does not stack hooks, so a
    module imported twice cannot double-report.
    """
    global _client
    _client = Client(
        base_url=base_url,
        key=key,
        environment=environment,
        release=release,
        server_name=server_name,
        in_app_include=in_app_include,
        timeout=timeout,
    )
    _install_hooks(capture_logging=capture_logging, logging_level=logging_level)
    return _client


class BsioMiddleware:
    """
    ASGI middleware: reports unhandled exceptions on the request path.

    Works with FastAPI, Starlette and anything else ASGI:

        app.add_middleware(BsioMiddleware)

    It re-raises, so the framework still produces its own 500 — this observes, it does
    not change behaviour.
    """

    def __init__(self, app: object) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):  # type: ignore[no-untyped-def]
        if scope.get("type") != "http":
            await self.app(scope, receive, send)  # type: ignore[operator]
            return
        try:
            await self.app(scope, receive, send)  # type: ignore[operator]
        except Exception as exc:
            if _client is not None:
                query = scope.get("query_string", b"")
                _client.capture_exception(
                    exc,
                    mechanism="asgi",
                    handled=False,
                    transaction="{} {}".format(
                        scope.get("method", ""), scope.get("path", "")
                    ),
                    request={
                        "method": scope.get("method", ""),
                        "url": scope.get("path", ""),
                        "query_string": query.decode("latin-1") if query else "",
                    },
                )
            raise
