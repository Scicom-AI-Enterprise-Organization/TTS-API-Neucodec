"""OpenTelemetry spans on the TTS/VC hot path, on by default.

Answers, per request, the three questions the serving loop actually raises:

* how long the request sat in **dynamic batching** before the GPU touched it
  (`codec.batch_wait`, `codec.batch_prep`, `codec.compute_wait`),
* how long it waited on **vLLM** (`lm.connect`, `lm.first_token`, plus the
  `tts.lm_wait_s` aggregate on `tts.stream`),
* how long the **codec decode** itself took (`codec.gpu_decode`, wrapped by the
  per-chunk `tts.chunk` / `codec.decode` spans).

Spans are built only when all three of these hold: ``ENABLE_TRACING_SPANS`` (on by
default), an exporter is configured (``TRACING_EXPORTER_CONFIGURED``, i.e. something
will actually collect them), and opentelemetry is importable. Otherwise every helper
here is a shared :func:`contextlib.nullcontext` or a function that returns ``None``
before doing anything, so the hot path pays one module-global boolean check per call
site -- no tracer lookup, no ``time_ns()``, no span objects, no dict for the
cross-thread carrier. That matters twice over: the decode loop is GIL-bound (see
CLAUDE.md), and an SDK with no span processor still *builds* every span before
dropping it (~292us per request, measured), which is pure waste.

Spans are emitted through whatever tracer provider is installed -- in this app
that is the one ``wan.patch()`` configures, so they land in Tempo
as children of the FastAPI server span and share its trace id with the JSON log
lines. ``OTLP_ENDPOINT`` is therefore not optional: without it (or another
exporter variable) these spans are not created at all.

Two things are deliberate:

* **The parent context is always passed explicitly** to anything that can run
  off the request's task -- the batching threads and the LM producer task. The
  ambient contextvar is useless in a worker thread that is servicing a batch
  built from N different requests.
* **Worker-thread stages are recorded after the fact** with
  :func:`record_span` and explicit start/end timestamps, rather than held open
  across the queue hops. A stage that spans two threads and a queue cannot be a
  live ``with`` block, and timestamps taken at the hop are exactly as accurate.

Usage::

    from app import tracing

    with tracing.span('codec.decode', attrs={'codec.tokens': n}) as sp:
        ...
    tracing.record_span('codec.gpu_decode', t0, tracing.now_ns(), parent=ctx)
"""

import logging
import os
import time
from contextlib import contextmanager, nullcontext

from app.env import (
    ENABLE_TRACING_SPANS,
    TRACING_EXPORTER_CONFIGURED,
    TRACING_SPANS_REQUIRE_EXPORTER,
)

#: True only when spans were requested, something is configured to collect them, and an
#: OpenTelemetry API is importable. Any one of those missing => the free no-op path.
enabled = False
_tracer = None

#: (level, message) explaining that decision, emitted by :func:`log_status`. Not logged
#: here: app/main.py imports this module before wan.patch() configures
#: logging, so an import-time INFO goes to a root logger that drops it -- which is how the
#: "no exporter, spans disabled" notice managed to be invisible in the first place.
status = (logging.INFO, '')

#: One shared instance: nullcontext holds no state, so it is safe to reuse it
#: concurrently and reentrantly, and reusing it keeps the disabled path
#: allocation free.
_NULL = nullcontext()

if not ENABLE_TRACING_SPANS:
    status = (logging.INFO, 'hot-path spans disabled (ENABLE_TRACING_SPANS=false)')
elif TRACING_SPANS_REQUIRE_EXPORTER and not TRACING_EXPORTER_CONFIGURED:
    # Loud about it: "spans requested but none appear" is otherwise an hour of digging.
    status = (logging.WARNING,
              'hot-path spans requested but no span exporter is configured, so they are '
              'disabled rather than built and dropped. Set OTLP_ENDPOINT (or '
              'ENABLE_CONSOLE_SPAN_EXPORTER=true) to collect them, or '
              'TRACING_SPANS_REQUIRE_EXPORTER=false if a processor is installed in code.')
else:
    try:
        from opentelemetry import context as otel_context
        from opentelemetry import trace as otel_trace

        # Resolved lazily by the API: this runs before wan.patch()
        # installs the real provider, and a ProxyTracer picks it up afterwards.
        _tracer = otel_trace.get_tracer(os.environ.get('SERVICE_NAME', 'tts-api'))
        enabled = True
        status = (logging.INFO, 'hot-path spans enabled (ENABLE_TRACING_SPANS)')
    except ImportError as e:
        status = (logging.WARNING,
                  f'ENABLE_TRACING_SPANS=true but opentelemetry is not installed ({e}), '
                  'hot-path spans disabled. Install wan.')


def log_status():
    """Log why tracing is on or off. Call once logging is configured."""
    level, message = status
    if message:
        logging.log(level, message)


def now_ns():
    """Epoch nanoseconds for a retroactive span, or ``0`` when disabled.

    Epoch, not monotonic: OpenTelemetry span timestamps are UNIX epoch ns.
    ``0`` doubles as "no timestamp", which is what :func:`record_span` checks.
    """
    return time.time_ns() if enabled else 0


def _clean(attrs):
    """Drop ``None`` values and coerce anything the SDK would reject to ``str``."""
    if not attrs:
        return None
    out = {}
    for k, v in attrs.items():
        if v is None:
            continue
        out[k] = v if isinstance(v, (bool, str, int, float)) else str(v)
    return out


def current_context():
    """The active OpenTelemetry context, to hand to another task or thread."""
    return otel_context.get_current() if enabled else None


def context_with(span):
    """A context whose active span is ``span``; use as an explicit ``parent``."""
    if not enabled or span is None:
        return None
    return otel_trace.set_span_in_context(span)


@contextmanager
def _span(name, parent, attrs):
    span = _tracer.start_span(name, context=parent, attributes=_clean(attrs))
    # end_on_exit + record_exception: a decode that raises should show up in the
    # trace as the failed span, not vanish.
    with otel_trace.use_span(span, end_on_exit=True, record_exception=True):
        yield span


def span(name, parent=None, attrs=None):
    """Context manager yielding a live span, or a nullcontext (``None``) when off.

    The span is made current for the duration, so anything called inside it
    (including code that only ever calls ``current_context()``) nests under it.
    """
    if not enabled:
        return _NULL
    return _span(name, parent, attrs)


def start_span(name, parent=None, attrs=None):
    """A span the caller ends itself, for work that outlives one function.

    Not made current -- pass :func:`context_with` to children instead. Returns
    ``None`` when disabled, which every other helper here accepts.
    """
    if not enabled:
        return None
    return _tracer.start_span(name, context=parent, attributes=_clean(attrs))


def end_span(span, attrs=None):
    """End a :func:`start_span` span, optionally setting final attributes."""
    if span is None:
        return
    set_attributes(span, attrs)
    span.end()


def record_span(name, start_ns, end_ns=None, parent=None, attrs=None):
    """Record an already-finished span from explicit timestamps.

    Safe to call from any thread: nothing touches the ambient context and the
    parent is explicit, which is what lets a batching thread file its stage
    under the request that queued the work.
    """
    if not enabled or not start_ns:
        return None
    span = _tracer.start_span(
        name, context=parent, start_time=start_ns, attributes=_clean(attrs),
    )
    span.end(end_time=end_ns or time.time_ns())
    return span


def set_attributes(span, attrs=None):
    if span is None:
        return
    attrs = _clean(attrs)
    if attrs:
        span.set_attributes(attrs)


def add_event(span, name, attrs=None):
    if span is None:
        return
    span.add_event(name, attributes=_clean(attrs))


def record_exception(span, exc):
    if span is None:
        return
    span.record_exception(exc)
    span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, str(exc)))


def stage_meta(span):
    """The carrier queued alongside a batched request, or ``None`` when off.

    Holds the parent context plus the enqueue timestamp, and is mutated in place
    by each stage to hand the next one its start time. ``None`` is the whole
    point of the disabled path: the batching threads see it and skip the timing
    entirely instead of computing it and throwing it away.
    """
    if not enabled or span is None:
        return None
    return {'ctx': context_with(span), 't_enqueue': time.time_ns()}
