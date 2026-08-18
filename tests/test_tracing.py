"""app/tracing.py: the disabled path must cost nothing, the enabled path must nest.

No torch / GPU / API needed -- app.tracing only imports app.env. The enabled-path
tests skip when opentelemetry is not installed (it arrives with fastapi-loki-tempo).

    uv run --with pytest -- pytest tests/test_tracing.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import importlib
from contextlib import nullcontext

import pytest


def _reload(enabled):
    """Re-import app.tracing with ENABLE_TRACING_SPANS flipped.

    The flag is read once at import time (that is what makes the disabled path free),
    so a test that wants the other mode has to drop both modules and import again.
    """
    os.environ['ENABLE_TRACING_SPANS'] = 'true' if enabled else 'false'
    for name in ('app.tracing', 'app.env'):
        sys.modules.pop(name, None)
    return importlib.import_module('app.tracing')


@pytest.fixture(autouse=True)
def restore_env():
    previous = os.environ.get('ENABLE_TRACING_SPANS')
    yield
    if previous is None:
        os.environ.pop('ENABLE_TRACING_SPANS', None)
    else:
        os.environ['ENABLE_TRACING_SPANS'] = previous
    # leave no half-configured modules behind for the rest of the session
    for name in ('app.tracing', 'app.env'):
        sys.modules.pop(name, None)


# --- disabled explicitly (the default is on) -------------------------------------------------------

def test_enabled_by_default_but_degrades_without_otel():
    """Unset means on -- and still means off wherever opentelemetry is absent.

    The second half is the one that matters operationally: a box without the OTel
    packages (a bare NPU host, a slim image) must boot and serve, not raise.
    """
    os.environ.pop('ENABLE_TRACING_SPANS', None)
    for name in ('app.tracing', 'app.env'):
        sys.modules.pop(name, None)
    tracing = importlib.import_module('app.tracing')
    try:
        from opentelemetry import trace as _otel_trace  # noqa: F401
    except ImportError:
        assert tracing.enabled is False
    else:
        assert tracing.enabled is True


def test_disabled_span_is_a_shared_nullcontext():
    tracing = _reload(False)
    first = tracing.span('codec.decode', attrs={'codec.tokens': 3})
    second = tracing.span('lm.generate')
    assert isinstance(first, nullcontext)
    # the same instance every time: no allocation on the hot path
    assert first is second
    with first as sp:
        assert sp is None


def test_disabled_helpers_are_noops():
    tracing = _reload(False)
    assert tracing.now_ns() == 0
    assert tracing.current_context() is None
    assert tracing.context_with(None) is None
    assert tracing.start_span('x') is None
    assert tracing.record_span('x', 1, 2) is None
    assert tracing.stage_meta(None) is None
    # every helper has to tolerate the None the disabled path hands back
    tracing.set_attributes(None, {'a': 1})
    tracing.add_event(None, 'evt')
    tracing.end_span(None, attrs={'a': 1})
    tracing.record_exception(None, ValueError('boom'))


def test_disabled_nested_spans_still_run_the_body():
    tracing = _reload(False)
    ran = []
    with tracing.span('outer'):
        with tracing.span('inner'):
            ran.append(True)
    assert ran == [True]


# --- enabled ----------------------------------------------------------------------

@pytest.fixture
def spans():
    """(tracing module with spans on, finished-span exporter)."""
    pytest.importorskip('opentelemetry.sdk', reason='opentelemetry not installed')
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    exporter.clear()
    return _reload(True), exporter


def _by_name(exporter):
    return {s.name: s for s in exporter.get_finished_spans()}


def test_enabled_emits_and_nests(spans):
    tracing, exporter = spans
    assert tracing.enabled is True
    with tracing.span('tts.stream', attrs={'tts.model': 'TTS-model'}) as outer:
        with tracing.span('tts.chunk', attrs={'chunk.index': 0}):
            pass
    found = _by_name(exporter)
    assert set(found) == {'tts.stream', 'tts.chunk'}
    assert found['tts.chunk'].parent.span_id == found['tts.stream'].context.span_id
    assert found['tts.stream'].attributes['tts.model'] == 'TTS-model'
    assert outer.context.trace_id == found['tts.chunk'].context.trace_id


def test_explicit_parent_beats_ambient_context(spans):
    """What the batching threads rely on: a parent handed over, not inherited."""
    tracing, exporter = spans
    parent = tracing.start_span('codec.decode')
    ctx = tracing.context_with(parent)
    with tracing.span('somewhere.else'):
        # ambient parent is 'somewhere.else', but ctx wins
        tracing.record_span('codec.gpu_decode', tracing.now_ns(), parent=ctx)
    tracing.end_span(parent, attrs={'codec.tokens': 100})

    found = _by_name(exporter)
    assert found['codec.gpu_decode'].parent.span_id == found['codec.decode'].context.span_id
    assert found['codec.decode'].attributes['codec.tokens'] == 100


def test_record_span_uses_the_given_timestamps(spans):
    tracing, exporter = spans
    start = tracing.now_ns()
    end = start + 5_000_000          # 5 ms
    tracing.record_span('codec.batch_wait', start, end, attrs={'batch.size': 4})
    span = _by_name(exporter)['codec.batch_wait']
    assert span.start_time == start
    assert span.end_time == end
    assert span.attributes['batch.size'] == 4


def test_record_span_without_a_start_is_dropped(spans):
    """now_ns() returns 0 when a stage was queued before tracing had a timestamp."""
    tracing, exporter = spans
    assert tracing.record_span('codec.batch_prep', 0, tracing.now_ns()) is None
    assert exporter.get_finished_spans() == ()


def test_stage_meta_carries_context_and_enqueue_time(spans):
    tracing, _ = spans
    span = tracing.start_span('codec.decode')
    meta = tracing.stage_meta(span)
    assert meta['ctx'] is not None
    assert meta['t_enqueue'] > 0
    assert tracing.stage_meta(None) is None


def test_attributes_are_cleaned(spans):
    tracing, exporter = spans
    with tracing.span('x', attrs={'kept': 1, 'dropped': None, 'coerced': [1, 2]}):
        pass
    attrs = _by_name(exporter)['x'].attributes
    assert attrs['kept'] == 1
    assert 'dropped' not in attrs
    assert attrs['coerced'] == '[1, 2]'


#: app/main.py imports app.tracing *before* fastapi_loki_tempo.patch() installs the
#: tracer provider, so get_tracer() has to resolve lazily -- if it bound a no-op tracer
#: at import time every span would silently vanish. The global provider can only be set
#: once per process, so this runs in a fresh one.
_LAZY_PROVIDER = """
import os, sys
sys.path.insert(0, {root!r})
os.environ['ENABLE_TRACING_SPANS'] = 'true'

from app import tracing                       # no provider exists yet
assert tracing.enabled

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

provider = TracerProvider()
exporter = InMemorySpanExporter()
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)           # ... as fastapi_loki_tempo.patch() does

with tracing.span('codec.decode', attrs={{'codec.tokens': 7}}):
    pass

names = [s.name for s in exporter.get_finished_spans()]
assert names == ['codec.decode'], names
print('ok')
"""


def test_tracer_resolves_after_the_provider_is_installed():
    import subprocess

    pytest.importorskip('opentelemetry.sdk', reason='opentelemetry not installed')
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    done = subprocess.run(
        [sys.executable, '-c', _LAZY_PROVIDER.format(root=root)],
        capture_output=True, text=True,
    )
    assert done.returncode == 0, done.stderr
    assert 'ok' in done.stdout


def test_exception_marks_the_span(spans):
    tracing, exporter = spans
    from opentelemetry.trace import StatusCode

    with pytest.raises(ValueError):
        with tracing.span('codec.decode'):
            raise ValueError('decode blew up')
    span = _by_name(exporter)['codec.decode']
    assert span.status.status_code == StatusCode.ERROR
    assert span.events[0].name == 'exception'
