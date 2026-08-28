from app.env import *

import torch

torch._dynamo.config.recompile_limit = 128
torch.set_float32_matmul_precision('high')

from typing import Literal, Optional
import re
import json
import base64
import struct
import asyncio
import io
import time
import wave
import tempfile
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from fastapi import FastAPI, Request, HTTPException, File, Form
from fastapi.responses import StreamingResponse, FileResponse, Response
from pydantic import BaseModel, Field, AliasChoices
from huggingface_hub import hf_hub_download
from app.normalizer import load as load_normalizer, to_cardinal
from app.normalizer.chinese import normalize_chinese, is_chinese_dominant, CJK_RE, KANA_RE
from app.llm_normalizer import llm_normalize, LLMNormalizerError, NormalizerMode
from app import tracing
from app.context import (
    make_store, RequestContext, build_prompt, fit_context, select_voice,
    seconds_to_tokens, total_tokens, CONTEXT_MODES,
)
import torch.cuda as cuda
import uuid
import bisect
import threading
import queue as thread_queue
import concurrent.futures
import soundfile as sf
import numpy as np
import aiohttp
# Optional heavy front-end deps: only needed for text->LM generation / VC. In DUMMY_TOKENS
# mode (canned tokens) or on a bare NPU box these may be absent; degrade gracefully.
try:
    import librosa
except Exception:
    librosa = None
try:
    import sentry_sdk
except Exception:
    sentry_sdk = None
try:
    import wan
except Exception:
    try:
        # pre-rename package name; a venv provisioned before the repo became `wan` still
        # has this one, and the try/except above would otherwise silently drop JSON
        # logging and tracing on that box rather than fail loudly.
        import fastapi_loki_tempo as wan
    except Exception:
        wan = None
from app.rules import *
from app.wrapper import CUDAGraphsWrapper
from app.timestretch import WSOLA, float_to_pcm16, pcm16_to_float, MIN_RATE, MAX_RATE
from app.neucodec import NeuCodec

if sentry_sdk is not None and len(SENTRY_DSN):
    sentry_sdk.init(dsn=SENTRY_DSN, send_default_pii=True)

def _suppress_asgi_message_spans():
    """Default OTel's ASGI middleware to dropping its per-message spans.

    That instrumentation opens one span per ASGI *event*. A streaming TTS response reads
    the vLLM stream line by line and polls `request.is_disconnected()` as it goes -- and
    every such poll is a real ASGI receive -- so one request produced ~2 `http receive`
    spans per speech token: measured on an H20, 505 of them next to 19 real spans. They
    carry no information, bury the spans that do, and multiply Tempo's ingest for nothing.
    DISCONNECT_POLL_S cuts the count; this removes the rest, including the per-chunk
    `http send` spans that scale with utterance length.

    Done by defaulting the middleware's own kwarg rather than by re-instrumenting or by
    editing `app.user_middleware`:

    * `patch()` must stay the thing that instruments the app. It registers its
      request-logging middleware first on purpose, and on older instrumentation versions
      (which add OTel via `add_middleware`) instrumenting ahead of it flips the order and
      drops `traceID` from the `type=request` log line.
    * `opentelemetry-instrumentation-fastapi` >=0.50 does not use `add_middleware` at all
      -- it wraps `build_middleware_stack` and constructs `OpenTelemetryMiddleware`
      itself -- so there is no registered `Middleware` entry whose kwargs could be
      edited (verified: that approach patched 0 of them).

    Defaulting `__init__` covers both mechanisms. `exclude_spans` only exists in newer
    versions and an unknown kwarg would raise on every request, so it is feature-detected.
    """
    try:
        import functools
        import inspect
        from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware
    except ImportError:
        return
    try:
        if 'exclude_spans' not in inspect.signature(OpenTelemetryMiddleware.__init__).parameters:
            logging.info(
                'opentelemetry-instrumentation-asgi predates exclude_spans; '
                'keeping the per-message http receive/send spans'
            )
            return
        if getattr(OpenTelemetryMiddleware, '_tts_excludes_message_spans', False):
            return      # uvicorn --reload / a second patch() call
        original_init = OpenTelemetryMiddleware.__init__

        @functools.wraps(original_init)
        def _init(self, *args, **kwargs):
            # not setdefault: the fastapi instrumentation passes `exclude_spans=None`
            # explicitly, so the key is always present and a setdefault never fires.
            # Only an explicit non-empty choice by the caller wins over this default.
            if not kwargs.get('exclude_spans'):
                kwargs['exclude_spans'] = ['receive', 'send']
            return original_init(self, *args, **kwargs)

        OpenTelemetryMiddleware.__init__ = _init
        OpenTelemetryMiddleware._tts_excludes_message_spans = True
        logging.info('asgi per-message http receive/send spans suppressed')
    except Exception as e:
        # an observability tweak must never stop the app from booting
        logging.warning(f'could not suppress asgi message spans: {e}')


app = FastAPI()
if wan is not None:
    wan.patch(app=app)
    # Everything below is deliberately *after* patch(), which is what configures logging:
    # an info() before it goes to an unconfigured root logger and is dropped. Also fine
    # for the middleware tweak -- Starlette does not build the middleware stack (and so
    # does not construct the OTel middleware) until the first request.
    logging.info(f'observability via {wan.__name__} {getattr(wan, "__version__", "?")}')
    if not TRACE_ASGI_MESSAGE_SPANS:
        _suppress_asgi_message_spans()
# Likewise deferred until logging exists, so "spans are off because nothing collects
# them" is actually visible instead of being swallowed by the unconfigured root logger.
tracing.log_status()

# Cross-request speech context (`request_id` on /v1/audio/speech), shared across
# worker processes -- see app/context.py for the why and the format.
CONTEXT_MAX_TOKENS = seconds_to_tokens(CONTEXT_MAX_S)
if CONTEXT_MODE not in CONTEXT_MODES:
    raise ValueError(f'CONTEXT_MODE={CONTEXT_MODE!r} must be one of {CONTEXT_MODES}')
context_store = make_store(CONTEXT_STORE, CONTEXT_STORE_DIR, CONTEXT_TTL_S, CONTEXT_MAX_TOKENS)
if context_store is None:
    logging.info('speech context: off (CONTEXT_STORE=off) -- request_id is ignored')
else:
    logging.info(
        f'speech context: {CONTEXT_STORE} store'
        f'{" at " + context_store.directory if hasattr(context_store, "directory") else ""}, '
        f'keeps {CONTEXT_MAX_S:g}s ({CONTEXT_MAX_TOKENS} speech tokens) per id, '
        f'ttl {CONTEXT_TTL_S:g}s, LM window {LM_MAX_MODEL_LEN}'
    )

torch.set_grad_enabled(False)

if DEVICE:
    device = DEVICE
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Accelerator helper module: torch.cuda / torch.npu / None. CUDA graphs and CUDA/NPU
# streams are only used when this is not None; on cpu the decode falls back to eager.
if device == "cuda":
    dev = torch.cuda
elif device == "npu":
    import torch_npu  # noqa: F401  registers the 'npu' device backend
    dev = torch.npu
else:
    logging.warning("No CUDA/NPU device selected, will run using CPU.")
    dev = None
logging.info(f"decode device: {device}")

# Language detection + text normalization are only needed when generating from text via
# the LM. In DUMMY_TOKENS mode we replay canned speech tokens, so skip them (and fasttext).
lang_model = None
normalizer = None
if not DUMMY_TOKENS_FILE:
    # fasttext (language detection) is only used when a request sets normalize_malaysian=True.
    # Make it optional so the app still boots where a fasttext build is unavailable (e.g. NPU);
    # requests with normalize_malaysian=False are unaffected.
    try:
        import fasttext
        filename = hf_hub_download(
            repo_id="mesolitica/fasttext-language-detection-bahasa-en",
            filename="fasttext.ftz",
        )
        lang_model = fasttext.load_model(filename)
    except Exception as e:
        logging.warning(f"fasttext unavailable; malaysian language detection disabled: {e}")
    normalizer = load_normalizer()

# Canned speech tokens replayed in DUMMY_TOKENS mode (extracted from a reference audio).
DUMMY_TOKENS = ""
if DUMMY_TOKENS_FILE:
    with open(DUMMY_TOKENS_FILE) as f:
        DUMMY_TOKENS = f.read().strip()
    _n_dummy = len(re.findall(r's_(\d+)', DUMMY_TOKENS))
    logging.info(f"DUMMY_TOKENS mode: replaying {_n_dummy} tokens from {DUMMY_TOKENS_FILE}")

logging.info('loading audio encoder')

codec = NeuCodec.from_pretrained("neuphonic/neucodec").eval().to(device)
codebook_size = 50
sr = 24000
TOKEN_RE = re.compile(r'<\|s_(\d+)\|>')

logging.info('done load audio encoder')

# Streams are only used by the dynamic-batching path (DYNAMIC_BATCHING=true). Guard so a
# non-CUDA box (e.g. Ascend NPU with dev=torch.npu, or plain CPU) does not crash at import.
if dev is not None:
    h2d_stream = dev.Stream()
    compute_stream = dev.Stream()
else:
    h2d_stream = compute_stream = None

def fn(padded_token):
    return codec.decode_code(padded_token.unsqueeze(1))

buckets = {}
BUCKET_TOKENS = [int(codebook_size * i) for i in CUDA_GRAPH_BATCH]
BUCKET_TOKENS = sorted(BUCKET_TOKENS)
if len(BUCKET_TOKENS) and device == "cuda":
    if TORCH_COMPILE:
        logging.info("warming up with torch compile")
    else:
        logging.info("warming up with cuda graphs")

    for B in tqdm(BUCKET_BATCHES):
        for T in BUCKET_TOKENS:
            input = torch.zeros((B, T), dtype=torch.long, device=device)
            if TORCH_COMPILE:
                l = lambda x: fn(x)
                l = torch.compile(l)
                for _ in range(3):
                    l(input)
                fn_with_graph = l
            else:
                fn_with_graph = CUDAGraphsWrapper.wrap(fn, [input], stream=compute_stream)
            buckets[(B, T)] = fn_with_graph

    logging.info(f'keys: {buckets.keys()}')

compute_queue = thread_queue.Queue()
batch_queue = thread_queue.Queue()
dynamic_batch_queue = asyncio.Queue()

vc_compute_queue = thread_queue.Queue()
vc_batch_queue = thread_queue.Queue()
vc_dynamic_batch_queue = asyncio.Queue()

def thread_safe_set_result(loop, future, value):
    def _safe_set():
        if not future.done():
            future.set_result(value)
    loop.call_soon_threadsafe(_safe_set)

def make_pinned_batch(tokens, target_T, dtype=torch.long):
    B = len(tokens)
    pinned = torch.empty((B, target_T), dtype=dtype)
    if dev is not None:
        # pinned host memory enables the async H2D copy; without an accelerator
        # pin_memory() raises "Cannot access accelerator device".
        pinned = pinned.pin_memory()
    pinned.fill_(0)

    for i, t in enumerate(tokens):
        n = len(t)
        if isinstance(t, torch.Tensor):
            pinned[i, :n].copy_(t)
        else:
            pinned[i, :n] = torch.tensor(t, dtype=dtype)
    return pinned

def choose_bucket_len(max_len):
    idx = bisect.bisect_left(BUCKET_TOKENS, max_len)
    if idx < len(BUCKET_TOKENS):
        return BUCKET_TOKENS[idx]
    return max_len

def thread_safe_set_exception(loop, futures, exc):
    for fut in futures:
        def _safe_set_exc(f=fut, ex=exc):
            if not f.done():
                f.set_exception(ex)
        loop.call_soon_threadsafe(_safe_set_exc)

# --- tracing of the batching stages ----------------------------------------------
# A batch is built from N different requests, so the worker threads cannot use the
# ambient span context: each item carries its own `meta` (tracing.stage_meta) holding
# that request's parent context plus the timestamp of the previous hop. Stages are
# recorded after the fact with explicit start/end times -- a stage that crosses a
# thread boundary and a queue cannot be a live `with` block. Every call site guards on
# `tracing.enabled` so nothing below runs, or is even iterated, when tracing is off.

def _trace_batch_stage(metas, t_end, batch_size, padded_shape):
    """Batch formed -> H2D copy issued (the batch thread), per request."""
    for meta in metas:
        if meta is None:
            continue
        tracing.record_span(
            'codec.batch_prep', meta.get('t_batched'), t_end, parent=meta['ctx'],
            attrs={
                'batch.size': batch_size,
                'batch.padded_tokens': int(padded_shape[1]),
            },
        )
        meta['t_h2d'] = t_end

def _trace_compute_stage(metas, t_start, t_end, padded_shape, cuda_graph):
    """H2D issued -> compute start -> decode done (the compute thread), per request."""
    for meta in metas:
        if meta is None:
            continue
        tracing.record_span(
            'codec.compute_wait', meta.get('t_h2d'), t_start, parent=meta['ctx'],
        )
        tracing.record_span(
            'codec.gpu_decode', t_start, t_end, parent=meta['ctx'],
            attrs={
                'batch.size': int(padded_shape[0]),
                'batch.padded_tokens': int(padded_shape[1]),
                'codec.cuda_graph': cuda_graph,
                'codec.device': device,
            },
        )

def compute_thread_fn(loop):
    while True:
        item = compute_queue.get()
        try:
            _compute_one(loop, item)
        except Exception as e:
            # An unhandled exception here used to kill the thread outright. The process
            # stayed up, compute_queue was never drained again, and every subsequent
            # decode hung forever on its future with nothing logged. Fail the batch,
            # log it, keep the worker alive.
            logging.exception(f'{item[0]}, compute_thread_fn failed: {e}')
            thread_safe_set_exception(loop, item[3], e)

def _compute_one(loop, item):
        uuid_str, padded_token, padded_token_len, futures, metas = item
        logging.debug(f'{uuid_str}, enter compute_thread_fn')
        t_compute = tracing.now_ns()
        shapes = padded_token.shape

        if shapes[1] == 0:
            logging.warning(f'{uuid_str}, skipping empty token batch with shape {shapes}')
            empty = torch.zeros((shapes[0], 1, 0), dtype=torch.float32)
            for i, fut in enumerate(futures):
                thread_safe_set_result(loop, fut, empty[i:i+1])
            return

        cuda_graph = shapes in buckets
        with torch.no_grad():
            if dev is not None:
                with dev.stream(compute_stream):
                    compute_stream.wait_stream(h2d_stream)
                    if cuda_graph:
                        logging.debug(f'{uuid_str}, Hit compute shape {shapes}')
                        recon = buckets[shapes](padded_token)
                    else:
                        recon = fn(padded_token)
                    logging.debug(f'{uuid_str}, done compute shape {shapes}')

                    ys = recon.cpu()
                compute_stream.synchronize()
            else:
                # no accelerator: plain eager decode, no streams to synchronize
                ys = fn(padded_token).cpu()
                logging.debug(f'{uuid_str}, done compute shape {shapes} (cpu)')
            if tracing.enabled:
                _trace_compute_stage(metas, t_compute, tracing.now_ns(), shapes, cuda_graph)
            for i, fut in enumerate(futures):
                out_len = padded_token_len[i] * 480
                ys_ = ys[i:i+1, :, :out_len]
                thread_safe_set_result(loop, fut, ys_)

def batch_thread_fn(loop):
    while True:
        uuid_str, batch = batch_queue.get()
        try:
            _batch_one(uuid_str, batch)
        except Exception as e:
            logging.exception(f'{uuid_str}, batch_thread_fn failed: {e}')
            thread_safe_set_exception(loop, [b[0] for b in batch], e)

def _batch_one(uuid_str, batch):
        logging.debug(f'{uuid_str}, enter batch_thread_fn')
        futures, tokens, metas = zip(*[(b[0], b[1], b[2]) for b in batch])

        padded_token_len = [len(t) for t in tokens]
        max_len = max(padded_token_len)
        target_T = choose_bucket_len(max_len)
        padded_token = make_pinned_batch(tokens, target_T)
        shapes = padded_token.shape
        logging.debug(f'{uuid_str}, batch shape {shapes} cpu')

        if dev is not None:
            with dev.stream(h2d_stream):
                padded_token_gpu = padded_token.to(device, non_blocking=True)
        else:
            padded_token_gpu = padded_token
        if tracing.enabled:
            _trace_batch_stage(metas, tracing.now_ns(), len(batch), shapes)
        compute_queue.put((uuid_str, padded_token_gpu, padded_token_len, futures, metas))

async def dynamic_batching():
    need_sleep = True
    while True:
        if need_sleep:
            await asyncio.sleep(MICROSLEEP)
        need_sleep = True
        batch = []
        while not dynamic_batch_queue.empty():
            try:
                request = await asyncio.wait_for(dynamic_batch_queue.get(), timeout=1e-9)
                batch.append(request)
                if len(batch) >= MAX_BATCH_SIZE:
                    need_sleep = False
                    break
            except asyncio.TimeoutError:
                break
        if not len(batch):
            continue

        uuid_str = str(uuid.uuid4())
        logging.debug(f'{uuid_str}, dynamic batching size {len(batch)}')
        if tracing.enabled:
            # queued -> picked up by this collector: the wait that MICROSLEEP and a
            # busy event loop add before the request is even part of a batch.
            t_batched = tracing.now_ns()
            for _, _, meta in batch:
                if meta is None:
                    continue
                tracing.record_span(
                    'codec.batch_wait', meta['t_enqueue'], t_batched, parent=meta['ctx'],
                    attrs={'batch.size': len(batch)},
                )
                meta['t_batched'] = t_batched
        try:
            batch_queue.put((uuid_str, batch))
        except Exception as e:
            # this task is the only feeder for the decode threads -- if it dies the whole
            # service stops decoding with no error surfaced anywhere. Never let it exit.
            logging.exception(f'{uuid_str}, dynamic_batching failed to enqueue: {e}')
            for fut, _, _ in batch:
                if not fut.done():
                    fut.set_exception(e)

def vc_compute_thread_fn(loop):
    while True:
        uuid_str, ys, futures, metas = vc_compute_queue.get()
        logging.debug(f'{uuid_str}, enter vc_compute_thread_fn, batch size {len(ys)}')
        t_compute = tracing.now_ns()
        try:
            tokens = batch_encode(ys)
            if tracing.enabled:
                t_done = tracing.now_ns()
                for meta in metas:
                    if meta is None:
                        continue
                    tracing.record_span(
                        'codec.encode_wait', meta['t_enqueue'], t_compute, parent=meta['ctx'],
                    )
                    tracing.record_span(
                        'codec.encode_gpu', t_compute, t_done, parent=meta['ctx'],
                        attrs={'batch.size': len(ys), 'codec.device': device},
                    )
            for i, fut in enumerate(futures):
                thread_safe_set_result(loop, fut, tokens[i])
        except Exception as e:
            logging.exception(f'{uuid_str}, vc_compute_thread_fn failed: {e}')
            thread_safe_set_exception(loop, futures, e)

def vc_batch_thread_fn(loop):
    while True:
        uuid_str, batch = vc_batch_queue.get()
        logging.debug(f'{uuid_str}, enter vc_batch_thread_fn, batch size {len(batch)}')
        try:
            futures, ys, metas = zip(*[(b[0], b[1], b[2]) for b in batch])
            vc_compute_queue.put((uuid_str, list(ys), futures, metas))
        except Exception as e:
            logging.exception(f'{uuid_str}, vc_batch_thread_fn failed: {e}')
            thread_safe_set_exception(loop, [b[0] for b in batch], e)

async def vc_dynamic_batching():
    need_sleep = True
    while True:
        if need_sleep:
            await asyncio.sleep(MICROSLEEP)
        need_sleep = True
        batch = []
        while not vc_dynamic_batch_queue.empty():
            try:
                request = await asyncio.wait_for(vc_dynamic_batch_queue.get(), timeout=1e-9)
                batch.append(request)
                if len(batch) >= MAX_BATCH_SIZE:
                    need_sleep = False
                    break
            except asyncio.TimeoutError:
                break
        if not len(batch):
            continue

        uuid_str = str(uuid.uuid4())
        logging.debug(f'{uuid_str}, vc dynamic batching size {len(batch)}')
        try:
            vc_batch_queue.put((uuid_str, batch))
        except Exception as e:
            logging.exception(f'{uuid_str}, vc_dynamic_batching failed to enqueue: {e}')
            for fut, _, _ in batch:
                if not fut.done():
                    fut.set_exception(e)

async def await_batched(future, what):
    """Wait on a batched worker future, bounded by BATCH_TIMEOUT.

    An unbounded wait means one wedged worker thread silently hangs every request
    that follows it -- the service looks alive and answers nothing.
    """
    if not BATCH_TIMEOUT:
        return await future
    try:
        return await asyncio.wait_for(future, timeout=BATCH_TIMEOUT)
    except asyncio.TimeoutError:
        logging.error(f'{what} timed out after {BATCH_TIMEOUT}s -- worker may be wedged')
        raise HTTPException(
            status_code=503,
            detail=f'{what} timed out after {BATCH_TIMEOUT}s',
        )

async def encode_audio(y):
    with tracing.span(
        'codec.encode',
        attrs={'codec.samples': len(y), 'codec.dynamic_batching': DYNAMIC_BATCHING},
    ) as sp:
        if DYNAMIC_BATCHING:
            future = asyncio.Future()
            await vc_dynamic_batch_queue.put((future, y, tracing.stage_meta(sp)))
            tokens = await await_batched(future, 'encode_audio')
        else:
            tokens = batch_encode([y])
            tokens = tokens[0]
        tracing.set_attributes(sp, {'codec.tokens': len(tokens)})
    return tokens

async def decode_speech_token(speech_token, parent=None):
    """`parent` pins the span to the request that asked for the decode.

    Needed because this is awaited from the streaming generator, which Starlette
    iterates in its own task -- and because everything downstream of the queue runs
    in the batching threads, where the ambient context is meaningless.
    """
    numbers = re.findall(r's_(\d+)', speech_token)
    d = list(map(int, numbers))
    if not d:
        logging.warning(f'decode_speech_token received empty tokens from: {speech_token[:100]}')
        return (sr, np.array([], dtype=np.float64))
    with tracing.span(
        'codec.decode', parent=parent,
        attrs={'codec.tokens': len(d), 'codec.dynamic_batching': DYNAMIC_BATCHING},
    ) as sp:
        if DYNAMIC_BATCHING:
            future = asyncio.Future()
            await dynamic_batch_queue.put((future, d, tracing.stage_meta(sp)))
            y_gen = await await_batched(future, 'decode_speech_token')
        else:
            audio_codes = torch.tensor(d)[None, None]
            y_gen = codec.decode_code(audio_codes.to(device))

        return (sr, y_gen[0, 0].cpu().numpy())

async def time_stretch_pcm16(gen, rate, sr):
    """Apply a speaking-rate change to a stream of int16 PCM chunks (see app/timestretch.py)."""
    ts = WSOLA(rate, sr)
    async for chunk in gen:
        b = float_to_pcm16(ts.process(pcm16_to_float(chunk)))
        if b:
            yield b
    tail = float_to_pcm16(ts.flush())
    if tail:
        yield tail

async def stream_speech(
    prompt,
    model,
    max_tokens,
    temperature,
    repetition_penalty,
    playback_speed,
    playback_overlap_speed,
    response_format,
    stream,
    request,
    stream_format="audio",
    stream_normalize=STREAM_NORMALIZE,
    speaking_rate=DEFAULT_SPEAKING_RATE,
    context=None,
    extra_headers=None,
):
    """`context` is the request's RequestContext (app/context.py) or None: when set,
    the prompt already carries the previous turns and the LM reader commits this
    turn's tokens to the store once generation finishes cleanly. `extra_headers`
    are added to every response shape (the X-Context-* headers)."""
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }
    if TTS_API_KEY:
        headers['Authorization'] = f'Bearer {TTS_API_KEY}'

    json_data = {
        'model': model,
        'prompt': prompt,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'repetition_penalty': repetition_penalty,
        'stream': True,
    }

    # `tts.stream` covers the whole response, which outlives this function: the
    # generator below is iterated by Starlette after we return, so the span is started
    # by hand here and ended in with_producer_cleanup(). Its context is the explicit
    # parent for everything that runs off this task (the LM producer, each decode).
    traced = tracing.enabled
    stream_span = tracing.start_span('tts.stream', attrs={
        'tts.model': model,
        'tts.max_tokens': max_tokens,
        'tts.temperature': temperature,
        'tts.prompt_chars': len(prompt),
        'tts.playback_speed': playback_speed,
        'tts.playback_overlap_speed': playback_overlap_speed,
        'tts.response_format': response_format,
        'tts.stream': stream,
        'tts.stream_format': stream_format,
        'tts.stream_normalize': stream_normalize,
        'tts.speaking_rate': speaking_rate,
        'tts.crossfade': STREAM_CROSSFADE,
        'tts.context_id': context.key if context is not None else '',
        'tts.context_mode': context.mode if context is not None else '',
        'tts.context_turns': len(context.turns) if context is not None else 0,
        'tts.context_tokens': context.tokens if context is not None else 0,
    })
    stream_ctx = tracing.context_with(stream_span)
    t_stream_start = time.perf_counter() if traced else 0.0
    # Aggregates, attached to `tts.stream` when the response finishes. Aggregated
    # rather than spanned because a request streams thousands of LM tokens: a span
    # each would bury the trace (and cost more than the decode it measures).
    stats = {
        'tts.lm_wait_s': 0.0,       # time the stitcher sat blocked on the LM queue
        'tts.decode_wait_s': 0.0,   # time it sat blocked on codec decodes
        'tts.decodes': 0,
        'tts.chunks': 0,
        'tts.audio_bytes': 0,       # decoded (pre speaking-rate stretch) bytes
        'tts.ttfb_s': None,
    }

    def counted(b):
        """Count an emitted chunk (and the first one's latency)."""
        if traced:
            stats['tts.chunks'] += 1
            stats['tts.audio_bytes'] += len(b)
            if stats['tts.ttfb_s'] is None:
                stats['tts.ttfb_s'] = time.perf_counter() - t_stream_start
        return b

    async def decode_counted(speech_token, parent=None):
        """decode_speech_token + the aggregate "waiting for the codec" bookkeeping.

        The wait covers queueing, batching and the GPU alike -- `codec.batch_wait` /
        `codec.gpu_decode` inside the decode's own span say which of those it was.
        """
        t0 = time.perf_counter() if traced else 0.0
        out = await decode_speech_token(speech_token, parent=parent or stream_ctx)
        if traced:
            stats['tts.decode_wait_s'] += time.perf_counter() - t0
            stats['tts.decodes'] += 1
        return out

    queue = asyncio.Queue()

    async def generate_audio_stream():
        # The consumer blocks on `queue` until it sees a terminator, so EVERY exit path
        # of this producer must leave one behind -- including cancellation and unexpected
        # exceptions. A missing terminator strands the consumer forever, which under load
        # silently accumulates dead tasks until the event loop stops serving requests.
        #
        # `lm.generate` is the whole "waiting on vLLM" story from the producer's side:
        # lm.connect (POST -> response headers) + lm.first_token (headers -> first
        # speech token, i.e. prefill) + the tail of the token stream. Started by hand
        # rather than with a `with` so a cancelled producer -- the normal outcome of a
        # client disconnect -- is an event, not an ERROR span.
        lm_span = tracing.start_span('lm.generate', parent=stream_ctx, attrs={
            'lm.url': TTS_API,
            'lm.model': model,
            'lm.max_tokens': max_tokens,
            'lm.temperature': temperature,
            'lm.repetition_penalty': repetition_penalty,
            'lm.prompt_chars': len(prompt),
            'lm.dummy_tokens': bool(DUMMY_TOKENS_FILE),
        })
        lm_ctx = tracing.context_with(lm_span)
        n_deltas = 0
        # Speech context (`request_id`): everything the LM emits, saved as the next
        # turn once the stream ended cleanly. A `length` finish means the tokens stop
        # short of the text -- a misaligned pair is worse context than none.
        lm_text = []
        lm_done = False
        finish_reason = None
        # `request.is_disconnected()` performs a real ASGI receive on every call, and
        # aiohttp yields two lines per SSE event (the data line and the blank one), so
        # polling it per line meant ~2 receives per speech token -- pure overhead on the
        # GIL-bound loop, and one instrumented span each when tracing is on. Poll on a
        # time budget instead; the consumer closing the stream still cancels this task
        # immediately (with_producer_cleanup), so this only bounds how long a vanished
        # client can keep the LM generating.
        next_disconnect_check = 0.0

        async def client_gone():
            nonlocal next_disconnect_check
            now = time.monotonic()
            if now < next_disconnect_check:
                return False
            next_disconnect_check = now + DISCONNECT_POLL_S
            return await request.is_disconnected()

        try:
            # DUMMY_TOKENS mode: replay canned speech tokens (simulating the vLLM SSE
            # stream) instead of calling the LM. One token per event, DUMMY_REPEAT times.
            if DUMMY_TOKENS_FILE:
                toks = re.findall(r"<\|s_\d+\|>", DUMMY_TOKENS)
                for _ in range(max(1, DUMMY_REPEAT)):
                    for t in toks:
                        if await client_gone():
                            break
                        await queue.put({'result': t})
                        n_deltas += 1
                        if DUMMY_TOKEN_DELAY > 0:
                            await asyncio.sleep(DUMMY_TOKEN_DELAY)
                return

            # Speech context: with history in the prompt the LM sometimes decides the
            # utterance is already over and emits end-of-speech after a handful of tokens
            # (see FALLBACK_* in app/context.py). Hold the first tokens back -- always
            # fewer than the first decode window, so nothing is delayed -- and if the LM
            # stops before that many, regenerate this chunk from the plain prompt: what
            # a request without request_id would have produced. Nothing has reached the
            # stitcher yet, so the switch is invisible to the client.
            hold_tokens = (
                context.fallback_hold_tokens()
                if (CONTEXT_FALLBACK and context is not None) else 0
            )
            attempt_prompt = prompt
            fallback_used = False
            while True:
                held = []                   # deltas withheld from the stitcher so far
                held_tokens = 0
                released = hold_tokens == 0
                lm_text = []
                lm_done = False
                finish_reason = None
                json_data['prompt'] = attempt_prompt
                t_post = tracing.now_ns()
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        TTS_API,
                        headers=headers,
                        json=json_data,
                    ) as resp:
                        t_headers = tracing.now_ns()
                        tracing.record_span(
                            'lm.connect', t_post, t_headers, parent=lm_ctx,
                            attrs={'http.response.status_code': resp.status},
                        )
                        tracing.set_attributes(lm_span, {'http.response.status_code': resp.status})
                        if resp.status != 200:
                            error_text = await resp.text()
                            logging.error(f"Backend error: {resp.status} - {error_text}")
                            queue.put_nowait({'error': f'backend returned {resp.status}: {error_text[:200]}'})
                            return

                        async for line in resp.content:
                            if await client_gone():
                                tracing.add_event(lm_span, 'client_disconnected')
                                break
                            if line.startswith(b"data: "):
                                data_str = line.decode("utf-8").strip()[6:]
                                if data_str == "[DONE]":
                                    lm_done = True
                                    break
                                try:
                                    data_json = json.loads(data_str)
                                    delta = data_json["choices"][0]
                                    finish_reason = delta.get("finish_reason") or finish_reason
                                    if "text" in delta:
                                        if not n_deltas:
                                            # prefill: what the request actually waited for
                                            tracing.record_span(
                                                'lm.first_token', t_headers, tracing.now_ns(),
                                                parent=lm_ctx,
                                            )
                                        n_deltas += 1
                                        text = delta["text"]
                                        if context is not None:
                                            lm_text.append(text)
                                        if released:
                                            await queue.put({'result': text})
                                        else:
                                            held.append(text)
                                            held_tokens += text.count('<|s_')
                                            if held_tokens >= hold_tokens:
                                                for h in held:
                                                    await queue.put({'result': h})
                                                held = []
                                                released = True
                                except json.JSONDecodeError:
                                    continue

                if not released and lm_done and context is not None and not fallback_used:
                    logging.warning(
                        f'context {context.key} ({context.mode}): LM stopped after '
                        f'{held_tokens} speech tokens (< {hold_tokens}) -- regenerating without context'
                    )
                    tracing.add_event(lm_span, 'context_fallback', {
                        'context.mode': context.mode, 'lm.tokens': held_tokens, 'lm.hold': hold_tokens,
                    })
                    tracing.set_attributes(stream_span, {'tts.context_fallback': True})
                    fallback_used = True
                    attempt_prompt = context.plain_prompt()
                    continue
                for h in held:              # short but final: release what there is
                    await queue.put({'result': h})
                break

        except asyncio.CancelledError:
            tracing.add_event(lm_span, 'cancelled')
            raise
        except Exception as e:
            logging.exception(f'generate_audio_stream failed: {e}')
            tracing.record_exception(lm_span, e)
            queue.put_nowait({'error': str(e)})
        finally:
            tracing.end_span(lm_span, attrs={
                'lm.deltas': n_deltas, 'lm.finish_reason': finish_reason or '',
            })
            if context is not None:
                if lm_done and finish_reason != 'length':
                    # Synchronous and *before* the terminator below: the stitcher
                    # cannot finish the response until it sees None, so by the time
                    # the client has chunk N, chunk N is in the store for chunk N+1
                    # -- whichever worker that one lands on.
                    ids = [int(m) for m in TOKEN_RE.findall(''.join(lm_text))]
                    turns = context.commit(ids)
                    if turns is not None:
                        logging.info(
                            f'context {context.key}: +{len(ids)} tokens -> '
                            f'{len(turns)} turns / {total_tokens(turns)} tokens stored'
                        )
                else:
                    logging.info(
                        f'context {context.key}: turn not stored '
                        f'(lm_done={lm_done}, finish_reason={finish_reason})'
                    )
            # unbounded queue -> put_nowait cannot block or raise, and is safe to run
            # while the task is being cancelled. A duplicate terminator is harmless:
            # the consumer stops at the first one and drops the queue.
            queue.put_nowait(None)

    producer_task = asyncio.create_task(generate_audio_stream())

    samples_per_token = sr // codebook_size                  # 480 samples / speech token
    chunk_size = int(playback_speed * codebook_size)         # tokens finalized per step (hop)
    overlap = int(playback_overlap_speed * codebook_size)    # neighbour-context tokens / side
    overlap_chunk = int((overlap / codebook_size) * sr)      # (legacy path only)
    ctx = max(1, overlap)                                     # crossfade needs >=1 token of context

    # crossfade width in samples, bounded so the ramp fits inside the context and
    # leaves a non-empty chunk core.
    xf = int(sr * (CROSSFADE_MS / 1000.0))
    xf = max(2, min(xf, ctx * samples_per_token, (chunk_size * samples_per_token) // 2))
    half = xf // 2

    all_ids = []

    def cos_ramp(n):
        # raised-cosine 0->1: derivative 0 at both ends, and up + reversed(up) == 1
        # -> equal-gain crossfade with no slope kink at the ramp edges.
        return 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, n)))

    def snap_to_zero_crossing(y_, margin=256):
        """(legacy path) trim to the nearest zero crossing near the chunk end."""
        if len(y_) < margin * 2:
            return y_
        search_region = y_[-margin:]
        zero_crossings = np.where(np.diff(np.signbit(search_region)))[0]
        if len(zero_crossings) > 0:
            cut = len(y_) - margin + zero_crossings[-1]
            return y_[:cut]
        return y_

    async def next_output():
        # Block on the queue rather than spinning on get_nowait()/sleep(1e-9): the spin
        # burns a full core on the single-threaded event loop, and if the producer ever
        # dies without a terminator it spins forever and starves every other request.
        if not traced:
            return await queue.get()
        t0 = time.perf_counter()
        out = await queue.get()
        # The vLLM wait from the consumer's side: how long the stitcher had nothing to
        # decode. Big here => the LM is what the request is waiting on, not the codec.
        stats['tts.lm_wait_s'] += time.perf_counter() - t0
        return out

    async def audio_stream_crossfade():
        """Streaming decode with context-primed windows + a raised-cosine crossfade
        at every chunk boundary.

        The NeuCodec decoder is non-causal (bidirectional attention over the window,
        conv receptive field, ISTFT 'same' padding), so a token decoded at the cold
        edge of an isolated chunk differs from the same token decoded with real
        neighbours -> splicing two chunks produces a click. Here each chunk is
        decoded with `ctx` tokens of real neighbour context on both sides (whose
        audio only warms the codec edges and is discarded), and adjacent chunk
        waveforms are blended over `xf` samples so the join is continuous.
        """
        all_ids.clear()
        text_buf = ""
        prev_xf = None        # samples held back straddling the last emitted boundary
        seg_start = 0                    # token index where the next window begins
        seg_len = chunk_size             # current window length; grows geometrically
        max_seg_tokens = max(chunk_size, int(STREAM_MAX_CHUNK_S * codebook_size))
        count = 0
        # running loudness state (STREAM_NORMALIZE): cumulative sum-of-squares / count of
        # active samples across the utterance so far, the running raw peak, and the
        # currently applied gain.
        norm_sq = 0.0
        norm_n = 0
        norm_peak = 0.0
        norm_gain_db = None
        # Freeze the gain once this much voiced audio has been seen: a causal AGC that
        # keeps adapting produces audible mid-utterance gain drift ("damping"); a single
        # per-utterance trim from the first second is within ~1 dB of the full-utterance
        # estimate and moves nothing afterwards. Later peaks are absorbed by the limiter.
        norm_lock_n = int(1.0 * sr)
        # Peaky voices (crest factor ~19 dB, raw peaks already near full scale) hard-clip
        # audibly when RMS-boosted. Cap the boost so gained peaks stay <= LIMITER_DRIVE,
        # and round whatever still exceeds the knee with a tanh soft clip instead of the
        # flat-top hard clip (which crackles).
        LIMITER_KNEE = 0.85
        LIMITER_DRIVE = 1.4

        def normalize_chunk(y):
            nonlocal norm_sq, norm_n, norm_peak, norm_gain_db
            locked = norm_gain_db is not None and norm_n >= norm_lock_n
            if not locked:
                active = y[np.abs(y) > 10 ** (-50 / 20)]    # gate out silence (< -50 dBFS)
                if len(active):
                    norm_sq += float(np.sum(active.astype(np.float64) ** 2))
                    norm_n += len(active)
                    norm_peak = max(norm_peak, float(np.abs(y).max()))
                if not norm_n:
                    return y
                est_db = 10 * np.log10(norm_sq / norm_n)    # == 20*log10(active RMS)
                want_db = float(np.clip(TARGET_RMS_DB - est_db, -MAX_GAIN_DB, MAX_GAIN_DB))
                if norm_peak > 0:
                    want_db = min(want_db, 20 * np.log10(LIMITER_DRIVE / norm_peak))
                if norm_gain_db is None:
                    norm_gain_db = want_db
                else:
                    norm_gain_db += float(np.clip(want_db - norm_gain_db, -GAIN_SLEW_DB, GAIN_SLEW_DB))
            y = y * (10 ** (norm_gain_db / 20))
            over = np.abs(y) > LIMITER_KNEE
            if np.any(over):
                y = np.where(
                    over,
                    np.sign(y) * (LIMITER_KNEE + (1.0 - LIMITER_KNEE)
                                  * np.tanh((np.abs(y) - LIMITER_KNEE) / (1.0 - LIMITER_KNEE))),
                    y,
                )
            return y

        def to_bytes(y):
            if stream_normalize:
                y = normalize_chunk(y)
            return (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

        # left/past context: free latency-wise (tokens already exist), decoded then
        # sliced off; right/future context stays small because it delays emission.
        past_ctx = max(ctx, int(STREAM_PAST_CONTEXT_S * codebook_size))

        async def emit_step(s, e, is_last):
            nonlocal prev_xf, count
            if is_last:
                e = len(all_ids)
            ds = max(0, s - past_ctx)
            de = len(all_ids) if is_last else min(len(all_ids), e + ctx)
            # one span per emitted chunk: the unit a client actually waits for, and the
            # parent of the decode so the batching stages hang off the right chunk.
            with tracing.span('tts.chunk', parent=stream_ctx, attrs={
                'chunk.index': stats['tts.chunks'],
                'chunk.tokens': e - s,
                'chunk.decode_tokens': de - ds,
                'chunk.is_last': is_last,
            }) as chunk_span:
                _, a = await decode_counted(
                    "".join(f"<|s_{i}|>" for i in all_ids[ds:de]),
                    parent=tracing.context_with(chunk_span),
                )
                if len(a) == 0:
                    return None
                off = ds * samples_per_token
                bk = s * samples_per_token        # boundary with the previous window
                be = e * samples_per_token        # boundary with the next window
                parts = []
                if s == 0:
                    emit_lo = 0
                else:
                    cx = a[bk - half - off: bk + half - off]
                    n = min(len(cx), len(prev_xf) if prev_xf is not None else 0)
                    if n > 0:
                        r = cos_ramp(n)
                        parts.append(prev_xf[:n] * (1.0 - r) + cx[:n] * r)
                    emit_lo = bk + half
                if is_last:
                    parts.append(a[emit_lo - off:])
                    prev_xf = None
                else:
                    parts.append(a[emit_lo - off: be - half - off])
                    prev_xf = a[be - half - off: be + half - off].copy()
                y = np.concatenate(parts) if len(parts) != 1 else parts[0]
                if len(y) == 0:
                    return None
                if DEBUG_AUDIO:
                    sf.write(f'/app/app/{count}.wav', y, sr)
                    count += 1
                b = to_bytes(y)
                tracing.set_attributes(chunk_span, {
                    'chunk.samples': len(y),
                    'chunk.audio_bytes': len(b),
                })
                return b

        while True:
            output = await next_output()
            if output is None:
                break
            if "error" in output:
                raise HTTPException(status_code=400, detail=output["error"])
            text_buf += output["result"]
            last = 0
            for m in TOKEN_RE.finditer(text_buf):
                all_ids.append(int(m.group(1)))
                last = m.end()
            text_buf = text_buf[last:]
            # emit every window that now has its full right-hand context available.
            # Windows grow geometrically (STREAM_CHUNK_GROWTH, capped at
            # STREAM_MAX_CHUNK_S): the first stays at chunk_size for TTFB, later ones
            # decode near-one-shot, shrinking the non-causal decoder's window tilt.
            while seg_start + seg_len + ctx <= len(all_ids):
                b = await emit_step(seg_start, seg_start + seg_len, is_last=False)
                if b:
                    yield counted(b)
                    await asyncio.sleep(0)
                seg_start += seg_len
                seg_len = min(int(seg_len * max(1.0, STREAM_CHUNK_GROWTH)), max_seg_tokens)

        # flush the tail: one final decode covering all remaining tokens
        if seg_start < len(all_ids):
            b = await emit_step(seg_start, len(all_ids), is_last=True)
            if b:
                yield counted(b)
                await asyncio.sleep(0)
        elif prev_xf is not None and len(prev_xf):
            yield counted(to_bytes(prev_xf))
            await asyncio.sleep(0)

    async def audio_stream_legacy():
        buffer = []
        to_yield = 0
        count = 0
        leftover = np.array([], dtype=np.float64)

        while True:
            output = await next_output()

            if output is None:
                break

            if "error" in output:
                raise HTTPException(status_code=400, detail=output["error"])

            output = output["result"]

            buffer.append(output)

            if len(buffer) % chunk_size == 0:
                _, y = await decode_counted("".join(buffer))
                if len(y) == 0:
                    continue
                y_ = y[to_yield : -overlap_chunk]
                y_ = np.clip(y_, -1.0, 1.0)
                y_ = np.concatenate([leftover, y_]) if len(leftover) else y_
                trimmed = snap_to_zero_crossing(y_)
                leftover = y_[len(trimmed):]

                if DEBUG_AUDIO:
                    sf.write(f'/app/app/{count}.wav', trimmed, sr)

                yield counted((trimmed * 32767).astype(np.int16).tobytes())
                await asyncio.sleep(0)

                if to_yield == 0:
                    to_yield = len(y) - to_yield - overlap_chunk

                buffer = buffer[-chunk_size:]
                count += 1

        if len(buffer):
            _, y = await decode_counted("".join(buffer))
            if len(y) == 0 and len(leftover):
                yield counted((leftover * 32767).astype(np.int16).tobytes())
                await asyncio.sleep(0)
                return
            elif len(y) == 0:
                return
            y_ = y[to_yield :]
            y_ = np.clip(y_, -1.0, 1.0)
            y_ = np.concatenate([leftover, y_]) if len(leftover) else y_

            if DEBUG_AUDIO:
                sf.write(f'/app/app/{count}.wav', y_, sr)

            yield counted((y_ * 32767).astype(np.int16).tobytes())
            await asyncio.sleep(0)
        elif len(leftover):
            yield counted((leftover * 32767).astype(np.int16).tobytes())
            await asyncio.sleep(0)

    def with_producer_cleanup(gen):
        """Cancel the LM-reader task whenever the consumer stops.

        On client disconnect the response generator is closed but the producer task
        keeps running; under sustained load those orphans pile up and hold aiohttp
        sessions/sockets open until the app stops responding.
        """
        async def _wrapped():
            try:
                async for chunk in gen:
                    yield chunk
            except Exception as e:
                tracing.record_exception(stream_span, e)
                raise
            finally:
                if not producer_task.done():
                    producer_task.cancel()
                # the only place that runs for every exit path of the response,
                # streaming or not -- so it is where `tts.stream` ends.
                tracing.end_span(stream_span, attrs=stats)
        return _wrapped()

    stitched = audio_stream_crossfade() if STREAM_CROSSFADE else audio_stream_legacy()
    if abs(speaking_rate - 1.0) > 1e-6:
        # speaking rate = pitch-preserving WSOLA on the stitched PCM (app/timestretch.py).
        # Sits after crossfade + loudness normalization and before the format/transport
        # layers, so every response mode (pcm/wav, raw/SSE, buffered) gets it and the
        # token/decode pipeline is untouched. Stateful across chunks, ~55 ms lookahead.
        stitched = time_stretch_pcm16(stitched, speaking_rate, sr)
    func = with_producer_cleanup(stitched)
    stream_headers = {
        'Cache-Control': 'no-cache, no-store',
        'X-Accel-Buffering': 'no',
    }
    if extra_headers:
        stream_headers.update(extra_headers)

    # OpenAI-compatible SSE streaming (stream_format="sse"). The livekit
    # openai TTS plugin (>=1.x) requests this and parses speech.audio.delta /
    # speech.audio.done events; it cannot consume the raw audio/pcm stream.
    # Each delta carries base64(int16 PCM @ {sr}Hz mono) — exactly what the
    # plugin decodes and pushes (mime audio/pcm, sample_rate {sr}).
    if stream and stream_format == 'sse':
        async def sse_stream():
            try:
                async for chunk in func:
                    audio_b64 = base64.b64encode(chunk).decode('ascii')
                    evt = json.dumps({'type': 'speech.audio.delta', 'audio': audio_b64})
                    yield f'data: {evt}\n\n'
                    await asyncio.sleep(0)
            finally:
                done = json.dumps({
                    'type': 'speech.audio.done',
                    'usage': {'input_tokens': 0, 'output_tokens': 0},
                })
                yield f'data: {done}\n\n'
                yield 'data: [DONE]\n\n'
        return StreamingResponse(
            sse_stream(),
            media_type='text/event-stream',
            headers=stream_headers,
        )

    if stream:
        if response_format == 'wav':
            async def wav_stream():
                max_data = 0x7FFFFFFF - 36
                wav_header = struct.pack('<4sI4s', b'RIFF', max_data + 36, b'WAVE')
                wav_header += struct.pack('<4sIHHIIHH', b'fmt ', 16, 1, 1, sr, sr * 2, 2, 16)
                wav_header += struct.pack('<4sI', b'data', max_data)
                yield wav_header
                async for chunk in func:
                    yield chunk
            return StreamingResponse(wav_stream(), media_type="audio/wav", headers=stream_headers)
        else:
            stream_headers['X-Audio-Sample-Rate'] = str(sr)
            stream_headers['X-Audio-Channels'] = '1'
            stream_headers['X-Audio-Bit-Depth'] = '16'
            return StreamingResponse(func, media_type="audio/pcm", headers=stream_headers)
    else:
        ys = []
        async for y_ in func:
            ys.append(y_)
        merged_bytes = b"".join(ys)

        if response_format == 'wav':
            bio = io.BytesIO()
            with wave.open(bio, 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes(merged_bytes)

            wav_bytes = bio.getvalue()
            resp_headers = {
                "Content-Type": "audio/wav",
                "Accept-Ranges": "bytes",
                "Content-Length": str(len(wav_bytes)),
            }
            if extra_headers:
                resp_headers.update(extra_headers)
            return Response(content=wav_bytes, headers=resp_headers)

        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pcm")
            tmp.write(merged_bytes)
            tmp.close()
            return FileResponse(
                path=tmp.name,
                media_type="audio/L16; rate=24000; channels=1",
                filename="merged_audio.pcm",
                headers=dict(extra_headers or {}),
            )

class NormalizeRequest(BaseModel):
    input: str = "Hello! How can I help you?"
    normalize_malaysian: bool = DEFAULT_NORMALIZE_MALAYSIAN
    # "rule" = the built-in rule-based pipeline below; "llm" = OpenAI-compatible LLM
    # normalizer (app/llm_normalizer.py), requires OPENAI_BASE_URL/OPENAI_MODEL_NAME.
    # In llm mode normalize_malaysian is ignored. Defaults come from
    # DEFAULT_NORMALIZER_MODE / DEFAULT_NORMALIZE_MALAYSIAN in the environment.
    mode: NormalizerMode = NormalizerMode(DEFAULT_NORMALIZER_MODE)

class TTSRequest(NormalizeRequest):
    voice: str = DEFAULT_SPEAKER
    model: str = MODEL_NAME
    response_format: Literal["pcm", "wav"] = "pcm"
    # "audio" = raw audio bytes (default, back-compat); "sse" = OpenAI-style
    # Server-Sent Events with base64 PCM deltas (required by livekit openai TTS plugin).
    stream_format: Literal["audio", "sse"] = "audio"
    temperature: float = DEFAULT_TEMPERATURE
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY
    max_tokens: int = DEFAULT_MAX_TOKENS
    stream: bool = True
    playback_speed: float = DEFAULT_PLAYBACK_SPEED
    playback_overlap_speed: float = DEFAULT_PLAYBACK_OVERLAP_SPEED
    # per-request override of the STREAM_NORMALIZE env default (utterance loudness
    # normalization toward TARGET_RMS_DB in the crossfade stitcher).
    stream_normalize: bool = STREAM_NORMALIZE
    # speaking rate: 1.0 = as generated, 1.3 = 30% faster, 0.8 = slower; pitch preserved
    # (WSOLA time stretch on the decoded audio, app/timestretch.py). `speed` is accepted as
    # an alias so OpenAI-compatible clients (e.g. the livekit openai TTS plugin) work as is.
    speaking_rate: float = Field(
        DEFAULT_SPEAKING_RATE, ge=MIN_RATE, le=MAX_RATE,
        validation_alias=AliasChoices('speaking_rate', 'speed'),
    )
    # Cross-request speech context (app/context.py): requests that share a request_id
    # are prompted with the previous turns' text + speech tokens, so text an agent
    # chunks into several TTS calls (LiveKit sentence chunks) keeps one continuous
    # prosody instead of restarting cold at every chunk. Alias `context_id`; the
    # `X-Context-Id` request header works too, for clients that cannot add body
    # fields. Omit it for independent utterances. Ignored when CONTEXT_STORE=off.
    request_id: Optional[str] = Field(
        None, max_length=256,
        validation_alias=AliasChoices('request_id', 'context_id'),
    )
    # how the history is prompted (see app/context.py build_prompt): "turns" = closed
    # VC-style turns then a new turn; "continue" = one turn, previous tokens as prefix,
    # the LM resumes mid-utterance. Default CONTEXT_MODE.
    context_mode: Literal['turns', 'continue'] = CONTEXT_MODE

# pydantic does not validate defaults, so a bad env value would bypass the range above.
if not MIN_RATE <= DEFAULT_SPEAKING_RATE <= MAX_RATE:
    raise ValueError(
        f'DEFAULT_SPEAKING_RATE={DEFAULT_SPEAKING_RATE} must be within [{MIN_RATE}, {MAX_RATE}]'
    )

def _pre_normalize(s):
    s = sanitize_markdown(s)
    s = s.replace('\n', ' ')
    s = re.sub(r'[ ]+', ' ', s).strip()

    for k, v in before_replace_mapping.items():
        s = s.replace(k, v)
    return s


def _post_normalize(s):
    for k, v in replace_mapping.items():
        s = s.replace(k, v)

    if not s.endswith('.'):
        s = s + '.'

    return re.sub(r'[ ]+', ' ', s).strip()


def normalize_malaysian_text(s, normalize_malaysian=False):
    s = _pre_normalize(s)

    if normalize_malaysian:
        lang = lang_model.predict(s, k = 3)[0][0]
        normalize_in_english = 'english' in lang
        chinese_dominant = is_chinese_dominant(s)

        def replace_range(match):
            num1 = int(match.group(1))
            num2 = int(match.group(2))
            words1 = to_cardinal(num1, english=normalize_in_english)
            words2 = to_cardinal(num2, english=normalize_in_english)
            phrase = match.group(3)
            if normalize_in_english:
                to = 'to'
            else:
                to = 'hingga'
            return f"{words1} {to} {words2} {phrase}"

        s = expand_contractions(s)
        s, _protected_phones = protect_phone_numbers(s)
        s = pattern_range.sub(replace_range, s)
        s = restore_phone_numbers(s, _protected_phones)
        if not chinese_dominant:
            # letter/digit splitting is meant for Malay/English word-level tokenization; for Chinese
            # it would insert spaces inside glued tokens like "RM500" and break Chinese verbalization below.
            new_s = []
            for w in s.split():
                splitted = split_alpha_num(w).split()
                for i in range(len(splitted)):
                    if len(splitted[i]) == 1:
                        splitted[i] = splitted[i].upper()
                splitted = ' '.join(splitted)
                new_s.append(splitted)
            s = ' '.join(new_s)

        logging.info(f'out from internal normalizer: {s}')

        segments = re.split(r'(\S*[^\x00-\x7F]\S*)', s)
        normalized_parts = []
        for seg in segments:
            if re.search(r'[^\x00-\x7F]', seg):
                if CJK_RE.search(seg) and not KANA_RE.search(seg):
                    normalized_parts.append(normalize_chinese(seg))
                else:
                    normalized_parts.append(seg)
            elif seg.strip():
                leading_ws = seg[:len(seg) - len(seg.lstrip())]
                trailing_ws = seg[len(seg.rstrip()):]
                if chinese_dominant:
                    converted = normalize_chinese(seg.strip())
                    if converted != seg.strip():
                        normalized_parts.append(leading_ws + converted + trailing_ws)
                        continue
                result = normalizer.normalize(
                    seg.strip(),
                    normalize_hingga = False,
                    normalize_text = False,
                    normalize_word_rules = False,
                    normalize_cardinal = False,
                    normalize_ordinal = False,
                    normalize_time = True,
                    normalize_url = True,
                    normalize_email = True,
                    normalize_in_english=normalize_in_english,
                )
                normalized_parts.append(leading_ws + result['normalize'] + trailing_ws)
            else:
                normalized_parts.append(seg)
        s = ''.join(normalized_parts)

        logging.info(f'out from malaya normalizer: {s}')

        original_s = s
        s = apply_pronunciation_replacements(s)
        if s != original_s:
            logging.info(f'Pronunciation replacements applied: "{original_s}" -> "{s}"')

    return _post_normalize(s)


async def normalize_request_text(data, fallback_to_rule=False):
    """Dispatch on data.mode. llm mode keeps the same pre/post cleanup (markdown
    sanitization, replace mappings) around the LLM call. With fallback_to_rule
    (TTS path) an LLM failure degrades to the rule-based pipeline instead of
    failing the whole speech request."""
    with tracing.span('tts.normalize', attrs={
        'normalizer.mode': data.mode.value,
        'normalizer.malaysian': data.normalize_malaysian,
        'normalizer.chars_in': len(data.input),
    }) as sp:
        if data.mode == NormalizerMode.llm:
            if not (OPENAI_BASE_URL and OPENAI_MODEL_NAME):
                if not fallback_to_rule:
                    raise HTTPException(
                        status_code=400,
                        detail='mode="llm" requires OPENAI_BASE_URL and OPENAI_MODEL_NAME to be configured',
                    )
                logging.warning('llm normalizer not configured, falling back to rule-based')
                tracing.set_attributes(sp, {'normalizer.fallback': 'unconfigured'})
            else:
                s = _pre_normalize(data.input)
                try:
                    # its own span: an unreachable/slow LLM endpoint is a common cause of
                    # a TTS request that looks stalled before a single token is generated.
                    with tracing.span('normalize.llm', attrs={
                        'llm.model': OPENAI_MODEL_NAME,
                        'llm.url': OPENAI_BASE_URL,
                        'llm.timeout_s': OPENAI_TIMEOUT,
                    }):
                        s = await llm_normalize(s)
                    logging.info(f'out from llm normalizer: {s}')
                    s = _post_normalize(s)
                    tracing.set_attributes(sp, {'normalizer.chars_out': len(s)})
                    return s
                except LLMNormalizerError as e:
                    if not fallback_to_rule:
                        raise HTTPException(status_code=502, detail=f'llm normalizer failed: {e}')
                    logging.warning(f'llm normalizer failed, falling back to rule-based: {e}')
                    tracing.set_attributes(sp, {'normalizer.fallback': 'error'})
        with tracing.span('normalize.rule'):
            s = normalize_malaysian_text(data.input, normalize_malaysian=data.normalize_malaysian)
        tracing.set_attributes(sp, {'normalizer.chars_out': len(s)})
        return s


@app.post('/v1/audio/normalize')
async def normalize_text(data: NormalizeRequest):
    s = await normalize_request_text(data)
    return {'output': s, 'mode': data.mode}


@app.get('/v1/audio/speaker')
async def speaker():
    return SPEAKERS

def load_context(key, voice, text, max_tokens, mode=CONTEXT_MODE):
    """Read a context id's history and size it for this request (app/context.py).

    Returns (RequestContext, max_tokens): the turns that go into the prompt -- the
    trailing run in this voice, left-trimmed to CONTEXT_MAX_S and to the LM window --
    and the request's max_tokens clamped so prompt + generation fit LM_MAX_MODEL_LEN.
    A store read failure degrades to no context rather than failing the request.
    """
    with tracing.span('tts.context', attrs={'context.id': key}) as sp:
        try:
            history = context_store.get(key)
        except Exception as e:
            logging.warning(f'context {key}: store read failed, generating without context: {e}')
            history = []
        turns, fitted = fit_context(
            select_voice(history, voice), text, max_tokens,
            LM_MAX_MODEL_LEN, CONTEXT_MAX_TOKENS, CONTEXT_MIN_GEN_TOKENS,
        )
        ctx = RequestContext(store=context_store, key=key, voice=voice, text=text, turns=turns, mode=mode)
        logging.info(
            f'context {key} ({mode}): {len(turns)}/{len(history)} turns, {ctx.tokens} speech tokens in prompt'
            + (f', max_tokens {max_tokens} -> {fitted}' if fitted != max_tokens else '')
        )
        tracing.set_attributes(sp, {
            'context.mode': mode,
            'context.history_turns': len(history),
            'context.turns': len(turns),
            'context.tokens': ctx.tokens,
            'context.max_tokens': fitted,
        })
        return ctx, fitted


def prompt_for_log(prompt):
    """Collapse speech-token runs so a context prompt (thousands of tokens) logs as one line."""
    return re.sub(
        r'(?:<\|s_\d+\|>)+',
        lambda m: f'<{m.group(0).count("<|s_")} speech tokens>',
        prompt,
    )


@app.get('/v1/audio/context/{context_id}')
async def get_context(context_id: str):
    """Inspect a context id: the stored turns (text + token counts, not the tokens)."""
    if context_store is None:
        raise HTTPException(status_code=400, detail='speech context is disabled (CONTEXT_STORE=off)')
    turns = context_store.get(context_id)
    return {
        'context_id': context_id,
        'turns': [
            {'voice': t.voice, 'text': t.text, 'tokens': len(t.tokens), 'ts': t.ts}
            for t in turns
        ],
        'tokens': total_tokens(turns),
        'max_tokens': CONTEXT_MAX_TOKENS,
        'ttl_s': CONTEXT_TTL_S,
    }


@app.delete('/v1/audio/context/{context_id}')
async def delete_context(context_id: str):
    """Forget a context id (e.g. when the agent session ends); the next request on it
    starts cold. Idle ids expire by themselves after CONTEXT_TTL_S."""
    if context_store is None:
        raise HTTPException(status_code=400, detail='speech context is disabled (CONTEXT_STORE=off)')
    return {'context_id': context_id, 'deleted': context_store.delete(context_id)}


@app.post('/v1/audio/speech')
async def tts_stream(data: TTSRequest, request: Request = None):
    speaker = data.voice
    max_tokens = data.max_tokens
    context = None
    # body field first; the header is for clients that cannot add body fields
    context_key = data.request_id or (
        request.headers.get('x-context-id') if request is not None else None
    )

    if DUMMY_TOKENS_FILE:
        # tokens are replayed from DUMMY_TOKENS_FILE; the prompt is unused.
        prompt = ''
    else:
        s = await normalize_request_text(data, fallback_to_rule=True)
        logging.info(f'normalized: {s}')
        if context_key and context_store is not None:
            context, max_tokens = load_context(context_key, speaker, s, data.max_tokens, data.context_mode)
        # with no context turns this is exactly the plain single-turn prompt
        prompt = build_prompt(
            context.turns if context is not None else [], speaker, s,
            mode=context.mode if context is not None else 'turns',
        )
        logging.info(f'prompt: {prompt_for_log(prompt)}')

    return await stream_speech(
        prompt=prompt,
        model=data.model,
        max_tokens=max_tokens,
        temperature=data.temperature,
        repetition_penalty=data.repetition_penalty,
        playback_speed=data.playback_speed,
        playback_overlap_speed=data.playback_overlap_speed,
        response_format=data.response_format,
        stream=data.stream,
        request=request,
        stream_format=data.stream_format,
        stream_normalize=data.stream_normalize,
        speaking_rate=data.speaking_rate,
        context=context,
        extra_headers=context.headers() if context is not None else None,
    )

def batch_encode(ys):
    with torch.no_grad():
        ys_pt = [codec._prepare_audio(torch.tensor(ys[i])[None, None])[0, 0] for i in range(len(ys))]
        features = codec.feature_extractor(ys_pt, sampling_rate=16_000, return_tensors="pt")
        semantic_features = features.input_features.to('cuda')
        padded = pad_sequence(ys_pt, batch_first=True)[:,None]
        acoustic_emb = codec.CodecEnc(padded.to('cuda'))
        acoustic_emb = acoustic_emb.transpose(1, 2)
        semantic_output = (
            codec.semantic_model(semantic_features).hidden_states[16].transpose(1, 2)
        )
        semantic_encoded = codec.SemanticEncoder_module(semantic_output)
        if acoustic_emb.shape[-1] != semantic_encoded.shape[-1]:
            min_len = min(acoustic_emb.shape[-1], semantic_encoded.shape[-1])
            acoustic_emb = acoustic_emb[:, :, :min_len]
            semantic_encoded = semantic_encoded[:, :, :min_len]        
        concat_emb = torch.cat([semantic_encoded, acoustic_emb], dim=1)
        concat_emb = codec.fc_prior(concat_emb.transpose(1, 2)).transpose(1, 2)

        _, fsq_codes, _ = codec.generator(concat_emb, vq=True)
        lens = features.attention_mask.sum(dim=1)
        fsq_codes = fsq_codes.cpu()

        tokens = []
        for i in range(lens.shape[0]):
            tokens.append(fsq_codes[i, 0, :lens[i]])
        
        return tokens


@app.post('/v1/audio/vc')
async def vc_stream(
    reference_audio: bytes = File(..., description="Reference audio"),
    reference_text: str = Form(..., description="Reference text"),
    generate_text: str = Form(..., description="Text to generate"),
    model: str = Form(default=MODEL_NAME, description="Model name"),
    response_format: str = Form(default="pcm", description="Response format: pcm or wav"),
    temperature: float = Form(default=DEFAULT_TEMPERATURE, description="Temperature"),
    repetition_penalty: float = Form(default=DEFAULT_REPETITION_PENALTY, description="Repetition penalty"),
    max_tokens: int = Form(default=DEFAULT_MAX_TOKENS, description="Max tokens"),
    stream: bool = Form(default=True, description="Stream response"),
    playback_speed: float = Form(default=DEFAULT_PLAYBACK_SPEED, description="Playback speed"),
    playback_overlap_speed: float = Form(default=DEFAULT_PLAYBACK_OVERLAP_SPEED, description="Playback overlap speed"),
    speaking_rate: float = Form(default=DEFAULT_SPEAKING_RATE, ge=MIN_RATE, le=MAX_RATE,
                                description="Speaking rate (1.0 = as generated, 1.3 = faster), pitch preserved"),
    request: Request = None
):
    file_like = io.BytesIO(reference_audio)
    with tracing.span('vc.load_audio', attrs={'audio.bytes': len(reference_audio)}):
        y, _ = librosa.load(file_like, sr=16000)

    codes = await encode_audio(y)

    reference_text = sanitize_markdown(reference_text)
    generate_text = sanitize_markdown(generate_text)

    tokens = ''.join([f'<|s_{i}|>' for i in codes])
    prompt = f"<|im_start|>{reference_text}<|speech_start|>{tokens}<|im_end|><|im_start|>{generate_text}<|speech_start|>"
    logging.debug(f'prompt: {prompt}')

    return await stream_speech(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        playback_speed=playback_speed,
        playback_overlap_speed=playback_overlap_speed,
        response_format=response_format,
        stream=stream,
        request=request,
        speaking_rate=speaking_rate,
    )

if len(SENTRY_DSN):
    @app.get("/sentry-debug")
    async def trigger_error():
        division_by_zero = 1 / 0

app.state.background_dynamic_batching = asyncio.create_task(dynamic_batching())
app.state.background_vc_dynamic_batching = asyncio.create_task(vc_dynamic_batching())

loop = asyncio.get_running_loop()
t1 = threading.Thread(
    target=batch_thread_fn,
    args=(loop,),
    daemon=True
)
t1.start()
t2 = threading.Thread(
    target=compute_thread_fn,
    args=(loop,),
    daemon=True
)
t2.start()
t3 = threading.Thread(
    target=vc_batch_thread_fn,
    args=(loop,),
    daemon=True
)
t3.start()
t4 = threading.Thread(
    target=vc_compute_thread_fn,
    args=(loop,),
    daemon=True
)
t4.start()
app.state.batch_thread = t1
app.state.compute_thread = t2
app.state.vc_batch_thread = t3
app.state.vc_compute_thread = t4