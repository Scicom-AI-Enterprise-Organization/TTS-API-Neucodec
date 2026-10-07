"""In-process LM (LM_BACKEND=inprocess): speech tokens from a vLLM engine this process owns.

The deployed path POSTs each prompt to a vLLM server (`TTS_API`, OpenAI /v1/completions,
stream=True) and reads text deltas off the SSE stream. `stream()` yields the same deltas from
vLLM's async engine inside this process, built from the same request body, so the producer in
`app/main.py` and everything after it (interleave hold-back, stitcher, loudness, fade-in) run
unchanged.

Why: to put a checkpoint through the serving path with ONE process per GPU or TP group, instead
of a vLLM server plus an API server (two ports) per model. bench/serve_inprocess.py launches it.

Two rules keep it the same LM as production:
  - Sampling is exactly what the HTTP body sends: temperature, repetition_penalty, max_tokens.
    Nothing else, so top-k and top-p stay vLLM's defaults (off), as on the served engine.
  - The checkpoint is relabelled the way the served engine relabels it (LM_HF_OVERRIDES,
    default architectures=Qwen3ForCausalLM).
"""
import asyncio
import json
import logging
import uuid

from app.env import (LM_DTYPE, LM_ENFORCE_EAGER, LM_GPU_MEMORY_UTILIZATION, LM_HF_OVERRIDES,
                     LM_MAX_MODEL_LEN, LM_MAX_NUM_SEQS, LM_MODEL, LM_TENSOR_PARALLEL_SIZE)

_engine = None
_lock = None


async def get_engine():
    """The engine, created once, on first use (or at startup, see app.main). Must run inside the
    serving event loop: the async engine starts its output handler on the running loop."""
    global _engine, _lock
    if _engine is not None:
        return _engine
    if _lock is None:
        _lock = asyncio.Lock()
    async with _lock:
        if _engine is None:
            from vllm import AsyncEngineArgs, AsyncLLMEngine
            args = AsyncEngineArgs(
                model=LM_MODEL, dtype=LM_DTYPE, max_model_len=LM_MAX_MODEL_LEN,
                tensor_parallel_size=LM_TENSOR_PARALLEL_SIZE,
                gpu_memory_utilization=LM_GPU_MEMORY_UTILIZATION,
                enforce_eager=LM_ENFORCE_EAGER,
                hf_overrides=json.loads(LM_HF_OVERRIDES) if LM_HF_OVERRIDES else None,
                **({'max_num_seqs': LM_MAX_NUM_SEQS} if LM_MAX_NUM_SEQS else {}),
            )
            logging.info(f'in-process LM: loading {LM_MODEL} (tp={LM_TENSOR_PARALLEL_SIZE}, '
                         f'gpu_memory_utilization={LM_GPU_MEMORY_UTILIZATION}, '
                         f'max_model_len={LM_MAX_MODEL_LEN}, eager={LM_ENFORCE_EAGER})')
            _engine = AsyncLLMEngine.from_engine_args(args)
    return _engine


def sampling_params(body):
    """The HTTP body's sampling fields as vLLM SamplingParams, and nothing more."""
    from vllm import SamplingParams
    return SamplingParams(temperature=float(body['temperature']),
                          repetition_penalty=float(body['repetition_penalty']),
                          max_tokens=int(body['max_tokens']))


def delta(text, sent):
    """(new text, new high-water mark) from a CUMULATIVE output (vLLM's default output kind)."""
    return text[sent:], len(text)


async def stream(prompt, body, engine=None):
    """Yield (text_delta, finish_reason) the way the SSE path sees them: finish_reason stays None
    until the last item. A consumer that stops early (client gone, task cancelled) aborts the
    request in the engine, so nothing keeps generating for nobody."""
    engine = engine or await get_engine()
    rid = f'tts-{uuid.uuid4().hex}'
    sent, finished = 0, False
    gen = engine.generate(prompt, sampling_params(body), rid)
    try:
        async for out in gen:
            o = out.outputs[0]
            d, sent = delta(o.text, sent)
            fr = o.finish_reason if out.finished else None
            if d or fr:
                yield d, fr
            if out.finished:
                finished = True
    finally:
        if not finished:
            try:
                await engine.abort(rid)
            except Exception as e:  # noqa: BLE001 -- best effort on the way out
                logging.warning(f'in-process LM: abort {rid} failed: {e}')
        # close the engine's generator now, not whenever it is garbage-collected
        await gen.aclose()
