import os
import logging

TTS_API = os.environ.get('TTS_API', 'http://tts-engine:9093')
if '/v1/completions' not in TTS_API:
    TTS_API = TTS_API + '/v1/completions'
TTS_API_KEY = os.environ.get('TTS_API_KEY', '')
MODEL_NAME = os.environ.get('MODEL_NAME', 'TTS-model')
DEFAULT_SPEAKER = os.environ.get('DEFAULT_SPEAKER', 'TM_English')
SPEAKERS = os.environ.get('SPEAKERS', 'TM_Mandarin,TM_English')
# 0.6 is what the Whisper-CER guardrail and all bench results were measured at;
# higher temperatures also increase utterance-to-utterance loudness/prosody variance.
DEFAULT_TEMPERATURE = float(os.environ.get('DEFAULT_TEMPERATURE', '0.6'))
DEFAULT_REPETITION_PENALTY = float(os.environ.get('DEFAULT_REPETITION_PENALTY', '1.15'))
DEFAULT_MAX_TOKENS = int(os.environ.get('DEFAULT_MAX_TOKENS', '3072'))
# 2.0 => first decode window = 100 tokens = 2.0s of audio.
DEFAULT_PLAYBACK_SPEED = float(os.environ.get('DEFAULT_PLAYBACK_SPEED', '2.0'))
DEFAULT_PLAYBACK_OVERLAP_SPEED = float(os.environ.get('DEFAULT_PLAYBACK_OVERLAP_SPEED', '0.2'))
# Request-level normalization defaults (overridable per request). DEFAULT_NORMALIZER_MODE
# must be 'rule' or 'llm' (validated against NormalizerMode in app/main.py at import).
DEFAULT_NORMALIZE_MALAYSIAN = os.environ.get('DEFAULT_NORMALIZE_MALAYSIAN', 'false').lower() == 'true'
DEFAULT_NORMALIZER_MODE = os.environ.get('DEFAULT_NORMALIZER_MODE', 'rule')
# Streaming stitcher: overlap-add with context-primed windows + a raised-cosine
# crossfade at every chunk boundary, to remove the boundary 'snap'/click. The
# NeuCodec decoder is non-causal (bidirectional attention + conv + ISTFT 'same'
# padding), so each chunk decoded in isolation has different edge context; a hard
# splice of two such chunks clicks. Set STREAM_CROSSFADE=false for the old hard-cut
# behaviour. CROSSFADE_MS is the blend width (bounded to the context/chunk size).
STREAM_CROSSFADE = os.environ.get('STREAM_CROSSFADE', 'true').lower() == 'true'
CROSSFADE_MS = float(os.environ.get('CROSSFADE_MS', '12.0'))
# Streaming loudness normalization (on by default): the LM's sampled speech tokens carry
# loudness, so utterance level varies run-to-run (measured 3.5-10 dB active-RMS spread on
# identical text at temp 0.6-0.7) and hot utterances hard-clip at full scale. The
# crossfade stitcher keeps a running active-RMS estimate over the utterance's emitted
# audio and applies a gain toward TARGET_RMS_DB, slewed by <=GAIN_SLEW_DB per chunk
# (first chunk jumps straight to the estimate) and clamped to +/-MAX_GAIN_DB.
# Set STREAM_NORMALIZE=false for the raw decoder level.
STREAM_NORMALIZE = os.environ.get('STREAM_NORMALIZE', 'true').lower() == 'true'
# Growing decode windows for the crossfade stitcher: the first chunk stays at
# chunk_size (TTFB unchanged), each next window is chunk_size * GROWTH^k, capped at
# STREAM_MAX_CHUNK_S seconds. Bigger windows decode closer to the one-shot result
# (the non-causal decoder tilts/ripples the envelope ~0.7-1.5 dB in small windows).
# Set STREAM_CHUNK_GROWTH=1.0 for the old fixed-size behaviour. Note: big windows
# need the LM to stay ahead of playback; the ~10x-realtime vLLM easily does.
STREAM_CHUNK_GROWTH = float(os.environ.get('STREAM_CHUNK_GROWTH', '2.0'))
STREAM_MAX_CHUNK_S = float(os.environ.get('STREAM_MAX_CHUNK_S', '10.0'))
# Past-token context included in every decode window (sliced off after decode).
# Past tokens are already generated, so unlike the future/right context
# (playback_overlap_speed) this costs no latency, only a bigger decode — and it
# gives the non-causal decoder real history, pulling each window's interior
# toward the one-shot result. Seconds of audio (x50 = tokens).
STREAM_PAST_CONTEXT_S = float(os.environ.get('STREAM_PAST_CONTEXT_S', '3.0'))
TARGET_RMS_DB = float(os.environ.get('TARGET_RMS_DB', '-16.0'))
MAX_GAIN_DB = float(os.environ.get('MAX_GAIN_DB', '12.0'))
GAIN_SLEW_DB = float(os.environ.get('GAIN_SLEW_DB', '1.0'))
# Batches concurrent decode calls; essential under concurrency and free at
# concurrency 1 (a batch of one), so on by default.
DYNAMIC_BATCHING = os.environ.get('DYNAMIC_BATCHING', 'true').lower() == 'true'
MICROSLEEP = float(os.environ.get('MICROSLEEP', '1e-4'))
MAX_BATCH_SIZE = int(os.environ.get('MAX_BATCH_SIZE', '16'))
# Upper bound on how long a request waits for its decode/encode future to resolve.
# Without it, a wedged worker thread turns every in-flight and future request into a
# permanent silent hang; with it the request fails loudly and the service stays usable.
# Set to 0 to disable the timeout.
BATCH_TIMEOUT = float(os.environ.get('BATCH_TIMEOUT', '120'))
CUDA_GRAPH_BATCH = eval(os.environ.get('CUDA_GRAPH_BATCH', '[]'))
"""
Empty = eager decode. On CUDA, enabling buckets is the single biggest codec win (~1.7x).
Buckets are seconds x50 tokens and must cover the decode window sizes: with growing
windows + past context they reach STREAM_MAX_CHUNK_S + STREAM_PAST_CONTEXT_S (~13.5s);
oversize shapes silently fall back to eager. Nice value:
[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 13.5]
"""
TORCH_COMPILE = os.environ.get('TORCH_COMPILE', 'false').lower() == 'true'

# LLM-based text normalizer (mode="llm" on /v1/audio/normalize and TTS requests):
# any OpenAI-compatible /chat/completions endpoint. Note OPENAI_MODEL_NAME is
# deliberately separate from MODEL_NAME (the TTS model served by vLLM).
# Empty OPENAI_BASE_URL disables llm mode (requests get a 400).
OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL', '').rstrip('/')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
OPENAI_MODEL_NAME = os.environ.get('OPENAI_MODEL_NAME', '')
# Bounds how long a TTS request can stall on a hung LLM-normalizer endpoint before
# the rule-based fallback kicks in (normalize calls measure 1-2s in practice).
OPENAI_TIMEOUT = float(os.environ.get('OPENAI_TIMEOUT', '10'))
DEBUG_AUDIO = os.environ.get('DEBUG_AUDIO', 'false').lower() == 'true'
SENTRY_DSN = os.environ.get('SENTRY_DSN', '')

# Hot-path OpenTelemetry spans: how long a request spent queued in dynamic batching,
# waiting on vLLM, and inside the codec decode. Off by default -- when off every helper
# in app/tracing.py is a shared nullcontext / no-op, so the GIL-bound decode loop pays
# nothing at all (see app/tracing.py for the span tree). Needs fastapi-loki-tempo (the
# OTel SDK) installed, plus OTLP_ENDPOINT for the spans to reach Tempo; the rest of the
# tracing config (SERVICE_NAME, OTLP_*, TRACING_SAMPLE) belongs to that library.
ENABLE_TRACING_SPANS = os.environ.get('ENABLE_TRACING_SPANS', 'false').lower() == 'true'

# Compute device. Empty = auto-detect (cuda -> npu -> cpu). Set to 'npu' to run the
# codec decode on a Huawei Ascend NPU (via torch_npu). On non-cuda devices the CUDA
# graph warmup and CUDA streams are skipped automatically (see app/main.py).
DEVICE = os.environ.get('DEVICE', '')

# Dummy streaming-tokens mode: instead of streaming speech tokens from the vLLM LM,
# replay a canned "<|s_N|>..." token string read from this file (e.g. tokens extracted
# from a reference audio). Lets the decode/serving pipeline run without vLLM. Empty = off.
DUMMY_TOKENS_FILE = os.environ.get('DUMMY_TOKENS_FILE', '')
# How many times to replay the canned tokens per request ("keep streaming").
DUMMY_REPEAT = int(os.environ.get('DUMMY_REPEAT', '1'))
# Seconds to sleep between emitted dummy tokens (>0 simulates the LM's token pace).
DUMMY_TOKEN_DELAY = float(os.environ.get('DUMMY_TOKEN_DELAY', '0.0'))

SPEAKERS = [s.strip() for s in SPEAKERS.split(',')]

BUCKET_BATCHES = list(range(1, MAX_BATCH_SIZE + 1, 1))
