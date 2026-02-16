import os
import logging

TTS_API = os.environ.get('TTS_API', 'http://tts-engine:9093')
if '/v1/completions' not in TTS_API:
    TTS_API = TTS_API + '/v1/completions'
MODEL_NAME = os.environ.get('MODEL_NAME', 'TTS-model')
DEFAULT_SPEAKER = os.environ.get('DEFAULT_SPEAKER', 'husein')
SPEAKERS = os.environ.get('SPEAKERS', 'husein,idayu,jenny')
DEFAULT_TEMPERATURE = float(os.environ.get('DEFAULT_TEMPERATURE', '0.7'))
DEFAULT_REPETITION_PENALTY = float(os.environ.get('DEFAULT_REPETITION_PENALTY', '1.15'))
DEFAULT_MAX_TOKENS = int(os.environ.get('DEFAULT_MAX_TOKENS', '3072'))
DEFAULT_PLAYBACK_SPEED = float(os.environ.get('DEFAULT_PLAYBACK_SPEED', '1.5'))
DEFAULT_PLAYBACK_OVERLAP_SPEED = float(os.environ.get('DEFAULT_PLAYBACK_OVERLAP_SPEED', '0.2'))
DYNAMIC_BATCHING = os.environ.get('DYNAMIC_BATCHING', 'false').lower() == 'true'
MICROSLEEP = float(os.environ.get('MICROSLEEP', '1e-4'))
MAX_BATCH_SIZE = int(os.environ.get('MAX_BATCH_SIZE', '16'))
CUDA_GRAPH_BATCH = eval(os.environ.get('CUDA_GRAPH_BATCH', '[]'))
"""
Nice value is [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 10.0]
"""
TORCH_COMPILE = os.environ.get('TORCH_COMPILE', 'false').lower() == 'true'
DEBUG_AUDIO = os.environ.get('DEBUG_AUDIO', 'false').lower() == 'true'
SENTRY_DSN = os.environ.get('SENTRY_DSN', '')

SPEAKERS = [s.strip() for s in SPEAKERS.split(',')]

BUCKET_BATCHES = list(range(1, MAX_BATCH_SIZE + 1, 1))
