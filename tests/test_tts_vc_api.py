"""
Integration tests for /v1/audio/speech (TTS) and /v1/audio/vc (Voice Conversion) endpoints.
Tests against a live API server. Set TTS_TEST_URL env var to override the default URL.

Run with:
    python -m pytest tests/test_tts_vc_api.py -v
    TTS_TEST_URL=http://localhost:9091 python -m pytest tests/test_tts_vc_api.py -v
"""

import sys
import os
import struct
import wave
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import requests

BASE_URL = os.environ.get('TTS_TEST_URL', 'http://localhost:9091')
JENNY_WAV = os.path.join(os.path.dirname(__file__), '..', 'jenny.wav')
JENNY_REF_TEXT = 'I wonder if I shall ever be happy enough to have real lace on my clothes and bows on my caps.'

try:
    r = requests.get(f'{BASE_URL}/v1/audio/speaker', timeout=5)
    APP_AVAILABLE = r.status_code == 200
    SKIP_REASON = '' if APP_AVAILABLE else f'API returned {r.status_code}'
except Exception as e:
    APP_AVAILABLE = False
    SKIP_REASON = str(e)


class LiveClient:
    """Wrapper around requests to match TestClient-like interface."""

    def get(self, path, **kwargs):
        return requests.get(f'{BASE_URL}{path}', timeout=120, **kwargs)

    def post(self, path, json=None, data=None, files=None, **kwargs):
        return requests.post(f'{BASE_URL}{path}', json=json, data=data,
                             files=files, timeout=120, **kwargs)


client = LiveClient()

skipif_no_app = pytest.mark.skipif(
    not APP_AVAILABLE,
    reason=f"API not reachable at {BASE_URL}: {SKIP_REASON}",
)


def is_valid_wav(data: bytes) -> bool:
    """Check if bytes contain a valid WAV header."""
    if len(data) < 44:
        return False
    return data[:4] == b'RIFF' and data[8:12] == b'WAVE'


def is_valid_pcm(data: bytes) -> bool:
    """Check if bytes look like valid PCM int16 data (non-empty, even length)."""
    return len(data) > 0 and len(data) % 2 == 0


def wav_sample_rate(data: bytes) -> int:
    """Extract sample rate from WAV header."""
    return struct.unpack_from('<I', data, 24)[0]


def wav_channels(data: bytes) -> int:
    """Extract number of channels from WAV header."""
    return struct.unpack_from('<H', data, 22)[0]


def wav_bit_depth(data: bytes) -> int:
    """Extract bits per sample from WAV header."""
    return struct.unpack_from('<H', data, 34)[0]


# ===========================================================================
# TTS endpoint tests
# ===========================================================================
@skipif_no_app
class TestTTSSpeechBasic:
    """Basic TTS endpoint tests."""

    def test_tts_default_returns_200(self):
        r = client.post('/v1/audio/speech', json={
            'input': 'Hello world.',
            'stream': False,
            'response_format': 'wav',
        })
        assert r.status_code == 200

    def test_tts_wav_format(self):
        r = client.post('/v1/audio/speech', json={
            'input': 'Selamat pagi.',
            'stream': False,
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert is_valid_wav(r.content)
        assert wav_sample_rate(r.content) == 24000
        assert wav_channels(r.content) == 1
        assert wav_bit_depth(r.content) == 16

    def test_tts_pcm_format(self):
        r = client.post('/v1/audio/speech', json={
            'input': 'Selamat pagi.',
            'stream': False,
            'response_format': 'pcm',
        })
        assert r.status_code == 200
        assert is_valid_pcm(r.content)

    def test_tts_streaming_wav(self):
        r = client.post('/v1/audio/speech', json={
            'input': 'Hello world.',
            'stream': True,
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert r.headers.get('content-type') == 'audio/wav'
        data = r.content
        assert is_valid_wav(data)

    def test_tts_streaming_pcm(self):
        r = client.post('/v1/audio/speech', json={
            'input': 'Hello world.',
            'stream': True,
            'response_format': 'pcm',
        })
        assert r.status_code == 200
        assert 'audio/pcm' in r.headers.get('content-type', '')
        assert r.headers.get('x-audio-sample-rate') == '24000'
        assert r.headers.get('x-audio-channels') == '1'
        assert r.headers.get('x-audio-bit-depth') == '16'
        assert is_valid_pcm(r.content)

    def test_tts_produces_audio_data(self):
        r = client.post('/v1/audio/speech', json={
            'input': 'Testing audio output.',
            'stream': False,
            'response_format': 'wav',
        })
        assert r.status_code == 200
        # WAV header is 44 bytes, audio data should add more
        assert len(r.content) > 44


@skipif_no_app
class TestTTSSpeakers:
    """Test different speaker voices."""

    def test_tts_speaker_husein(self):
        r = client.post('/v1/audio/speech', json={
            'input': 'Hello.',
            'voice': 'husein',
            'stream': False,
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert is_valid_wav(r.content)

    def test_tts_speaker_jenny(self):
        r = client.post('/v1/audio/speech', json={
            'input': 'Hello.',
            'voice': 'jenny',
            'stream': False,
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert is_valid_wav(r.content)

    def test_tts_speaker_idayu(self):
        r = client.post('/v1/audio/speech', json={
            'input': 'Hello.',
            'voice': 'idayu',
            'stream': False,
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert is_valid_wav(r.content)

    def test_speaker_list_endpoint(self):
        r = client.get('/v1/audio/speaker')
        assert r.status_code == 200
        speakers = r.json()
        assert isinstance(speakers, list)
        assert len(speakers) > 0


@skipif_no_app
class TestTTSNormalization:
    """Test that text normalization works end-to-end through TTS."""

    def test_tts_with_markdown(self):
        r = client.post('/v1/audio/speech', json={
            'input': '**Selamat pagi** semua.',
            'stream': False,
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert is_valid_wav(r.content)
        assert len(r.content) > 44

    def test_tts_with_html(self):
        r = client.post('/v1/audio/speech', json={
            'input': '<b>Hello</b> world.',
            'stream': False,
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert is_valid_wav(r.content)

    def test_tts_with_link(self):
        r = client.post('/v1/audio/speech', json={
            'input': 'Visit [Google](https://google.com) today.',
            'stream': False,
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert is_valid_wav(r.content)

    def test_tts_normalize_malaysian(self):
        r = client.post('/v1/audio/speech', json={
            'input': 'Harga RM10.5 hubungi 012-1234567.',
            'normalize_malaysian': True,
            'stream': False,
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert is_valid_wav(r.content)
        assert len(r.content) > 44

    def test_tts_normalize_malaysian_with_markdown(self):
        r = client.post('/v1/audio/speech', json={
            'input': '**Harga** RM10.5 hubungi [kami](https://example.com).',
            'normalize_malaysian': True,
            'stream': False,
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert is_valid_wav(r.content)


@skipif_no_app
class TestTTSParameters:
    """Test different parameter combinations."""

    def test_tts_low_temperature(self):
        r = client.post('/v1/audio/speech', json={
            'input': 'Hello.',
            'temperature': 0.3,
            'stream': False,
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert is_valid_wav(r.content)

    def test_tts_high_temperature(self):
        r = client.post('/v1/audio/speech', json={
            'input': 'Hello.',
            'temperature': 0.9,
            'stream': False,
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert is_valid_wav(r.content)

    def test_tts_custom_playback_speed(self):
        r = client.post('/v1/audio/speech', json={
            'input': 'Hello.',
            'playback_speed': 1.0,
            'stream': False,
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert is_valid_wav(r.content)

    def test_tts_max_tokens(self):
        r = client.post('/v1/audio/speech', json={
            'input': 'Hello.',
            'max_tokens': 1024,
            'stream': False,
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert is_valid_wav(r.content)


@skipif_no_app
class TestTTSTextVariations:
    """Test different text inputs for TTS."""

    def test_tts_short_text(self):
        r = client.post('/v1/audio/speech', json={
            'input': 'Hi.',
            'stream': False,
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert is_valid_wav(r.content)

    def test_tts_long_text(self):
        r = client.post('/v1/audio/speech', json={
            'input': 'Selamat pagi semua. Hari ini kita akan berbincang tentang teknologi terkini dan bagaimana ia memberi kesan kepada kehidupan seharian kita.',
            'stream': False,
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert is_valid_wav(r.content)

    def test_tts_english_text(self):
        r = client.post('/v1/audio/speech', json={
            'input': 'Good morning everyone. Today we will discuss the latest technology.',
            'stream': False,
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert is_valid_wav(r.content)

    def test_tts_numbers_and_symbols(self):
        r = client.post('/v1/audio/speech', json={
            'input': 'Harga RM500 untuk 2.5kg barangan.',
            'normalize_malaysian': True,
            'stream': False,
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert is_valid_wav(r.content)

    def test_tts_multilingual_chinese(self):
        r = client.post('/v1/audio/speech', json={
            'input': '你好世界.',
            'stream': False,
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert is_valid_wav(r.content)


# ===========================================================================
# VC endpoint tests
# ===========================================================================
@skipif_no_app
class TestVCBasic:
    """Basic Voice Conversion endpoint tests."""

    @pytest.fixture
    def jenny_audio(self):
        with open(JENNY_WAV, 'rb') as f:
            return f.read()

    def test_vc_returns_200(self, jenny_audio):
        r = client.post('/v1/audio/vc', files={
            'reference_audio': ('jenny.wav', jenny_audio, 'audio/wav'),
        }, data={
            'reference_text': JENNY_REF_TEXT,
            'generate_text': 'Hello, how can I help you today?',
            'stream': 'false',
            'response_format': 'wav',
        })
        assert r.status_code == 200

    def test_vc_wav_format(self, jenny_audio):
        r = client.post('/v1/audio/vc', files={
            'reference_audio': ('jenny.wav', jenny_audio, 'audio/wav'),
        }, data={
            'reference_text': JENNY_REF_TEXT,
            'generate_text': 'Good morning everyone.',
            'stream': 'false',
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert is_valid_wav(r.content)
        assert wav_sample_rate(r.content) == 24000
        assert wav_channels(r.content) == 1
        assert wav_bit_depth(r.content) == 16

    def test_vc_pcm_format(self, jenny_audio):
        r = client.post('/v1/audio/vc', files={
            'reference_audio': ('jenny.wav', jenny_audio, 'audio/wav'),
        }, data={
            'reference_text': JENNY_REF_TEXT,
            'generate_text': 'Good morning everyone.',
            'stream': 'false',
            'response_format': 'pcm',
        })
        assert r.status_code == 200
        assert is_valid_pcm(r.content)

    def test_vc_streaming_wav(self, jenny_audio):
        r = client.post('/v1/audio/vc', files={
            'reference_audio': ('jenny.wav', jenny_audio, 'audio/wav'),
        }, data={
            'reference_text': JENNY_REF_TEXT,
            'generate_text': 'Hello world.',
            'stream': 'true',
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert r.headers.get('content-type') == 'audio/wav'
        assert is_valid_wav(r.content)

    def test_vc_streaming_pcm(self, jenny_audio):
        r = client.post('/v1/audio/vc', files={
            'reference_audio': ('jenny.wav', jenny_audio, 'audio/wav'),
        }, data={
            'reference_text': JENNY_REF_TEXT,
            'generate_text': 'Hello world.',
            'stream': 'true',
            'response_format': 'pcm',
        })
        assert r.status_code == 200
        assert 'audio/pcm' in r.headers.get('content-type', '')
        assert r.headers.get('x-audio-sample-rate') == '24000'

    def test_vc_produces_audio(self, jenny_audio):
        r = client.post('/v1/audio/vc', files={
            'reference_audio': ('jenny.wav', jenny_audio, 'audio/wav'),
        }, data={
            'reference_text': JENNY_REF_TEXT,
            'generate_text': 'Testing voice conversion output.',
            'stream': 'false',
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert len(r.content) > 44


@skipif_no_app
class TestVCTextVariations:
    """Test VC with different text inputs."""

    @pytest.fixture
    def jenny_audio(self):
        with open(JENNY_WAV, 'rb') as f:
            return f.read()

    def test_vc_malay_text(self, jenny_audio):
        r = client.post('/v1/audio/vc', files={
            'reference_audio': ('jenny.wav', jenny_audio, 'audio/wav'),
        }, data={
            'reference_text': JENNY_REF_TEXT,
            'generate_text': 'Selamat pagi, apa khabar?',
            'stream': 'false',
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert is_valid_wav(r.content)

    def test_vc_english_text(self, jenny_audio):
        r = client.post('/v1/audio/vc', files={
            'reference_audio': ('jenny.wav', jenny_audio, 'audio/wav'),
        }, data={
            'reference_text': JENNY_REF_TEXT,
            'generate_text': 'The weather is beautiful today.',
            'stream': 'false',
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert is_valid_wav(r.content)

    def test_vc_long_generate_text(self, jenny_audio):
        r = client.post('/v1/audio/vc', files={
            'reference_audio': ('jenny.wav', jenny_audio, 'audio/wav'),
        }, data={
            'reference_text': JENNY_REF_TEXT,
            'generate_text': 'Good morning everyone. Today we will be discussing the latest developments in artificial intelligence and machine learning.',
            'stream': 'false',
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert is_valid_wav(r.content)

    def test_vc_short_generate_text(self, jenny_audio):
        r = client.post('/v1/audio/vc', files={
            'reference_audio': ('jenny.wav', jenny_audio, 'audio/wav'),
        }, data={
            'reference_text': JENNY_REF_TEXT,
            'generate_text': 'Hi.',
            'stream': 'false',
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert is_valid_wav(r.content)


@skipif_no_app
class TestVCMarkdownSanitization:
    """Test that markdown is stripped from VC text inputs."""

    @pytest.fixture
    def jenny_audio(self):
        with open(JENNY_WAV, 'rb') as f:
            return f.read()

    def test_vc_markdown_in_generate_text(self, jenny_audio):
        r = client.post('/v1/audio/vc', files={
            'reference_audio': ('jenny.wav', jenny_audio, 'audio/wav'),
        }, data={
            'reference_text': JENNY_REF_TEXT,
            'generate_text': '**Selamat pagi** semua, _apa khabar_?',
            'stream': 'false',
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert is_valid_wav(r.content)

    def test_vc_markdown_in_reference_text(self, jenny_audio):
        r = client.post('/v1/audio/vc', files={
            'reference_audio': ('jenny.wav', jenny_audio, 'audio/wav'),
        }, data={
            'reference_text': '**' + JENNY_REF_TEXT + '**',
            'generate_text': 'Hello world.',
            'stream': 'false',
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert is_valid_wav(r.content)

    def test_vc_html_in_generate_text(self, jenny_audio):
        r = client.post('/v1/audio/vc', files={
            'reference_audio': ('jenny.wav', jenny_audio, 'audio/wav'),
        }, data={
            'reference_text': JENNY_REF_TEXT,
            'generate_text': '<b>Hello</b> <i>world</i>.',
            'stream': 'false',
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert is_valid_wav(r.content)

    def test_vc_link_in_generate_text(self, jenny_audio):
        r = client.post('/v1/audio/vc', files={
            'reference_audio': ('jenny.wav', jenny_audio, 'audio/wav'),
        }, data={
            'reference_text': JENNY_REF_TEXT,
            'generate_text': 'Visit [our website](https://example.com) today.',
            'stream': 'false',
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert is_valid_wav(r.content)

    def test_vc_code_block_in_generate_text(self, jenny_audio):
        r = client.post('/v1/audio/vc', files={
            'reference_audio': ('jenny.wav', jenny_audio, 'audio/wav'),
        }, data={
            'reference_text': JENNY_REF_TEXT,
            'generate_text': 'Use the `print` function.',
            'stream': 'false',
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert is_valid_wav(r.content)


@skipif_no_app
class TestVCParameters:
    """Test VC with different parameter combinations."""

    @pytest.fixture
    def jenny_audio(self):
        with open(JENNY_WAV, 'rb') as f:
            return f.read()

    def test_vc_low_temperature(self, jenny_audio):
        r = client.post('/v1/audio/vc', files={
            'reference_audio': ('jenny.wav', jenny_audio, 'audio/wav'),
        }, data={
            'reference_text': JENNY_REF_TEXT,
            'generate_text': 'Hello world.',
            'temperature': '0.3',
            'stream': 'false',
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert is_valid_wav(r.content)

    def test_vc_high_temperature(self, jenny_audio):
        r = client.post('/v1/audio/vc', files={
            'reference_audio': ('jenny.wav', jenny_audio, 'audio/wav'),
        }, data={
            'reference_text': JENNY_REF_TEXT,
            'generate_text': 'Hello world.',
            'temperature': '0.9',
            'stream': 'false',
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert is_valid_wav(r.content)

    def test_vc_custom_playback_speed(self, jenny_audio):
        r = client.post('/v1/audio/vc', files={
            'reference_audio': ('jenny.wav', jenny_audio, 'audio/wav'),
        }, data={
            'reference_text': JENNY_REF_TEXT,
            'generate_text': 'Hello world.',
            'playback_speed': '1.0',
            'stream': 'false',
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert is_valid_wav(r.content)

    def test_vc_max_tokens(self, jenny_audio):
        r = client.post('/v1/audio/vc', files={
            'reference_audio': ('jenny.wav', jenny_audio, 'audio/wav'),
        }, data={
            'reference_text': JENNY_REF_TEXT,
            'generate_text': 'Hello world.',
            'max_tokens': '1024',
            'stream': 'false',
            'response_format': 'wav',
        })
        assert r.status_code == 200
        assert is_valid_wav(r.content)
