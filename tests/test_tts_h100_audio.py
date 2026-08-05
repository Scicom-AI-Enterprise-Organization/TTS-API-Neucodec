"""
Regression test for the NVIDIA / CUDA deployment (e.g. 1x H100 SXM RunPod).

Why this exists: app/neucodec now uses a vendored Rotary Positional Embedding
(app/neucodec/_rope.py) instead of torchtune.modules.RotaryPositionalEmbeddings
(torchtune drags in torchao, which is unimportable on Python 3.9 and needed to be
dropped for the Ascend NPU build). The vendored class is a byte-for-byte copy of
torchtune's numerics, but this test guards that the swap did not break decode:

  1. test_rope_*  -- GPU-free unit checks on the vendored RoPE module itself
                     (imports, shape, and the rotation invariant that RoPE must
                     preserve vector norm). Runs anywhere torch is installed.
  2. test_tts_decode_saves_audio -- end-to-end: pushes 5 texts through the live
                     /v1/audio/speech endpoint, asserts each returns valid, non-silent
                     24 kHz audio, and saves every clip to the audio/ folder. Auto-runs
                     wherever the API is reachable (skips otherwise), so on the H100
                     RunPod (app:9091 + vLLM up) it runs automatically.

Run:
    # unit checks only (no server needed):
    python -m pytest tests/test_tts_h100_audio.py -v -k rope
    # full, on the H100 pod with the stack up:
    TTS_TEST_URL=http://localhost:9091 python -m pytest tests/test_tts_h100_audio.py -v

Env:
    TTS_TEST_URL        API base URL (default http://localhost:9091)
    TTS_TEST_AUDIO_DIR  where to write the wavs (default <repo>/audio)
"""

import io
import os
import sys
import wave

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest

BASE_URL = os.environ.get('TTS_TEST_URL', 'http://localhost:9091')
AUDIO_DIR = os.environ.get(
    'TTS_TEST_AUDIO_DIR', os.path.join(os.path.dirname(__file__), '..', 'audio')
)

# 5 texts spanning the app's real use-cases: English, Malay, and code-switch.
TTS_CASES = [
    ("en_1", "husein", "Hello there, how can I help you today?"),
    ("en_2", "husein", "Artificial intelligence is transforming the way we live and work every single day."),
    ("ms_1", "husein", "Selamat pagi, apa yang boleh saya bantu encik hari ini?"),
    ("ms_2", "husein", "Terima kasih kerana menghubungi kami, sila tunggu sebentar."),
    ("cs_1", "husein", "Okay encik, your booking is confirmed, nanti saya hantar details melalui email ya."),
]


# ---------------------------------------------------------------------------
# 1. GPU-free unit checks on the vendored RoPE (guards the code change directly)
# ---------------------------------------------------------------------------
torch = pytest.importorskip("torch", reason="torch not installed")


def test_rope_import_and_shape():
    """The neucodec package imports (no torchtune) and RoPE keeps the input shape."""
    from app.neucodec._rope import RotaryPositionalEmbeddings

    b, s, n_h, h_d = 2, 16, 4, 64
    rope = RotaryPositionalEmbeddings(dim=h_d, max_seq_len=32)
    x = torch.randn(b, s, n_h, h_d)
    y = rope(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_rope_preserves_norm():
    """RoPE is a rotation, so it must preserve the L2 norm of each token/head vector.
    This is implementation-independent and would fail if the swap corrupted the math."""
    from app.neucodec._rope import RotaryPositionalEmbeddings

    b, s, n_h, h_d = 1, 20, 2, 128
    rope = RotaryPositionalEmbeddings(dim=h_d, max_seq_len=64)
    x = torch.randn(b, s, n_h, h_d)
    y = rope(x)
    nx = x.float().norm(dim=-1)
    ny = y.float().norm(dim=-1)
    assert torch.allclose(nx, ny, atol=1e-4), "RoPE changed vector norms (not a pure rotation)"


def test_neucodec_package_imports_without_torchtune():
    """Importing the codec must not require torchtune/torchao."""
    import app.neucodec  # noqa: F401
    assert "torchtune" not in sys.modules, "torchtune got imported; vendored RoPE not used"


# ---------------------------------------------------------------------------
# 2. Live-API integration: 5 texts -> audio/ (auto-runs on the H100 RunPod)
# ---------------------------------------------------------------------------
requests = pytest.importorskip("requests", reason="requests not installed")
np = pytest.importorskip("numpy", reason="numpy not installed")

try:
    _r = requests.get(f'{BASE_URL}/v1/audio/speaker', timeout=5)
    APP_AVAILABLE = _r.status_code == 200
    SKIP_REASON = '' if APP_AVAILABLE else f'API returned {_r.status_code}'
except Exception as e:  # noqa: BLE001
    APP_AVAILABLE = False
    SKIP_REASON = str(e)

skipif_no_app = pytest.mark.skipif(
    not APP_AVAILABLE, reason=f"API not reachable at {BASE_URL}: {SKIP_REASON}"
)


def _wav_stats(data: bytes):
    """(sample_rate, n_frames, duration_s, rms) from wav bytes."""
    with wave.open(io.BytesIO(data), 'rb') as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    rms = float((a ** 2).mean() ** 0.5) if a.size else 0.0
    return sr, n, (n / sr if sr else 0.0), rms


@skipif_no_app
@pytest.mark.parametrize(
    "case_id,voice,text", TTS_CASES, ids=[c[0] for c in TTS_CASES]
)
def test_tts_decode_saves_audio(case_id, voice, text):
    """Decode each text on the live server and save the wav; assert real, non-silent audio."""
    os.makedirs(AUDIO_DIR, exist_ok=True)

    r = requests.post(
        f'{BASE_URL}/v1/audio/speech',
        timeout=180,
        json={
            'input': text,
            'voice': voice,
            'response_format': 'wav',
            'stream': False,
        },
    )
    assert r.status_code == 200, f'{case_id}: HTTP {r.status_code}: {r.text[:200]}'

    data = r.content
    assert len(data) > 44 and data[:4] == b'RIFF' and data[8:12] == b'WAVE', \
        f'{case_id}: response is not a valid WAV'

    sr, n_frames, dur, rms = _wav_stats(data)

    out_path = os.path.join(AUDIO_DIR, f'{case_id}_{voice}.wav')
    with open(out_path, 'wb') as f:
        f.write(data)

    # regression assertions: real 24 kHz audio, non-trivial length, not silence
    assert sr == 24000, f'{case_id}: sample rate {sr} != 24000'
    assert dur >= 0.5, f'{case_id}: audio too short ({dur:.2f}s) -- decode likely broken'
    assert rms > 1e-3, f'{case_id}: audio is silent (rms={rms:.5f}) -- decode likely broken'

    print(f'[{case_id}] {voice}: {dur:.2f}s @ {sr}Hz rms={rms:.4f} -> {out_path}')


@skipif_no_app
def test_tts_end_to_end_distinct_from_text():
    """True end-to-end guard: the server must actually generate speech from text via the LM,
    not replay canned tokens. A short and a long sentence must produce different, and
    proportionally different-length, audio. This fails under DUMMY_TOKENS replay (identical
    output for every input) and would catch a decode that ignores its input."""
    def synth(text):
        r = requests.post(
            f'{BASE_URL}/v1/audio/speech',
            timeout=180,
            json={'input': text, 'voice': 'husein', 'response_format': 'wav', 'stream': False},
        )
        assert r.status_code == 200, f'HTTP {r.status_code}: {r.text[:200]}'
        assert r.content[:4] == b'RIFF', 'not a WAV'
        return r.content

    short_wav = synth("Hello.")
    long_wav = synth(
        "Artificial intelligence is transforming the way we live and work, from how we "
        "search for information to how we drive our cars and manage our homes every single day."
    )
    _, _, dur_short, rms_short = _wav_stats(short_wav)
    _, _, dur_long, rms_long = _wav_stats(long_wav)

    assert short_wav != long_wav, \
        "identical audio for different texts -> server is not generating from text (dummy replay?)"
    assert rms_short > 1e-3 and rms_long > 1e-3, "one of the clips is silent"
    assert dur_long > dur_short + 1.0, \
        f"long text ({dur_long:.2f}s) not clearly longer than short text ({dur_short:.2f}s) " \
        f"-> LM likely not driving duration end-to-end"
    print(f'[e2e] short={dur_short:.2f}s long={dur_long:.2f}s (distinct, LM-driven)')
