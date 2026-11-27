"""
app/tiny_encoder.py: the distilled encoder behind VC_ENCODER.

No GPU, no weights, no network. The first tests need nothing; the rest need torch (and transformers for the filter
bank and the model test) and are skipped without them:

    uv run --with pytest -- pytest tests/test_tiny_encoder.py -q
    uv run --with pytest --with torch --with transformers --with numpy -- pytest tests/test_tiny_encoder.py -q

That the released weights write the codes they were measured with is checked on a GPU box, against the encoder the
numbers were taken with (Tiny-NeuCodec-Encoder, scripts/check_app_encoder.py): 9,824 of 9,824 codes equal.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import tiny_encoder as T  # noqa: E402


# ---------------------------------------------------------------- where VC_ENCODER points

def test_import_needs_no_torch_at_module_level():
    src = open(T.__file__).read()
    head = src.split('def _squash')[0]
    assert 'import torch' not in head and 'import transformers' not in head


def test_resolve_local_file_and_directory(tmp_path):
    f = tmp_path / 'S4' / 'student.pt'
    f.parent.mkdir()
    f.write_bytes(b'x')
    assert T.resolve(str(f)) == str(f)
    assert T.resolve(str(f.parent)) == str(f)


@pytest.mark.parametrize('source, repo, want', [
    ('S4', 'Scicom-intl/Tiny-NeuCodec-Encoder', ('Scicom-intl/Tiny-NeuCodec-Encoder', 'S4/student.pt')),
    ('S2', 'me/mirror', ('me/mirror', 'S2/student.pt')),
    ('org/name', 'unused', ('org/name', 'student.pt')),
    ('org/name/S4', 'unused', ('org/name', 'S4/student.pt')),
    ('org/name/a/b/', 'unused', ('org/name', 'a/b/student.pt')),
])
def test_resolve_hub_names(monkeypatch, source, repo, want):
    import huggingface_hub

    seen = []
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', lambda r, f: seen.append((r, f)) or '/cache/student.pt')
    assert T.resolve(source, repo) == '/cache/student.pt'
    assert seen == [want]


# ---------------------------------------------------------------- the quantizer's arithmetic

torch = pytest.importorskip('torch')


def test_codes_are_eight_base4_digits():
    # a latent far below / above every rounding boundary lands on the lowest / highest level of each digit
    assert int(T.codes_from_z(torch.full((8,), -10.0))) == 0
    assert int(T.codes_from_z(torch.full((8,), 10.0))) == 4 ** 8 - 1
    z = torch.full((8,), -10.0)
    z[3] = 10.0                                   # digit 3 at its top level: 3 * 4**3
    assert int(T.codes_from_z(z)) == 3 * 4 ** 3
    assert T.codes_from_z(torch.zeros(5, 7, 8)).shape == (5, 7)


def test_bound_is_squashed_twice_and_stays_in_range():
    z = torch.linspace(-20, 20, 401)
    b = T.bound(z)
    assert b.min() > -2.01 and b.max() < 1.01     # rounds to -2..1, i.e. four levels
    assert torch.all(b[1:] >= b[:-1])
    assert not torch.allclose(b, T._squash(z))


def test_prepare_audio_pads_like_neucodec():
    for n in (1, 319, 320, 321, 16000):
        y = T.prepare_audio(torch.zeros(n))
        assert y.shape[:2] == (1, 1) and y.shape[-1] % T.HOP == 0
        assert y.shape[-1] == n + (T.HOP - n % T.HOP)     # a full hop when already a multiple, as NeuCodec does


def test_linear_parts_reproduce_fc_prior_then_project_in():
    class Codec:
        pass

    torch.manual_seed(0)
    c = Codec()
    c.fc_prior = torch.nn.Linear(2048, 2048)
    c.generator = Codec()
    c.generator.quantizer = Codec()
    c.generator.quantizer.project_in = torch.nn.Linear(2048, 8)
    A_s, A_a, bias = T.linear_parts(c)
    s, a = torch.randn(11, 1024), torch.randn(11, 1024)
    want = c.generator.quantizer.project_in(c.fc_prior(torch.cat([s, a], -1)))
    assert torch.allclose(s @ A_s.T + a @ A_a.T + bias, want, atol=1e-4)


# ---------------------------------------------------------------- filter bank and model (transformers)

def test_torch_fbank_matches_the_feature_extractor():
    transformers = pytest.importorskip('transformers')
    np = pytest.importorskip('numpy')
    fe = transformers.SeamlessM4TFeatureExtractor(feature_size=80, num_mel_bins=80, sampling_rate=16000, stride=2, padding_value=1)
    TorchFbank, _, _ = T._modules()
    rng = np.random.default_rng(0)
    wav = (0.1 * rng.standard_normal(16000 * 2)).astype('float32')
    y = T.prepare_audio(wav)
    want = fe(y.squeeze(0), sampling_rate=16000, return_tensors='pt').input_features[0]
    got, raw = TorchFbank(fe)(y)
    assert got.shape == want.shape and raw.shape[-1] == 80
    assert float((got - want).abs().max()) < 2e-3


def test_load_runs_end_to_end_on_a_small_stand_in(tmp_path):
    """A NeuCodec-shaped object with a tiny w2v-BERT: the loader builds the student, loads its state, frees the
    teacher's transformer and writes one code per 20 ms."""
    transformers = pytest.importorskip('transformers')
    cfg = transformers.Wav2Vec2BertConfig(hidden_size=1024, num_hidden_layers=1, num_attention_heads=4, intermediate_size=64,
                                          feature_projection_input_dim=160, adapter_kernel_size=3, conv_depthwise_kernel_size=3)
    fe = transformers.SeamlessM4TFeatureExtractor(feature_size=80, num_mel_bins=80, sampling_rate=16000, stride=2, padding_value=1)

    class Acoustic(torch.nn.Module):
        def forward(self, y):                      # [1, 1, T] -> [1, T / 320, 1024], as NeuCodec's CodecEnc
            return torch.zeros(1, y.shape[-1] // T.HOP, 1024)

    class Codec(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.semantic_model = transformers.Wav2Vec2BertModel(cfg)
            self.feature_extractor = fe
            self.SemanticEncoder_module = torch.nn.Conv1d(1024, 1024, 1)
            self.CodecEnc = Acoustic()
            self.fc_prior = torch.nn.Linear(2048, 2048)
            self.generator = torch.nn.Module()
            self.generator.quantizer = torch.nn.Module()
            self.generator.quantizer.project_in = torch.nn.Linear(2048, 8)

    torch.manual_seed(0)
    codec = Codec().eval()
    _, Student, _ = T._modules()
    st = Student(codec, depth=1)
    path = tmp_path / 'S1'
    path.mkdir()
    torch.save({'depth': 1, 'unified': False, 'level': False, 'wave_stem': False, 'step': 7,
                'state': {k: v for k, v in st.state_dict().items() if k != 'A_s'}}, path / 'student.pt')

    enc, info = T.load_tiny_encoder(str(path), codec, 'cpu', free_teacher=True)
    assert info == {'depth': 1, 'unified': False, 'level': False, 'step': 7}
    assert codec.semantic_model is None and codec.CodecEnc is not None       # an S model keeps the convolutional branch
    wav = 0.1 * torch.randn(16000)
    codes = enc.encode_code(wav)
    assert codes.dtype == torch.long and codes.ndim == 1
    assert abs(len(codes) - 50) <= 1 and int(codes.min()) >= 0 and int(codes.max()) < 65536
    assert torch.equal(codes, enc.encode_code(wav))                           # deterministic


def test_a_checkpoint_that_does_not_fit_is_refused(tmp_path):
    pytest.importorskip('transformers')
    torch.save({'depth': 1, 'wave_stem': True, 'state': {}}, tmp_path / 'student.pt')
    with pytest.raises(ValueError, match='waveform stem'):
        T.load_tiny_encoder(str(tmp_path), None, 'cpu')
