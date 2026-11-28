"""
app/tiny_decoder.py: the decoder backbone behind CODEC_DECODER.

No GPU, no weights, no network. The first test needs nothing; the rest need torch and safetensors and are skipped
without them:

    uv run --with pytest -- pytest tests/test_tiny_decoder.py -q
    uv run --with pytest --with torch --with safetensors -- pytest tests/test_tiny_decoder.py -q

That the released weights decode to the samples they were measured with is checked on a GPU box, against the training
code (Tiny-NeuCodec-Encoder, scripts/merge_public.py): largest sample difference 0.00e+00 for both models.
"""
import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import tiny_decoder as D  # noqa: E402


def test_import_needs_no_torch_at_module_level():
    head = open(D.__file__).read().split('def _conv_trunk')[0]
    assert 'import torch' not in head and 'import safetensors' not in head


try:
    import safetensors  # noqa: F401
    import torch
    nn = torch.nn
except ImportError:                                             # the rest needs torch; the import test above does not
    torch = None
    nn = types.SimpleNamespace(Module=object)
needs_torch = pytest.mark.skipif(torch is None, reason='needs torch and safetensors')


def _codec(backbone):
    head = nn.Module()
    head.out = nn.Linear(D.DIM, 1922)
    g = nn.Module()
    g.backbone, g.head = backbone, head
    codec = types.SimpleNamespace(generator=g)
    return codec


def _write(folder, name, cfg, sd):
    from safetensors.torch import save_file
    os.makedirs(folder / name)
    save_file({k: v.contiguous() for k, v in sd.items()}, str(folder / name / 'model.safetensors'))
    (folder / name / 'config.json').write_text(json.dumps(cfg))


CONV = {'arch': 'conv', 'dim': 32, 'layers': 3, 'mult': 2, 'kernel': 7, 'dilations': [1, 2, 4]}


def _conv_weights():
    trunk = D._conv_trunk(CONV)
    out = nn.Linear(CONV['dim'], 1922)
    return trunk, {**trunk.state_dict(), **{f'out.{k}': v for k, v in out.state_dict().items()}}


@needs_torch
def test_conv_replaces_backbone_and_head(tmp_path):
    trunk, sd = _conv_weights()
    _write(tmp_path, 'decoder-conv', CONV, sd)
    codec = _codec(nn.Identity())
    D.load_decoder(codec, 'decoder-conv', local_dir=str(tmp_path))
    g = codec.generator
    x = torch.randn(2, 40, D.DIM)
    assert torch.equal(g.backbone(x), trunk(x))
    assert g.head.out.in_features == CONV['dim'] and torch.equal(g.head.out.weight, sd['out.weight'])
    assert g.backbone.span() == 3 + 1 * 3 + 2 * 3 + 4 * 3
    assert codec.tiny_decoder['name'] == 'decoder-conv'


@needs_torch
def test_name_may_be_the_model_folder_itself(tmp_path):
    _, sd = _conv_weights()
    _write(tmp_path, 'decoder-conv', CONV, sd)
    codec = _codec(nn.Identity())
    D.load_decoder(codec, str(tmp_path / 'decoder-conv'), repo='nobody/never-downloaded')
    assert codec.generator.head.out.in_features == CONV['dim']


class _Transformerish(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformers = nn.Sequential(nn.Linear(D.DIM, D.DIM), nn.Dropout(0.1))


@needs_torch
def test_neucodec_arch_loads_in_place_and_drops_dropout(tmp_path):
    src = _Transformerish()
    out = nn.Linear(D.DIM, 1922)
    sd = {**{f'backbone.{k}': v for k, v in src.state_dict().items()}, **{f'out.{k}': v for k, v in out.state_dict().items()}}
    _write(tmp_path, 'decoder-finetuned', {'arch': 'neucodec'}, sd)
    dst = _Transformerish()
    codec = _codec(dst)
    D.load_decoder(codec, 'decoder-finetuned', local_dir=str(tmp_path))
    assert codec.generator.backbone is dst                      # same module, new weights
    assert torch.equal(dst.transformers[0].weight, src.transformers[0].weight)
    assert torch.equal(codec.generator.head.out.weight, out.weight)
    assert dst.transformers[1].p == 0.0


@needs_torch
def test_neucodec_arch_refuses_a_conv_backbone(tmp_path):
    _write(tmp_path, 'decoder-finetuned', {'arch': 'neucodec'}, {'out.weight': torch.zeros(1922, D.DIM), 'out.bias': torch.zeros(1922)})
    with pytest.raises(ValueError):
        D.load_decoder(_codec(nn.Identity()), 'decoder-finetuned', local_dir=str(tmp_path))
