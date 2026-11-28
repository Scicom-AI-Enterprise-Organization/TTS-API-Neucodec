"""Tiny NeuCodec Decoder (https://huggingface.co/Scicom-intl/Tiny-NeuCodec): another backbone for NeuCodec's decoder,
at the same 24 kHz.

    codes [F] -- codebook -- fc_post_a -- BACKBONE -- linear -> log-magnitude, phase -- inverse STFT -- 24 kHz audio

Only the backbone and the linear layer after it change. The codebook, `fc_post_a` and the inverse STFT are NeuCodec's
own, so the codes are NeuCodec's codes and nothing upstream (encoder, language model) has to know.

    from neucodec import NeuCodec
    from tiny_decoder import load_decoder

    codec = NeuCodec.from_pretrained("neuphonic/neucodec").eval().cuda()
    load_decoder(codec, "decoder-finetuned")              # or "decoder-conv"
    audio = codec.decode_code(codes)                       # [B, 1, F] codes -> [B, 1, F * 480] samples

Two kinds of model:

  arch "neucodec"   NeuCodec's own backbone (12 transformer blocks, 184M) trained further against recordings.
                    The weights are loaded into the modules that are already there.
  arch "conv"       ConvNeXt blocks (depthwise conv, kernel 7, dilated), about 50M. `codec.generator.backbone` and
                    `codec.generator.head.out` are replaced. Each output frame sees a fixed span of codes
                    (`codec.generator.backbone.span()` frames each side), so a window that carries that much context
                    plus two frames decodes to the same samples as the whole clip.

Give a batch one length. A decoder reads the codes around a frame, so padding a short sequence changes its audio.
torch is imported inside the functions, so importing this file needs nothing.
"""
from __future__ import annotations

import json
import os

REPO = "Scicom-intl/Tiny-NeuCodec"
DIM = 1024                  # what fc_post_a hands the backbone


def _conv_trunk(cfg):
    import torch
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, dim, mult, kernel, dilation):
            super().__init__()
            self.dw = nn.Conv1d(dim, dim, kernel, padding=dilation * (kernel - 1) // 2, dilation=dilation, groups=dim)
            self.norm = nn.LayerNorm(dim, eps=1e-6)
            self.fc1 = nn.Linear(dim, mult * dim)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(mult * dim, dim)
            self.gamma = nn.Parameter(torch.ones(dim))

        def forward(self, x):                                   # [B, F, dim]
            y = self.dw(x.transpose(1, 2)).transpose(1, 2)
            return x + self.gamma * self.fc2(self.act(self.fc1(self.norm(y))))

    class ConvTrunk(nn.Module):
        def __init__(self, dim, layers, mult, kernel, dilations):
            super().__init__()
            self.cfg = {"dim": dim, "layers": layers, "mult": mult, "kernel": kernel, "dilations": list(dilations)}
            self.embed = nn.Conv1d(DIM, dim, 7, padding=3)
            self.norm_in = nn.LayerNorm(dim, eps=1e-6)
            self.blocks = nn.ModuleList(Block(dim, mult, kernel, dilations[i % len(dilations)]) for i in range(layers))
            self.norm_out = nn.LayerNorm(dim, eps=1e-6)

        def span(self):
            """Frames of context each side that reach an output frame (the inverse STFT adds two more)."""
            c = self.cfg
            return 3 + sum(c["dilations"][i % len(c["dilations"])] * (c["kernel"] - 1) // 2 for i in range(c["layers"]))

        def forward(self, x):                                   # [B, F, 1024] -> [B, F, dim]
            x = self.norm_in(self.embed(x.transpose(1, 2)).transpose(1, 2))
            for b in self.blocks:
                x = b(x)
            return self.norm_out(x)

    return ConvTrunk(cfg["dim"], cfg["layers"], cfg["mult"], cfg["kernel"], cfg["dilations"])


def load_decoder(codec, name="decoder-finetuned", repo=REPO, token=None, local_dir=None):
    """Put the backbone `name` of this repository into a loaded NeuCodec, in place. Returns the codec.

    `local_dir`: a folder that holds `<name>/config.json` and `<name>/model.safetensors` (no download then).
    `name` may also be such a model folder itself (a path holding config.json)."""
    import torch.nn as nn
    from safetensors.torch import load_file

    if not local_dir and os.path.isfile(os.path.join(name, "config.json")):
        local_dir, name = os.path.split(os.path.abspath(name))
    if local_dir:
        cfg_path, w_path = os.path.join(local_dir, name, "config.json"), os.path.join(local_dir, name, "model.safetensors")
    else:
        from huggingface_hub import hf_hub_download

        cfg_path = hf_hub_download(repo, f"{name}/config.json", token=token)
        w_path = hf_hub_download(repo, f"{name}/model.safetensors", token=token)
    cfg = json.load(open(cfg_path))
    sd = load_file(w_path)
    out = {k[len("out."):]: v for k, v in sd.items() if k.startswith("out.")}
    g = codec.generator
    p = next(g.head.parameters())
    if cfg["arch"] == "neucodec":
        if not hasattr(g.backbone, "transformers"):
            raise ValueError("this NeuCodec already carries a conv backbone; load a fresh NeuCodec first")
        g.backbone.load_state_dict({k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")})
        g.head.out.load_state_dict(out)
        for m in g.backbone.modules():                          # the released model trains with dropout; inference has none
            if isinstance(m, nn.Dropout):
                m.p = 0.0
    else:
        trunk = _conv_trunk(cfg)
        trunk.load_state_dict({k: v for k, v in sd.items() if not k.startswith("out.")})
        lin = nn.Linear(cfg["dim"], out["weight"].shape[0])
        lin.load_state_dict(out)
        g.backbone = trunk.to(p.device, p.dtype).eval()
        g.head.out = lin.to(p.device, p.dtype).eval()
    codec.tiny_decoder = {"name": name, **cfg}
    return codec
