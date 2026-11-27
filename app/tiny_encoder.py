"""A distilled NeuCodec encoder for /v1/audio/vc (`VC_ENCODER`, off by default).

NeuCodec's encoder reads the reference clip twice:

    wav 16 kHz --+-- CodecEnc (convolutions) -------------------------- a --+
                 |                                                          +-- fc_prior -- project_in -- z [F, 8] -- FSQ -- code
                 +-- fbank -- w2v-BERT 2.0 (layer 16) -- SemanticEncoder -- s --+

A Tiny NeuCodec Encoder (https://huggingface.co/Scicom-intl/Tiny-NeuCodec-Encoder) is trained to hand the quantizer
the same eight numbers `z` with a few transformer layers instead of sixteen. It writes NeuCodec's own codes: the
codebook, the quantizer and the decoder are untouched, so the LM reads them as it reads NeuCodec's.

  S models keep NeuCodec's convolutional branch and replace the w2v-BERT layers ("depth" layers instead of 16).
  U models replace both branches (filter bank -> layers -> a linear head) and read the clip's level as an extra input.

Everything after the two branches is linear up to the quantizer:

    z = A_s s + A_a a + bias,     A = P W,     bias = P b + p

with W, b the weights of fc_prior and P, p those of the quantizer's project_in. The quantizer is FSQ with eight
dimensions of four levels, so a code is eight base-4 digits, digit d = round(bound(z_d)) + 2.

One clip per call. Clips are not padded into a batch: padding changes the codes (it changes NeuCodec's own too).
torch and transformers are imported inside the functions, so `import app.tiny_encoder` works anywhere.
"""
from __future__ import annotations

import copy
import math
import os

LEVELS = 4
DIMS = 8
SR = 16_000
HOP = 320                   # 50 codes a second
_EPS = 1e-3
_HALF_L = (LEVELS - 1) * (1 + _EPS) / 2
_OFFSET = 0.5
_SHIFT = math.atanh(_OFFSET / _HALF_L)


def _squash(z):
    import torch

    return torch.tanh(z + _SHIFT) * _HALF_L - _OFFSET


def bound(z):
    """The value the quantizer rounds. FSQ squashes z, and the residual wrapper around it squashes once more before
    handing z to FSQ, so the squashing is applied twice (with one, only 20% of the codes come out right)."""
    return _squash(_squash(z))


def codes_from_z(z):
    """[..., 8] latent -> [...] codes in 0..65535."""
    import torch

    d = (torch.round(bound(z)) + LEVELS // 2).long().clamp_(0, LEVELS - 1)
    return (d * (LEVELS ** torch.arange(DIMS, device=d.device))).sum(-1)


def linear_parts(codec):
    """A_s [8, 1024], A_a [8, 1024], bias [8] of z = A_s s + A_a a + bias."""
    W, b = codec.fc_prior.weight.detach().float(), codec.fc_prior.bias.detach().float()
    proj = codec.generator.quantizer.project_in
    P, p = proj.weight.detach().float(), proj.bias.detach().float()
    A = P @ W
    return A[:, :1024].contiguous(), A[:, 1024:].contiguous(), (P @ b + p).contiguous()


def prepare_audio(wav):
    """As NeuCodec._prepare_audio: 1-D float waveform at 16 kHz -> [1, 1, T'] padded to a multiple of the hop."""
    import torch
    import torch.nn.functional as F

    wav = torch.as_tensor(wav, dtype=torch.float32).reshape(1, 1, -1)
    return F.pad(wav, (0, HOP - (wav.shape[-1] % HOP)))


def _modules():
    """The classes, built on first use so that importing this file needs no torch."""
    import torch
    import torch.nn as nn

    class TorchFbank(nn.Module):
        """w2v-BERT's input features on the model's device (its own extractor runs in numpy on one CPU core): Kaldi filter
        bank, log, per-clip normalization per bin, two frames stacked. The window and the mel filters are taken from the
        extractor itself, so only floating point differs."""

        def __init__(self, feature_extractor):
            super().__init__()
            self.register_buffer("window", torch.tensor(feature_extractor.window, dtype=torch.float64))
            self.register_buffer("mel", torch.tensor(feature_extractor.mel_filters, dtype=torch.float64))     # [257, 80]
            self.stride = feature_extractor.stride

        def logmel(self, y):
            w = y.to(torch.float64) * 32768.0
            fr = w.unfold(-1, 400, 160)
            fr = fr - fr.mean(-1, keepdim=True)
            fr = torch.cat([fr[..., :1] * (1 - 0.97), fr[..., 1:] - 0.97 * fr[..., :-1]], -1) * self.window
            spec = torch.fft.rfft(fr, n=512).abs() ** 2
            return torch.log(torch.clamp(spec @ self.mel, min=1.192092955078125e-07)).float()

        def stats(self, x):
            """What the normalization throws away: mean and standard deviation of every bin. [B, N, 80] -> [B, 160]."""
            k = max(x.shape[1], 2)
            mean = x.sum(1) / k
            var = ((x - mean[:, None]) ** 2).sum(1) / (k - 1)
            return torch.cat([mean, torch.sqrt(var + 1e-7)], -1)

        def forward(self, y):
            """[T] waveform (already padded as the codec pads it) -> features [F, 160] and the log-mel before normalization."""
            raw = self.logmel(y.reshape(-1))
            x = (raw - raw.mean(0, keepdim=True)) / torch.sqrt(raw.var(0, unbiased=True, keepdim=True) + 1e-7)
            n = x.shape[0] - x.shape[0] % self.stride
            return x[:n].reshape(n // self.stride, -1), raw

    class Student(nn.Module):
        """`depth` Conformer layers in place of w2v-BERT's sixteen. `unified` also replaces the convolutional branch."""

        def __init__(self, codec, depth, unified=False, level=False):
            super().__init__()
            from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel

            cfg = copy.deepcopy(codec.semantic_model.config)
            cfg.num_hidden_layers = depth
            cfg.layerdrop = 0.0
            cfg.apply_spec_augment = False
            for k in ("attention_dropout", "hidden_dropout", "activation_dropout", "feat_proj_dropout",
                      "conformer_conv_dropout", "final_dropout"):
                if hasattr(cfg, k):
                    setattr(cfg, k, 0.0)
            cfg.output_hidden_states = False
            self.depth, self.unified, self.level = depth, unified, level
            self.model = Wav2Vec2BertModel(Wav2Vec2BertConfig.from_dict(cfg.to_dict()))
            self.sem = copy.deepcopy(codec.SemanticEncoder_module)
            A_s, _, _ = linear_parts(codec)
            if unified:
                self.head = nn.Linear(1024, DIMS)
            else:
                self.register_buffer("A_s", A_s.clone())
            self.lvl = nn.Linear(160, 1024) if level else None
            if level:
                self.register_buffer("lvl_shift", torch.cat([torch.full((80,), 12.0), torch.full((80,), 3.0)]))
                self.register_buffer("lvl_scale", torch.cat([torch.full((80,), 4.0), torch.full((80,), 1.5)]))

        def forward(self, feats, ca=None, stats=None):
            """[1, F, 160] features -> z [1, F, 8] (float32)."""
            h, _ = self.model.feature_projection(feats)
            if self.lvl is not None:
                h = h + self.lvl(((stats - self.lvl_shift) / self.lvl_scale).to(h.dtype))[:, None]
            h = self.model.encoder(h).last_hidden_state
            s = self.sem(h.transpose(1, 2).float()).transpose(1, 2)
            return self.head(s) if self.unified else s @ self.A_s.T + ca

    class TinyEncoder(nn.Module):
        """`encode_code(wav_16k)` -> [F] codes, the codes NeuCodec's decoder and the LM read."""

        def __init__(self, codec, student, bf16=True):
            super().__init__()
            self.student, self.bf16 = student, bf16
            self.fbank = TorchFbank(codec.feature_extractor)
            self.acoustic = None if student.unified else codec.CodecEnc
            _, A_a, bias = linear_parts(codec)
            self.register_buffer("A_a", A_a)
            self.register_buffer("bias", bias)

        @torch.no_grad()
        def encode_code(self, wav):
            dev = self.A_a.device
            y = prepare_audio(wav).to(dev)
            feats, raw = self.fbank(y)
            n, ca = feats.shape[0], None
            if self.acoustic is not None:
                a = self.acoustic(y)[0].float()
                n = min(n, a.shape[0])
                ca = (a[:n] @ self.A_a.T + self.bias)[None]
            # level: over the frames of the audio that is encoded, as in training
            st = self.fbank.stats(raw[None, : 2 * n - 2]) if self.student.level else None
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=self.bf16 and dev.type == "cuda"):
                z = self.student(feats[None, :n], ca, st)
            return codes_from_z(z[0].float())

    return TorchFbank, Student, TinyEncoder


REPO = "Scicom-intl/Tiny-NeuCodec-Encoder"


def resolve(source, repo=REPO):
    """Where `source` points, as a local `student.pt`:

      S4                                      a model's name: that folder of `repo` on Hugging Face
      org/name  or  org/name/folder           a Hugging Face repository, or a folder inside one
      /path/to/dir  or  /path/to/student.pt   a local copy
    """
    if os.path.isfile(source):
        return source
    if os.path.isdir(source):
        return os.path.join(source, "student.pt")
    from huggingface_hub import hf_hub_download

    parts = source.strip("/").split("/")
    if len(parts) == 1:
        return hf_hub_download(repo, f"{parts[0]}/student.pt")
    return hf_hub_download("/".join(parts[:2]), "/".join(parts[2:] + ["student.pt"]))


def load_tiny_encoder(source, codec, device="cuda", bf16=True, free_teacher=False, repo=REPO):
    """Build the encoder named by `source` on `device`, around an already loaded NeuCodec.

    `free_teacher` drops the parts of NeuCodec's own encoder the student replaces (w2v-BERT, and for a U model the
    convolutional branch too), so they stop holding memory. After that `codec.encode_code` no longer works; decoding
    is unaffected. `source`: see `resolve`. Returns the encoder and a dict describing it.
    """
    import torch

    _, Student, TinyEncoder = _modules()
    ck = torch.load(resolve(source, repo), map_location="cpu", weights_only=True)
    if ck.get("wave_stem"):
        raise ValueError("this build does not load students with a waveform stem")
    student = Student(codec, ck["depth"], ck.get("unified", False), ck.get("level", False))
    missing, unexpected = student.load_state_dict(ck["state"], strict=False)
    if set(missing) - {"A_s"} or unexpected:
        raise ValueError(f"checkpoint does not fit the model: missing {sorted(missing)}, unexpected {sorted(unexpected)}")
    enc = TinyEncoder(codec, student, bf16=bf16).to(device).eval()
    info = {"depth": student.depth, "unified": student.unified, "level": student.level, "step": ck.get("step")}
    if free_teacher:
        codec.semantic_model = None
        if student.unified:
            codec.CodecEnc = None
    return enc, info
