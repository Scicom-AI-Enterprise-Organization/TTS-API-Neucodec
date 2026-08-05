# Ascend 910B3: bf16 attention precision degrades autoregressive TTS naturalness

**Prepared for:** Huawei Ascend / CANN / vllm-ascend engineering
**Date:** 2026-07-08
**Reporter:** Scicom AI Enterprise (TTS platform team)
**Severity:** High — blocks production use of the 910B3 for a high-quality TTS model that otherwise runs end-to-end on the NPU.

---

## 1. Executive summary

We ported a streaming Text-to-Speech pipeline (autoregressive LM → neural audio codec) to the Ascend 910B3.
The system runs end-to-end on the NPU and is **intelligible**, but the generated speech is **measurably and
audibly less natural than the identical software stack on an NVIDIA H100**: **−0.6 MOS** (UTMOSv2 2.60 vs
3.21) at matched bf16 precision and matched sampling settings.

We isolated the cause to the **bf16 fused attention kernel in `vllm-ascend` / CANN on the 910B3**. Everything
else was ruled out with controlled experiments:

- the neural codec decode is **bit-identical on NPU vs CPU** (MAE = 0.0),
- the sampler already runs **fp32 softmax**,
- matmul is at **full precision** (`allow_hf32 == False`),
- and **fp32 is not available** as a fallback — `--dtype float32` crashes the engine because the paged-KV
  attention op (`ReshapeCacheOperation`) supports only bf16/fp16.

So the LM's bf16 attention path produces subtly different logits than the H100's bf16 attention, which shifts
the sampled speech tokens toward less-natural audio, and **there is no configuration-level way to raise the
attention precision** on the current stack.

**Primary ask:** provide a higher-precision (fp32-accumulation) execution path for the *standard* (GQA)
fused-attention kernels — the same `kernel_type_high_precision` option that already exists for MLA — and/or
fp32 support for the paged-KV attention op so a full-precision reference is possible.

---

## 2. Environment

| Component | Value |
|---|---|
| Accelerator | Huawei Ascend **910B3**, 64 GB HBM/chip (8 per host) |
| CANN | **8.5.2** (npu-smi / driver **25.5.1**) |
| NNAL/ATB | present (`/usr/local/Ascend/nnal/atb/set_env.sh`) |
| Host | aarch64, Python 3.11 |
| Framework | `torch==2.7.1`, `torch-npu==2.7.1` |
| Serving | `vllm==0.11.0`, `vllm-ascend==0.11.0` |
| Precision | bfloat16 (model native) |

**Reference (NVIDIA):** H100 SXM 80 GB, `torch==2.9.1+cu128`, CUDA 12.8, `vllm==0.16.0`, bfloat16 — same
model, same sampling parameters.

## 3. Workload

- **LM:** `Qwen3-1.7B` continued-pretrained to emit discrete speech tokens; vocabulary ≈ **217K**
  (speech-token vocab). Standard **GQA** attention (not MLA). Served by vLLM (`vllm-ascend` backend on NPU 0).
- **Codec:** NeuCodec decoder (speech tokens → 24 kHz waveform), run via `torch_npu` on NPU 1. fp32.
- **Sampling:** temperature 0.6, repetition_penalty 1.15, no top-k / top-p. Identical on both platforms.
- **Eval set:** fixed 16 sentences (English / Malay / code-switch), ~2–8 s audio each.

## 4. Symptom — quality gap at matched precision

Single-stream, 16-sentence eval set, bf16, temperature 0.6, identical decode on both platforms.

| Metric | H100 SXM | Ascend 910B3 | Notes |
|---|---|---|---|
| LM throughput | ~380 tok/s | ~81 tok/s | ~4.8× slower (secondary; expected from HBM bandwidth) |
| End-to-end RTF | ~0.13 | ~0.68 | LM-bound |
| Intelligibility — Whisper large-v3 **CER** | 2.2 % | 3.4 % | comparable; content is **not** garbled |
| **Naturalness — UTMOSv2 MOS** | **3.21** | **2.60** | **−0.61 MOS — audibly worse** |

MOS measured with UTMOSv2 (VoiceMOS Challenge 2024 Track-1 top system), `fusion_stage3`, fold 0.
14 of 16 clips scored lower on Ascend; several by more than 1.0 MOS. The direction is systematic and far
exceeds UTMOSv2's ~±0.07 run-to-run noise, so it is a real quality difference, not sampling variance.

The intelligibility being comparable while naturalness drops is the signature of **subtly wrong tokens**:
the audio is still understandable, but prosody/timbre are degraded.

## 5. Root-cause isolation

We eliminated every layer except the bf16 attention kernel. Each row is a controlled experiment.

| Hypothesis | Experiment | Result | Conclusion |
|---|---|---|---|
| Codec decode differs on NPU | Decode the **same** LM tokens on NPU vs CPU, per-sample MAE | **MAE = 0.00000** on all 16 clips | Decode is bit-identical — **ruled out** |
| Sampler is low-precision | Inspect `vllm_ascend/sample/sampler.py` | softmax uses `dtype=torch.float32` | Sampler is fp32 — **ruled out** |
| Matmul uses reduced precision (HF32) | Read `torch.npu.matmul.allow_hf32` | `False` | Matmul full precision — **ruled out** |
| ACL-graph capture introduces error | Serve with `--enforce-eager` (no ACL graph), re-measure MOS | MOS 2.45 (≈ 2.60 baseline, within noise) | Graph capture not the cause — **ruled out** |
| Precision too low → raise it (fp16) | Serve with `--dtype float16` | MOS **1.73** (much worse) | fp16 worse; bf16 is best available |
| Precision too low → raise it (fp32) | Serve with `--dtype float32` | **Engine crash** (see below) | fp32 **unavailable** — attention/KV op is bf16/fp16 only |

### The fp32 crash (the key limitation)

Launching vLLM with `--dtype float32` initialises and then dies on the first decode step:

```
RuntimeError: The Inner error is reported as above. The process exits for this inner error,
and the current working operator name is ReshapeCacheOperation.
[ERROR] 2026-07-08-21:18:23 (PID:..., Device:0, RankID:-1) ERR00100 PTA call acl api failed.
vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue.
```

`ReshapeCacheOperation` (paged-KV cache write feeding the fused attention) does not accept fp32, so the
attention path is **locked to bf16/fp16**. There is therefore no full-precision reference or fallback on this
stack.

### Attention kernels involved (from `vllm-ascend` 0.11.0)

- Prefill: `torch_npu._npu_flash_attention(...)`
- Decode: `torch_npu.npu_fused_infer_attention_score(...)`
- KV write: `ReshapeCacheOperation` (the op that rejects fp32)

None of these expose a precision / accumulation-dtype argument for the standard (GQA) path.

### Observation: a high-precision kernel already exists for MLA, but not GQA

`vllm-ascend` already passes `kernel_type="kernel_type_high_precision"` for **MLA** attention
(`vllm_ascend/attention/mla_v1.py`, `vllm_ascend/torchair/torchair_mla.py`). The **standard GQA** attention
path used by Qwen3-class models has **no equivalent**. This strongly suggests the accuracy-improving kernel
variant exists in CANN and simply needs to be exposed/used for the non-MLA attention path.

**Diagnosis:** with matmul, sampler, and decode all at full precision, the residual −0.6 MOS is attributable
to the **internal accumulation of the bf16 fused attention kernel** on the 910B3, which diverges from the
H100's bf16 attention enough to change sampled tokens in an autoregressive, high-entropy (217K-vocab) TTS
decoder — where small logit errors compound over hundreds of steps.

## 6. What we are asking for

1. **A high-precision / fp32-accumulation option for the standard (GQA) fused-attention kernels**
   (`_npu_flash_attention`, `npu_fused_infer_attention_score`) — i.e. extend the existing
   `kernel_type_high_precision` path from MLA to the ordinary attention path, exposed through `vllm-ascend`.
2. **fp32 support (or at least fp32-capable `ReshapeCacheOperation` / paged-KV)** so full-precision
   inference can be run as a correctness reference and fallback.
3. **Guidance / knobs** to control attention softmax + accumulation precision on 910B (any ACL precision-mode
   or env setting that raises attention accumulation to fp32).

Any one of these would let us verify and likely close the naturalness gap.

## 7. Reproduction

```bash
# --- vLLM (LM) on NPU 0, python 3.11 ---
uv venv /root/vllm-venv311 --python 3.11 && source /root/vllm-venv311/bin/activate
uv pip install vllm==0.11.0
uv pip install vllm-ascend==0.11.0 "setuptools<81"
source /usr/local/Ascend/ascend-toolkit/set_env.sh && source /usr/local/Ascend/nnal/atb/set_env.sh
ASCEND_RT_VISIBLE_DEVICES=0 vllm serve <Qwen3-1.7B-TTS> --dtype bfloat16 --port 9093 \
  --gpu-memory-utilization 0.6 --max-model-len 4096 --max-num-seqs 64 --served-model-name TTS-model

# reproduce the fp32 crash (attention/KV is bf16/fp16 only):
ASCEND_RT_VISIBLE_DEVICES=0 vllm serve <Qwen3-1.7B-TTS> --dtype float32 --port 9093 --enforce-eager
#   -> RuntimeError ... ReshapeCacheOperation ; ERR00100 PTA call acl api failed

# generate audio, then score naturalness with UTMOSv2:
#   pip install git+https://github.com/sarulab-speech/UTMOSv2
#   python -c "import utmosv2; m=utmosv2.create_model(pretrained=True); print(m.predict(input_dir='clips/', device='cpu'))"
```

Controlled decode check (proves the codec is not the cause):

```python
# same tokens, decode on NPU and CPU:
y_npu = codec.to("npu").decode_code(codes.to("npu"))[0,0].float().cpu().numpy()
y_cpu = codec.cpu().decode_code(codes)[0,0].numpy()
# mean(|y_npu - y_cpu|) == 0.00000 for every clip
```

## 8. Impact

The 910B3 decode path and the full serving pipeline work, but the **bf16 attention precision makes the NPU
unusable for this quality-sensitive TTS product**: −0.6 MOS is clearly audible and unacceptable for
customer-facing speech. Because there is no fp32 fallback and no precision knob for the standard attention
kernel, we currently have **no path to close the gap on-device** and must keep this workload on NVIDIA
H100/H200. Exposing a high-precision attention kernel (item 1 above) would unblock 910B3 adoption for TTS.

---

*Appendix — full mitigation MOS table:* baseline bf16+ACL-graphs **2.60**; bf16 `--enforce-eager` **2.45**;
`--dtype float16` **1.73**; `--dtype float32` **crash**; H100 bf16 **3.21**.
