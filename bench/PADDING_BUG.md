# The decode batcher was padding windows, and padding corrupts a non-causal decoder

**Found 2026-09-22**, from a report that enabling `CUDA_GRAPH_BATCH` made the voice sound worse.
It does, the effect is large, and the repo's standing claim that it cannot was wrong.

![padding bug](../docs/img/padding_bug.png)

![cuda graphs](../docs/img/cuda_graphs.png)

![precision matrix](../docs/img/precision_matrix.png)


## The claim that was wrong

CLAUDE.md said, of CUDA graphs, MPS and multi-worker:

> All optimizations … are **bit-identical decode operations** — no weight, precision, or sampling
> change — so accuracy cannot regress by construction.

The graph *replay* is bit-identical: decoding the same tensor twice gives `max|a−b| = 0`. What is
not identical is **what gets fed to it**. Turning buckets on changes the input.

## The mechanism

`_batch_one` padded every decode up to `choose_bucket_len(max_len)` — the next CUDA-graph bucket,
or the longest item in the batch when no buckets are configured — and `make_pinned_batch` fills
the padding with **speech token id 0**. `_compute_one` then sliced each request's own samples back
out with `ys[i:i+1, :, :out_len]`.

Slicing removes the padding's *output*. It does not remove its *influence*. The NeuCodec decoder is
non-causal — global attention over the window, a conv receptive field, ISTFT `'same'` padding — so
those pad frames are right-context that the unpadded decode never had, and they change the samples
that are kept.

## Measured, on real LM tokens

| case | SNR vs decoding it alone | max\|Δ\| (signal peak 0.78–1.19) |
|---|---|---|
| 340-token window padded to graph bucket 500 | **4.7 dB** | 1.433 |
| 340-token window padded to bucket 675 | **4.3 dB** | 1.164 |
| 180-token window batched with a 470-token one, **buckets empty** | **−1.3 dB** | 1.606 |
| 180-token window batched with another **180**-token one | **72.3 dB** | **0.000** |

The third row is the important one: this is not a CUDA-graph bug, it is a **padding** bug, and
padding also happens with graphs off whenever dynamic batching puts different-length windows
together. The fourth row is why grouping is a fix rather than a mitigation — equal lengths are
bit-exact.

And the damage is not at the seam. Splitting the error by position for a 160-token pad:

| region | max\|Δ\| | rms |
|---|---|---|
| first 25% | 0.840 | −23.2 dB |
| **middle 50%** | **1.433** | **−18.6 dB** |
| last 25% | 0.365 | −26.2 dB |

Worst in the middle, which is what global attention over a corrupted region does.

Audible examples: `ucc_ai_research/evaluation/tts/synthetic-audio/2026-09-22-padding-bug/` —
the same 238 tokens decoded clean, batched with a longer request, padded to a bucket, and the
difference signal on its own.

## Why nothing caught it

The Whisper-CER guardrail runs at **concurrency 1** with buckets empty. One item per batch means
`max_len` is that request's own length, so nothing is padded and the broken path never executes.
The guardrail was structurally incapable of seeing this.

## The fix

1. **One decode per distinct token length** (`app/batching.py:group_by_length`). A batch of mixed
   lengths becomes several GPU calls; every call pads nothing.
2. **`choose_bucket_len` is no longer on the decode path.** Rounding up *requires* padding, so the
   bucket idea and correctness are incompatible.
3. **CUDA graphs are now captured lazily on the exact shape** (`CUDA_GRAPH_LAZY`, default on,
   bounded by `CUDA_GRAPH_MAX_SHAPES`, default 64). This is what makes (1) and (2) affordable:
   fixed buckets are useless once you stop padding — measured, **0 of 150** real decodes landed on
   a configured bucket, because the stitcher's schedule produces lengths like 47 / 121 / 269, not
   multiples of 50. The real shapes are few and repeat, so a cache keyed on the exact shape hits
   **80–84%**, with the top three shapes alone covering ~64% of decodes.
4. **`CUDAGraphsWrapper.wrap` captures with `capture_error_mode="thread_local"`.** Under CUDA's
   default `"global"` mode the whole context enters capture and the batch thread's `pin_memory()`
   dies with *"operation not permitted when stream is capturing"* — 30 of 48 requests failed on the
   first attempt at lazy capture. Fine when every graph is captured at startup; fatal once capture
   happens mid-serving.

## After

| | value |
|---|---|
| audio vs the eager reference | **223 dB SNR — identical** |
| errors | **0** |
| graph hit rate | **80–84%** |

Enabling graphs now changes nothing audible, which is what the old claim promised and did not
deliver.

### What graphs are worth, and what correctness cost

Three builds on one GPU, one worker each, benchmark passes **interleaved** so drift on the shared
card hits every arm equally (audio-s/s):

| concurrency | eager (prod default) | patched + lazy graphs | **graphs vs eager** | pre-fix graphs (wrong audio) | **cost of correctness** |
|---|---|---|---|---|---|
| 8 | 66.7 | 65.8 | **1.0×** | 67.6 | −3% |
| 32 | 132.2 | 190.6 | **1.44×** | 209.4 | −9% |
| 64 | 138.8 | 213.4 | **1.54×** | 296.2 | −28% |

Two things to read off this:

- **CUDA graphs only pay when the codec GPU is the constraint.** At c=8 they are worth nothing —
  the GPU sits around 37% and there is no launch overhead worth removing. At c=32–64 they are
  worth **1.44–1.54×**, which broadly confirms the ~1.7× in the CLAUDE.md table rather than
  refuting it. Benchmarking this at low concurrency measures the wrong thing.
- **Correctness costs throughput, and the cost grows with load** — 3% at c=8, 28% at c=64. Grouping
  by exact length splits what used to be one padded batch into several smaller ones, so the more
  concurrent requests there are at differing window sizes, the more the batching is fragmented.
  That is the real price of the fix, and it is worth paying: the alternative is 2.1× the throughput
  at 10.6 dB SNR.

If that 28% matters, the lever is making window lengths *collide* rather than padding them apart —
e.g. quantising the stitcher's schedule so concurrent requests land on the same few lengths. That
keeps batches whole without ever padding, and is the only way to get both.

⚠ Two measurement traps met on the way:

- **A hash is the wrong test.** Batch *size* alone shifts the output ~1e-4 (cuBLAS picking
  different reduction orders per shape), which flips a SHA while sitting 90+ dB down. Compare audio
  distance, not digests — the first end-to-end run looked like a total failure (7/8 "DIFF") purely
  because of this.
- **Firing N requests simultaneously does not produce mixed-length batches.** They run in lockstep
  on the same window schedule, so they batch at equal lengths and the bug hides. Serial-vs-
  concurrent measured 58–69 dB on the *unfixed* code for exactly this reason. Staggered arrivals
  and differing final-flush windows are what trigger it in production.

## A/B on identical hardware, and what UTMOSv2 says

Three builds, same GPU, same single worker, same texts, temperature 0, **serial** — so
batching is not a variable and the only difference is the code. Buckets pad every window
even at concurrency 1, which is why the reported defect shows up without any load at all:

| build | SNR vs the eager reference |
|---|---|
| pre-fix, graphs ON | **median 10.6 dB** (range 6.8–14.2) |
| patched, graphs ON | **median 222.7 dB** — identical |

**UTMOSv2 cannot see this defect.** Scored through `tm-h20-utmosv2`, 6 reps per file:

| build | mean MOS | within-file sd (the noise floor) |
|---|---|---|
| eager reference | 3.313 | 0.170 |
| pre-fix + graphs (10.6 dB SNR) | **3.367** | 0.116 |
| patched + graphs (bit-identical) | 3.283 | 0.129 |

The corrupted build scored **higher** than the reference, and the bit-identical build
scored lower — both deltas inside the scorer's own noise. Since the patched build is
bit-identical to the reference, its −0.030 is a direct read of that noise floor, and the
corrupted build's +0.054 is smaller still.

⚠ **Do not use UTMOSv2 to validate this class of change.** It scores random crops for
naturalness and is blind to broadband additive corruption, the same way it was flat
against the continuity change in `bench/INTERLEAVE_AB.md`. A sample-wise diff against a
known-good reference is the test that works; UTMOSv2 will happily certify corrupted audio.

## Reproducing

```bash
# tensor level, no LM in the loop — the decisive test
python - <<'PY'
from app.neucodec import NeuCodec; import torch, numpy as np
c = NeuCodec.from_pretrained('neuphonic/neucodec').eval().cuda()
d = lambda ids: c.decode_code(torch.tensor(ids)[None,None].cuda())[0,0].float().cpu().numpy()
w = [...]                      # any real speech-token stream
bare, pad = d(w), d(w + [0]*160)[:len(d(w))]
print('SNR', 20*np.log10(np.sqrt(np.mean(bare**2))/np.sqrt(np.mean((pad-bare)**2))))
PY

# end to end: graphs-on must equal graphs-off, at temperature 0
python bench/latency_bench.py --url http://127.0.0.1:9095 --concurrency 8
pytest tests/test_decode_batching.py -q      # runs anywhere, imports nothing
```
