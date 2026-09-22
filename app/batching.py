"""How a decode batch is split before it reaches the GPU.

One rule, and it is the whole module: **never pad one request's tokens up to another's
length.** NeuCodec's decoder is non-causal -- global attention over the window, a conv
receptive field, ISTFT 'same' padding -- so trailing pad tokens are right-context that
the unpadded decode never had, and their influence reaches the samples that are kept.
Slicing the padding's own output back off does not undo it.

Measured on real LM tokens (2026-09-22, `bench/PADDING_BUG.md`):

    180-token window batched with a 470-token one   -1.3 dB SNR vs decoding it alone
    340-token window padded to CUDA-graph bucket 500  4.7 dB SNR
    two windows of the SAME length, batched          -90 dB, max|d| 0.000  (identical)

The error is *louder than the speech* in the first case, and worst in the MIDDLE of the
window rather than at the seam -- which is what global attention over the padded region
does. Equal-length batching is bit-exact, so grouping is a fix and not a mitigation.

This module imports nothing, so `tests/test_decode_batching.py` runs anywhere -- unlike
anything that reaches through `app.main`, which cannot even be imported outside a running
event loop.
"""
from __future__ import annotations


def group_by_length(batch):
    """Split a decode batch into one group per distinct token length, order preserved.

    `batch` items are the `(future, tokens, meta)` triples `dynamic_batching` builds.
    Everything in a returned group has the identical token length, so padding to the
    group's length copies every row in full and pads nothing.
    """
    groups: dict[int, list] = {}
    for item in batch:
        groups.setdefault(len(item[1]), []).append(item)
    return groups
