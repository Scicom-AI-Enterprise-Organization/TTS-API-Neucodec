"""The decode batcher must never pad one request's tokens up to another's length.

NeuCodec's decoder is non-causal, so trailing pad tokens are right-context that the
unpadded decode never had, and their influence reaches the samples that are kept --
slicing the padding's own output back off does not undo it. Measured on real LM tokens
(2026-09-22): a 180-token window batched with a 470-token one came back at -1.3 dB SNR
against decoding it alone, i.e. the error louder than the speech, and worst in the middle
of the window rather than at the seam. Two windows of the SAME length are bit-identical
batched or not. Hence: one decode per distinct length.

    uv run --with pytest --with torch -- pytest tests/test_decode_batching.py -v
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.batching import group_by_length          # noqa: E402 -- imports nothing itself


def item(tokens, tag=None):
    """A batch entry as dynamic_batching builds it: (future, tokens, meta)."""
    return (tag or object(), tokens, None)


def test_equal_lengths_stay_one_group():
    batch = [item([1] * 50), item([2] * 50), item([3] * 50)]
    groups = group_by_length(batch)
    assert list(groups) == [50]
    assert len(groups[50]) == 3


def test_mixed_lengths_are_split():
    batch = [item([1] * 180), item([2] * 470), item([3] * 180)]
    groups = group_by_length(batch)
    assert sorted(groups) == [180, 470]
    assert len(groups[180]) == 2 and len(groups[470]) == 1


def test_every_item_survives_exactly_once():
    batch = [item([0] * n, tag=f't{i}') for i, n in enumerate([47, 121, 47, 269, 121, 47])]
    groups = group_by_length(batch)
    tags = [it[0] for g in groups.values() for it in g]
    assert sorted(tags) == sorted(f't{i}' for i in range(6))


def test_order_within_a_group_is_preserved():
    batch = [item([0] * 47, tag='a'), item([0] * 99, tag='x'),
             item([0] * 47, tag='b'), item([0] * 47, tag='c')]
    assert [it[0] for it in group_by_length(batch)[47]] == ['a', 'b', 'c']


def test_a_group_never_needs_padding():
    """The invariant the whole fix rests on: within a group, max == min length, so
    make_pinned_batch(tokens, n) copies every row fully and pads nothing."""
    batch = [item([0] * n) for n in (47, 121, 47, 269, 121, 47, 269)]
    for n, group in group_by_length(batch).items():
        assert {len(it[1]) for it in group} == {n}


def test_single_item_batch():
    groups = group_by_length([item([5] * 12)])
    assert list(groups) == [12] and len(groups[12]) == 1
