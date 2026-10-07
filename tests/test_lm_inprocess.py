"""app/lm_inprocess.py: the in-process LM must hand the producer the same deltas the SSE path
does, and must not leave the engine generating for a consumer that went away.

No vLLM needed: the engine is a fake with vLLM's generate()/abort() shape (cumulative outputs,
the default RequestOutputKind)."""
import asyncio
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from app import lm_inprocess as L  # noqa: E402


class FakeEngine:
    def __init__(self, texts, finish='stop'):
        self.texts, self.finish, self.aborted = texts, finish, []

    async def generate(self, prompt, params, rid):
        for i, t in enumerate(self.texts):
            last = i == len(self.texts) - 1
            yield SimpleNamespace(finished=last, outputs=[SimpleNamespace(
                text=t, finish_reason=self.finish if last else None)])
            await asyncio.sleep(0)

    async def abort(self, rid):
        self.aborted.append(rid)


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def collect(engine, monkeypatch, stop_after=None):
    monkeypatch.setattr(L, 'sampling_params', lambda body: None)   # no vllm import

    async def go():
        out, gen = [], L.stream('p', {}, engine=engine)
        async for item in gen:
            out.append(item)
            if stop_after is not None and len(out) >= stop_after:
                break
        await gen.aclose()
        return out
    return run(go())


def test_cumulative_outputs_become_deltas(monkeypatch):
    e = FakeEngine(['<|s_1|>', '<|s_1|><|s_2|>', '<|s_1|><|s_2|><|s_3|>'])
    out = collect(e, monkeypatch)
    assert [d for d, _ in out] == ['<|s_1|>', '<|s_2|>', '<|s_3|>']
    assert ''.join(d for d, _ in out) == '<|s_1|><|s_2|><|s_3|>'


def test_finish_reason_only_on_the_last_item(monkeypatch):
    out = collect(FakeEngine(['a', 'ab', 'abc'], finish='length'), monkeypatch)
    assert [fr for _, fr in out] == [None, None, 'length']


def test_a_finish_with_no_new_text_still_reports_the_reason(monkeypatch):
    out = collect(FakeEngine(['ab', 'ab']), monkeypatch)
    assert out == [('ab', None), ('', 'stop')]


def test_a_consumer_that_stops_early_aborts_the_request(monkeypatch):
    e = FakeEngine(['a', 'ab', 'abc', 'abcd'])
    out = collect(e, monkeypatch, stop_after=2)
    assert len(out) == 2 and len(e.aborted) == 1


def test_a_finished_request_is_not_aborted(monkeypatch):
    e = FakeEngine(['a', 'ab'])
    collect(e, monkeypatch)
    assert e.aborted == []


def test_sampling_is_exactly_the_http_body():
    """temperature, repetition_penalty, max_tokens and nothing else: top-k/top-p stay off,
    as on the served engine (the body never sends them)."""
    import pytest
    vllm = pytest.importorskip('vllm')
    p = L.sampling_params({'temperature': 0.6, 'repetition_penalty': 1.15, 'max_tokens': 3072,
                           'model': 'TTS-model', 'prompt': 'x', 'stream': True})
    assert (p.temperature, p.repetition_penalty, p.max_tokens) == (0.6, 1.15, 3072)
    assert p.top_k in (0, -1) and p.top_p == 1.0
