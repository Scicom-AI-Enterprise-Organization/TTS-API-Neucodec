"""response_format=tokens (app/speech_tokens.py): parsing, the SSE reader, the response body. No torch, no GPU,
no server: runs anywhere (python -m pytest tests/test_speech_tokens.py)."""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from app import speech_tokens as st  # noqa: E402


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_parse_ids_keeps_order_and_ignores_other_text():
    assert st.parse_ids('<|s_12|><|s_7|>junk<|s_65535|><|im_end|>') == [12, 7, 65535]
    assert st.parse_ids('') == []
    assert st.parse_ids(None) == []


def test_collect_joins_deltas_and_keeps_the_last_finish_reason():
    async def deltas():
        for d in [('<|s_1|>', None), ('', None), ('<|s_2|><|s_3|>', None), ('', 'stop')]:
            yield d
    text, fr = run(st.collect(deltas()))
    assert text == '<|s_1|><|s_2|><|s_3|>' and fr == 'stop'


class _Resp:
    def __init__(self, status, lines, text=''):
        self.status, self._text = status, text

        async def content():
            for l in lines:
                yield l
        self.content = content()

    async def text(self):
        return self._text


class _Post:
    """aiohttp's session.post shape: an async context manager; records the body it was given."""
    def __init__(self, resp):
        self.resp, self.calls = resp, []

    def __call__(self, url, headers=None, json=None):
        self.calls.append(json)
        resp = self.resp

        class _CM:
            async def __aenter__(self):
                return resp

            async def __aexit__(self, *a):
                return False
        return _CM()


def _sse(text, fr=None):
    return b'data: ' + json.dumps({'choices': [{'text': text, 'finish_reason': fr}]}).encode() + b'\n'


def test_http_deltas_reads_vllm_sse_and_forces_stream():
    post = _Post(_Resp(200, [_sse('<|s_4|>'), b'\n', _sse('<|s_5|>'), _sse('', 'length'), b'data: [DONE]\n']))
    text, fr = run(st.collect(st.http_deltas(post, 'http://lm/v1/completions', {}, {'prompt': 'p', 'stream': False})))
    assert text == '<|s_4|><|s_5|>' and fr == 'length'
    assert post.calls[0]['stream'] is True and post.calls[0]['prompt'] == 'p'


def test_http_deltas_raises_on_backend_error():
    post = _Post(_Resp(500, [], text='boom'))
    try:
        run(st.collect(st.http_deltas(post, 'u', {}, {})))
    except RuntimeError as e:
        assert '500' in str(e) and 'boom' in str(e)
    else:
        raise AssertionError('expected RuntimeError')


def test_response_body_shape():
    b = st.response_body(model='TTS-model', voice='TM_English_Normal', text_in='Hi?', normalized='Hi?',
                         completion='<|s_1|><|s_2|><|im_end|>', finish_reason='stop')
    assert b['object'] == 'speech.tokens' and b['tokens'] == [1, 2] and b['n_tokens'] == 2
    assert b['speech'] == '<|s_1|><|s_2|>' and b['seconds'] == 0.04 and b['finish_reason'] == 'stop'
    assert 'prompt' not in b and 'interleave' not in b
    b = st.response_body(model='m', voice='v', text_in='x', normalized='x', completion='', finish_reason=None,
                         prompt='<|im_start|>...', interleave={'id': 'k', 'history_turns': 2, 'stored': True})
    assert b['tokens'] == [] and b['prompt'] == '<|im_start|>...' and b['interleave']['stored'] is True
