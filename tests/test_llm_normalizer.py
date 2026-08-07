"""
Unit tests for the LLM-based normalizer (app/llm_normalizer.py).

No GPU / torch / network required: HTTP behaviour is tested against a local fake
OpenAI-compatible server. Run with:

    uv run --with aiohttp --with pytest pytest tests/test_llm_normalizer.py -v

The TestLive class talks to the real endpoint and only runs when OPENAI_BASE_URL,
OPENAI_API_KEY and OPENAI_MODEL_NAME are set (e.g. `set -a; source .env`).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import asyncio
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app.llm_normalizer import (
    build_messages,
    parse_normalized,
    llm_normalize,
    LLMNormalizerError,
    NormalizerMode,
)
from app.prompt import SYSTEM_PROMPT, EXAMPLES, JSON_SCHEMA


class TestBuildMessages:
    def test_shape(self):
        msgs = build_messages('hello')
        assert msgs[0] == {'role': 'system', 'content': SYSTEM_PROMPT}
        assert len(msgs) == 2 + 2 * len(EXAMPLES)
        assert msgs[-1] == {'role': 'user', 'content': 'hello'}

    def test_few_shot_pairs_alternate(self):
        msgs = build_messages('x')[1:-1]
        for i, m in enumerate(msgs):
            assert m['role'] == ('user' if i % 2 == 0 else 'assistant')

    def test_assistant_replies_are_json_normalized(self):
        for m in build_messages('x')[1:-1]:
            if m['role'] == 'assistant':
                out = json.loads(m['content'])
                assert isinstance(out['normalized'], str) and out['normalized']

    def test_examples_match_prompt_pairs(self):
        msgs = build_messages('x')[1:-1]
        for (before, after), (u, a) in zip(EXAMPLES, zip(msgs[::2], msgs[1::2])):
            assert u['content'] == before
            assert json.loads(a['content'])['normalized'] == after

    def test_chinese_not_ascii_escaped(self):
        blob = json.dumps(build_messages('x'), ensure_ascii=False)
        assert '令吉' in blob  # ensure_ascii=False keeps CJK readable for the model


class TestJsonSchema:
    def test_schema_is_strict_single_string_field(self):
        schema = JSON_SCHEMA['schema']
        assert JSON_SCHEMA['strict'] is True
        assert schema['required'] == ['normalized']
        assert schema['properties']['normalized'] == {'type': 'string'}
        assert schema['additionalProperties'] is False


class TestParseNormalized:
    def test_clean_json(self):
        assert parse_normalized('{"normalized": "lima puluh ringgit"}') == 'lima puluh ringgit'

    def test_code_fenced_json(self):
        assert parse_normalized('```json\n{"normalized": "abc"}\n```') == 'abc'

    def test_bare_fence(self):
        assert parse_normalized('```\n{"normalized": "abc"}\n```') == 'abc'

    def test_bare_json_string(self):
        assert parse_normalized('"just the text"') == 'just the text'

    def test_plain_text_fallback(self):
        assert parse_normalized('lima puluh ringgit') == 'lima puluh ringgit'

    def test_unicode(self):
        assert parse_normalized('{"normalized": "五十令吉"}') == '五十令吉'

    def test_empty_is_none(self):
        assert parse_normalized('') is None
        assert parse_normalized(None) is None
        assert parse_normalized('{"normalized": "  "}') is None

    def test_json_without_key_is_none(self):
        assert parse_normalized('{"text": "abc"}') is None
        assert parse_normalized('[1, 2]') is None


class FakeOpenAI(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible /chat/completions stub. Behaviour is driven by
    class attributes set per-test; every request body is recorded."""

    requests = []
    reply_content = '{"normalized": "ok"}'
    status = 200
    reject_response_format = False  # first 400s json_schema, like older backends

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        FakeOpenAI.requests.append({'path': self.path, 'headers': dict(self.headers), 'body': body})

        if self.reject_response_format and 'response_format' in body:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "response_format is not supported"}')
            return
        self.send_response(self.status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        reply = {'choices': [{'message': {'role': 'assistant', 'content': self.reply_content}}]}
        self.wfile.write(json.dumps(reply).encode())

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_server():
    FakeOpenAI.requests = []
    FakeOpenAI.reply_content = '{"normalized": "ok"}'
    FakeOpenAI.status = 200
    FakeOpenAI.reject_response_format = False
    server = HTTPServer(('127.0.0.1', 0), FakeOpenAI)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f'http://127.0.0.1:{server.server_port}/v1'
    server.shutdown()


def run(coro):
    return asyncio.run(coro)


class TestLLMNormalize:
    def test_success(self, fake_server):
        FakeOpenAI.reply_content = '{"normalized": "lima puluh ringgit"}'
        out = run(llm_normalize('RM50', base_url=fake_server, api_key='k', model='m'))
        assert out == 'lima puluh ringgit'

    def test_request_payload(self, fake_server):
        run(llm_normalize('RM50', base_url=fake_server, api_key='secret', model='my-model'))
        (req,) = FakeOpenAI.requests
        assert req['path'] == '/v1/chat/completions'
        assert req['headers']['Authorization'] == 'Bearer secret'
        body = req['body']
        assert body['model'] == 'my-model'
        assert body['temperature'] == 0
        assert body['response_format'] == {'type': 'json_schema', 'json_schema': JSON_SCHEMA}
        assert body['messages'][-1] == {'role': 'user', 'content': 'RM50'}

    def test_no_auth_header_when_key_empty(self, fake_server):
        run(llm_normalize('x', base_url=fake_server, api_key='', model='m'))
        assert 'Authorization' not in FakeOpenAI.requests[0]['headers']

    def test_retries_without_response_format_on_400(self, fake_server):
        FakeOpenAI.reject_response_format = True
        FakeOpenAI.reply_content = '{"normalized": "retried"}'
        out = run(llm_normalize('x', base_url=fake_server, api_key='k', model='m'))
        assert out == 'retried'
        first, second = FakeOpenAI.requests
        assert 'response_format' in first['body']
        assert 'response_format' not in second['body']

    def test_server_error_raises(self, fake_server):
        FakeOpenAI.status = 500
        with pytest.raises(LLMNormalizerError, match='500'):
            run(llm_normalize('x', base_url=fake_server, api_key='k', model='m'))

    def test_unusable_reply_raises(self, fake_server):
        FakeOpenAI.reply_content = '{"wrong_key": "abc"}'
        with pytest.raises(LLMNormalizerError, match='unusable'):
            run(llm_normalize('x', base_url=fake_server, api_key='k', model='m'))

    def test_plain_text_reply_accepted(self, fake_server):
        FakeOpenAI.reply_content = 'lima puluh ringgit'
        out = run(llm_normalize('RM50', base_url=fake_server, api_key='k', model='m'))
        assert out == 'lima puluh ringgit'

    def test_unconfigured_raises(self):
        with pytest.raises(LLMNormalizerError, match='not configured'):
            run(llm_normalize('x', base_url='', api_key='', model=''))

    def test_unreachable_raises(self):
        with pytest.raises(LLMNormalizerError, match='failed'):
            run(llm_normalize('x', base_url='http://127.0.0.1:1/v1', api_key='k', model='m', timeout=2))


class TestNormalizerMode:
    def test_values(self):
        assert NormalizerMode('rule') == NormalizerMode.rule
        assert NormalizerMode('llm') == NormalizerMode.llm
        with pytest.raises(ValueError):
            NormalizerMode('nope')


LIVE_READY = all(os.environ.get(k) for k in ('OPENAI_BASE_URL', 'OPENAI_API_KEY', 'OPENAI_MODEL_NAME'))


@pytest.mark.skipif(not LIVE_READY, reason='OPENAI_* env not set (set -a; source .env)')
class TestLive:
    """Hits the real OPENAI_BASE_URL — verifies the LLM actually normalizes."""

    def test_malay_money(self):
        out = run(llm_normalize('baki saya tinggal RM50'))
        assert 'lima puluh ringgit' in out.lower()
        assert not re.search(r'\d', out)

    def test_english_ic(self):
        out = run(llm_normalize('my IC is 960314875079'))
        assert not re.search(r'\d', out)
        assert 'nine' in out.lower()

    def test_chinese_money(self):
        out = run(llm_normalize('余额还剩RM50'))
        assert '五十' in out
        assert not re.search(r'\d', out)

    def test_passthrough(self):
        out = run(llm_normalize('apa khabar semua?'))
        assert out.rstrip('.') == 'apa khabar semua?'
