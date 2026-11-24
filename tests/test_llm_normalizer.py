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
    needs_normalization,
    clock_readings,
    clock_hint,
    number_language,
    rejected_reason,
    malay_hundreds,
    mail_words,
    LLMNormalizerError,
    NormalizerMode,
)
from app.prompt import (SYSTEM_PROMPT, EXAMPLES, JSON_SCHEMA, CALLER_PROMPT_HEADER, CLOCK_HINT_HEADER,
                        NUMBER_LANGUAGE_HINT)


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

    def test_prompt_in_system_and_user_turn(self):
        msgs = build_messages('RM50', prompt='normalize this in Malay')
        assert msgs[0]['content'] == (
            f'{SYSTEM_PROMPT}\n\n{CALLER_PROMPT_HEADER}\nnormalize this in Malay'
        )
        assert msgs[-1] == {'role': 'user', 'content': '[normalize this in Malay]\nRM50'}
        # the few-shots are untouched
        assert msgs[1:-1] == build_messages('RM50')[1:-1]

    def test_prompt_is_stripped(self):
        msgs = build_messages('x', prompt='  read numbers in English \n')
        assert msgs[0]['content'].endswith(f'{CALLER_PROMPT_HEADER}\nread numbers in English')
        assert msgs[-1]['content'] == '[read numbers in English]\nx'

    @pytest.mark.parametrize('prompt', [None, '', '   ', '\n'])
    def test_blank_prompt_is_no_prompt(self, prompt):
        assert build_messages('x', prompt=prompt) == build_messages('x')


class TestClockTime:
    """A written time is 24-hour unless am/pm or a period word is written, and is read on the
    12-hour clock with its period. The reading goes to the LLM with the text (clock_hint)."""

    @pytest.mark.parametrize('text,readings', [
        ('pada pukul pagi 8:00', [('pukul pagi 8:00', 'pukul pagi lapan')]),        # the prod report
        ('pada pukul pagi 8.00', [('pukul pagi 8.00', 'pukul pagi lapan')]),
        ('pukul 8:00', [('pukul 8:00', 'pukul lapan pagi')]),
        ('jam 8:00 pagi', [('jam 8:00 pagi', 'jam lapan pagi')]),
        ('mesyuarat 13:30', [('13:30', 'pukul satu tiga puluh minit petang')]),
        ('Kereta api bertolak pada 14:05.', [('14:05', 'pukul dua lima minit petang')]),
        ('pukul 12:15', [('pukul 12:15', 'pukul dua belas lima belas minit tengah hari')]),
        ('pukul 00:30', [('pukul 00:30', 'pukul dua belas tiga puluh minit tengah malam')]),
        ('Gangguan dijangka dari 23:00 hingga 02:00.',
         [('23:00', 'pukul sebelas malam'), ('02:00', 'pukul dua pagi')]),
        ('Kami akan hubungi anda sebelum 9.30am.', [('9.30am', 'pukul sembilan tiga puluh minit pagi')]),
        ('at 8:00', [('8:00', 'eight a m')]),
        ('at 20:00', [('20:00', 'eight p m')]),
        ('at 8:05 am', [('8:05 am', 'eight zero five a m')]),
        ('Our technician will arrive at 9:00 AM tomorrow.', [('9:00 AM', 'nine A M')]),
        ('Operating hours are 8:00 am to 5:00 pm.', [('8:00 am', 'eight a m'), ('5:00 pm', 'five p m')]),
        ('Your appointment is on Sept. 4, 2026 at 12:00 noon.', [('12:00', 'twelve')]),
        # no marker word either way: the dictionaries decide
        ('bas pertama 7:05', [('7:05', 'pukul tujuh lima minit pagi')]),
        ('bas terakhir 20:00, pertama 7:05',
         [('20:00', 'pukul lapan malam'), ('7:05', 'pukul tujuh lima minit pagi')]),
        ('kaunter dibuka 8:30', [('8:30', 'pukul lapan tiga puluh minit pagi')]),
        ('last bus 20:00', [('20:00', 'eight p m')]),
        ('Gate 7:05', [('7:05', 'seven zero five a m')]),
        ('在早上8:00', [('早上8:00', '早上八点')]),
        ('14:00开会', [('14:00', '下午两点')]),
        ('凌晨0:30', [('凌晨0:30', '凌晨零点三十分')]),
        ('停机时间是23:00到02:00。', [('23:00', '晚上十一点'), ('02:00', '凌晨两点')]),
    ])
    def test_readings(self, text, readings):
        assert clock_readings(text) == readings

    @pytest.mark.parametrize('text', [
        'harga RM8.00', 'suhu 37.50 darjah', 'Interest is charged at 3.25 per annum.', '4.8 bintang',
        'RM1,250.50', 'StaywithUnifi 5.0', 'score 3-1', '03-12345678', 'apa khabar semua?',
        'pukul 3 petang',                       # no minutes, no am/pm: the LLM reads it as before
        'கூட்டம் காலை 10:45 மணிக்கு',             # Tamil: no Tamil reading here
    ])
    def test_not_a_clock_time(self, text):
        assert clock_readings(text) == [] and clock_hint(text) is None

    def test_hint_in_system_and_user_turn(self):
        msgs = build_messages('pada pukul pagi 8:00')
        hint = f'{CLOCK_HINT_HEADER}\npukul pagi 8:00 -> pukul pagi lapan'
        assert msgs[0]['content'] == f'{SYSTEM_PROMPT}\n\n{hint}'
        assert msgs[-1]['content'] == f'[{hint}]\npada pukul pagi 8:00'
        assert msgs[1:-1] == build_messages('x')[1:-1]          # few-shots untouched

    def test_hint_after_caller_prompt(self):
        msgs = build_messages('pukul 8:00', prompt='normalize in Malay')
        hint = f'{CLOCK_HINT_HEADER}\npukul 8:00 -> pukul lapan pagi'
        assert msgs[0]['content'] == f'{SYSTEM_PROMPT}\n\n{CALLER_PROMPT_HEADER}\nnormalize in Malay\n\n{hint}'
        assert msgs[-1]['content'] == f'[normalize in Malay]\n[{hint}]\npukul 8:00'

    @pytest.mark.parametrize('reply,spoken,kept', [
        ('pada pukul pagi lapan sifar sifar', 'pukul pagi lapan', False),
        ('pada pukul pagi lapan.', 'pukul pagi lapan', True),
        ('Kereta api bertolak pada pukul dua lima minit petang.', 'pukul dua lima minit petang', True),
        ('pukul dua empat lima minit petang', 'pukul dua lima minit petang', False),
        ('at eight p m', 'eight a m', False),
        ('at eight a m, then', 'eight a m', True),
        ('at eight o clock', 'eight', False),
        ('at twelve noon', 'twelve', True),
        ('在早上八点', '早上八点', True),
        ('在早上八点零零', '早上八点', False),
        ('下午两点开会', '下午两点', True),
    ])
    def test_clock_kept(self, reply, spoken, kept):
        assert (rejected_reason('x 8:00', reply, clocks=[spoken]) != 'clock time changed') is kept

    def test_no_time_no_hint(self):
        assert build_messages('my balance is RM50')[0]['content'] == SYSTEM_PROMPT


class TestNumberLanguage:
    """Malay text has its numbers read in Malay: the LLM is told, and a reply with English
    number words is rejected. "Okay, 100 ringgit, betul?" came back "one hundred ringgit"
    10 of 10 (gemma-4-31b, 2026-11-24)."""

    @pytest.mark.parametrize('text', [
        '100 ringgit, betul?', 'Okay, 100 ringgit, betul?', '100 ringgit, boleh?',
        '250 ringgit sahaja.', 'RM89.90, betul?', '3 bulan, betul?',
        'Encik perlu bayar 350 ringgit sebelum hujung bulan.',
        'Baik, jumlahnya 100 ringgit, betul?',
    ])
    def test_malay(self, text):
        assert number_language(text) == 'ms'

    @pytest.mark.parametrize('text', [
        '100 ringgit, right?', "That's 50 ringgit.", "Okay, so that's 100 ringgit, right?",
        'RM50',                                 # no words: the caller's prompt decides
        '100 ringgit.', 'Total 100 ringgit.',   # no marker word: left to the LLM
        'RM100, ok?', '100 ringgit, ok?',       # "ok" is said in both
        'Bil anda RM250 for this month',        # the amount sits in the English part
        'Nanti saya call balik at 3pm ya',      # a clock time keeps its own reading
        'apa khabar semua?',                    # no number
        '余额还剩RM50', 'கூட்டம் 10 மணிக்கு',
    ])
    def test_not_malay(self, text):
        assert number_language(text) is None

    def test_hint_in_system_and_user_turn(self):
        msgs = build_messages('100 ringgit, betul?')
        assert msgs[0]['content'] == f'{SYSTEM_PROMPT}\n\n{NUMBER_LANGUAGE_HINT}'
        assert msgs[-1]['content'] == f'[{NUMBER_LANGUAGE_HINT}]\n100 ringgit, betul?'
        assert msgs[1:-1] == build_messages('x')[1:-1]

    def test_no_hint_with_caller_prompt(self):
        msgs = build_messages('100 ringgit, betul?', prompt='read it in English')
        assert NUMBER_LANGUAGE_HINT not in msgs[0]['content'] + msgs[-1]['content']

    def test_hint_after_clock_hint(self):
        msgs = build_messages('Bayar 50 ringgit sebelum 17:00, boleh?')
        hint = f'{CLOCK_HINT_HEADER}\n17:00 -> pukul lima petang'
        assert msgs[0]['content'] == f'{SYSTEM_PROMPT}\n\n{hint}\n\n{NUMBER_LANGUAGE_HINT}'

    @pytest.mark.parametrize('text,reply,rejected', [
        ('Okay, 100 ringgit, betul?', 'Okay, one hundred ringgit, betul?', True),
        ('250 ringgit sahaja.', 'two hundred fifty ringgit sahaja.', True),
        ('100 ringgit, betul?', 'seratus ringgit, betul?', False),
        ('Pelan One 100 ringgit, betul?', 'Pelan One seratus ringgit, betul?', False),   # the text's own "One"
    ])
    def test_reply_check(self, text, reply, rejected):
        reason = rejected_reason(text, reply, number_lang=number_language(text))
        assert (reason == 'English numbers in Malay text') is rejected

    def test_english_clock_reading_passes(self):
        text = 'Saya call balik at 3pm, nombor 2 ya?'
        assert number_language(text) == 'ms'
        reply = 'Saya call balik at three p m, nombor dua ya?'
        assert rejected_reason(text, reply, clocks=['three p m'], number_lang='ms') is None


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
        FakeOpenAI.reply_content = '{"normalized": "fifty ringgit"}'
        run(llm_normalize('RM50', base_url=fake_server, api_key='secret', model='my-model'))
        (req,) = FakeOpenAI.requests
        assert req['path'] == '/v1/chat/completions'
        assert req['headers']['Authorization'] == 'Bearer secret'
        body = req['body']
        assert body['model'] == 'my-model'
        assert body['temperature'] == 0
        assert body['response_format'] == {'type': 'json_schema', 'json_schema': JSON_SCHEMA}
        assert body['messages'][-1] == {'role': 'user', 'content': 'RM50'}
        assert body['messages'][0] == {'role': 'system', 'content': SYSTEM_PROMPT}

    def test_prompt_sent(self, fake_server):
        FakeOpenAI.reply_content = '{"normalized": "fifty ringgit"}'
        run(llm_normalize('RM50', base_url=fake_server, api_key='k', model='m',
                          prompt='normalize this in English'))
        msgs = FakeOpenAI.requests[0]['body']['messages']
        assert msgs[0]['content'].endswith(f'{CALLER_PROMPT_HEADER}\nnormalize this in English')
        assert msgs[-1] == {'role': 'user', 'content': '[normalize this in English]\nRM50'}

    def test_prompt_kept_on_retry_without_response_format(self, fake_server):
        FakeOpenAI.reject_response_format = True
        run(llm_normalize('x', base_url=fake_server, api_key='k', model='m', prompt='in Malay'))
        first, second = FakeOpenAI.requests
        assert first['body']['messages'] == second['body']['messages']
        assert second['body']['messages'][0]['content'].endswith('in Malay')

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

    def test_reply_that_drops_the_number_raises(self, fake_server):
        FakeOpenAI.reply_content = '{"normalized": "Did we get that right?"}'
        with pytest.raises(LLMNormalizerError, match='number dropped'):
            run(llm_normalize('750412105678. Did we get that right?', base_url=fake_server, api_key='k', model='m'))

    def test_reply_that_ignores_the_clock_reading_raises(self, fake_server):
        FakeOpenAI.reply_content = '{"normalized": "pada pukul pagi lapan sifar sifar"}'
        with pytest.raises(LLMNormalizerError, match='clock time changed'):
            run(llm_normalize('pada pukul pagi 8:00', base_url=fake_server, api_key='k', model='m'))

    def test_reply_that_keeps_the_clock_reading_passes(self, fake_server):
        FakeOpenAI.reply_content = '{"normalized": "pada pukul pagi lapan"}'
        assert run(llm_normalize('pada pukul pagi 8:00', base_url=fake_server, api_key='k', model='m')) \
            == 'pada pukul pagi lapan'

    def test_ordinal_after_a_clock_reading_passes(self, fake_server):
        reply = 'bas terakhir pukul lapan malam, pertama pukul tujuh lima minit pagi'
        FakeOpenAI.reply_content = json.dumps({'normalized': reply})
        assert run(llm_normalize('bas terakhir 20:00, pertama 7:05', base_url=fake_server,
                                 api_key='k', model='m')) == reply

    def test_satu_ratus_comes_back_seratus(self, fake_server):
        FakeOpenAI.reply_content = '{"normalized": "Bayaran satu ratus dua puluh ringgit."}'
        assert run(llm_normalize('Bayaran RM120.00.', base_url=fake_server, api_key='k', model='m')) \
            == 'Bayaran seratus dua puluh ringgit.'

    def test_mail_domain_comes_back_mel(self, fake_server):
        FakeOpenAI.reply_content = '{"normalized": "emel saya ali at yopmail dot com"}'
        assert run(llm_normalize('emel saya ali@yopmail.com', base_url=fake_server, api_key='k',
                                 model='m')) == 'emel saya ali at yopmel dot com'

    def test_english_numbers_for_malay_text_raise(self, fake_server):
        FakeOpenAI.reply_content = '{"normalized": "Okay, one hundred ringgit, betul?"}'
        with pytest.raises(LLMNormalizerError, match='English numbers in Malay text'):
            run(llm_normalize('Okay, 100 ringgit, betul?', base_url=fake_server, api_key='k', model='m'))
        assert NUMBER_LANGUAGE_HINT in FakeOpenAI.requests[-1]['body']['messages'][-1]['content']

    def test_prompt_may_ask_for_english_numbers(self, fake_server):
        FakeOpenAI.reply_content = '{"normalized": "one hundred ringgit, betul?"}'
        assert run(llm_normalize('100 ringgit, betul?', base_url=fake_server, api_key='k', model='m',
                                 prompt='read numbers in English')) == 'one hundred ringgit, betul?'

    def test_chinese_numerals_for_english_text_raise(self, fake_server):
        FakeOpenAI.reply_content = '{"normalized": "八 八 零 一 二 三 四 五"}'
        with pytest.raises(LLMNormalizerError, match='Chinese'):
            run(llm_normalize('8 8 0 1 2 3 4 5', base_url=fake_server, api_key='k', model='m'))

    def test_prompt_may_ask_for_chinese_numerals(self, fake_server):
        FakeOpenAI.reply_content = '{"normalized": "一五零零"}'
        out = run(llm_normalize('1500', base_url=fake_server, api_key='k', model='m',
                                prompt='please spell one by one in zh'))
        assert out == '一五零零'

    def test_unconfigured_raises(self):
        with pytest.raises(LLMNormalizerError, match='not configured'):
            run(llm_normalize('x', base_url='', api_key='', model=''))

    def test_unreachable_raises(self):
        with pytest.raises(LLMNormalizerError, match='failed'):
            run(llm_normalize('x', base_url='http://127.0.0.1:1/v1', api_key='k', model='m', timeout=2))


class TestMalayHundreds:
    @pytest.mark.parametrize('text,reply,want', [
        ('RM120', 'satu ratus dua puluh ringgit', 'seratus dua puluh ringgit'),
        ('RM180,000', 'satu ratus lapan puluh ribu ringgit', 'seratus lapan puluh ribu ringgit'),
        ('RM1,100', 'seribu satu ratus ringgit', 'seribu seratus ringgit'),
        ('RM1,150,000', 'satu juta satu ratus lima puluh ribu ringgit',
         'satu juta seratus lima puluh ribu ringgit'),
        ('101 aduan', 'Satu ratus satu aduan', 'Seratus satu aduan'),
        ('RM1,000', 'satu ribu ringgit', 'seribu ringgit'),
        ('RM1,001,000', 'satu juta satu ribu ringgit', 'satu juta seribu ringgit'),
        # "satu ribu" ending a larger group is right as it is
        ('RM21,000', 'dua puluh satu ribu ringgit', 'dua puluh satu ribu ringgit'),
        ('RM201,000', 'dua ratus satu ribu ringgit', 'dua ratus satu ribu ringgit'),
        ('RM101,000', 'satu ratus satu ribu ringgit', 'seratus satu ribu ringgit'),
        ('RM11,000', 'sebelas ribu ringgit', 'sebelas ribu ringgit'),
        ('RM120', 'one hundred twenty ringgit', 'one hundred twenty ringgit'),
    ])
    def test_reply(self, text, reply, want):
        assert malay_hundreds(text, reply) == want

    def test_phrase_the_text_wrote_is_kept(self):
        assert malay_hundreds('satu ratus 20 orang', 'satu ratus dua puluh orang') \
            == 'satu ratus dua puluh orang'


class TestMailWords:
    @pytest.mark.parametrize('text,want', [
        ('Tidak mengapa. tiran benar at yopmail dot com. Betul ke?',
         'Tidak mengapa. tiran benar at yopmel dot com. Betul ke?'),
        ('ali at hotmail dot com dot my', 'ali at hotmel dot com dot my'),
        ('ali at Yopmail titik com', 'ali at Yopmel titik com'),
        ('ali@yopmail.com', 'ali@yopmel.com'),
        ('ali at mailinator dot com', 'ali at melinator dot com'),
        ('ali at YOPMAIL dot com', 'ali at YOPMEL dot com'),
        # Malay spelled "at"
        ('d tiga empat tiga lapan kosong lima pada yopmail dot com. Betul?',
         'd tiga empat tiga lapan kosong lima pada yopmel dot com. Betul?'),
        ('ali di hotmail titik com', 'ali di hotmel titik com'),
        # known to the LM, or not glued: as written
        ('ali at gmail dot com', 'ali at gmail dot com'),
        ('ali at email dot com', 'ali at email dot com'),
        ('ali at mail dot yahoo dot com', 'ali at mail dot yahoo dot com'),
        # not an email domain
        ('saya hantar ke hotmail tadi', 'saya hantar ke hotmail tadi'),
        ('look at hotmail. Then sign in', 'look at hotmail. Then sign in'),
        ('yopmail at site dot com', 'yopmail at site dot com'),
        ('that hotmail dot com', 'that hotmail dot com'),
        ('padahotmail dot com', 'padahotmail dot com'),
        ('saya pada hotmail tadi', 'saya pada hotmail tadi'),
    ])
    def test_reply(self, text, want):
        assert mail_words(text) == want


class TestNormalizerMode:
    def test_values(self):
        assert NormalizerMode('rule') == NormalizerMode.rule
        assert NormalizerMode('llm') == NormalizerMode.llm
        with pytest.raises(ValueError):
            NormalizerMode('nope')


class TestNeedsNormalization:
    """The LLM-skip gate: plain text must be recognised as plain (the whole TTFB win),
    and anything the prompt asks the model to rewrite must still reach the LLM."""

    PLAIN = [
        'Hello there, how can I help you today?',
        'Thank you for calling our support line. Please tell me your account number and I will look into the issue right away.',
        'Selamat pagi, apa yang boleh saya bantu encik hari ini?',
        'Okay encik, your booking is confirmed, nanti saya hantar details melalui email ya.',
        "Don't worry, we'll sort it out — it's no problem at all.",
        'She said “thank you” and left… (quietly).',
        'Sekiranya anda mempunyai sebarang pertanyaan lanjut, jangan teragak-agak untuk menghubungi kami.',
        '你好，请问有什么可以帮您？',
        '总共是一千二百五十令吉。',
        'I am here; the meeting is at half past three in the afternoon.',
        'Encik Ahmad akan datang esok pagi.',
        'The first, second and third items are ready.',
    ]
    NEEDS = [
        'Your total is RM1,250.50.',                 # digits
        'Dr. Lim will call you.',                    # abbreviation (dotted)
        'Dr Lim will call you.',                     # abbreviation (bare)
        'Scicom Sdn Bhd',                            # abbreviation (bare)
        'Please bring your IC and the OTP.',         # acronyms
        'Up by twenty percent, or 20%.',             # digit + symbol
        'up by twenty %',                            # symbol alone
        'email me at husein@site.com',               # symbol + dotted token
        'visit site.com for details',                # dotted token
        'see e.g. the manual',                       # dotted token
        '**bold** markdown and snake_case',          # markdown / underscore
        'It is 3pm.',                                # digit
        'Meet at St. John street',                   # dotted-only abbreviation
        'Terima kasih Pn. Siti',                     # Malay honorific, dotted
        'a few km away',                             # unit
        '总共RM50',                                   # digits inside Chinese
        '我叫侯赛因，身份证号是960314875079',
        'Q&A session',                               # symbol
        'price is 50/50',                            # digits + slash
    ]

    @pytest.mark.parametrize('text', PLAIN)
    def test_plain_text_skips_llm(self, text):
        assert not needs_normalization(text)

    @pytest.mark.parametrize('text', NEEDS)
    def test_normalizable_text_calls_llm(self, text):
        assert needs_normalization(text)

    def test_common_words_are_not_abbreviations(self):
        # "no", "am", "sat", "min" are only abbreviations when written with a period
        assert not needs_normalization('no problem, I am sure she sat there for a minute')
        assert needs_normalization('ref no. 4471'.replace('4471', 'four'))   # "no." still gates

    def test_empty(self):
        assert not needs_normalization('')
        assert not needs_normalization(None)


class TestKeepProductNames:
    """A product name spelled out before the LLM must come back unchanged: gemma knows the
    promotion and wrote "five point oh" back as Malay "lima perpuluhan sifar"."""

    TEXT = 'Kekal bersama Unifi dengan promosi StaywithUnifi 5.0! Langganan dari RM129 sebulan.'

    def spans(self):
        from app.rules import product_names
        spans = []
        out = product_names(self.TEXT, spans)
        return out, spans

    def test_spans_are_what_was_written(self):
        out, spans = self.spans()
        assert spans == ['Stay with Unifi', 'Unifi five point oh']
        assert all(sp in out for sp in spans)

    def test_reply_that_changes_the_name_is_rejected(self):
        out, spans = self.spans()
        reply = ('Kekal bersama Unifi dengan promosi Stay with Unifi lima perpuluhan sifar, '
                 'Langganan dari seratus dua puluh sembilan ringgit sebulan.')
        assert rejected_reason(out, reply, keep=spans) == 'spelled-out phrase changed'

    def test_reply_that_keeps_it_passes(self):
        out, spans = self.spans()
        reply = ('Kekal bersama Unifi dengan promosi Stay with Unifi five point oh, '
                 'Langganan dari seratus dua puluh sembilan ringgit sebulan.')
        assert rejected_reason(out, reply, keep=spans) is None

    def test_caller_prompt_decides(self):
        out, spans = self.spans()
        assert rejected_reason(out, 'promosi Stay with Unifi lima titik kosong, seratus dua puluh sembilan ringgit',
                               prompt='baca nama produk dalam bahasa Melayu', keep=spans) is None


class TestRejectedReason:
    @pytest.mark.parametrize('text,reply,reason', [
        ('750412105678. Did we get that right?', 'Did we get that right?', 'number dropped'),
        ('Is 0123456789 the number on file with us?', 'Is the number on file with us?', 'number dropped'),
        ('Your IC is 750412-10-5678, correct?', 'Your IC is seven five zero four one two, correct?',
         'read-back digits dropped'),
        ('6 4 0 9 2 3 0 8 7 7 6 6.', 'six four zero nine two three.', 'read-back digits dropped'),
        ('D 3 6 7 6 T S 1 5.', 'D 3 6 7 6 T S one five.', 'digits left'),
        ('The 1Gbps plan.', 'The 1Gbps plan.', 'digits left'),
        ('8 8 0 1 2 3 4 5', '八 八 零 一 二 三 四 五', 'Chinese numerals in non-Chinese text'),
        ('A 5 5 2 0 9 1 3.', 'A 五 五 二 零 九 一 三.', 'Chinese numerals in non-Chinese text'),
    ])
    def test_rejected(self, text, reply, reason):
        assert rejected_reason(text, reply) == reason

    @pytest.mark.parametrize('text,reply', [
        ('750412105678. Did we get that right?',
         'seven five zero four one two one zero five six seven eight. Did we get that right?'),
        ('Call 012-3456789 now.', 'Call zero one double two three four five six seven eight nine now.'),
        ('Your IC is 750412-10-5678.', 'Your IC is seven five zero four one two, one zero, five six seven eight.'),
        ('Pay before 2026-07-25.', 'Pay before the twenty-fifth of July twenty twenty-six.'),
        ('About 1000000 people.', 'About one million people.'),
        ('You are 3rd in the queue.', 'You are third in the queue.'),
        ('Anda yang ke-3.', 'Anda yang ketiga.'),
        ('总共RM1,250.50', '总共一千二百五十令吉五十仙'),
        ('D 3 7 1 5 H O S P 1 6。', 'D 三 七 一 五 H O S P 一 六。'),     # Chinese punctuation: a Chinese call
        ('Kami buka 24/7.', 'Kami buka dua puluh empat jam tujuh hari seminggu.'),
        ('2024ல் நடந்தது', '2024ல் நடந்தது'),                                  # Tamil is not checked
        ('No number here.', 'No number here.'),
    ])
    def test_accepted(self, text, reply):
        assert rejected_reason(text, reply) is None


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

    # "pada pukul pagi 8:00" was spoken "lapan sifar sifar" (prod, 2026-11-16)
    @pytest.mark.parametrize('text,want', [
        ('pada pukul pagi 8:00', 'pada pukul pagi lapan'),
        ('pada pukul pagi 8.00', 'pada pukul pagi lapan'),
        ('mesyuarat 13:30', 'mesyuarat pukul satu tiga puluh minit petang'),
        ('pukul 20:00', 'pukul lapan malam'),
        ('at 20:00', 'at eight p m'),
        ('14:00开会', '下午两点开会'),
    ])
    def test_clock_time(self, text, want):
        assert run(llm_normalize(text)).rstrip('.') == want

    # "100 ringgit, betul?" was spoken "one hundred ringgit" (2026-11-24)
    @pytest.mark.parametrize('text,want', [
        ('Okay, 100 ringgit, betul?', 'Okay, seratus ringgit, betul?'),
        ('100 ringgit, boleh?', 'seratus ringgit, boleh?'),
        ('100 ringgit, right?', 'one hundred ringgit, right?'),
    ])
    def test_number_language(self, text, want):
        assert run(llm_normalize(text)) == want

    def test_passthrough(self):
        out = run(llm_normalize('apa khabar semua?'))
        assert out.rstrip('.') == 'apa khabar semua?'

    # A bare amount carries no language, and without context the LLM read "RM50" as
    # "五十令吉" (2026-10-07, gemma-4-31b). The prompt is what tells it.
    def test_prompt_sets_language_malay(self):
        out = run(llm_normalize('RM50', prompt='normalize this in Malay'))
        assert 'lima puluh ringgit' in out.lower()

    def test_prompt_sets_language_english(self):
        out = run(llm_normalize('RM50', prompt='normalize this in English'))
        assert 'fifty ringgit' in out.lower()

    # The prompt carries what a number *is*, not only its language: with "normalize it in
    # english" alone, 50450 came back "fifty thousand four hundred fifty".
    def test_prompt_postcode_chinese(self):
        out = run(llm_normalize('08000', prompt='this is a postcode, normalize it in chinese'))
        assert out.rstrip('.。') == '零八零零零'

    def test_prompt_postcode_english_digit_by_digit(self):
        out = run(llm_normalize('50450', prompt='this is a postcode, normalize it in english'))
        assert out.lower().rstrip('.') == 'five zero four five zero'

    # Spelling digits one by one in a language other than the sentence's. With the prompt in
    # the system prompt only, both of these were ignored ("一千五百", Malay digits).
    def test_prompt_spell_one_by_one(self):
        out = run(llm_normalize('1500', prompt='please spell one by one in en'))
        assert out.lower().rstrip('.') == 'one five zero zero'

    def test_prompt_language_overrides_sentence_language(self):
        out = run(llm_normalize('nombor akaun saya 1234234', prompt='please spell one by one in zh'))
        assert out.rstrip('.。') == 'nombor akaun saya 一二三四二三四'

    def test_prompt_does_not_leak(self):
        out = run(llm_normalize('Order 12 sudah sampai', prompt='read the numbers in English'))
        assert '[' not in out and 'read the numbers' not in out.lower()
        assert 'twelve' in out.lower()

    def test_prompt_leaves_plain_text_alone(self):
        # LLM_NORMALIZER_SKIP_PLAIN still applies with a prompt; this is why that is safe
        out = run(llm_normalize('apa khabar semua?', prompt='normalize this in English'))
        assert out.rstrip('.') == 'apa khabar semua?'
