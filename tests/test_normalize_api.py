"""
Integration tests for /v1/audio/normalize endpoint.
Tests sanitize_markdown + normalize_malaysian_text together via the API.

Requires the full app to be importable (fasttext, malaya normalizer, etc.).
Run with: python -m pytest tests/test_normalize_api.py -v

If models are not available, these tests will be skipped.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest

try:
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    APP_AVAILABLE = True
except Exception as e:
    APP_AVAILABLE = False
    SKIP_REASON = str(e)

skipif_no_app = pytest.mark.skipif(not APP_AVAILABLE, reason=f"App not loadable: {SKIP_REASON if not APP_AVAILABLE else ''}")


@skipif_no_app
class TestNormalizeEndpointBasic:
    """Tests without normalize_malaysian (just sanitize_markdown + cleanup)."""

    def test_plain_text(self):
        r = client.post('/v1/audio/normalize', json={'input': 'Hello world'})
        assert r.status_code == 200
        assert r.json()['output'] == 'Hello world.'

    def test_trailing_period_not_doubled(self):
        r = client.post('/v1/audio/normalize', json={'input': 'Hello world.'})
        assert r.json()['output'] == 'Hello world.'

    def test_markdown_bold(self):
        r = client.post('/v1/audio/normalize', json={'input': '**Selamat pagi** semua'})
        assert r.json()['output'] == 'Selamat pagi semua.'
        assert '**' not in r.json()['output']

    def test_markdown_italic(self):
        r = client.post('/v1/audio/normalize', json={'input': '*italic* text'})
        assert r.json()['output'] == 'italic text.'

    def test_markdown_link(self):
        r = client.post('/v1/audio/normalize', json={'input': 'Visit [Google](https://google.com)'})
        out = r.json()['output']
        assert 'Google' in out
        assert 'https://' not in out

    def test_markdown_heading(self):
        r = client.post('/v1/audio/normalize', json={'input': '# Title here'})
        assert r.json()['output'] == 'Title here.'

    def test_html_tags(self):
        r = client.post('/v1/audio/normalize', json={'input': '<b>bold</b> and <i>italic</i>'})
        out = r.json()['output']
        assert '<b>' not in out
        assert '<i>' not in out
        assert 'bold' in out
        assert 'italic' in out

    def test_code_block_removed(self):
        r = client.post('/v1/audio/normalize', json={'input': 'before ```python\nprint("hi")\n``` after'})
        out = r.json()['output']
        assert '```' not in out
        assert 'print' not in out
        assert 'before' in out
        assert 'after' in out

    def test_inline_code(self):
        r = client.post('/v1/audio/normalize', json={'input': 'use `print()` function'})
        assert r.json()['output'] == 'use print() function.'

    def test_multiple_spaces_normalized(self):
        r = client.post('/v1/audio/normalize', json={'input': 'too   many    spaces'})
        assert r.json()['output'] == 'too many spaces.'

    def test_newlines_collapsed(self):
        r = client.post('/v1/audio/normalize', json={'input': 'line one\n\nline two'})
        out = r.json()['output']
        assert 'line one' in out
        assert 'line two' in out

    def test_ic_number_preserved(self):
        r = client.post('/v1/audio/normalize', json={'input': 'IC saya 880101-14-5678'})
        assert '880101-14-5678' in r.json()['output']

    def test_phone_number_preserved(self):
        r = client.post('/v1/audio/normalize', json={'input': 'Call 012-345-6789'})
        assert '012-345-6789' in r.json()['output']

    def test_website_url_preserved(self):
        r = client.post('/v1/audio/normalize', json={'input': 'Visit https://www.google.com'})
        assert 'https://www.google.com' in r.json()['output']

    def test_email_preserved(self):
        r = client.post('/v1/audio/normalize', json={'input': 'Email to ahmad@example.com'})
        assert 'ahmad@example.com' in r.json()['output']

    def test_empty_string(self):
        r = client.post('/v1/audio/normalize', json={'input': ''})
        assert r.status_code == 200

    def test_complex_llm_response(self):
        text = """**Key Points:**
1. The price is **RM500,000**
2. Located in *Bangsar*
3. Contact: [Agent Ali](tel:+60123456789)

> Note: This is subject to change.

For more details, visit [our website](https://property.com.my)."""

        r = client.post('/v1/audio/normalize', json={'input': text})
        out = r.json()['output']
        assert '**' not in out
        assert '>' not in out
        assert '[' not in out
        assert 'Key Points:' in out
        assert 'RM500,000' in out
        assert 'Bangsar' in out
        assert 'Agent Ali' in out
        assert 'our website' in out


@skipif_no_app
class TestNormalizeEndpointMalaysian:
    """Tests with normalize_malaysian=True."""

    def test_basic_malay(self):
        r = client.post('/v1/audio/normalize', json={
            'input': 'Selamat pagi',
            'normalize_malaysian': True,
        })
        assert r.status_code == 200
        assert len(r.json()['output']) > 0

    def test_markdown_stripped_before_normalize(self):
        r = client.post('/v1/audio/normalize', json={
            'input': '**Selamat pagi** semua',
            'normalize_malaysian': True,
        })
        out = r.json()['output']
        assert '**' not in out
        assert 'Selamat' in out or 'selamat' in out.lower()

    def test_ic_number_normalized(self):
        r = client.post('/v1/audio/normalize', json={
            'input': 'IC saya 880101-14-5678',
            'normalize_malaysian': True,
        })
        out = r.json()['output']
        # IC should be expanded to digits
        assert '880101-14-5678' not in out
        assert len(out) > 10

    def test_phone_number_normalized(self):
        r = client.post('/v1/audio/normalize', json={
            'input': 'Hubungi 012-345-6789',
            'normalize_malaysian': True,
        })
        out = r.json()['output']
        assert len(out) > 0

    def test_url_normalized(self):
        r = client.post('/v1/audio/normalize', json={
            'input': 'Lawati https://www.google.com',
            'normalize_malaysian': True,
        })
        out = r.json()['output']
        # URL should be expanded (dots become "dot", etc.)
        assert 'dot' in out.lower() or 'google' in out.lower()

    def test_email_normalized(self):
        r = client.post('/v1/audio/normalize', json={
            'input': 'Email ahmad@example.com',
            'normalize_malaysian': True,
        })
        out = r.json()['output']
        assert 'dot' in out.lower() or 'di' in out.lower() or 'at' in out.lower()

    def test_contraction_expanded(self):
        r = client.post('/v1/audio/normalize', json={
            'input': "I can't do it",
            'normalize_malaysian': True,
        })
        out = r.json()['output']
        assert "can't" not in out.lower()
        assert 'cannot' in out.lower()

    def test_money_normalized(self):
        r = client.post('/v1/audio/normalize', json={
            'input': 'Harga rumah RM500000',
            'normalize_malaysian': True,
        })
        out = r.json()['output']
        assert len(out) > 0

    def test_html_stripped_with_malaysian(self):
        r = client.post('/v1/audio/normalize', json={
            'input': '<b>Selamat pagi</b> semua',
            'normalize_malaysian': True,
        })
        out = r.json()['output']
        assert '<b>' not in out
        assert '</b>' not in out

    def test_full_markdown_document_malaysian(self):
        text = """# Pengumuman Penting

**Tarikh:** 15 April 2026

Sila hubungi kami di:
- Telefon: 012-345-6789
- Email: info@syarikat.com
- Laman web: [Syarikat Kami](https://syarikat.com.my)

> Harga bermula dari RM100,000

IC pemohon: **880101-14-5678**

~~Tawaran lama~~ sudah tamat."""

        r = client.post('/v1/audio/normalize', json={
            'input': text,
            'normalize_malaysian': True,
        })
        out = r.json()['output']

        # All markdown stripped
        assert '**' not in out
        assert '~~' not in out
        assert '# ' not in out
        assert '[' not in out
        assert '](' not in out
        assert '<' not in out

        # Content preserved (some may be normalized)
        assert 'pengumuman' in out.lower() or 'Pengumuman' in out
        assert 'tamat' in out.lower()

    def test_english_text_with_malaysian_flag(self):
        r = client.post('/v1/audio/normalize', json={
            'input': 'The meeting is at 3:00 PM',
            'normalize_malaysian': True,
        })
        out = r.json()['output']
        assert r.status_code == 200
        assert len(out) > 0

    def test_mixed_language(self):
        r = client.post('/v1/audio/normalize', json={
            'input': 'Saya pergi ke meeting pukul 3 petang',
            'normalize_malaysian': True,
        })
        out = r.json()['output']
        assert r.status_code == 200
        assert len(out) > 0

    def test_unicode_tamil_preserved(self):
        r = client.post('/v1/audio/normalize', json={
            'input': 'சுப்பிரமணியம் வணக்கம்',
            'normalize_malaysian': True,
        })
        out = r.json()['output']
        # Non-ASCII segments should pass through untouched
        assert 'சுப்பிரமணியம்' in out

    def test_range_normalized(self):
        r = client.post('/v1/audio/normalize', json={
            'input': 'Harga 100-200 ringgit',
            'normalize_malaysian': True,
        })
        out = r.json()['output']
        # Range should be expanded (e.g., "seratus hingga dua ratus ringgit")
        assert '100-200' not in out
        assert len(out) > 10
