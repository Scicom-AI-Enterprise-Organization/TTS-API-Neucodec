"""
Unit tests for sanitize_markdown in app/rules.py.
Run with: python -m pytest tests/test_sanitize_markdown.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.rules import sanitize_markdown


class TestBoldItalicStrikethrough:
    def test_bold_double_star(self):
        assert sanitize_markdown('**hello**') == 'hello'

    def test_bold_double_underscore(self):
        assert sanitize_markdown('__hello__') == 'hello'

    def test_italic_single_star(self):
        assert sanitize_markdown('*hello*') == 'hello'

    def test_italic_single_underscore(self):
        assert sanitize_markdown('_hello_') == 'hello'

    def test_bold_italic_triple_star(self):
        assert sanitize_markdown('***hello***') == 'hello'

    def test_bold_italic_triple_underscore(self):
        assert sanitize_markdown('___hello___') == 'hello'

    def test_strikethrough(self):
        assert sanitize_markdown('~~deleted~~') == 'deleted'

    def test_bold_in_sentence(self):
        assert sanitize_markdown('ini **penting** sekali') == 'ini penting sekali'

    def test_italic_in_sentence(self):
        assert sanitize_markdown('ini *penting* sekali') == 'ini penting sekali'

    def test_mixed_bold_italic(self):
        result = sanitize_markdown('**bold** and *italic* and ~~strike~~')
        assert result == 'bold and italic and strike'

    def test_nested_bold_in_italic(self):
        result = sanitize_markdown('***bold and italic***')
        assert result == 'bold and italic'


class TestHeadings:
    def test_h1(self):
        assert sanitize_markdown('# Heading').strip() == 'Heading'

    def test_h2(self):
        assert sanitize_markdown('## Heading').strip() == 'Heading'

    def test_h3(self):
        assert sanitize_markdown('### Heading').strip() == 'Heading'

    def test_h6(self):
        assert sanitize_markdown('###### Heading').strip() == 'Heading'

    def test_heading_multiline(self):
        text = '# Title\nSome text\n## Subtitle'
        result = sanitize_markdown(text)
        assert '# ' not in result
        assert '## ' not in result
        assert 'Title' in result
        assert 'Subtitle' in result


class TestLinks:
    def test_link_keeps_text(self):
        assert sanitize_markdown('[click here](https://example.com)') == 'click here'

    def test_link_in_sentence(self):
        result = sanitize_markdown('Sila layari [laman web](https://example.com) untuk maklumat.')
        assert result == 'Sila layari laman web untuk maklumat.'

    def test_image_keeps_alt(self):
        assert sanitize_markdown('![alt text](image.png)') == 'alt text'

    def test_image_empty_alt(self):
        assert sanitize_markdown('![](image.png)') == ''

    def test_link_with_title(self):
        result = sanitize_markdown('[text](https://example.com "title")')
        assert 'https://' not in result
        assert 'text' in result


class TestCodeBlocks:
    def test_inline_code(self):
        assert sanitize_markdown('use `print()` function') == 'use print() function'

    def test_fenced_code_block(self):
        text = 'before\n```python\nprint("hello")\n```\nafter'
        result = sanitize_markdown(text)
        assert '```' not in result
        assert 'print' not in result
        assert 'before' in result
        assert 'after' in result

    def test_fenced_code_block_no_lang(self):
        text = 'before\n```\nsome code\n```\nafter'
        result = sanitize_markdown(text)
        assert '```' not in result
        assert 'some code' not in result


class TestBlockquotes:
    def test_blockquote(self):
        assert sanitize_markdown('> This is a quote').strip() == 'This is a quote'

    def test_multiline_blockquote(self):
        text = '> Line one\n> Line two'
        result = sanitize_markdown(text)
        assert '>' not in result
        assert 'Line one' in result
        assert 'Line two' in result


class TestLists:
    def test_unordered_dash(self):
        text = '- item one\n- item two'
        result = sanitize_markdown(text)
        assert result.strip() == 'item one\nitem two'

    def test_unordered_star(self):
        text = '* item one\n* item two'
        result = sanitize_markdown(text)
        assert result.strip() == 'item one\nitem two'

    def test_unordered_plus(self):
        text = '+ item one\n+ item two'
        result = sanitize_markdown(text)
        assert result.strip() == 'item one\nitem two'

    def test_ordered_list(self):
        text = '1. first\n2. second\n3. third'
        result = sanitize_markdown(text)
        assert result.strip() == 'first\nsecond\nthird'

    def test_ordered_list_double_digit(self):
        text = '10. tenth item'
        result = sanitize_markdown(text)
        assert result.strip() == 'tenth item'


class TestHorizontalRule:
    def test_dashes(self):
        assert sanitize_markdown('---').strip() == ''

    def test_stars(self):
        assert sanitize_markdown('***').strip() == ''

    def test_underscores(self):
        assert sanitize_markdown('___').strip() == ''

    def test_long_dashes(self):
        assert sanitize_markdown('----------').strip() == ''


class TestHTMLTags:
    def test_simple_tag(self):
        assert sanitize_markdown('<b>bold</b>') == 'bold'

    def test_paragraph_tag(self):
        assert sanitize_markdown('<p>paragraph</p>') == 'paragraph'

    def test_div_tag(self):
        assert sanitize_markdown('<div>content</div>') == 'content'

    def test_br_tag(self):
        result = sanitize_markdown('line one<br>line two')
        assert '<br>' not in result
        assert 'line one' in result
        assert 'line two' in result

    def test_self_closing_tag(self):
        result = sanitize_markdown('text<br/>more')
        assert '<br/>' not in result

    def test_tag_with_attributes(self):
        result = sanitize_markdown('<a href="https://example.com">link</a>')
        assert '<a' not in result
        assert '</a>' not in result
        assert 'link' in result

    def test_nested_html(self):
        result = sanitize_markdown('<div><p>nested</p></div>')
        assert '<' not in result
        assert 'nested' in result

    def test_span_with_style(self):
        result = sanitize_markdown('<span style="color:red">red text</span>')
        assert '<span' not in result
        assert 'red text' in result


class TestWebsiteURLs:
    def test_raw_url_preserved(self):
        # Raw URLs without markdown syntax should be preserved (normalizer handles them)
        text = 'visit https://www.google.com for search'
        result = sanitize_markdown(text)
        assert 'https://www.google.com' in result

    def test_markdown_link_url_removed(self):
        text = 'visit [Google](https://www.google.com) for search'
        result = sanitize_markdown(text)
        assert result == 'visit Google for search'
        assert 'https://' not in result

    def test_multiple_urls(self):
        text = '[A](https://a.com) and [B](https://b.com)'
        result = sanitize_markdown(text)
        assert result == 'A and B'


class TestICNumber:
    def test_ic_number_preserved(self):
        text = 'IC saya 880101-14-5678'
        result = sanitize_markdown(text)
        assert '880101-14-5678' in result

    def test_ic_in_markdown(self):
        text = 'IC beliau **880101-14-5678** sudah disahkan'
        result = sanitize_markdown(text)
        assert '880101-14-5678' in result
        assert '**' not in result


class TestPhoneNumber:
    def test_phone_number_preserved(self):
        text = 'Hubungi 012-345-6789'
        result = sanitize_markdown(text)
        assert '012-345-6789' in result

    def test_phone_with_country_code(self):
        text = 'Call +60 12-345 6789'
        result = sanitize_markdown(text)
        assert '+60' in result

    def test_phone_in_bold(self):
        text = 'Hubungi **012-345-6789** sekarang'
        result = sanitize_markdown(text)
        assert '012-345-6789' in result
        assert '**' not in result


class TestComplexMarkdown:
    def test_full_document(self):
        text = """# Selamat Pagi

**Nama saya** Ahmad. Saya tinggal di _Kuala Lumpur_.

## Maklumat Peribadi

- IC: 880101-14-5678
- Telefon: 012-345-6789
- Email: ahmad@example.com

> Ini adalah petikan penting

Sila layari [laman web kami](https://example.com) untuk maklumat lanjut.

~~Ini sudah dibatalkan~~

1. Perkara pertama
2. Perkara kedua

---

Terima kasih."""

        result = sanitize_markdown(text)

        # No markdown syntax remaining
        assert '**' not in result
        assert '__' not in result
        assert '# ' not in result
        assert '## ' not in result
        assert '~~' not in result
        assert '[' not in result
        assert '](' not in result
        assert '---' not in result
        assert '> ' not in result

        # Content preserved
        assert 'Selamat Pagi' in result
        assert 'Nama saya' in result
        assert 'Ahmad' in result
        assert 'Kuala Lumpur' in result
        assert '880101-14-5678' in result
        assert '012-345-6789' in result
        assert 'ahmad@example.com' in result
        assert 'petikan penting' in result
        assert 'laman web kami' in result
        assert 'sudah dibatalkan' in result
        assert 'Perkara pertama' in result
        assert 'Terima kasih.' in result

    def test_llm_style_response(self):
        """Test typical LLM markdown response that might be fed to TTS."""
        text = """Here's what I found:

**Key Points:**
1. The price is **RM500,000**
2. Located in *Bangsar*
3. Contact: [Agent Ali](tel:+60123456789)

> Note: This is subject to change.

For more details, visit [our website](https://property.com.my)."""

        result = sanitize_markdown(text)

        assert '**' not in result
        assert '*' not in result
        assert '>' not in result
        assert '[' not in result
        assert 'Key Points:' in result
        assert 'RM500,000' in result
        assert 'Bangsar' in result
        assert 'Agent Ali' in result
        assert 'our website' in result

    def test_code_with_explanation(self):
        text = "Gunakan fungsi `calculate()` untuk mengira. Contoh:\n```\nresult = calculate(10)\n```\nIni akan return 10."
        result = sanitize_markdown(text)
        assert '`' not in result
        assert '```' not in result
        assert 'calculate()' in result
        assert 'result = calculate' not in result  # code block removed
        assert 'Ini akan return 10.' in result


class TestEdgeCases:
    def test_empty_string(self):
        assert sanitize_markdown('') == ''

    def test_no_markdown(self):
        text = 'Ini adalah teks biasa tanpa markdown.'
        assert sanitize_markdown(text) == text

    def test_numbers_with_dots(self):
        # Should not strip "3.14" thinking it's a list
        text = 'Pi is approximately 3.14'
        result = sanitize_markdown(text)
        assert '3.14' in result

    def test_underscore_in_variable(self):
        # my_variable should not be treated as italic
        text = 'set my_variable to 5'
        result = sanitize_markdown(text)
        assert 'my_variable' in result

    def test_asterisk_in_math(self):
        # "2 * 3" should not be treated as italic
        text = '2 * 3 = 6'
        result = sanitize_markdown(text)
        assert '2' in result
        assert '6' in result

    def test_single_star_no_closing(self):
        text = 'rating: 4.5 * out of 5'
        result = sanitize_markdown(text)
        assert '4.5' in result

    def test_table_pipes(self):
        text = '| Name | Age |\n|------|-----|\n| Ali  | 30  |'
        result = sanitize_markdown(text)
        assert 'Ali' in result

    def test_escape_chars(self):
        text = r'This is \*not bold\*'
        result = sanitize_markdown(text)
        # Escaped markdown should ideally not be processed,
        # but at minimum the content should be preserved
        assert 'not bold' in result

    def test_multiple_newlines_preserved(self):
        text = 'Line 1\n\nLine 2'
        result = sanitize_markdown(text)
        assert 'Line 1' in result
        assert 'Line 2' in result

    def test_unicode_preserved(self):
        text = '**சுப்பிரமணியம்** வணக்கம்'
        result = sanitize_markdown(text)
        assert 'சுப்பிரமணியம்' in result
        assert 'வணக்கம்' in result
        assert '**' not in result

    def test_malay_text_with_markdown(self):
        text = '**Selamat pagi** semua! _Apa khabar_?'
        result = sanitize_markdown(text)
        assert result == 'Selamat pagi semua! Apa khabar?'

    def test_chinese_text_with_markdown(self):
        text = '**你好** 世界'
        result = sanitize_markdown(text)
        assert '你好' in result
        assert '**' not in result
