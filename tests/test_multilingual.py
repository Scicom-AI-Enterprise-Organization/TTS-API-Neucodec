"""
Tests for multilingual text normalization.
Verifies that non-Latin scripts (Chinese, Korean, Tamil, Arabic, Japanese,
Thai, Hindi/Devanagari) pass through untouched while ASCII content alongside
them is still normalized correctly.

The normalize_malaysian_text pipeline splits text into non-ASCII and ASCII
segments. Non-ASCII segments pass through unchanged; ASCII segments go
through the normalizer.

Run with: python -m pytest tests/test_multilingual.py -v
"""

import sys
import os
import re
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore', category=FutureWarning)

import pytest
from app.rules import (
    sanitize_markdown, before_replace_mapping, expand_contractions, pattern_range,
    protect_phone_numbers, restore_phone_numbers,
)
from app.normalizer import load, to_cardinal
from app.normalizer.chinese import normalize_chinese, is_chinese_dominant, CJK_RE, KANA_RE

normalizer = load()


def pipeline(s, english=False):
    """Mirrors the ASCII/CJK segment-routing steps of normalize_malaysian_text
    (normalize_malaysian=True), minus the fasttext language-detection call
    (`english` stands in for that) and the split_alpha_num digit-splitting pass
    (covered separately by TestSplitAlphaNumExhaustive in test_malaysian_rules.py).
    Must be kept in sync with app/main.py or bugs like the pattern_range/phone-number
    clash can hide behind a passing test."""
    s = sanitize_markdown(s)
    s = s.replace('\n', ' ')
    s = re.sub(r'[ ]+', ' ', s).strip()

    for k, v in before_replace_mapping.items():
        s = s.replace(k, v)

    chinese_dominant = is_chinese_dominant(s)

    def replace_range(match):
        num1 = int(match.group(1))
        num2 = int(match.group(2))
        words1 = to_cardinal(num1, english=english)
        words2 = to_cardinal(num2, english=english)
        phrase = match.group(3)
        to = 'to' if english else 'hingga'
        return f'{words1} {to} {words2} {phrase}'

    s = expand_contractions(s)
    s, protected_phones = protect_phone_numbers(s)
    s = pattern_range.sub(replace_range, s)
    s = restore_phone_numbers(s, protected_phones)

    segments = re.split(r'(\S*[^\x00-\x7F]\S*)', s)
    normalized_parts = []
    for seg in segments:
        if re.search(r'[^\x00-\x7F]', seg):
            if CJK_RE.search(seg) and not KANA_RE.search(seg):
                normalized_parts.append(normalize_chinese(seg))
            else:
                normalized_parts.append(seg)
        elif seg.strip():
            leading_ws = seg[:len(seg) - len(seg.lstrip())]
            trailing_ws = seg[len(seg.rstrip()):]
            if chinese_dominant:
                converted = normalize_chinese(seg.strip())
                if converted != seg.strip():
                    normalized_parts.append(leading_ws + converted + trailing_ws)
                    continue
            result = normalizer.normalize(
                seg.strip(),
                normalize_hingga=False, normalize_text=False,
                normalize_word_rules=False, normalize_time=True,
                normalize_cardinal=False, normalize_ordinal=False,
                normalize_url=True, normalize_email=True,
                normalize_in_english=english,
            )
            normalized_parts.append(leading_ws + result['normalize'] + trailing_ws)
        else:
            normalized_parts.append(seg)
    return ''.join(normalized_parts)


# ---------------------------------------------------------------------------
# Chinese (Simplified & Traditional)
# ---------------------------------------------------------------------------
class TestChinese:
    def test_pure_chinese(self):
        assert pipeline('你好世界') == '你好世界'

    def test_chinese_traditional(self):
        assert pipeline('你好世界') == '你好世界'

    def test_chinese_with_money(self):
        out = pipeline('价格是 RM500')
        assert '价格是' in out
        assert '五百令吉' in out

    def test_chinese_with_email(self):
        out = pipeline('请联系 test@mail.com')
        assert '请联系' in out
        assert 'TEST at MAIL dot COM' in out

    def test_chinese_with_phone(self):
        out = pipeline('电话 012-1234567')
        assert '电话' in out
        assert '零一二一二三四五六七' in out

    def test_chinese_with_temperature(self):
        out = pipeline('今天温度是 36.5c')
        assert '今天温度是' in out
        assert '摄氏三十六点五度' in out

    def test_chinese_with_url(self):
        out = pipeline('访问 https://www.google.com')
        assert '访问' in out
        assert 'HTTPS' in out
        assert 'WWW' in out
        assert 'dot' in out

    def test_chinese_mixed_english(self):
        out = pipeline('中文 mixed with English')
        assert '中文' in out
        assert 'mixed' in out
        assert 'English' in out

    def test_chinese_markdown_bold(self):
        out = pipeline('**你好** 世界')
        assert '**' not in out
        assert '你好' in out
        assert '世界' in out

    def test_chinese_number_attached(self):
        # Numbers glued directly to Hanzi (the normal way to write Chinese) are now verbalized too.
        out = pipeline('价格是RM500')
        assert '价格是五百令吉' in out

    def test_chinese_number_spaced(self):
        # Separated by a space, still Chinese wording since the sentence is Chinese-dominant.
        out = pipeline('价格是 RM500')
        assert '令吉' in out

    def test_chinese_sentence_complex(self):
        out = pipeline('中文 email test@mail.com 电话 012-1234567')
        assert '中文' in out
        assert 'TEST at MAIL dot COM' in out
        assert '电话' in out
        assert '零一二一二三四五六七' in out


# ---------------------------------------------------------------------------
# Korean
# ---------------------------------------------------------------------------
class TestKorean:
    def test_pure_korean(self):
        assert pipeline('안녕하세요') == '안녕하세요'

    def test_korean_with_money(self):
        out = pipeline('가격은 RM100 입니다')
        assert '가격은' in out
        assert 'seratus ringgit' in out
        assert '입니다' in out

    def test_korean_with_phone(self):
        out = pipeline('전화번호는 012-1234567')
        assert '전화번호는' in out
        assert 'kosong satu dua' in out

    def test_korean_with_email(self):
        out = pipeline('이메일 user@test.com 입니다')
        assert '이메일' in out
        assert 'di TEST dot COM' in out
        assert '입니다' in out

    def test_korean_markdown(self):
        out = pipeline('**안녕** 하세요 RM100')
        assert '**' not in out
        assert '안녕' in out
        assert 'seratus ringgit' in out

    def test_korean_with_percent(self):
        out = pipeline('할인 50% 입니다')
        assert '할인' in out
        assert 'peratus' in out
        assert '입니다' in out

    def test_korean_long_sentence(self):
        out = pipeline('안녕하세요 저는 한국인입니다')
        assert '안녕하세요' in out
        assert '한국인입니다' in out


# ---------------------------------------------------------------------------
# Tamil
# ---------------------------------------------------------------------------
class TestTamil:
    def test_pure_tamil(self):
        out = pipeline('வணக்கம்')
        assert 'வணக்கம்' in out

    def test_tamil_with_money(self):
        out = pipeline('விலை RM50 ஆகும்')
        assert 'விலை' in out
        assert 'lima puluh ringgit' in out
        assert 'ஆகும்' in out

    def test_tamil_with_phone(self):
        out = pipeline('தொலைபேசி 012-3456789')
        assert 'தொலைபேசி' in out
        assert 'kosong satu dua' in out

    def test_tamil_with_email(self):
        out = pipeline('மின்னஞ்சல் test@mail.com')
        assert 'மின்னஞ்சல்' in out
        assert 'di MAIL dot COM' in out

    def test_tamil_markdown(self):
        out = pipeline('**வணக்கம்** நண்பர்களே')
        assert '**' not in out
        assert 'வணக்கம்' in out
        assert 'நண்பர்களே' in out

    def test_tamil_mixed_sentence(self):
        out = pipeline('வணக்கம் hello world')
        assert 'வணக்கம்' in out
        assert 'hello' in out

    def test_tamil_with_ic(self):
        out = pipeline('அடையாள அட்டை 880101-14-5678')
        assert 'அடையாள' in out
        assert 'lapan lapan kosong satu' in out


# ---------------------------------------------------------------------------
# Arabic
# ---------------------------------------------------------------------------
class TestArabic:
    def test_pure_arabic(self):
        assert pipeline('مرحبا') == 'مرحبا'

    def test_arabic_greeting(self):
        assert pipeline('السلام عليكم') == 'السلام عليكم'

    def test_arabic_with_money(self):
        out = pipeline('السعر RM200')
        assert 'السعر' in out
        assert 'dua ratus ringgit' in out

    def test_arabic_with_date(self):
        out = pipeline('العربية RM100.50 和 25/12/2025')
        assert 'العربية' in out
        assert 'seratus ringgit' in out
        assert 'Disember' in out or 'disember' in out

    def test_arabic_with_email(self):
        out = pipeline('البريد user@test.com')
        assert 'البريد' in out
        assert 'di TEST dot COM' in out

    def test_arabic_markdown(self):
        out = pipeline('**مرحبا** بالعالم')
        assert '**' not in out
        assert 'مرحبا' in out
        assert 'بالعالم' in out

    def test_arabic_rtl_mixed(self):
        out = pipeline('مرحبا hello مرحبا')
        assert 'مرحبا' in out
        assert 'hello' in out


# ---------------------------------------------------------------------------
# Japanese
# ---------------------------------------------------------------------------
class TestJapanese:
    def test_pure_hiragana(self):
        assert pipeline('こんにちは') == 'こんにちは'

    def test_pure_katakana(self):
        assert pipeline('コンピューター') == 'コンピューター'

    def test_kanji(self):
        assert pipeline('日本語') == '日本語'

    def test_japanese_with_money(self):
        out = pipeline('価格は RM300 です')
        assert '価格は' in out
        assert 'tiga ratus ringgit' in out
        assert 'です' in out

    def test_japanese_with_time(self):
        out = pipeline('こんにちは pukul 3:45PM harga RM99.99')
        assert 'こんにちは' in out
        assert 'tiga' in out
        assert 'empat puluh lima' in out
        assert 'sembilan puluh sembilan ringgit' in out

    def test_japanese_markdown_heading(self):
        out = pipeline('# こんにちは\n価格は RM300')
        assert '#' not in out
        assert 'こんにちは' in out
        assert 'tiga ratus ringgit' in out

    def test_japanese_mixed_script(self):
        out = pipeline('日本語テスト test email info@test.com')
        assert '日本語テスト' in out
        assert 'di TEST dot COM' in out


# ---------------------------------------------------------------------------
# Thai
# ---------------------------------------------------------------------------
class TestThai:
    def test_pure_thai(self):
        out = pipeline('สวัสดี')
        assert 'สวัสดี' in out

    def test_thai_greeting(self):
        out = pipeline('สวัสดีครับ')
        assert 'สวัสดีครับ' in out

    def test_thai_with_money(self):
        out = pipeline('ราคา RM150')
        assert 'ราคา' in out
        assert 'seratus lima puluh ringgit' in out

    def test_thai_with_phone(self):
        out = pipeline('โทร 012-1234567')
        assert 'โทร' in out
        assert 'kosong satu dua' in out

    def test_thai_with_email(self):
        out = pipeline('อีเมล user@test.com')
        assert 'อีเมล' in out
        assert 'di TEST dot COM' in out

    def test_thai_markdown(self):
        out = pipeline('**สวัสดี** ครับ')
        assert '**' not in out
        assert 'สวัสดี' in out

    def test_thai_with_percent(self):
        out = pipeline('ลด 30% วันนี้')
        assert 'ลด' in out
        assert 'peratus' in out
        assert 'วันนี้' in out


# ---------------------------------------------------------------------------
# Hindi / Devanagari
# ---------------------------------------------------------------------------
class TestHindi:
    def test_pure_hindi(self):
        out = pipeline('नमस्ते')
        assert 'नमस्ते' in out

    def test_hindi_sentence(self):
        out = pipeline('नमस्ते दुनिया')
        assert 'नमस्ते' in out
        assert 'दुनिया' in out

    def test_hindi_with_money(self):
        out = pipeline('कीमत RM250')
        assert 'कीमत' in out
        assert 'dua ratus lima puluh ringgit' in out

    def test_hindi_with_phone(self):
        out = pipeline('फोन 012-3456789')
        assert 'फोन' in out
        assert 'kosong satu dua' in out

    def test_hindi_with_email(self):
        out = pipeline('ईमेल user@test.com')
        assert 'ईमेल' in out
        assert 'di TEST dot COM' in out

    def test_hindi_markdown(self):
        out = pipeline('**नमस्ते** दोस्तों')
        assert '**' not in out
        assert 'नमस्ते' in out
        assert 'दोस्तों' in out


# ---------------------------------------------------------------------------
# Mixed multilingual
# ---------------------------------------------------------------------------
class TestMixedMultilingual:
    def test_all_scripts_together(self):
        out = pipeline('你好 hello สวัสดี 안녕 வணக்கம்')
        assert '你好' in out
        assert 'hello' in out
        assert 'สวัสดี' in out
        assert '안녕' in out
        assert 'வணக்கம்' in out

    def test_arabic_chinese_with_data(self):
        out = pipeline('العربية RM100.50 和 25/12/2025')
        assert 'العربية' in out
        assert 'ringgit' in out
        assert '和' in out
        assert 'Disember' in out or 'disember' in out

    def test_japanese_malay_english(self):
        out = pipeline('こんにちは pukul 3:45PM harga RM99.99')
        assert 'こんにちは' in out
        assert 'ringgit' in out

    def test_korean_malay_email(self):
        out = pipeline('안녕하세요 email info@company.com 감사합니다')
        assert '안녕하세요' in out
        assert 'di COMPANY dot COM' in out
        assert '감사합니다' in out

    def test_tamil_malay_ic(self):
        out = pipeline('அடையாள எண் 880101-14-5678 நன்றி')
        assert 'அடையாள' in out
        assert 'lapan lapan kosong satu' in out
        assert 'நன்றி' in out

    def test_thai_malay_money(self):
        out = pipeline('ราคา RM500 บาท สวัสดี')
        assert 'ราคา' in out
        assert 'lima ratus ringgit' in out
        assert 'สวัสดี' in out

    def test_hindi_malay_phone(self):
        out = pipeline('फोन नंबर 012-1234567 धन्यवाद')
        assert 'फोन' in out
        assert 'नंबर' in out
        assert 'kosong satu dua' in out
        assert 'धन्यवाद' in out

    def test_multilingual_markdown_complex(self):
        text = """# 你好世界

**안녕하세요** வணக்கம் مرحبا

- Email: test@mail.com
- Phone: 012-1234567
- Price: RM99.99

> こんにちは สวัสดี नमस्ते"""

        out = pipeline(text)

        # All markdown removed
        assert '**' not in out
        assert '#' not in out
        assert '>' not in out

        # All scripts preserved
        assert '你好世界' in out
        assert '안녕하세요' in out
        assert 'வணக்கம்' in out
        assert 'مرحبا' in out
        assert 'こんにちは' in out
        assert 'สวัสดี' in out
        assert 'नमस्ते' in out

        # Normalizable content processed
        assert 'di MAIL dot COM' in out
        assert 'kosong satu dua' in out
        assert 'ringgit' in out


# ---------------------------------------------------------------------------
# Non-ASCII attached to ASCII (edge cases)
# ---------------------------------------------------------------------------
class TestNonASCIIAttached:
    """When non-ASCII chars are attached to ASCII without spaces, the whole token is treated as
    non-ASCII. For most scripts that means it passes through raw; Chinese is the exception, since
    normalize_chinese() verbalizes numbers/currency embedded in it instead of leaving them raw."""

    def test_chinese_rm_attached(self):
        # No space between CJK and RM500 => whole token is non-ASCII, but Chinese still verbalizes it
        out = pipeline('价格是RM500')
        assert '价格是五百令吉' in out

    def test_chinese_number_attached(self):
        out = pipeline('温度是36.5c')
        assert '温度是摄氏三十六点五度' in out

    def test_chinese_rm_spaced(self):
        # With space => still Chinese wording (sentence is Chinese-dominant)
        out = pipeline('价格是 RM500')
        assert '令吉' in out

    def test_korean_number_attached(self):
        out = pipeline('가격RM100')
        assert '가격RM100' in out

    def test_korean_number_spaced(self):
        out = pipeline('가격 RM100')
        assert 'ringgit' in out

    def test_arabic_number_attached(self):
        out = pipeline('السعرRM200')
        assert 'السعرRM200' in out

    def test_arabic_number_spaced(self):
        out = pipeline('السعر RM200')
        assert 'ringgit' in out


# ---------------------------------------------------------------------------
# Emoji
# ---------------------------------------------------------------------------
class TestEmoji:
    def test_emoji_preserved(self):
        out = pipeline('hello 🔥 world')
        assert 'hello' in out
        assert 'world' in out

    def test_emoji_with_money(self):
        out = pipeline('harga RM50 🎉')
        assert 'lima puluh ringgit' in out

    def test_emoji_only(self):
        out = pipeline('🔥🎉💯')
        assert len(out) > 0


# ---------------------------------------------------------------------------
# Malay text (the primary use case)
# ---------------------------------------------------------------------------
class TestMalayPrimary:
    def test_basic_malay(self):
        out = pipeline('Selamat pagi semua')
        assert 'Selamat pagi semua' == out

    def test_malay_with_all_types(self):
        out = pipeline('Bayar RM50 sebelum 25/12/2025 hubungi 012-1234567 email info@test.com suhu 36.5c')
        assert 'ringgit' in out
        assert 'Disember' in out or 'disember' in out
        assert 'kosong satu dua' in out
        assert 'di TEST dot COM' in out
        assert 'celsius' in out

    def test_phone_number_not_swallowed_by_range(self):
        # Regression: pattern_range used to greedily match "012-1234567 email info"
        # as a numeric range (12 to 1234567), garbling the phone number instead of
        # reading it digit-by-digit.
        out = pipeline('hubungi 012-1234567 email info@test.com')
        assert 'kosong satu dua, satu dua tiga empat lima enam tujuh' in out
        assert 'hingga' not in out

    def test_malay_english_mixed(self):
        out = pipeline('Meeting at 3:00PM please contact 012-1234567')
        assert 'Meeting' in out
        assert 'tiga' in out or 'three' in out

    def test_malay_with_markdown(self):
        out = pipeline('**Penting:** Bayar RM100 sebelum [tarikh akhir](https://example.com)')
        assert '**' not in out
        assert '[' not in out
        assert 'Penting' in out
        assert 'seratus ringgit' in out
        assert 'tarikh akhir' in out

    def test_malay_html(self):
        out = pipeline('<b>Sila</b> bayar <span style="color:red">RM50</span>')
        assert '<' not in out
        assert 'Sila' in out
        assert 'lima puluh ringgit' in out
