"""
Extensive unit tests for the text normalizer.
Tests the app.normalizer module directly (no GPU required) and
app.rules functions (sanitize_markdown, expand_contractions, split_alpha_num).

Run with: python -m pytest tests/test_normalizer.py -v
"""

import sys
import os
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore', category=FutureWarning)

import pytest
from app.rules import (
    sanitize_markdown,
    expand_contractions,
    split_alpha_num,
    before_replace_mapping,
    replace_mapping,
)
from app.normalizer import load

normalizer = load()


def norm(text, english=False, **kwargs):
    """Helper that calls normalizer.normalize with the same flags as normalize_malaysian_text."""
    defaults = dict(
        normalize_hingga=False,
        normalize_text=False,
        normalize_word_rules=False,
        normalize_time=True,
        normalize_cardinal=False,
        normalize_ordinal=False,
        normalize_url=True,
        normalize_email=True,
        normalize_in_english=english,
    )
    defaults.update(kwargs)
    return normalizer.normalize(text, **defaults)['normalize']


def full_norm(text, english=False, **kwargs):
    """Helper that calls normalizer.normalize with all default flags (full normalization)."""
    return normalizer.normalize(text, normalize_in_english=english, **kwargs)['normalize']


# ---------------------------------------------------------------------------
# Email normalization
# ---------------------------------------------------------------------------
class TestEmailNormalization:
    def test_email_malay(self):
        out = norm('email saya ialah husein.zol05@gmail.com')
        assert out == 'email saya ialah HUSEIN dot ZOL kosong lima di GMAIL dot COM'

    def test_email_english(self):
        out = norm('email saya ialah husein.zol05@gmail.com', english=True)
        assert out == 'email saya ialah HUSEIN dot ZOL zero five at GMAIL dot COM'

    def test_email_standalone(self):
        out = norm('husein.zol05@gmail.com')
        assert out == 'HUSEIN dot ZOL kosong lima di GMAIL dot COM'

    def test_email_standalone_english(self):
        out = norm('husein.zol05@gmail.com', english=True)
        assert out == 'HUSEIN dot ZOL zero five at GMAIL dot COM'

    def test_email_in_sentence(self):
        out = norm('sila hantar ke test@mail.com sekarang')
        assert 'TEST di MAIL dot COM' in out
        assert 'sila' in out
        assert 'sekarang' in out

    def test_email_in_sentence_english(self):
        out = norm('please send to test@mail.com now', english=True)
        assert 'TEST at MAIL dot COM' in out

    def test_email_with_numbers(self):
        out = norm('ahmad123@yahoo.com')
        assert 'di' in out
        assert 'dot' in out

    def test_email_with_numbers_english(self):
        out = norm('ahmad123@yahoo.com', english=True)
        assert 'at' in out
        assert 'dot' in out


# ---------------------------------------------------------------------------
# URL normalization
# ---------------------------------------------------------------------------
class TestURLNormalization:
    def test_url_https(self):
        out = norm('https://huseinhouse.com')
        assert out == 'HTTPS huseinhouse dot com'

    def test_url_https_english(self):
        out = norm('https://huseinhouse.com', english=True)
        assert out == 'HTTPS huseinhouse dot com'

    def test_url_www(self):
        out = norm('https://www.google.com')
        assert 'HTTPS' in out
        assert 'WWW' in out
        assert 'dot' in out
        assert 'google' in out

    def test_url_in_sentence_malay(self):
        out = norm('lawati https://www.google.com untuk maklumat')
        assert 'lawati' in out
        assert 'HTTPS' in out
        assert 'dot' in out
        assert 'maklumat' in out

    def test_url_in_sentence_english(self):
        out = norm('visit https://www.google.com for info', english=True)
        assert 'visit' in out
        assert 'HTTPS' in out
        assert 'dot' in out


# ---------------------------------------------------------------------------
# Phone number normalization
# ---------------------------------------------------------------------------
class TestPhoneNormalization:
    def test_phone_malay(self):
        out = norm('012-1234567')
        assert out == 'kosong satu dua, satu dua tiga empat lima enam tujuh'

    def test_phone_english(self):
        out = norm('012-1234567', english=True)
        assert out == 'zero one two, one two three four five six seven'

    def test_phone_in_sentence_malay(self):
        out = norm('hubungi 012-1234567')
        assert out == 'hubungi kosong satu dua, satu dua tiga empat lima enam tujuh'

    def test_phone_in_sentence_english(self):
        out = norm('hubungi 012-1234567', english=True)
        assert out == 'hubungi zero one two, one two three four five six seven'

    def test_phone_another_format(self):
        out = norm('03-12345678')
        assert 'kosong tiga' in out

    def test_phone_another_format_english(self):
        out = norm('03-12345678', english=True)
        assert 'zero three' in out


# ---------------------------------------------------------------------------
# IC number normalization
# ---------------------------------------------------------------------------
class TestICNormalization:
    def test_ic_malay(self):
        out = norm('880101-14-5678')
        assert out == 'lapan lapan kosong satu kosong satu satu empat lima enam tujuh lapan'

    def test_ic_english(self):
        out = norm('880101-14-5678', english=True)
        assert out == 'eight eight zero one zero one one four five six seven eight'

    def test_ic_in_sentence_malay(self):
        out = norm('IC saya 880101-14-5678')
        assert 'IC saya' in out
        assert 'lapan lapan kosong satu kosong satu' in out

    def test_ic_in_sentence_english(self):
        out = norm('IC saya 880101-14-5678', english=True)
        assert 'IC saya' in out
        assert 'eight eight zero one zero one' in out

    def test_ic_another_number(self):
        out = norm('911111-01-1111')
        assert 'sembilan' in out
        assert 'kosong' in out


# ---------------------------------------------------------------------------
# Money normalization
# ---------------------------------------------------------------------------
class TestMoneyNormalization:
    def test_rm_malay(self):
        out = norm('RM10.5')
        assert out == 'sepuluh ringgit lima puluh sen'

    def test_rm_english(self):
        out = norm('RM10.5', english=True)
        assert out == 'ten ringgit fifty sen'

    def test_dollar_malay(self):
        out = norm('$10.5')
        assert out == 'sepuluh dollar lima puluh cent'

    def test_dollar_english(self):
        out = norm('$10.5', english=True)
        assert out == 'ten dollar fifty cent'

    def test_dollar_k_malay(self):
        out = norm('$10.4K')
        assert 'ribu' in out
        assert 'dollar' in out

    def test_rm_in_sentence(self):
        out = norm('harga RM10.5')
        assert out == 'harga sepuluh ringgit lima puluh sen'

    def test_rm_in_sentence_english(self):
        out = norm('harga RM10.5', english=True)
        assert out == 'harga ten ringgit fifty sen'

    def test_sen_malay(self):
        out = norm('1015 sen')
        assert out == 'sepuluh ringgit lima belas sen'


# ---------------------------------------------------------------------------
# Time normalization
# ---------------------------------------------------------------------------
class TestTimeNormalization:
    def test_time_pm_malay(self):
        out = norm('2:01pm')
        assert out == 'pukul dua satu minit pagi'

    def test_time_pm_english(self):
        out = norm('2:01pm', english=True)
        assert out == 'at two one minute am'

    def test_time_am_malay(self):
        out = norm('2AM')
        assert out == 'pukul dua pagi'

    def test_time_am_english(self):
        out = norm('2AM', english=True)
        assert out == 'at two am'

    def test_time_morning(self):
        out = norm('10:30am')
        assert 'pukul' in out
        assert 'sepuluh' in out
        assert 'tiga puluh' in out

    def test_time_in_sentence_malay(self):
        out = norm('mesyuarat pukul 2:01pm')
        assert 'mesyuarat' in out
        assert 'dua' in out
        assert 'satu minit' in out


# ---------------------------------------------------------------------------
# Percentage normalization
# ---------------------------------------------------------------------------
class TestPercentNormalization:
    def test_percent_malay(self):
        out = norm('61.2%')
        assert out == 'enam puluh satu perpuluhan dua peratus'

    def test_percent_english(self):
        out = norm('61.2%', english=True)
        assert out == 'sixty one point two percent'

    def test_percent_in_sentence_malay(self):
        out = norm('kadar 61.2%')
        assert out == 'kadar enam puluh satu perpuluhan dua peratus'

    def test_percent_in_sentence_english(self):
        out = norm('kadar 61.2%', english=True)
        assert out == 'kadar sixty one point two percent'


# ---------------------------------------------------------------------------
# Unit normalization (temperature, weight, distance, volume, data)
# ---------------------------------------------------------------------------
class TestUnitNormalization:
    def test_celsius_malay(self):
        out = norm('36.5c')
        assert out == 'tiga puluh enam perpuluhan lima celsius'

    def test_celsius_english(self):
        out = norm('36.5c', english=True)
        assert out == 'thirty six point five celcius'

    def test_kg_malay(self):
        out = norm('61.2kg')
        assert out == 'enam puluh satu perpuluhan dua kilogram'

    def test_kg_english(self):
        out = norm('61.2kg', english=True)
        assert out == 'sixty one point two kilogram'

    def test_km_malay(self):
        out = norm('10km')
        assert out == 'sepuluh kilometer'

    def test_km_english(self):
        out = norm('10km', english=True)
        assert out == 'ten kilometer'

    def test_liter_malay(self):
        out = norm('2.5l')
        assert out == 'dua perpuluhan lima liter'

    def test_liter_english(self):
        out = norm('2.5l', english=True)
        assert out == 'two point five liter'

    def test_mb_malay(self):
        out = norm('500mb')
        assert out == 'lima ratus megabit'

    def test_mb_english(self):
        out = norm('500mb', english=True)
        assert out == 'five hundred megabits'

    def test_celsius_in_sentence(self):
        out = norm('suhu 36.5c')
        assert out == 'suhu tiga puluh enam perpuluhan lima celsius'

    def test_kg_in_sentence(self):
        out = norm('berat 61.2kg')
        assert out == 'berat enam puluh satu perpuluhan dua kilogram'

    def test_km_in_sentence(self):
        out = norm('jarak 10km')
        assert out == 'jarak sepuluh kilometer'


# ---------------------------------------------------------------------------
# Date normalization
# ---------------------------------------------------------------------------
class TestDateNormalization:
    def test_date_malay(self):
        out = norm('1/11/2021')
        assert 'Januari' in out or 'januari' in out

    def test_date_english(self):
        out = norm('1/11/2021', english=True)
        assert 'January' in out or 'january' in out


# ---------------------------------------------------------------------------
# Number with zero prefix
# ---------------------------------------------------------------------------
class TestZeroPrefixNumber:
    def test_zero_prefix_malay(self):
        out = norm('01234')
        assert out == 'kosong satu dua tiga empat'

    def test_zero_prefix_english(self):
        out = norm('01234', english=True)
        assert out == 'zero one two three four'


# ---------------------------------------------------------------------------
# Passport-like pattern
# ---------------------------------------------------------------------------
class TestPassportPattern:
    def test_passport_malay(self):
        out = norm('A12345678')
        assert 'A' in out
        assert 'satu' in out or 'dua' in out

    def test_passport_english(self):
        out = norm('A12345678', english=True)
        assert 'A' in out
        assert 'one' in out or 'two' in out


# ---------------------------------------------------------------------------
# Full normalization (all flags enabled) - cardinal, ordinal, fraction, etc.
# ---------------------------------------------------------------------------
class TestFullNormalizationCardinal:
    def test_cardinal_malay(self):
        out = full_norm('123')
        assert out == 'seratus dua puluh tiga'

    def test_cardinal_english(self):
        out = full_norm('123', english=True)
        assert out == 'one hundred and twenty three'

    def test_cardinal_large(self):
        out = full_norm('1000000')
        assert 'juta' in out or 'satu' in out

    def test_cardinal_in_sentence(self):
        out = full_norm('ada 123 orang')
        assert 'seratus dua puluh tiga' in out
        assert 'ada' in out
        assert 'orang' in out


class TestFullNormalizationOrdinal:
    def test_ordinal_malay(self):
        out = full_norm('ke-21')
        assert out == 'kedua puluh satu'

    def test_ordinal_english(self):
        out = full_norm('ke-21', english=True)
        assert out == 'twenty-first'

    def test_ordinal_roman(self):
        out = full_norm('ke-XXI')
        assert out == 'kedua puluh satu'

    def test_ordinal_in_sentence(self):
        out = full_norm('ini adalah ke-3 kali')
        assert 'ketiga' in out


class TestFullNormalizationFraction:
    def test_fraction_malay(self):
        out = full_norm('10/4')
        assert out == 'sepuluh per empat'

    def test_fraction_in_sentence(self):
        out = full_norm('kadar 10/4 adalah betul')
        assert 'sepuluh per empat' in out


class TestFullNormalizationMultiplier:
    def test_x_kali_malay(self):
        out = full_norm('10x')
        assert out == 'sepuluh kali'

    def test_x_kali_english(self):
        out = full_norm('10x', english=True)
        assert out == 'ten times'

    def test_x_kali_in_sentence(self):
        out = full_norm('saya sokong 10x')
        assert 'sepuluh kali' in out


class TestFullNormalizationHingga:
    def test_hingga_malay(self):
        out = full_norm('2011 - 2019')
        assert 'hingga' in out

    def test_hingga_english(self):
        out = full_norm('2011 - 2019', english=True)
        assert 'until' in out


class TestFullNormalizationElongated:
    def test_elongated(self):
        out = full_norm('saayyyyaa ttttaaak ssssukaaa')
        assert 'tak' in out or 'suka' in out

    def test_xkisah(self):
        out = full_norm('xkisah')
        assert 'tak' in out


# ---------------------------------------------------------------------------
# Contractions (from rules.py)
# ---------------------------------------------------------------------------
class TestContractions:
    def test_cant(self):
        assert expand_contractions("I can't do it") == 'I cannot do it'

    def test_shes(self):
        assert expand_contractions("She's going") == 'She is going'

    def test_theyve(self):
        assert expand_contractions("They've been here") == 'They have been here'

    def test_were(self):
        assert expand_contractions("We're leaving") == 'We are leaving'

    def test_wont(self):
        assert expand_contractions("Won't you come?") == 'Will not you come?'

    def test_its(self):
        assert expand_contractions("It's raining") == 'It is raining'

    def test_isnt(self):
        assert expand_contractions("That isn't right") == 'That is not right'

    def test_dont(self):
        assert expand_contractions("I don't know") == 'I do not know'

    def test_couldnt(self):
        assert expand_contractions("I couldn't see") == 'I could not see'

    def test_wouldnt(self):
        assert expand_contractions("I wouldn't do that") == 'I would not do that'

    def test_havent(self):
        assert expand_contractions("I haven't eaten") == 'I have not eaten'

    def test_lets(self):
        assert expand_contractions("Let's go") == 'Let us go'

    def test_preserves_case(self):
        assert expand_contractions("Can't do it") == 'Cannot do it'

    def test_no_contractions(self):
        text = 'no contractions here'
        assert expand_contractions(text) == text

    def test_multiple_contractions(self):
        out = expand_contractions("I can't and won't do it")
        assert out == 'I cannot and will not do it'


# ---------------------------------------------------------------------------
# Alpha-num splitting (from rules.py)
# ---------------------------------------------------------------------------
class TestSplitAlphaNum:
    def test_alpha_then_num(self):
        assert split_alpha_num('abc123') == 'abc 123'

    def test_num_then_alpha(self):
        assert split_alpha_num('123abc') == '123 abc'

    def test_rm_number(self):
        assert split_alpha_num('RM500') == 'RM 500'

    def test_passport(self):
        assert split_alpha_num('A12345') == 'A 12345'

    def test_pure_alpha(self):
        assert split_alpha_num('hello') == 'hello'

    def test_pure_num(self):
        assert split_alpha_num('12345') == '12345'

    def test_mixed_multiple(self):
        assert split_alpha_num('abc123def') == 'abc 123 def'


# ---------------------------------------------------------------------------
# Before/after replace mappings
# ---------------------------------------------------------------------------
class TestReplaceMappings:
    def test_before_replace_en_dash(self):
        text = 'hello–world'
        for k, v in before_replace_mapping.items():
            text = text.replace(k, v)
        assert '–' not in text

    def test_before_replace_semicolon(self):
        text = 'hello; world'
        for k, v in before_replace_mapping.items():
            text = text.replace(k, v)
        assert text == 'hello, world'

    def test_before_replace_exclamation(self):
        text = 'hello! world'
        for k, v in before_replace_mapping.items():
            text = text.replace(k, v)
        assert text == 'hello, world'

    def test_replace_mapping_dash(self):
        text = 'hello-world'
        for k, v in replace_mapping.items():
            text = text.replace(k, v)
        assert text == 'hello world'

    def test_replace_mapping_brackets(self):
        text = 'hello (world) [test]'
        for k, v in replace_mapping.items():
            text = text.replace(k, v)
        assert '(' not in text
        assert ')' not in text
        assert '[' not in text
        assert ']' not in text


# ---------------------------------------------------------------------------
# Sanitize markdown (comprehensive)
# ---------------------------------------------------------------------------
class TestSanitizeMarkdownComprehensive:
    def test_bold_star(self):
        assert sanitize_markdown('**tebal**') == 'tebal'

    def test_bold_underscore(self):
        assert sanitize_markdown('__tebal__') == 'tebal'

    def test_italic_star(self):
        assert sanitize_markdown('*condong*') == 'condong'

    def test_italic_underscore(self):
        assert sanitize_markdown('_condong_') == 'condong'

    def test_bold_italic(self):
        assert sanitize_markdown('***tebal condong***') == 'tebal condong'

    def test_strikethrough(self):
        assert sanitize_markdown('~~potong~~') == 'potong'

    def test_heading(self):
        assert sanitize_markdown('# Tajuk').strip() == 'Tajuk'

    def test_link(self):
        assert sanitize_markdown('[Google](https://google.com)') == 'Google'

    def test_image(self):
        assert sanitize_markdown('![gambar](img.png)') == 'gambar'

    def test_inline_code(self):
        assert sanitize_markdown('guna `print()`') == 'guna print()'

    def test_fenced_code(self):
        text = 'sebelum\n```\nkod\n```\nselepas'
        out = sanitize_markdown(text)
        assert '```' not in out
        assert 'kod' not in out
        assert 'sebelum' in out
        assert 'selepas' in out

    def test_blockquote(self):
        assert sanitize_markdown('> petikan').strip() == 'petikan'

    def test_unordered_list(self):
        out = sanitize_markdown('- satu\n- dua')
        assert out.strip() == 'satu\ndua'

    def test_ordered_list(self):
        out = sanitize_markdown('1. satu\n2. dua')
        assert out.strip() == 'satu\ndua'

    def test_horizontal_rule(self):
        assert sanitize_markdown('---').strip() == ''

    def test_html_simple(self):
        assert sanitize_markdown('<b>tebal</b>') == 'tebal'

    def test_html_with_attrs(self):
        out = sanitize_markdown('<a href="url">pautan</a>')
        assert out == 'pautan'

    def test_html_self_closing(self):
        out = sanitize_markdown('baris<br/>seterusnya')
        assert '<br/>' not in out

    def test_preserves_plain_text(self):
        text = 'Ini teks biasa tanpa markdown.'
        assert sanitize_markdown(text) == text

    def test_preserves_ic(self):
        assert '880101-14-5678' in sanitize_markdown('IC **880101-14-5678**')

    def test_preserves_phone(self):
        assert '012-345-6789' in sanitize_markdown('telefon **012-345-6789**')

    def test_preserves_email(self):
        assert 'test@mail.com' in sanitize_markdown('email test@mail.com')

    def test_preserves_url(self):
        assert 'https://google.com' in sanitize_markdown('lawati https://google.com')

    def test_preserves_underscore_in_var(self):
        assert 'my_variable' in sanitize_markdown('set my_variable to 5')

    def test_preserves_unicode(self):
        text = '**சுப்பிரமணியம்** வணக்கம்'
        out = sanitize_markdown(text)
        assert 'சுப்பிரமணியம்' in out
        assert '**' not in out


# ---------------------------------------------------------------------------
# Combined pipeline tests (sanitize + normalizer)
# ---------------------------------------------------------------------------
class TestCombinedPipeline:
    """Test sanitize_markdown followed by normalizer, simulating normalize_malaysian_text."""

    def _pipeline(self, text, english=False):
        s = sanitize_markdown(text)
        s = s.replace('\n', ' ')
        import re
        s = re.sub(r'[ ]+', ' ', s).strip()
        return norm(s, english=english)

    def test_bold_email(self):
        out = self._pipeline('email saya **husein.zol05@gmail.com**')
        assert '**' not in out
        assert 'HUSEIN dot ZOL kosong lima di GMAIL dot COM' in out

    def test_bold_email_english(self):
        out = self._pipeline('email saya **husein.zol05@gmail.com**', english=True)
        assert '**' not in out
        assert 'HUSEIN dot ZOL zero five at GMAIL dot COM' in out

    def test_italic_phone(self):
        out = self._pipeline('hubungi *012-1234567*')
        assert '*' not in out
        assert 'kosong satu dua' in out

    def test_link_with_url(self):
        out = self._pipeline('lawati [Google](https://www.google.com)')
        assert 'https://' not in out
        assert 'Google' in out

    def test_heading_with_money(self):
        out = self._pipeline('# Harga RM10.5')
        assert '#' not in out
        assert 'sepuluh ringgit lima puluh sen' in out

    def test_markdown_ic(self):
        out = self._pipeline('IC beliau **880101-14-5678** sudah disahkan')
        assert '**' not in out
        assert 'lapan lapan kosong satu kosong satu' in out
        assert 'sudah disahkan' in out

    def test_markdown_ic_english(self):
        out = self._pipeline('IC beliau **880101-14-5678** sudah disahkan', english=True)
        assert '**' not in out
        assert 'eight eight zero one zero one' in out

    def test_code_block_plus_email(self):
        text = 'email: ```skip this``` contact@test.com'
        out = self._pipeline(text)
        assert '```' not in out
        assert 'skip' not in out
        assert 'CONTACT di TEST dot COM' in out

    def test_html_plus_phone(self):
        out = self._pipeline('<b>hubungi</b> 012-1234567')
        assert '<b>' not in out
        assert 'hubungi' in out
        assert 'kosong satu dua' in out

    def test_full_markdown_document(self):
        text = """# Pengumuman

**Tarikh:** 15 April 2026

- Telefon: 012-1234567
- Email: info@syarikat.com
- IC: 880101-14-5678

> Harga bermula dari RM10.5

Layari [laman web](https://syarikat.com.my)."""

        out = self._pipeline(text)
        assert '**' not in out
        assert '# ' not in out
        assert '~~' not in out
        assert '> ' not in out
        assert '[' not in out
        assert 'Pengumuman' in out
        assert 'kosong satu dua' in out
        assert 'di SYARIKAT dot COM' in out
        assert 'lapan lapan kosong satu' in out
        assert 'sepuluh ringgit lima puluh sen' in out

    def test_full_markdown_document_english(self):
        text = """# Announcement

**Date:** 15 April 2026

- Phone: 012-1234567
- Email: info@syarikat.com
- IC: 880101-14-5678

> Price starts from RM10.5

Visit [website](https://syarikat.com.my)."""

        out = self._pipeline(text, english=True)
        assert '**' not in out
        assert 'Announcement' in out
        assert 'zero one two' in out
        assert 'at SYARIKAT dot COM' in out
        assert 'eight eight zero one' in out
        assert 'ten ringgit fifty sen' in out


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_empty_string(self):
        out = norm('')
        assert out == ''

    def test_whitespace_only(self):
        out = norm('   ')
        assert out.strip() == ''

    def test_only_punctuation(self):
        out = norm('...')
        assert '.' in out

    def test_multiple_emails(self):
        out = norm('a@b.com dan c@d.com')
        assert out.count('dot') == 2
        assert out.count('di') == 2

    def test_multiple_phones(self):
        out = norm('012-1234567 atau 013-7654321')
        assert out.count('kosong') >= 2

    def test_mixed_ic_and_email(self):
        out = norm('IC beliau 880101-14-5678 email test@mail.com')
        assert 'lapan lapan kosong satu' in out
        assert 'TEST di MAIL dot COM' in out

    def test_long_text(self):
        text = 'Sila hubungi kami di 012-1234567 atau email ke info@test.com untuk harga RM10.5 setiap unit 61.2kg'
        out = norm(text)
        assert 'kosong satu dua' in out
        assert 'di' in out
        assert 'dot' in out
        assert 'sepuluh ringgit' in out
        assert 'kilogram' in out

    def test_unicode_tamil_passthrough(self):
        # Tokenizer may split Tamil chars; verify content is preserved not dropped
        out = norm('வணக்கம்')
        assert 'வண' in out

    def test_chinese_passthrough(self):
        out = norm('你好世界')
        assert '你好' in out
