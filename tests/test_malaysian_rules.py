"""
Stress tests for Malaysian text normalization rules.
Tests the normalizer with realistic Malaysian text patterns:
money (RM/USD), IC numbers, phone numbers, email, URL, time, dates,
units, percentages, years, ordinals, fractions, ranges, pronunciation
replacements, contractions, and complex multi-type sentences.

Run with: python -m pytest tests/test_malaysian_rules.py -v
"""

import sys
import os
import re
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore', category=FutureWarning)

import pytest
from app.rules import (
    sanitize_markdown,
    expand_contractions,
    split_alpha_num,
    apply_pronunciation_replacements,
    before_replace_mapping,
    replace_mapping,
    pattern_range,
)
from app.normalizer import load, to_cardinal

normalizer = load()


def norm(text, english=False):
    """Normalize with the same flags as normalize_malaysian_text."""
    r = normalizer.normalize(
        text,
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
    return r['normalize']


def full(text, english=False, **kw):
    """Full normalization with all flags enabled."""
    return normalizer.normalize(text, normalize_in_english=english, **kw)['normalize']


# ---------------------------------------------------------------------------
# Money (RM) - exhaustive
# ---------------------------------------------------------------------------
class TestMoneyRM:
    def test_rm_whole_number(self):
        assert norm('RM10') == 'sepuluh ringgit'

    def test_rm_with_sen(self):
        assert norm('RM10.5') == 'sepuluh ringgit lima puluh sen'

    def test_rm_with_sen_99(self):
        assert norm('RM99.99') == 'sembilan puluh sembilan ringgit sembilan puluh sembilan sen'

    def test_rm_zero_sen(self):
        out = norm('RM50.00')
        assert 'lima puluh ringgit' in out

    def test_rm_only_sen(self):
        out = norm('RM0.50')
        assert 'lima puluh sen' in out

    def test_rm_in_sentence(self):
        out = norm('Bayar RM50.00 sekarang')
        assert 'lima puluh ringgit' in out
        assert 'Bayar' in out

    def test_rm_english(self):
        assert norm('RM10.5', english=True) == 'ten ringgit fifty sen'

    def test_rm_99_english(self):
        out = norm('RM99.99', english=True)
        assert 'ninety nine ringgit' in out

    def test_sen_standalone(self):
        assert norm('1015 sen') == 'sepuluh ringgit lima belas sen'


class TestMoneyUSD:
    def test_dollar_with_cents(self):
        assert norm('$10.5') == 'sepuluh dollar lima puluh cent'

    def test_dollar_english(self):
        assert norm('$10.5', english=True) == 'ten dollar fifty cent'

    def test_dollar_k(self):
        out = norm('$10.4K')
        assert 'ribu' in out
        assert 'dollar' in out

    def test_dollar_m(self):
        out = norm('$1.5M')
        assert 'juta' in out
        assert 'dollar' in out

    def test_dollar_m_english(self):
        out = norm('$1.5M', english=True)
        assert 'million' in out
        assert 'dollar' in out


# ---------------------------------------------------------------------------
# IC numbers - exhaustive
# ---------------------------------------------------------------------------
class TestICNumbers:
    def test_ic_standard(self):
        assert norm('880101-14-5678') == 'lapan lapan kosong satu kosong satu satu empat lima enam tujuh lapan'

    def test_ic_standard_english(self):
        assert norm('880101-14-5678', english=True) == 'eight eight zero one zero one one four five six seven eight'

    def test_ic_young(self):
        out = norm('050901-10-1234')
        assert 'kosong lima kosong sembilan kosong satu' in out

    def test_ic_zeros(self):
        out = norm('000101-14-0001')
        assert out.startswith('kosong kosong kosong')

    def test_ic_in_sentence(self):
        out = norm('IC beliau 950315-10-1234 sudah disahkan')
        assert 'IC beliau' in out
        assert 'sembilan lima kosong tiga satu lima' in out
        assert 'sudah disahkan' in out

    def test_ic_multiple(self):
        out = norm('IC 880101-14-5678 dan 950315-10-1234')
        assert 'lapan lapan kosong satu' in out
        assert 'sembilan lima kosong tiga' in out


# ---------------------------------------------------------------------------
# Phone numbers - exhaustive
# ---------------------------------------------------------------------------
class TestPhoneNumbers:
    def test_mobile_012(self):
        assert norm('012-1234567') == 'kosong satu dua, satu dua tiga empat lima enam tujuh'

    def test_mobile_012_english(self):
        assert norm('012-1234567', english=True) == 'zero one two, one two three four five six seven'

    def test_mobile_011(self):
        out = norm('011-12345678')
        assert 'kosong satu satu' in out

    def test_landline_03(self):
        out = norm('03-87654321')
        assert 'kosong tiga' in out
        assert 'lapan tujuh enam lima' in out

    def test_landline_english(self):
        out = norm('03-87654321', english=True)
        assert 'zero three' in out
        assert 'eight seven six five' in out

    def test_phone_in_sentence(self):
        out = norm('Hubungi 012-3456789 untuk tempahan')
        assert 'Hubungi' in out
        assert 'kosong satu dua' in out
        assert 'tempahan' in out

    def test_phone_multiple(self):
        out = norm('012-1234567 atau 013-7654321')
        assert 'kosong satu dua' in out
        assert 'kosong satu tiga' in out


# ---------------------------------------------------------------------------
# Email - exhaustive
# ---------------------------------------------------------------------------
class TestEmailExhaustive:
    def test_email_basic(self):
        assert norm('husein.zol05@gmail.com') == 'HUSEIN dot ZOL kosong lima di GMAIL dot COM'

    def test_email_basic_english(self):
        assert norm('husein.zol05@gmail.com', english=True) == 'HUSEIN dot ZOL zero five at GMAIL dot COM'

    def test_email_subdomain(self):
        out = norm('nama.penuh123@domain.co.my')
        assert 'NAMA dot PENUH' in out
        assert 'di DOMAIN dot CO dot MY' in out

    def test_email_subdomain_english(self):
        out = norm('nama.penuh123@domain.co.my', english=True)
        assert 'at DOMAIN dot CO dot MY' in out

    def test_email_in_sentence(self):
        out = norm('Hantar ke info@syarikat.com segera')
        assert 'Hantar ke' in out
        assert 'di SYARIKAT dot COM' in out
        assert 'segera' in out

    def test_email_multiple(self):
        out = norm('a@b.com dan c@d.com')
        assert out.count('dot') == 2
        assert out.count('di') == 2


# ---------------------------------------------------------------------------
# URL - exhaustive
# ---------------------------------------------------------------------------
class TestURLExhaustive:
    def test_url_https(self):
        assert norm('https://huseinhouse.com') == 'HTTPS huseinhouse dot com'

    def test_url_www(self):
        out = norm('https://www.google.com')
        assert 'HTTPS' in out
        assert 'WWW' in out
        assert 'google' in out
        assert 'dot com' in out

    def test_url_with_path(self):
        out = norm('https://www.malaysia-ai.org/models')
        assert 'HTTPS' in out
        assert 'WWW' in out
        assert 'dot' in out

    def test_url_ip_address(self):
        out = norm('http://192.168.1.1')
        assert 'HTTP' in out
        assert 'dot' in out

    def test_url_in_sentence(self):
        out = norm('Lawati https://www.google.com untuk maklumat')
        assert 'Lawati' in out
        assert 'maklumat' in out
        assert 'HTTPS' in out


# ---------------------------------------------------------------------------
# Time - exhaustive
# ---------------------------------------------------------------------------
class TestTimeExhaustive:
    def test_2pm(self):
        assert norm('2:01pm') == 'pukul dua satu minit pagi'

    def test_2pm_english(self):
        assert norm('2:01pm', english=True) == 'at two one minute am'

    def test_2am(self):
        assert norm('2AM') == 'pukul dua pagi'

    def test_2am_english(self):
        assert norm('2AM', english=True) == 'at two am'

    def test_midnight(self):
        out = norm('12:00am')
        assert 'dua belas' in out

    def test_midnight_english(self):
        out = norm('12:00am', english=True)
        assert 'twelve' in out

    def test_morning(self):
        out = norm('10:30am')
        assert 'pukul' in out
        assert 'sepuluh' in out
        assert 'tiga puluh' in out

    def test_late_night(self):
        out = norm('11:59pm')
        assert 'sebelas' in out
        assert 'lima puluh sembilan' in out

    def test_late_night_english(self):
        out = norm('11:59pm', english=True)
        assert 'eleven' in out
        assert 'fifty nine' in out


# ---------------------------------------------------------------------------
# Percentage - exhaustive
# ---------------------------------------------------------------------------
class TestPercentExhaustive:
    def test_decimal_percent(self):
        assert norm('61.2%') == 'enam puluh satu perpuluhan dua peratus'

    def test_decimal_percent_english(self):
        assert norm('61.2%', english=True) == 'sixty one point two percent'

    def test_100_percent(self):
        assert norm('100%') == 'seratus peratus'

    def test_100_percent_english(self):
        assert norm('100%', english=True) == 'one hundred percent'

    def test_small_percent(self):
        out = norm('0.5%')
        assert 'perpuluhan lima peratus' in out

    def test_percent_in_sentence(self):
        out = norm('Harga naik 10%')
        assert 'Harga naik' in out
        assert 'peratus' in out


# ---------------------------------------------------------------------------
# Units - exhaustive
# ---------------------------------------------------------------------------
class TestUnitsExhaustive:
    def test_celsius(self):
        assert norm('36.5c') == 'tiga puluh enam perpuluhan lima celsius'

    def test_celsius_english(self):
        assert norm('36.5c', english=True) == 'thirty six point five celcius'

    def test_kg(self):
        assert norm('61.2kg') == 'enam puluh satu perpuluhan dua kilogram'

    def test_kg_english(self):
        assert norm('61.2kg', english=True) == 'sixty one point two kilogram'

    def test_gram(self):
        assert norm('100g') == 'seratus gram'

    def test_gram_english(self):
        assert norm('100g', english=True) == 'one hundred gram'

    def test_km(self):
        assert norm('10km') == 'sepuluh kilometer'

    def test_km_english(self):
        assert norm('10km', english=True) == 'ten kilometer'

    def test_km_decimal(self):
        out = norm('5.0km')
        assert 'kilometer' in out

    def test_liter(self):
        assert norm('2.5l') == 'dua perpuluhan lima liter'

    def test_liter_english(self):
        assert norm('2.5l', english=True) == 'two point five liter'

    def test_ml(self):
        assert norm('250ml') == 'dua ratus lima puluh milliliter'

    def test_mb(self):
        assert norm('500mb') == 'lima ratus megabit'

    def test_mb_english(self):
        assert norm('500mb', english=True) == 'five hundred megabits'

    def test_gb(self):
        out = norm('1.5gb')
        assert 'gigabit' in out

    def test_gb_english(self):
        out = norm('1.5gb', english=True)
        assert 'gigabit' in out

    def test_units_in_sentence(self):
        out = norm('Suhu badan 37.5c dan berat 65kg')
        assert 'celsius' in out
        assert 'kilogram' in out


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------
class TestDates:
    def test_date_standard(self):
        out = norm('25/12/2025')
        assert 'Disember' in out or 'disember' in out
        assert 'dua puluh lima' in out

    def test_date_english(self):
        out = norm('25/12/2025', english=True)
        assert 'December' in out or 'december' in out
        assert 'twenty five' in out

    def test_date_january(self):
        out = norm('1/1/2000')
        assert 'Januari' in out

    def test_date_in_sentence(self):
        out = norm('Tarikh akhir 25/12/2025')
        assert 'Tarikh akhir' in out
        assert 'Disember' in out or 'disember' in out


# ---------------------------------------------------------------------------
# Zero-prefix numbers
# ---------------------------------------------------------------------------
class TestZeroPrefix:
    def test_zero_prefix(self):
        assert norm('01234') == 'kosong satu dua tiga empat'

    def test_zero_prefix_english(self):
        assert norm('01234', english=True) == 'zero one two three four'

    def test_single_zero(self):
        out = norm('001')
        assert 'kosong' in out


# ---------------------------------------------------------------------------
# Passport
# ---------------------------------------------------------------------------
class TestPassport:
    def test_passport_a(self):
        out = norm('A12345678')
        assert 'A' in out
        assert 'satu dua tiga' in out

    def test_passport_english(self):
        out = norm('A12345678', english=True)
        assert 'A' in out
        assert 'one two three' in out


# ---------------------------------------------------------------------------
# Full normalization - Year
# ---------------------------------------------------------------------------
class TestYearNormalization:
    def test_tahun_2024(self):
        assert full('tahun 2024') == 'tahun dua puluh dua puluh empat'

    def test_tahun_2024_english(self):
        out = full('tahun 2024', english=True)
        assert 'twenty' in out

    def test_tahun_1999(self):
        assert full('tahun 1999') == 'tahun sembilan belas sembilan puluh sembilan'

    def test_tahun_2000(self):
        assert full('tahun 2000') == 'tahun dua ribu'

    def test_tahun_1945(self):
        assert full('tahun 1945') == 'tahun sembilan belas empat puluh lima'


# ---------------------------------------------------------------------------
# Full normalization - Pada hari bulan
# ---------------------------------------------------------------------------
class TestPadaHariBulan:
    def test_pada_15_3(self):
        assert full('pada 15/3') == 'pada lima belas hari bulan tiga'

    def test_pada_15_3_english(self):
        assert full('pada 15/3', english=True) == 'on the fifteenth day of the third month'

    def test_pada_1_12(self):
        assert full('pada 1/12') == 'pada satu hari bulan dua belas'


# ---------------------------------------------------------------------------
# Full normalization - Ordinals
# ---------------------------------------------------------------------------
class TestOrdinals:
    def test_ke_1(self):
        assert full('ke-1') == 'pertama'

    def test_ke_1_english(self):
        assert full('ke-1', english=True) == 'first'

    def test_ke_21(self):
        assert full('ke-21') == 'kedua puluh satu'

    def test_ke_21_english(self):
        assert full('ke-21', english=True) == 'twenty-first'

    def test_ke_100(self):
        assert full('ke-100') == 'keseratus'

    def test_ke_100_english(self):
        assert full('ke-100', english=True) == 'one hundredth'

    def test_ke_roman(self):
        assert full('ke-XII') == 'kedua belas'

    def test_ke_in_sentence(self):
        out = full('ini adalah ke-3 kali')
        assert 'ketiga' in out


# ---------------------------------------------------------------------------
# Full normalization - Cardinals
# ---------------------------------------------------------------------------
class TestCardinals:
    def test_123(self):
        assert full('123') == 'seratus dua puluh tiga'

    def test_123_english(self):
        assert full('123', english=True) == 'one hundred and twenty three'

    def test_1000(self):
        out = full('1000')
        assert 'seribu' in out

    def test_1000000(self):
        out = full('1000000')
        assert 'juta' in out or 'satu' in out

    def test_cardinal_in_sentence(self):
        out = full('ada 500 orang')
        assert 'lima ratus' in out
        assert 'orang' in out


# ---------------------------------------------------------------------------
# Full normalization - Fractions
# ---------------------------------------------------------------------------
class TestFractions:
    def test_10_4(self):
        assert full('10/4') == 'sepuluh per empat'

    def test_1_2(self):
        assert full('1/2') == 'satu per dua'

    def test_3_4(self):
        assert full('3/4') == 'tiga per empat'


# ---------------------------------------------------------------------------
# Full normalization - Multiplier (x kali)
# ---------------------------------------------------------------------------
class TestMultiplier:
    def test_10x(self):
        assert full('10x') == 'sepuluh kali'

    def test_10x_english(self):
        assert full('10x', english=True) == 'ten times'

    def test_5x(self):
        assert full('5x') == 'lima kali'

    def test_5x_english(self):
        assert full('5x', english=True) == 'five times'

    def test_100x(self):
        assert full('100x') == 'seratus kali'


# ---------------------------------------------------------------------------
# Full normalization - Hingga (range)
# ---------------------------------------------------------------------------
class TestHingga:
    def test_range_malay(self):
        out = full('2011 - 2019')
        assert 'hingga' in out

    def test_range_english(self):
        out = full('2011 - 2019', english=True)
        assert 'until' in out


# ---------------------------------------------------------------------------
# Full normalization - Hijri year
# ---------------------------------------------------------------------------
class TestHijri:
    def test_hijri(self):
        out = full('1445H')
        assert 'Hijrah' in out
        assert 'seribu empat ratus empat puluh lima' in out


# ---------------------------------------------------------------------------
# Full normalization - Elongated words
# ---------------------------------------------------------------------------
class TestElongated:
    def test_besttt(self):
        assert full('besttt') == 'best'

    def test_takkkkk(self):
        assert full('takkkkk') == 'tak'

    def test_elongated_in_sentence(self):
        out = full('pergi dulu yaaaa')
        assert 'ya' in out


# ---------------------------------------------------------------------------
# Full normalization - xkisah (tak prefix)
# ---------------------------------------------------------------------------
class TestTakPrefix:
    def test_xkisah(self):
        out = full('xkisah')
        assert 'tak' in out

    def test_xtahu(self):
        out = full('xtahu')
        assert 'tak' in out


# ---------------------------------------------------------------------------
# Pronunciation replacements
# ---------------------------------------------------------------------------
class TestPronunciationReplacements:
    def test_dr(self):
        assert apply_pronunciation_replacements('dr Ahmad') == 'doctor Ahmad'

    def test_mr(self):
        assert apply_pronunciation_replacements('mr Ali') == 'mister Ali'

    def test_mrs(self):
        assert apply_pronunciation_replacements('mrs Siti') == 'missus Siti'

    def test_ms(self):
        assert apply_pronunciation_replacements('ms Aminah') == 'miss Aminah'

    def test_sdn_bhd(self):
        assert apply_pronunciation_replacements('Syarikat Sdn Bhd') == 'Syarikat Sendirian Berhad'

    def test_lrt(self):
        assert apply_pronunciation_replacements('naik LRT') == 'naik L R T'

    def test_mrt(self):
        assert apply_pronunciation_replacements('naik MRT') == 'naik M R T'

    def test_kl(self):
        assert apply_pronunciation_replacements('pergi ke KL') == 'pergi ke K L'

    def test_pdrm(self):
        assert apply_pronunciation_replacements('laporan pdrm') == 'laporan P D R M'

    def test_cctv(self):
        assert apply_pronunciation_replacements('pasang cctv') == 'pasang C C T V'

    def test_sop(self):
        assert apply_pronunciation_replacements('ikut sop') == 'ikut S O P'

    def test_lhdn(self):
        assert apply_pronunciation_replacements('pegawai lhdn') == 'pegawai L H D N'

    def test_umno(self):
        assert apply_pronunciation_replacements('parti UMNO') == 'parti umno'

    def test_5g(self):
        assert apply_pronunciation_replacements('jaringan 5G') == 'jaringan five G'

    def test_us(self):
        assert apply_pronunciation_replacements('di US') == 'di U S'

    def test_mba(self):
        assert apply_pronunciation_replacements('ada mba') == 'ada M B A'

    def test_msc(self):
        assert apply_pronunciation_replacements('ada msc') == 'ada M S C'

    def test_emgs(self):
        assert apply_pronunciation_replacements('sistem emgs') == 'sistem E M G S'

    def test_multiple_in_sentence(self):
        out = apply_pronunciation_replacements('dr Ahmad pergi ke KL naik LRT')
        assert out == 'doctor Ahmad pergi ke K L naik L R T'

    def test_no_partial_match(self):
        # 'mr' should not match inside 'timer'
        assert apply_pronunciation_replacements('timer') == 'timer'


# ---------------------------------------------------------------------------
# Pattern range (X-Y unit)
# ---------------------------------------------------------------------------
class TestPatternRange:
    def _replace(self, text, english=False):
        def replace_range(match):
            num1 = int(match.group(1))
            num2 = int(match.group(2))
            words1 = to_cardinal(num1, english=english)
            words2 = to_cardinal(num2, english=english)
            phrase = match.group(3)
            to = 'to' if english else 'hingga'
            return f'{words1} {to} {words2} {phrase}'
        return pattern_range.sub(replace_range, text)

    def test_100_200_ringgit(self):
        assert self._replace('100-200 ringgit') == 'seratus hingga dua ratus ringgit'

    def test_5_10_orang(self):
        assert self._replace('5-10 orang') == 'lima hingga sepuluh orang'

    def test_2024_2025_tahun(self):
        out = self._replace('2024-2025 tahun')
        assert 'dua ribu dua puluh empat' in out
        assert 'hingga' in out
        assert 'tahun' in out

    def test_50_100_peratus(self):
        out = self._replace('50-100 peratus')
        assert 'lima puluh hingga seratus peratus' == out

    def test_3_5_hari(self):
        assert self._replace('3-5 hari') == 'tiga hingga lima hari'

    def test_english_range(self):
        out = self._replace('100-200 ringgit', english=True)
        assert 'one hundred to two hundred ringgit' == out

    def test_range_in_sentence(self):
        out = self._replace('Harga 50-100 ringgit sahaja')
        assert 'lima puluh hingga seratus ringgit' in out
        assert 'Harga' in out
        assert 'sahaja' in out

    def test_no_match_no_unit(self):
        # Range without a following word should not match
        text = '100-200'
        assert self._replace(text) == text


# ---------------------------------------------------------------------------
# Contractions - exhaustive
# ---------------------------------------------------------------------------
class TestContractionsExhaustive:
    def test_aint(self):
        assert expand_contractions("ain't") == 'is not'

    def test_arent(self):
        assert expand_contractions("aren't") == 'are not'

    def test_didnt(self):
        assert expand_contractions("didn't") == 'did not'

    def test_doesnt(self):
        assert expand_contractions("doesn't") == 'does not'

    def test_hadnt(self):
        assert expand_contractions("hadn't") == 'had not'

    def test_hasnt(self):
        assert expand_contractions("hasn't") == 'has not'

    def test_hed(self):
        assert expand_contractions("he'd") == 'he would'

    def test_hell(self):
        assert expand_contractions("he'll") == 'he will'

    def test_hes(self):
        assert expand_contractions("he's") == 'he is'

    def test_id(self):
        assert expand_contractions("i'd") == 'i would'

    def test_ill(self):
        assert expand_contractions("i'll") == 'i will'

    def test_im(self):
        assert expand_contractions("i'm") == 'i am'

    def test_ive(self):
        assert expand_contractions("i've") == 'i have'

    def test_mightnt(self):
        assert expand_contractions("mightn't") == 'might not'

    def test_mustnt(self):
        assert expand_contractions("mustn't") == 'must not'

    def test_shant(self):
        assert expand_contractions("shan't") == 'shall not'

    def test_shed(self):
        assert expand_contractions("she'd") == 'she would'

    def test_shell(self):
        assert expand_contractions("she'll") == 'she will'

    def test_shouldnt(self):
        assert expand_contractions("shouldn't") == 'should not'

    def test_thats(self):
        assert expand_contractions("that's") == 'that is'

    def test_theres(self):
        assert expand_contractions("there's") == 'there is'

    def test_theyd(self):
        assert expand_contractions("they'd") == 'they would'

    def test_theyll(self):
        assert expand_contractions("they'll") == 'they will'

    def test_theyre(self):
        assert expand_contractions("they're") == 'they are'

    def test_wed(self):
        assert expand_contractions("we'd") == 'we would'

    def test_weve(self):
        assert expand_contractions("we've") == 'we have'

    def test_werent(self):
        assert expand_contractions("weren't") == 'were not'

    def test_whatll(self):
        assert expand_contractions("what'll") == 'what will'

    def test_whatre(self):
        assert expand_contractions("what're") == 'what are'

    def test_whats(self):
        assert expand_contractions("what's") == 'what is'

    def test_whove(self):
        assert expand_contractions("who've") == 'who have'

    def test_youd(self):
        assert expand_contractions("you'd") == 'you would'

    def test_youll(self):
        assert expand_contractions("you'll") == 'you will'

    def test_youre(self):
        assert expand_contractions("you're") == 'you are'

    def test_youve(self):
        assert expand_contractions("you've") == 'you have'

    def test_sentence_multiple(self):
        out = expand_contractions("I can't believe they've done it and she's leaving")
        assert out == "I cannot believe they have done it and she is leaving"

    def test_preserves_non_contraction(self):
        assert expand_contractions("hello world") == 'hello world'


# ---------------------------------------------------------------------------
# Split alpha-num
# ---------------------------------------------------------------------------
class TestSplitAlphaNumExhaustive:
    def test_abc123(self):
        assert split_alpha_num('abc123') == 'abc 123'

    def test_123abc(self):
        assert split_alpha_num('123abc') == '123 abc'

    def test_RM500(self):
        assert split_alpha_num('RM500') == 'RM 500'

    def test_A12345(self):
        assert split_alpha_num('A12345') == 'A 12345'

    def test_abc123def456(self):
        assert split_alpha_num('abc123def456') == 'abc 123 def 456'

    def test_pure_alpha(self):
        assert split_alpha_num('hello') == 'hello'

    def test_pure_num(self):
        assert split_alpha_num('12345') == '12345'

    def test_single_char(self):
        assert split_alpha_num('a') == 'a'


# ---------------------------------------------------------------------------
# Before/after replace mappings
# ---------------------------------------------------------------------------
class TestMappingsExhaustive:
    def test_before_en_dash(self):
        text = 'harga–mahal'
        for k, v in before_replace_mapping.items():
            text = text.replace(k, v)
        assert text == 'harga mahal'

    def test_before_semicolon(self):
        text = 'satu; dua'
        for k, v in before_replace_mapping.items():
            text = text.replace(k, v)
        assert text == 'satu, dua'

    def test_before_exclamation(self):
        text = 'hebat!'
        for k, v in before_replace_mapping.items():
            text = text.replace(k, v)
        assert text == 'hebat,'

    def test_before_smart_quote(self):
        text = 'it\u2019s'
        for k, v in before_replace_mapping.items():
            text = text.replace(k, v)
        assert text == "it's"

    def test_after_dash(self):
        text = 'satu-dua'
        for k, v in replace_mapping.items():
            text = text.replace(k, v)
        assert text == 'satu dua'

    def test_after_brackets(self):
        text = '(nota) [rujukan]'
        for k, v in replace_mapping.items():
            text = text.replace(k, v)
        assert text == 'nota rujukan'

    def test_all_before_mappings_applied(self):
        text = 'a–b; c\u2018d! e！f'
        for k, v in before_replace_mapping.items():
            text = text.replace(k, v)
        assert '–' not in text
        assert ';' not in text
        assert '!' not in text


# ---------------------------------------------------------------------------
# Complex multi-type sentences (stress test)
# ---------------------------------------------------------------------------
class TestComplexSentences:
    def test_bayar_rm_before_date(self):
        out = norm('Bayar RM50.00 sebelum 25/12/2025')
        assert 'lima puluh ringgit' in out
        assert 'Disember' in out or 'disember' in out

    def test_suhu_dan_berat(self):
        out = norm('Suhu badan 37.5c dan berat 65kg')
        assert 'celsius' in out
        assert 'kilogram' in out

    def test_ic_phone_email(self):
        out = norm('IC 950315-10-1234 telefon 012-3456789 email test@test.com')
        assert 'sembilan lima kosong tiga' in out
        assert 'kosong satu dua' in out
        assert 'TEST di TEST dot COM' in out

    def test_full_complex_malay(self):
        out = full('Pada tahun 2024, harga RM500 naik 10%, suhu 35c')
        assert 'tahun dua puluh dua puluh empat' in out
        assert 'lima ratus ringgit' in out
        assert 'sepuluh peratus' in out
        assert 'celsius' in out

    def test_full_complex_with_ordinal(self):
        out = full('ke-3 kali bayar RM50 untuk 2.5kg barangan')
        assert 'ketiga' in out
        assert 'lima puluh ringgit' in out
        assert 'kilogram' in out

    def test_full_complex_dr_date_time(self):
        out = full('Dr. Ahmad bayar RM99.99 pada 25/12/2025 pukul 3:45PM')
        assert 'sembilan puluh sembilan ringgit' in out
        assert 'Disember' in out or 'disember' in out
        assert 'tiga' in out
        assert 'empat puluh lima' in out

    def test_full_english_complex(self):
        out = full('The price is $10.5 at 3:00PM on 25/12/2025', english=True)
        assert 'ten dollar' in out
        assert 'December' in out or 'december' in out

    def test_markdown_then_normalize(self):
        text = '**Harga** RM10.5 untuk [produk](https://shop.com) dengan berat 2.5kg'
        s = sanitize_markdown(text)
        s = re.sub(r'[ ]+', ' ', s.replace('\n', ' ')).strip()
        out = norm(s)
        assert '**' not in out
        assert 'sepuluh ringgit' in out
        assert 'kilogram' in out
        assert 'https://' not in out
        assert 'produk' in out

    def test_markdown_email_ic_phone(self):
        text = '**IC:** 880101-14-5678\n**Tel:** 012-1234567\n**Email:** test@mail.com'
        s = sanitize_markdown(text)
        s = re.sub(r'[ ]+', ' ', s.replace('\n', ' ')).strip()
        out = norm(s)
        assert '**' not in out
        assert 'lapan lapan kosong satu' in out
        assert 'kosong satu dua' in out
        assert 'TEST di MAIL dot COM' in out

    def test_html_then_normalize(self):
        text = '<b>Hubungi</b> 012-1234567 atau <a href="mailto:a@b.com">a@b.com</a>'
        s = sanitize_markdown(text)
        out = norm(s)
        assert '<' not in out
        assert 'Hubungi' in out
        assert 'kosong satu dua' in out
        assert 'di' in out
        assert 'dot COM' in out

    def test_pronunciation_after_normalize(self):
        out = norm('dr Ahmad bayar RM10.5 di KL')
        out = apply_pronunciation_replacements(out)
        assert 'doctor' in out
        assert 'K L' in out
        assert 'sepuluh ringgit' in out

    def test_full_pipeline_realistic(self):
        """Simulate the full normalize_malaysian_text pipeline."""
        text = '**dr Ahmad** pergi ke [KL](https://maps.google.com) naik LRT, bayar RM5.50 pada 1/1/2025'
        # 1. sanitize markdown
        s = sanitize_markdown(text)
        # 2. cleanup
        s = s.replace('\n', ' ')
        s = re.sub(r'[ ]+', ' ', s).strip()
        # 3. before_replace_mapping
        for k, v in before_replace_mapping.items():
            s = s.replace(k, v)
        # 4. expand contractions
        s = expand_contractions(s)
        # 5. normalize
        out = norm(s)
        # 6. replace_mapping
        for k, v in replace_mapping.items():
            out = out.replace(k, v)
        # 7. pronunciation
        out = apply_pronunciation_replacements(out)

        assert '**' not in out
        assert 'https://' not in out
        assert 'doctor' in out or 'dr' in out
        assert 'K L' in out
        assert 'ringgit' in out
        assert 'Januari' in out or 'januari' in out
