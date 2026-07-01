"""
Unit tests for app/normalizer/chinese.py. Pure-Python, no GPU/model deps.

Run with: python -m pytest tests/test_chinese_normalizer.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.normalizer.chinese import (
    cn_int, cn_cardinal, cn_digit_string, normalize_chinese, is_chinese_dominant,
)


class TestCnInt:
    def test_zero(self):
        assert cn_int(0) == '零'

    def test_teens(self):
        assert cn_int(10) == '十'
        assert cn_int(11) == '十一'
        assert cn_int(19) == '十九'

    def test_tens(self):
        assert cn_int(20) == '二十'
        assert cn_int(99) == '九十九'

    def test_hundreds_with_internal_zero(self):
        assert cn_int(100) == '一百'
        assert cn_int(105) == '一百零五'
        assert cn_int(110) == '一百一十'

    def test_thousands_with_internal_zero(self):
        assert cn_int(1000) == '一千'
        assert cn_int(1001) == '一千零一'
        assert cn_int(1010) == '一千零一十'
        assert cn_int(1100) == '一千一百'

    def test_wan_magnitude(self):
        assert cn_int(10000) == '一万'
        assert cn_int(100000) == '十万'
        assert cn_int(100001) == '十万零一'
        assert cn_int(123456) == '十二万三千四百五十六'
        assert cn_int(1000000) == '一百万'

    def test_yi_magnitude(self):
        assert cn_int(100000000) == '一亿'
        assert cn_int(100000001) == '一亿零一'

    def test_negative(self):
        assert cn_int(-5) == '负五'
        assert cn_int(-50) == '负五十'


class TestCnCardinal:
    def test_integer_string(self):
        assert cn_cardinal('500') == '五百'

    def test_decimal(self):
        assert cn_cardinal('36.5') == '三十六点五'
        assert cn_cardinal('0.5') == '零点五'

    def test_negative_decimal(self):
        assert cn_cardinal('-3.14') == '负三点一四'

    def test_accepts_int(self):
        assert cn_cardinal(12) == '十二'


class TestCnDigitString:
    def test_uses_yao_for_one(self):
        assert cn_digit_string('012-1234567', use_yao=True) == '零一二一二三四五六七'

    def test_plain_digits_without_yao(self):
        assert cn_digit_string('2025', use_yao=False) == '二零二五'

    def test_strips_non_digits(self):
        assert cn_digit_string('880101-14-5678') == '八八零一零一一四五六七八'


class TestNormalizeChinese:
    def test_plain_text_unchanged(self):
        assert normalize_chinese('你好世界') == '你好世界'

    def test_money_glued(self):
        assert normalize_chinese('价格是RM500') == '价格是五百令吉'

    def test_money_spaced(self):
        assert normalize_chinese('价格是 RM500') == '价格是 五百令吉'

    def test_money_with_cents(self):
        assert normalize_chinese('RM10.50') == '十令吉五十仙'

    def test_dollar(self):
        assert normalize_chinese('$100') == '一百美元'

    def test_percent(self):
        assert normalize_chinese('打折50%') == '打折百分之五十'

    def test_phone(self):
        assert normalize_chinese('电话012-1234567') == '电话零一二一二三四五六七'

    def test_ic(self):
        assert '八八零一零一一四五六七八' in normalize_chinese('身份证880101-14-5678')

    def test_date_dmy(self):
        assert normalize_chinese('今天是25/12/2025') == '今天是二零二五年十二月二十五日'
        assert normalize_chinese('今天是25-12-2025') == '今天是二零二五年十二月二十五日'

    def test_date_ymd(self):
        assert normalize_chinese('今天是2025/12/25') == '今天是二零二五年十二月二十五日'
        assert normalize_chinese('今天是2025-12-25') == '今天是二零二五年十二月二十五日'

    def test_time_with_colon(self):
        assert normalize_chinese('现在是3:45PM') == '现在是下午三点四十五分'

    def test_bare_time(self):
        assert normalize_chinese('现在9.50AM') == '现在上午九点五十分'

    def test_temperature(self):
        assert normalize_chinese('温度是36.5c') == '温度是摄氏三十六点五度'

    def test_distance(self):
        assert normalize_chinese('距离5km') == '距离五公里'

    def test_weight(self):
        assert normalize_chinese('重量10kg') == '重量十公斤'

    def test_generic_number_glued(self):
        assert normalize_chinese('有50个人') == '有五十个人'

    def test_email_untouched(self):
        assert normalize_chinese('联系test@mail.com') == '联系test@mail.com'


class TestIsChineseDominant:
    def test_chinese_sentence(self):
        assert is_chinese_dominant('价格是 RM500 电话 012-1234567') is True

    def test_incidental_chinese_char_in_arabic_sentence(self):
        assert is_chinese_dominant('العربية RM100.50 和 25/12/2025') is False

    def test_no_chinese_at_all(self):
        assert is_chinese_dominant('안녕하세요') is False

    def test_pure_kanji_is_dominant(self):
        # No kana to disambiguate from Chinese; treated as Chinese (harmless when there's nothing
        # numeric to convert, which is how app/main.py additionally guards attached-token routing).
        assert is_chinese_dominant('日本語') is True

    def test_kanji_with_kana_not_dominant(self):
        assert is_chinese_dominant('価格は RM300 です') is False
