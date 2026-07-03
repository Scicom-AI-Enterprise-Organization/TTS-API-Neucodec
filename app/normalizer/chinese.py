"""
Mandarin (Chinese) text normalizer.

Unlike app/normalizer/normalizer.py (Malay/English, dictionary + stemming heavy), this module only
verbalizes numbers, currency, dates, times, phone/IC numbers, emails, and common units into Hanzi --
the parts that actually need to be spoken correctly by the TTS model. Plain Hanzi text is never
touched: every conversion function here is a targeted regex substitution, so text with nothing to
convert is returned unchanged.

Out of scope (no unambiguous Chinese TTS convention to fall back on, and not needed by the current
callers): ordinal numbers, fractions, and word-level dictionary correction.
"""

import re

_CN_DIGITS = '零一二三四五六七八九'
_CN_SMALL_UNITS = ['', '十', '百', '千']
_CN_BIG_UNITS = ['', '万', '亿', '万亿']

_ASCII_BOUND_L = r'(?<![0-9A-Za-z])'
_ASCII_BOUND_R = r'(?![0-9A-Za-z])'
_NUM_CORE = r"-?\d+(?:[.,]\d+)?"

CJK_RANGES = r'一-鿿㐀-䶿豈-﫿'
CJK_RE = re.compile(f'[{CJK_RANGES}]')
KANA_RE = re.compile(r'[぀-ヿ]')
_OTHER_SCRIPT_RE = re.compile(f'[^\\x00-\\x7F{CJK_RANGES}\\s]')

EMAIL_RE = re.compile(_ASCII_BOUND_L + r'[\w.+-]+@[\w-]+(?:\.[\w-]+)+' + _ASCII_BOUND_R)
IC_RE = re.compile(r'(?:[0-9]{2})(?:0[1-9]|1[0-2])(?:0[1-9]|[12][0-9]|3[01])-[0-9]{2}-[0-9]{4}')
PHONE_RE = re.compile(r'(?<![0-9])(?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?![0-9])')
MONEY_RE = re.compile(
    r'(?:RM|rm|Rm|rM|US\$|USD|usd|\$|£|€)\s*' + _NUM_CORE
    + r'(?:\s*(?:[kKmMbBjJ]|ribu|juta|thousand|million|billion|bilion))?' + _ASCII_BOUND_R
    + r'|' + _ASCII_BOUND_L + _NUM_CORE
    + r'(?:\s*(?:[kKmMbBjJ]|ribu|juta|thousand|million|billion|bilion))?'
    r'\s*(?:ringgit|sen|dollar|cent|pound|penny|euro)' + _ASCII_BOUND_R,
    re.IGNORECASE,
)
PERCENT_RE = re.compile(_ASCII_BOUND_L + _NUM_CORE + '%')
TEMPERATURE_RE = re.compile(
    _ASCII_BOUND_L + _NUM_CORE
    + r'\s*(?:Kelvin|kelvin|Farenheit|farenheit|Celcius|celcius|Celsius|celsius|K|F|C|c)' + _ASCII_BOUND_R
)
DISTANCE_RE = re.compile(_ASCII_BOUND_L + _NUM_CORE + r'\s*(?:km|KM|cm|CM|mm|MM|m|M)' + _ASCII_BOUND_R)
WEIGHT_RE = re.compile(_ASCII_BOUND_L + _NUM_CORE + r'\s*(?:kg|KG|g|G)' + _ASCII_BOUND_R)
VOLUME_RE = re.compile(_ASCII_BOUND_L + _NUM_CORE + r'\s*(?:ml|ML|l|L)' + _ASCII_BOUND_R)
DATE_RE = re.compile(
    r'(?<![0-9])(?:[12][0-9]{3}[-/.](?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12][0-9]|3[01])'
    r'|(?:0?[1-9]|[12][0-9]|3[01])[-/.](?:0?[1-9]|1[0-2])[-/.][12][0-9]{3})(?![0-9])'
)
TIME_RE = re.compile(
    r'(?<![0-9])([0-2]?[0-9])[:.]([0-5][0-9])(?::([0-5][0-9]))?'
    r'\s*(AM|PM|am|pm|a\.m\.|p\.m\.)?(?![0-9])',
    re.IGNORECASE,
)
BARE_TIME_RE = re.compile(r'(?<![0-9])([0-2]?[0-9])\s*(AM|PM|am|pm)' + _ASCII_BOUND_R, re.IGNORECASE)
NUMBER_RE = re.compile(_ASCII_BOUND_L + _NUM_CORE + _ASCII_BOUND_R)

UNIT_MAP_ZH = {
    'km': '公里', 'm': '米', 'cm': '厘米', 'mm': '毫米',
    'kg': '公斤', 'g': '克',
    'l': '升', 'ml': '毫升',
}

# Chinese states temperature as a prefix ("摄氏36.5度"), unlike the suffix-style units above.
TEMPERATURE_PREFIX_MAP = {
    'c': ('摄氏', '度'), 'celcius': ('摄氏', '度'), 'celsius': ('摄氏', '度'),
    'f': ('华氏', '度'), 'farenheit': ('华氏', '度'),
    'k': ('开尔文', ''), 'kelvin': ('开尔文', ''),
}

_CURRENCY_MAP = {
    'rm': ('令吉', '仙'),
    'ringgit': ('令吉', '仙'),
    'sen': ('令吉', '仙'),
    'us': ('美元', '分'),
    'usd': ('美元', '分'),
    '$': ('美元', '分'),
    'dollar': ('美元', '分'),
    'cent': ('美元', '分'),
    '£': ('英镑', '便士'),
    'pound': ('英镑', '便士'),
    'penny': ('英镑', '便士'),
    '€': ('欧元', '分'),
    'euro': ('欧元', '分'),
}

_MULTIPLIER_MAP = {
    'k': 1e3, 'ribu': 1e3, 'thousand': 1e3,
    'm': 1e6, 'j': 1e6, 'juta': 1e6, 'million': 1e6,
    'b': 1e9, 'bilion': 1e9, 'billion': 1e9,
}


def _convert_group(n):
    """Convert an int in [1, 9999] to Hanzi, with correct internal 零 insertion."""
    digits = str(n)
    length = len(digits)
    result = ''
    zero_pending = False
    for idx, ch in enumerate(digits):
        d = int(ch)
        place = length - idx - 1
        if d == 0:
            if result:
                zero_pending = True
            continue
        if zero_pending:
            result += '零'
            zero_pending = False
        result += _CN_DIGITS[d]
        if place > 0:
            result += _CN_SMALL_UNITS[place]
    return result


def cn_int(n):
    n = int(n)
    if n < 0:
        return '负' + cn_int(-n)
    if n == 0:
        return _CN_DIGITS[0]
    groups = []
    temp = n
    while temp > 0:
        groups.insert(0, temp % 10000)
        temp //= 10000
    num_groups = len(groups)
    result = ''
    prev_had_value = False
    prev_was_gap = False
    for i, g in enumerate(groups):
        place = num_groups - 1 - i
        if g == 0:
            if prev_had_value:
                prev_was_gap = True
            continue
        g_str = _convert_group(g)
        if prev_had_value and (prev_was_gap or g < 1000):
            result += _CN_DIGITS[0]
        result += g_str + _CN_BIG_UNITS[place]
        prev_had_value = True
        prev_was_gap = False
    if result.startswith('一十'):
        result = result[1:]
    return result


def cn_cardinal(x):
    """Convert an int/float/numeric-string (optionally with a leading '-') to Hanzi."""
    x = str(x).strip()
    negative = x.startswith('-')
    if negative:
        x = x[1:]
    x = x.replace(',', '')
    if '.' in x:
        int_part, dec_part = x.split('.', 1)
    else:
        int_part, dec_part = x, ''
    int_part = int_part or '0'
    result = cn_int(int(int_part))
    dec_digits = ''.join(_CN_DIGITS[int(d)] for d in dec_part if d.isdigit())
    if dec_digits:
        result += '点' + dec_digits
    if negative:
        result = '负' + result
    return result


def cn_digit_string(text, use_yao=True):
    """Read a string digit-by-digit (dashes/spaces stripped). use_yao is accepted but unused --
    kept so existing call sites don't need updating; plain digits are read for every value."""
    digits_only = re.sub(r'\D', '', text)
    return ''.join(_CN_DIGITS[int(ch)] for ch in digits_only)


def is_chinese_dominant(text):
    """True if CJK ideographs outnumber every other non-ASCII script character in the text."""
    cjk = len(CJK_RE.findall(text))
    if cjk == 0:
        return False
    other = len(_OTHER_SCRIPT_RE.findall(text))
    return cjk > other


def _email_repl(m):
    return m.group(0).upper().replace('.', ' dot ').replace('@', ' at ')


def _ic_repl(m):
    return cn_digit_string(m.group(0), use_yao=True)


def _phone_repl(m):
    return cn_digit_string(m.group(0), use_yao=True)


def _money_repl(m):
    x = m.group(0)
    xl = x.lower()
    currency = cent_unit = None
    for marker, (cur, cent) in _CURRENCY_MAP.items():
        if xl.startswith(marker) or xl.endswith(marker):
            currency, cent_unit = cur, cent
            break
    if currency is None:
        return x
    is_cent = xl.endswith(('sen', 'cent', 'penny')) and not xl.startswith(('rm', '$', '£', '€', 'us'))
    core = xl
    for marker in ('rm', 'ringgit', 'sen', 'usd', 'us', '$', 'dollar', 'cent', '£', 'pound', 'penny', '€', 'euro'):
        core = core.replace(marker, '')
    core = core.strip()
    match = re.match(r"(\d+(?:[.,]\d+)?)\s*([a-zA-Z]*)", core)
    if not match:
        return x
    num_str, suffix = match.group(1), match.group(2).lower()
    value = float(num_str.replace(',', ''))
    value *= _MULTIPLIER_MAP.get(suffix, 1)
    if is_cent:
        value /= 100
    if value == int(value):
        whole, frac = str(int(value)), ''
    else:
        whole, _, frac = repr(value).partition('.')
    result = cn_cardinal(whole) + currency
    if frac:
        cents = int((frac + '00')[:2])
        if cents:
            result += cn_cardinal(str(cents)) + cent_unit
    return result


def _percent_repl(m):
    return '百分之' + cn_cardinal(m.group(0).rstrip('%'))


def _unit_repl(m):
    x = m.group(0)
    num_match = re.match(_NUM_CORE, x)
    num_str = num_match.group(0)
    unit_str = x[num_match.end():].strip().lower()
    unit_zh = UNIT_MAP_ZH.get(unit_str, unit_str)
    return cn_cardinal(num_str) + unit_zh


def _temperature_repl(m):
    x = m.group(0)
    num_match = re.match(_NUM_CORE, x)
    num_str = num_match.group(0)
    unit_str = x[num_match.end():].strip().lower()
    prefix, suffix = TEMPERATURE_PREFIX_MAP.get(unit_str, ('', unit_str))
    return prefix + cn_cardinal(num_str) + suffix


def _date_repl(m):
    text = m.group(0)
    parts = re.split(r'[-/.]', text)
    if len(parts) != 3:
        return text
    if len(parts[0]) == 4:
        year, month, day = parts
    else:
        day, month, year = parts
    try:
        month_i, day_i = int(month), int(day)
    except ValueError:
        return text
    if not (1 <= month_i <= 12 and 1 <= day_i <= 31):
        return text
    year_str = cn_digit_string(year, use_yao=False)
    return f'{year_str}年{cn_cardinal(month_i)}月{cn_cardinal(day_i)}日'


def _time_from_hour_minute_second(hour, minute, second, desc):
    if desc == 'pm' and hour < 12:
        hour += 12
    elif desc == 'am' and hour == 12:
        hour = 0
    period = '上午' if hour < 12 else '下午'
    hour12 = hour % 12 or 12
    result = period + cn_cardinal(hour12) + '点'
    if minute > 0:
        result += cn_cardinal(minute) + '分'
    if second > 0:
        result += cn_cardinal(second) + '秒'
    return result


def _time_repl(m):
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    second = int(m.group(3) or 0)
    desc = (m.group(4) or '').lower().replace('.', '')
    return _time_from_hour_minute_second(hour, minute, second, desc)


def _bare_time_repl(m):
    hour = int(m.group(1))
    desc = m.group(2).lower()
    return _time_from_hour_minute_second(hour, 0, 0, desc)


def _number_repl(m):
    return cn_cardinal(m.group(0))


def normalize_chinese(text):
    """Verbalize numbers/currency/dates/times/phone/IC/units embedded in (or glued to) Chinese text."""
    if not text:
        return text
    text = EMAIL_RE.sub(_email_repl, text)
    text = IC_RE.sub(_ic_repl, text)
    text = PHONE_RE.sub(_phone_repl, text)
    text = MONEY_RE.sub(_money_repl, text)
    text = TEMPERATURE_RE.sub(_temperature_repl, text)
    text = DISTANCE_RE.sub(_unit_repl, text)
    text = WEIGHT_RE.sub(_unit_repl, text)
    text = VOLUME_RE.sub(_unit_repl, text)
    text = PERCENT_RE.sub(_percent_repl, text)
    text = DATE_RE.sub(_date_repl, text)
    text = TIME_RE.sub(_time_repl, text)
    text = BARE_TIME_RE.sub(_bare_time_repl, text)
    text = NUMBER_RE.sub(_number_repl, text)
    return text
