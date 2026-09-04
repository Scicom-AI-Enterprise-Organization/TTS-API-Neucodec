"""
Rule-based spoken-form normalizer: the LLM normalizer's contract (app/prompt.py), as regexes.

Rewrites what is not speakable as written -- numbers, money, IC/phone numbers, dates, times,
percentages, decimals, ordinals, units, emails, URLs, ids, a few abbreviations -- into words, in
the language of the sentence (see lang.py), and leaves every other word exactly as written.
Each handler is a regex with a callback; they run in a fixed order so that the more specific
shapes (email, IC, phone, date, time, money, ...) claim their digits before the generic number
rule sees what is left. Nothing here needs torch or a model; ~50 us per sentence.

Conventions follow bench/results/normalizer_truth.jsonl (the live LLM's outputs on
bench/normalizer_corpus.py), except where the LLM was inconsistent or wrong; those choices are
called out inline. Score it with bench/normalizer_agreement.py.
"""
import re

from . import numbers as N
from .lang import detect_lang

# --------------------------------------------------------------------------- vocabulary
V = {
    'en': dict(ringgit='ringgit', sen='sen', dollar=('dollar', 'dollars'), usd='US dollars', sgd='Singapore dollars',
               euro=('euro', 'euros'), pound=('pound', 'pounds'), percent='percent', plus='plus', to='to',
               at='at', dot='dot', am='a m', pm='p m', oclock="o'clock", number='number', hash='number',
               million='million', billion='billion', thousand='thousand'),
    'ms': dict(ringgit='ringgit', sen='sen', dollar='dolar', usd='dolar Amerika', sgd='dolar Singapura',
               euro='euro', pound='paun', percent='peratus', plus='tambah', to='hingga',
               at='at', dot='dot', am='pagi', pm='petang', oclock='', number='nombor', hash='nombor',
               million='juta', billion='bilion', thousand='ribu'),
    'zh': dict(ringgit='令吉', sen='仙', dollar='美元', usd='美元', sgd='新元', euro='欧元', pound='英镑',
               percent='百分之', plus='加', to='到', at='at', dot='dot', am='上午', pm='下午', oclock='点',
               number='号', hash='号', million='百万', billion='十亿', thousand='千'),
    'ta': dict(ringgit='ரிங்கிட்', sen='சென்', dollar='டாலர்', usd='அமெரிக்க டாலர்', sgd='சிங்கப்பூர் டாலர்',
               euro='யூரோ', pound='பவுண்ட்', percent='சதவீதம்', plus='பிளஸ்', to='முதல்',
               at='at', dot='dot', am='ஏ எம்', pm='பி எம்', oclock='', number='எண்', hash='எண்',
               million='மில்லியன்', billion='பில்லியன்', thousand='ஆயிரம்'),
}

# unit -> per-language (singular, plural) or one form
UNITS = {
    'kg': {'en': ('kilogram', 'kilograms'), 'ms': 'kilogram', 'zh': '公斤', 'ta': 'கிலோகிராம்'},
    'g': {'en': ('gram', 'grams'), 'ms': 'gram', 'zh': '克', 'ta': 'கிராம்'},
    'mg': {'en': ('milligram', 'milligrams'), 'ms': 'miligram', 'zh': '毫克', 'ta': 'மில்லிகிராம்'},
    'km': {'en': ('kilometer', 'kilometers'), 'ms': 'kilometer', 'zh': '公里', 'ta': 'கிலோமீட்டர்'},
    'm': {'en': ('meter', 'meters'), 'ms': 'meter', 'zh': '米', 'ta': 'மீட்டர்'},
    'cm': {'en': ('centimeter', 'centimeters'), 'ms': 'sentimeter', 'zh': '厘米', 'ta': 'சென்டிமீட்டர்'},
    'mm': {'en': ('millimeter', 'millimeters'), 'ms': 'milimeter', 'zh': '毫米', 'ta': 'மில்லிமீட்டர்'},
    'ml': {'en': ('milliliter', 'milliliters'), 'ms': 'mililiter', 'zh': '毫升', 'ta': 'மில்லிலிட்டர்'},
    'l': {'en': ('liter', 'liters'), 'ms': 'liter', 'zh': '升', 'ta': 'லிட்டர்'},
    'gb': {'en': ('gigabyte', 'gigabytes'), 'ms': 'gigabait', 'zh': 'GB', 'ta': 'ஜிபி'},
    'mb': {'en': ('megabyte', 'megabytes'), 'ms': 'megabait', 'zh': 'MB', 'ta': 'எம்பி'},
    'tb': {'en': ('terabyte', 'terabytes'), 'ms': 'terabait', 'zh': 'TB', 'ta': 'டிபி'},
    'kb': {'en': ('kilobyte', 'kilobytes'), 'ms': 'kilobait', 'zh': 'KB', 'ta': 'கேபி'},
    'mbps': {'en': 'megabits per second', 'ms': 'megabit sesaat', 'zh': '兆比特每秒', 'ta': 'எம்பிபிஎஸ்'},
    'kwh': {'en': ('kilowatt hour', 'kilowatt hours'), 'ms': 'kilowatt jam', 'zh': '千瓦时', 'ta': 'கிலோவாட் மணி'},
    '°c': {'en': ('degree Celsius', 'degrees Celsius'), 'ms': 'darjah Celsius', 'zh': ('摄氏', '度'), 'ta': 'டிகிரி செல்சியஸ்'},
    '°f': {'en': ('degree Fahrenheit', 'degrees Fahrenheit'), 'ms': 'darjah Fahrenheit', 'zh': ('华氏', '度'), 'ta': 'டிகிரி பாரன்ஹீட்'},
    '°': {'en': ('degree', 'degrees'), 'ms': 'darjah', 'zh': '度', 'ta': 'டிகிரி'},
}
_UNIT_ALIASES = {'kgs': 'kg', 'kilo': 'kg', 'kilos': 'kg', '℃': '°c', '℉': '°f', 'ltr': 'l', 'litre': 'l'}

MONTHS = {
    'en': [None, 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
           'October', 'November', 'December'],
    'ms': [None, 'Januari', 'Februari', 'Mac', 'April', 'Mei', 'Jun', 'Julai', 'Ogos', 'September',
           'Oktober', 'November', 'Disember'],
    'ta': [None, 'ஜனவரி', 'பிப்ரவரி', 'மார்ச்', 'ஏப்ரல்', 'மே', 'ஜூன்', 'ஜூலை', 'ஆகஸ்ட்', 'செப்டம்பர்',
           'அக்டோபர்', 'நவம்பர்', 'டிசம்பர்'],
}
_MONTH_NUM = {}
for _lang in ('en', 'ms', 'ta'):
    for _i, _m in enumerate(MONTHS[_lang]):
        if _m:
            _MONTH_NUM[_m.lower()] = _i
_MONTH_NUM.update({'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6, 'jul': 7, 'aug': 8, 'ogs': 8, 'sep': 9,
                   'sept': 9, 'oct': 10, 'okt': 10, 'nov': 11, 'dec': 12, 'dis': 12})
_MONTH_ALT = '|'.join(sorted((re.escape(m) for m in _MONTH_NUM), key=len, reverse=True))

# Words after which a following number is read digit by digit ("Room 305", "PIN 0457"). The
# LLM reads room and PIN numbers as digits, order/invoice numbers as digits, but house
# numbers after "No." ("No. 12") and short codes as cardinals -- hence the length rule in
# `_number`. Chinese reads room numbers as cardinals (三百零五号), so its list is short.
_DIGIT_CONTEXT = {
    'en': r'room|rooms|no\.?|number|order|pin|ref|reference|id|invoice|account|acc|acct|flight|ext|extension|lot|policy|ticket|case',
    'ms': r'bilik|no\.?|nombor|pesanan|pin|rujukan|invois|akaun|penerbangan|lot|blok|polisi|tiket|kes|id|order',
    'zh': r'密码|编号|号码|订单|参考|账号|工单',
    'ta': r'அறை|no\.?|எண்|ஆர்டர்|pin|குறிப்பு|order|invoice|id',
}
_ALWAYS_DIGITS_CONTEXT = r'room|rooms|bilik|pin|அறை|密码'

# Chinese: 2 before a measure word is 两 (两箱, 两点), not 二 -- except after 第 (第二次).
_ZH_MEASURE = '个箱件位次点天人张份块条台辆间只杯瓶本层种批小周组套页颗粒根支'


# --------------------------------------------------------------------------- helpers
def _card(n, lang):
    return N.cardinal(n, lang)


def _digits(s, lang):
    return N.digits(s, lang)


def _vocab(lang, key, plural=True):
    v = V[lang][key]
    if isinstance(v, tuple):
        return v[1] if plural else v[0]
    return v


def _unit_words(unit, lang, plural):
    u = UNITS[_UNIT_ALIASES.get(unit, unit)][lang]
    if lang == 'zh' and isinstance(u, tuple):      # (prefix, suffix): 摄氏三十二度
        return u
    if isinstance(u, tuple):
        return u[1] if plural else u[0]
    return u


def _join(parts, lang):
    parts = [p for p in parts if p]
    return ''.join(parts) if lang == 'zh' else ' '.join(parts)


def _num_words(int_part, frac_part, lang):
    """A plain amount: '1,250' / '2.5' -> words (commas dropped, decimals read digit by digit)."""
    int_part = int_part.replace(',', '')
    if frac_part:
        out = N.decimal(int_part, frac_part, lang)
        if lang == 'ta' and int(int_part or 0) == 1:
            out = 'ஒரு' + out[len('ஒன்று'):]      # ஒரு புள்ளி இரண்டு, as the LLM says it
        return out
    return _card(int(int_part), lang)


def _spell_chunk(chunk, lang):
    """Letter/digit runs of an id token: digits one by one, letter runs kept (MH, ABC), single
    letters as themselves. 'MH370' -> 'MH three seven zero', 'X-12340567' -> 'X one two ...'."""
    out = []
    for run in re.findall(r'\d+|[A-Za-z]+|[^A-Za-z\d]+', chunk):
        if run.isdigit():
            if lang != 'zh' and len(run) <= 2 and out and not out[-1].isdigit() and False:
                pass
            out.append(_digits(run, lang))
        elif run.isalpha():
            out.append(run)
        # separators (-, /) are dropped: they were only glue
    return _join(out, lang)


# --------------------------------------------------------------------------- handlers
EMAIL_RE = re.compile(r'(?<![A-Za-z0-9_.+-])([A-Za-z0-9_.+-]+)@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)(?![A-Za-z0-9_-])')


def _spell_domainish(s, lang):
    s = re.sub(r'\d+', lambda m: ' ' + _digits(m.group(0), lang) + ' ', s)
    s = s.replace('.', f" {V[lang]['dot']} ").replace('-', ' dash ').replace('_', ' underscore ').replace('+', ' plus ')
    return re.sub(r'\s+', ' ', s).strip()


def _email(m, lang):
    out = f"{_spell_domainish(m.group(1), lang)} {V[lang]['at']} {_spell_domainish(m.group(2), lang)}"
    return f' {out} ' if lang == 'zh' else out


URL_RE = re.compile(r'(?<![A-Za-z0-9_@])((?:https?://|www\.)[^\s<>"\'一-鿿㐀-䶿豈-﫿஀-௿]+)')


def _url(m, lang):
    url = m.group(1)
    trail = ''
    while url and url[-1] in '.,;:!?)』」':
        trail = url[-1] + trail
        url = url[:-1]
    out = url
    out = re.sub(r'^(https?)://', lambda mm: ' '.join(mm.group(1)) + ' colon slash slash ', out)
    out = re.sub(r'^www\.', 'w w w dot ', out)
    out = out.replace('/', ' slash ').replace('.', f" {V[lang]['dot']} ").replace('-', ' dash ') \
             .replace('_', ' underscore ').replace('?', ' question mark ').replace('=', ' equals ') \
             .replace('&', ' and ').replace('#', ' hash ')
    out = re.sub(r'\d+', lambda mm: ' ' + _digits(mm.group(0), lang) + ' ', out)
    out = re.sub(r'\s+', ' ', out).strip()
    return (f' {out} ' if lang == 'zh' else out) + trail


IC_RE = re.compile(r'(?<![\d-])(\d{6})-(\d{2})-(\d{4})(?![\d-])')


def _ic(m, lang):
    return _digits(''.join(m.groups()), lang)


# +60 12-345 6789 | 03-1234 5678 | 1-300-88-1234 | 0198765432 : 7..15 digits in 1..5 groups
PHONE_RE = re.compile(r'(?<![A-Za-z0-9_+.,-])(\+?\d{1,4}(?:[ -]\d{1,8}){1,5}|\+?\d{7,15})(?![A-Za-z0-9_-])')


def _phone(m, lang):
    s = m.group(1)
    n = sum(c.isdigit() for c in s)
    if n < 7 or n > 15 or re.fullmatch(r'\d{4}-\d{4}', s):       # 2020-2024 is a year range
        return s
    plus = V[lang]['plus'] + (' ' if lang != 'zh' else '') if s.startswith('+') else ''
    return plus + _digits(s, lang)      # zh joins the digits (零三一二三四五六七八), the rest space them


# 15/3/2024, 15-03-2024, 15.3.24, 2024-03-15, 3/15/2024 (US, when the first field cannot be a day)
DATE_NUM_RE = re.compile(r'(?<![\d/.-])(\d{1,4})([/.-])(\d{1,2})\2(\d{1,4})(?!\d)(?![/.-]\d)')


def _year_value(y):
    y = int(y)
    if y < 100:
        y += 2000 if y < 50 else 1900
    return y


def _date_words(d, mth, y, lang, style='dmy', comma=''):
    """Spoken date. y may be None. style 'dmy' or 'mdy' (the LLM says ISO dates month-first)."""
    if lang == 'zh':
        out = f'{N.year(y, "zh")}年' if y else ''
        return out + f'{_card(mth, "zh")}月{_card(d, "zh")}日'
    if lang == 'en':
        month = MONTHS['en'][mth]
        year = ' ' + N.year(y, 'en') if y else ''
        if style == 'mdy':
            return f'{month} {N.en_ordinal(d)}{comma}{year}'
        return f'the {N.en_ordinal(d)} of {month}{year}'
    month = MONTHS[lang][mth]
    year = ' ' + N.year(y, lang) if y else ''
    return f'{_card(d, lang)} {month}{year}'


def _date_numeric(m, lang):
    a, sep, b, c = m.group(1), m.group(2), int(m.group(3)), m.group(4)
    if len(a) == 4:                                 # y-m-d
        y, mth, d, style = int(a), b, int(c), 'mdy'
    elif len(c) in (2, 4) and len(a) <= 2:          # d-m-y (or m-d-y when a cannot be a month)
        a = int(a)
        if 1 <= b <= 12 and 1 <= a <= 31:
            d, mth = a, b
        elif 1 <= a <= 12 and 1 <= b <= 31:
            d, mth = b, a
        else:
            return m.group(0)
        y, style = _year_value(c), 'dmy'
    else:
        return m.group(0)
    if not (1 <= mth <= 12 and 1 <= d <= 31 and 1900 <= y <= 2199):
        return m.group(0)
    return _date_words(d, mth, y, lang, style)


# 15 March 2024 | 5 Jan 2026 | 15th March | March 15, 1998 | March 15
DATE_DMY_RE = re.compile(r'(?<![A-Za-z0-9_/.-])(\d{1,2})(?:st|nd|rd|th|hb)?\s+(' + _MONTH_ALT + r')(?![A-Za-z])\.?(?:,?\s+(\d{4})(?![A-Za-z0-9_]))?',
                         re.IGNORECASE)
DATE_MDY_RE = re.compile(r'(?<![A-Za-z])(' + _MONTH_ALT + r')(?![A-Za-z])\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?![A-Za-z0-9_])(?!\s*[:%.]\d)(?:(,?)\s+(\d{4})(?![A-Za-z0-9_]))?',
                         re.IGNORECASE)


def _month_name_out(written, lang):
    """The month as the sentence should say it: full name in the sentence language when the
    input was an abbreviation, otherwise as written (the LLM keeps 'June' inside Malay)."""
    key = written.lower().rstrip('.')
    num = _MONTH_NUM.get(key)
    if num is None:
        return written, None
    if lang in ('en', 'ms', 'ta') and (len(key) <= 4 and key not in ('mac', 'mei', 'jun', 'may', 'june', 'july')):
        return MONTHS[lang][num], num
    if lang == 'ta' and key in _MONTH_NUM and not any('஀' <= ch <= '௿' for ch in written):
        return MONTHS['ta'][num] if written.lower() in ('jan', 'feb', 'mar', 'apr', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec') else written, num
    return written, num


def _date_dmy(m, lang):
    d, month, y = int(m.group(1)), m.group(2), m.group(3)
    name, num = _month_name_out(month, lang)
    if not (1 <= d <= 31) or num is None:
        return m.group(0)
    if lang == 'en':
        return f'the {N.en_ordinal(d)} of {name}' + (' ' + N.year(y, 'en') if y else '')
    if lang == 'zh':
        return _date_words(d, num, int(y) if y else None, 'zh')
    return f'{_card(d, lang)} {name}' + (' ' + N.year(y, lang) if y else '')


def _date_mdy(m, lang):
    month, d, comma, y = m.group(1), int(m.group(2)), m.group(3) or '', m.group(4)
    name, num = _month_name_out(month, lang)
    if not (1 <= d <= 31) or num is None:
        return m.group(0)
    if lang == 'en':
        return f'{name} {N.en_ordinal(d)}' + (f'{comma} ' + N.year(y, 'en') if y else '')
    if lang == 'zh':
        return _date_words(d, num, int(y) if y else None, 'zh')
    return f'{name} {_card(d, lang)}' + (f'{comma} ' + N.year(y, lang) if y else '')


# 10:45 AM | 3:30 p.m. | 5pm | 9.30am | 14:05 | 8:00 | 17:00
_AMPM = r'([AaPp])\.?\s?[Mm](?![A-Za-z])\.?'
TIME_AMPM_RE = re.compile(r'(?<![\d:.])(\d{1,2})(?:[:.](\d{2}))?\s?' + _AMPM)
TIME_24_RE = re.compile(r'(?<![\d:.])(\d{1,2}):(\d{2})(?![\d:])')
TIME_RANGE_RE = re.compile(
    r'(?<![\d:.])(\d{1,2})(?:[:.](\d{2}))?\s?(?:([AaPp])\.?\s?[Mm](?![A-Za-z])\.?)?\s*(?:-|–|—|to|hingga|ke|到|至)\s*'
    r'(\d{1,2})(?:[:.](\d{2}))?\s?([AaPp])\.?\s?[Mm](?![A-Za-z])\.?')
_ZH_PERIOD_RE = re.compile(r'(上午|下午|中午|晚上|早上|凌晨|傍晚|夜里)\s*$')


def _time_words(h, mm, ampm, lang, before=''):
    """h: int hour, mm: '45' / '05' / None, ampm: 'a'/'p'/None (case of the source kept for en)."""
    if lang == 'zh':
        period = ''
        if ampm and not _ZH_PERIOD_RE.search(before):
            period = V['zh']['am'] if ampm.lower() == 'a' else V['zh']['pm']
        hour = '两' if h == 2 else _card(h, 'zh')
        out = period + hour + '点'
        if mm and mm != '00':
            out += ('零' if mm[0] == '0' else '') + _card(int(mm), 'zh') + '分'
        return out
    parts = [_card(h, lang)]
    if mm and mm != '00':
        parts.append(_digits(mm, lang) if mm[0] == '0' else _card(int(mm), lang))
    if ampm:
        word = V[lang]['am'] if ampm.lower() == 'a' else V[lang]['pm']
        if lang == 'en' and ampm.isupper():
            word = word.upper()
        parts.append(word)
    elif lang == 'en' and (mm == '00' or mm is None):
        parts.append(V['en']['oclock'])
    return ' '.join(parts)


def _time_ampm(m, lang):
    h = int(m.group(1))
    if h > 24:
        return m.group(0)
    return _time_words(h, m.group(2), m.group(3), lang, m.string[:m.start()][-6:])


def _time_24(m, lang):
    h, mm = int(m.group(1)), m.group(2)
    if h > 24 or int(mm) > 59:
        return m.group(0)
    return _time_words(h, mm, None, lang, m.string[:m.start()][-6:])


def _time_range(m, lang):
    h1, m1, ap1, h2, m2, ap2 = m.groups()
    if int(h1) > 24 or int(h2) > 24:
        return m.group(0)
    ap1 = ap1 or ap2
    a = _time_words(int(h1), m1, ap1, lang, m.string[:m.start()][-6:])
    b = _time_words(int(h2), m2, ap2, lang)
    if lang == 'zh':
        return a + V['zh']['to'] + b
    if lang == 'ta':
        return f'{a} {V["ta"]["to"]} {b} வரை'
    return f'{a} {V[lang]["to"]} {b}'


# RM1,250.50 | RM 3,400 | $25 | USD 10 | RM1.2 million | RM50k
MONEY_RE = re.compile(
    r'(?<![A-Za-z0-9_])(RM|MYR|USD|US\$|SGD|S\$|AUD|GBP|EUR|\$|€|£)\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?'
    r'(?:\s?(million|juta|billion|bilion|k|百万|万|亿|மில்லியன்|பில்லியன்)(?![A-Za-z0-9_]))?(?![A-Za-z0-9_])(?!\.\d)', re.IGNORECASE)
_CURRENCY_KEY = {'rm': 'ringgit', 'myr': 'ringgit', 'usd': 'usd', 'us$': 'usd', '$': 'dollar', 'sgd': 'sgd',
                 's$': 'sgd', 'aud': 'dollar', 'gbp': 'pound', '£': 'pound', 'eur': 'euro', '€': 'euro'}
_MULT_VALUE = {'million': 10 ** 6, 'juta': 10 ** 6, 'billion': 10 ** 9, 'bilion': 10 ** 9, 'k': 1000,
               '百万': 10 ** 6, '万': 10 ** 4, '亿': 10 ** 8, 'மில்லியன்': 10 ** 6, 'பில்லியன்': 10 ** 9}


def _money(m, lang):
    cur, whole, cents, mult = m.group(1).lower(), m.group(2).replace(',', ''), m.group(3), m.group(4)
    key = _CURRENCY_KEY[cur]
    if key == 'ringgit':
        major, minor = V[lang]['ringgit'], V[lang]['sen']
    else:
        minor = None
        amount_is_one = whole == '1' and not cents and not mult
        major = _vocab(lang, key, plural=not amount_is_one)
    if mult:
        mv = _MULT_VALUE[mult.lower()]
        if lang == 'zh':
            value = int(round(float(whole + ('.' + cents if cents else '')) * mv))
            return _card(value, 'zh') + major
        mword = {'million': V[lang]['million'], 'juta': V[lang]['million'], 'மில்லியன்': V[lang]['million'],
                 'billion': V[lang]['billion'], 'bilion': V[lang]['billion'], 'பில்லியன்': V[lang]['billion'],
                 'k': V[lang]['thousand']}[mult.lower()]
        amount = _num_words(whole, cents, lang)
        return f'{amount} {mword} {major}'
    if lang == 'zh':
        out = ''
        if int(whole) or not cents or int(cents) == 0:
            out += _card(int(whole), 'zh') + major
        if cents and int(cents):
            out += _card(int(cents.ljust(2, '0')), 'zh') + (minor or '')
        return out
    parts = []
    if int(whole) or not cents or int(cents) == 0:
        parts.append(f'{_card(int(whole), lang)} {major}')
    if cents and int(cents):
        c = int(cents.ljust(2, '0'))
        if minor:
            parts.append(f'{_card(c, lang)} {minor}')
        else:
            parts.append({'en': f'{_card(c, "en")} cents', 'ms': f'{_card(c, "ms")} sen', 'ta': f'{_card(c, "ta")} சென்'}[lang])
    return ' '.join(parts)


PERCENT_RE = re.compile(r'(?<![A-Za-z0-9_.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s?(%|％|percent|peratus)(?![A-Za-z0-9_])', re.IGNORECASE)


def _percent(m, lang):
    words = _num_words(m.group(1), m.group(2), lang)
    if lang == 'zh':
        return V['zh']['percent'] + words
    return f'{words} {V[lang]["percent"]}'


UNIT_RE = re.compile(
    r'(?<![A-Za-z0-9_.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s?(°\s?[CcFf]|℃|℉|°|kg|kgs|mg|g|km|cm|mm|m|ml|l|gb|mb|tb|kb|mbps|kwh)(?![A-Za-z0-9_])',
    re.IGNORECASE)


def _unit(m, lang):
    whole, frac, unit = m.group(1), m.group(2), m.group(3).lower().replace(' ', '')
    if unit in ('°c', '°f') or unit in ('℃', '℉'):
        unit = _UNIT_ALIASES.get(unit, unit)
    unit = _UNIT_ALIASES.get(unit, unit)
    words = _num_words(whole, frac, lang)
    plural = not (whole.replace(',', '') == '1' and not frac)
    u = _unit_words(unit, lang, plural)
    if lang == 'zh':
        if isinstance(u, tuple):
            return u[0] + words + u[1]
        return words + u
    return f'{words} {u}'


ORD_EN_RE = re.compile(r'(?<![A-Za-z0-9_.])(\d+)(st|nd|rd|th)(?![A-Za-z0-9_])')
ORD_MS_RE = re.compile(r'(?<![A-Za-z0-9_])ke-(\d+)(?![A-Za-z0-9_])')
ORD_ZH_RE = re.compile(r'第(\d+)')
ORD_TA_RE = re.compile(r'(?<![A-Za-z0-9_.])(\d+)\s?-?\s?(ஆவது|வது|ஆம்|ம்)')


def _ord_en(m, lang):
    return N.en_ordinal(int(m.group(1)))


def _ord_ms(m, lang):
    return N.ms_ordinal(int(m.group(1)))


def _ord_zh(m, lang):
    return '第' + N.zh_cardinal(int(m.group(1)))


def _ord_ta(m, lang):
    suffix = 'ஆம்' if m.group(2) in ('ஆம்', 'ம்') else 'ஆவது'
    return N.ta_ordinal(int(m.group(1)), suffix)


RANGE_RE = re.compile(r'(?<![A-Za-z0-9_.,/-])(\d{1,4})\s?[-–—]\s?(\d{1,4})(?![A-Za-z0-9_.,/-])')


def _range(m, lang):
    a, b = int(m.group(1)), int(m.group(2))
    f = N.year if (1900 <= a <= 2099 and 1900 <= b <= 2099 and len(m.group(1)) == 4) else _card
    if lang == 'zh':
        return f(a, 'zh') + V['zh']['to'] + f(b, 'zh')
    return f'{f(a, lang)} {V[lang]["to"]} {f(b, lang)}'


HASH_RE = re.compile(r'#\s?(\d+)')


def _hash(m, lang):
    if lang == 'zh':
        return V['zh']['hash'] + N.zh_digits(m.group(1))
    return f'{V[lang]["hash"]} {_digits(m.group(1), lang)}'


# tokens mixing letters and digits: X-12340567, ABC1234, MH370, 24A, A12345678, 1Z999AA1012345
ALNUM_RE = re.compile(r'(?<![A-Za-z0-9_])(?=[A-Za-z\d-]*\d)(?=[A-Za-z\d-]*[A-Za-z])[A-Za-z\d]+(?:-[A-Za-z\d]+)*(?![A-Za-z0-9_])')


def _alnum(m, lang):
    tok = m.group(0)
    seat = re.fullmatch(r'(\d{1,2})([A-Za-z])', tok)      # 24A -> twenty-four A
    if seat:
        return f'{_card(int(seat.group(1)), lang)} {seat.group(2)}' if lang != 'zh' else _card(int(seat.group(1)), 'zh') + seat.group(2)
    return _spell_chunk(tok, lang)


YEAR_RE = re.compile(r'(?<![A-Za-z0-9_.,])(1[89]\d{2}|20\d{2})(?![A-Za-z0-9_])(?![.,]\d)')


def _year(m, lang):
    return N.year(int(m.group(1)), lang)


NUMBER_RE = re.compile(r'(?<![A-Za-z0-9_.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?![A-Za-z0-9_])(?!,\d)')


def _number(m, lang):
    whole, frac = m.group(1), m.group(2)
    if frac is not None:
        return _num_words(whole, frac, lang)
    digits = whole.replace(',', '')
    before = m.string[:m.start()]
    ctx = re.search(r'(?:\b|^)(' + _DIGIT_CONTEXT[lang] + r')\s*(?:[:#]\s*)?(?:is|ialah|adalah|number|nombor|no\.?)?\s*$',
                    before[-24:], re.IGNORECASE)
    caps = re.search(r'(?<![A-Za-z])[A-Z]{2,}\s+$', before)
    if (',' not in whole) and (
            (caps and len(digits) >= 3)                                 # HTTP 404
            or (len(digits) > 1 and digits[0] == '0')                   # 0457
            or len(digits) >= 7                                          # phone-like run
            or (ctx and (len(digits) >= 4 or re.fullmatch(_ALWAYS_DIGITS_CONTEXT, ctx.group(1), re.IGNORECASE)))):
        return _digits(digits, lang)
    n = int(digits)
    after = m.string[m.end():m.end() + 2]
    if lang == 'zh' and n == 2 and after[:1] in _ZH_MEASURE and not before.endswith('第'):
        return '两'
    if lang == 'ta' and n == 1 and re.match(r'\s[஀-௿]', m.string[m.end():m.end() + 2]):
        return 'ஒரு'
    return _card(n, lang)


# --------------------------------------------------------------------------- Chinese date/time particles
ZH_YEAR_RE = re.compile(r'(?<!\d)(\d{4})年')
ZH_MONTH_RE = re.compile(r'(?<![\d.])(\d{1,2})月')
ZH_DAY_RE = re.compile(r'(?<![\d.])(\d{1,2})(日|号)')
ZH_HOUR_RE = re.compile(r'(?<![\d.:])(\d{1,2})点')
ZH_MINUTE_RE = re.compile(r'(?<![\d.:])(\d{1,2})分(?!之)')


def _zh_particles(s):
    s = ZH_YEAR_RE.sub(lambda m: N.zh_digits(m.group(1)) + '年', s)
    s = ZH_MONTH_RE.sub(lambda m: N.zh_cardinal(int(m.group(1))) + '月', s)
    s = ZH_DAY_RE.sub(lambda m: N.zh_cardinal(int(m.group(1))) + m.group(2), s)
    s = ZH_HOUR_RE.sub(lambda m: ('两' if int(m.group(1)) == 2 else N.zh_cardinal(int(m.group(1)))) + '点', s)
    s = ZH_MINUTE_RE.sub(lambda m: ('零' if m.group(1)[0] == '0' and len(m.group(1)) == 2 else '')
                         + N.zh_cardinal(int(m.group(1))) + '分', s)
    return s


# --------------------------------------------------------------------------- abbreviations
# Only what the LLM expands. It leaves Mr./Mrs./Ms./En./Pn./e.g./etc./Sdn Bhd and acronyms alone.
ABBREVIATIONS = {
    'en': [(r'\bDr\.(?=\s)', 'Doctor'), (r'\bProf\.(?=\s)', 'Professor'), (r'\bvs\.?(?=\s)', 'versus'),
           (r'\bapprox\.(?=\s)', 'approximately'), (r'\bNo\.(?=\s)', 'Number')],
    'ms': [(r'\bDr\.(?=\s)', 'Doktor'), (r'\bJln\.?(?=\s)', 'Jalan'), (r'\bNo\.(?=\s)', 'Nombor'),
           (r'\bno\.(?=\s)', 'nombor'), (r'\bcth\.', 'contohnya'), (r'\bdsb\.', 'dan sebagainya'),
           (r'\bdll\.', 'dan lain-lain'), (r'\bTmn\.?(?=\s)', 'Taman'), (r'\bKg\.?(?=\s[A-Z])', 'Kampung')],
    'zh': [],
    'ta': [],
}
_ABBR_COMPILED = {lang: [(re.compile(p), r) for p, r in rules] for lang, rules in ABBREVIATIONS.items()}


# --------------------------------------------------------------------------- pipeline
def _sub(pattern, handler, s, lang):
    return pattern.sub(lambda m: handler(m, lang), s)


PIPELINE = [
    (EMAIL_RE, _email), (URL_RE, _url), (IC_RE, _ic), (DATE_NUM_RE, _date_numeric),
    (DATE_DMY_RE, _date_dmy), (DATE_MDY_RE, _date_mdy), (PHONE_RE, _phone),
    (TIME_RANGE_RE, _time_range), (TIME_AMPM_RE, _time_ampm), (TIME_24_RE, _time_24),
    (MONEY_RE, _money), (PERCENT_RE, _percent), (UNIT_RE, _unit),
    (ORD_EN_RE, _ord_en), (ORD_MS_RE, _ord_ms), (ORD_ZH_RE, _ord_zh), (ORD_TA_RE, _ord_ta),
    (RANGE_RE, _range), (HASH_RE, _hash), (ALNUM_RE, _alnum), (YEAR_RE, _year), (NUMBER_RE, _number),
]


def normalize(text, lang=None):
    """Spoken form of `text`. lang: 'en' | 'ms' | 'zh' | 'ta', detected from the text when None."""
    if not text or not re.search(r'\d|[@#%°℃℉]|www\.|https?://', text):
        return _abbreviations(text or '', lang or detect_lang(text or ''))
    lang = lang or detect_lang(text)
    s = text
    if lang == 'zh':
        s = _zh_particles(s)
    for pattern, handler in PIPELINE:
        if handler is _ord_en and lang != 'en':
            continue
        s = _sub(pattern, handler, s, lang)
    s = _abbreviations(s, lang)
    return re.sub(r'[ \t]{2,}', ' ', s).strip()


def _abbreviations(s, lang):
    for pattern, repl in _ABBR_COMPILED.get(lang, []):
        s = pattern.sub(repl, s)
    return s
