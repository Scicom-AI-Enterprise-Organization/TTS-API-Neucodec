"""
Rule-based spoken-form normalizer: the LLM normalizer's contract (app/prompt.py), as regexes.

Rewrites what is not speakable as written -- numbers, money, IC/phone numbers, dates, times,
percentages, decimals, fractions, ordinals, units, emails, URLs, ids, a few abbreviations -- into
words, in the language of the sentence (see lang.py), and leaves every other word exactly as
written. Each handler is a regex with a callback; they run in a fixed order so that the more
specific shapes (email, IC, phone, date, time, money, ...) claim their digits before the generic
number rule sees what is left. Nothing here needs torch or a model; ~50 us per sentence.

Conventions follow bench/results/normalizer_truth.jsonl (the live LLM's outputs on
bench/normalizer_corpus.py), except where the LLM was inconsistent or wrong; those choices are
called out inline. Score it with bench/normalizer_agreement.py.
"""
import re

from . import numbers as N
from .lang import detect_lang, latin_scores, local_lang

# --------------------------------------------------------------------------- vocabulary
V = {
    'en': dict(ringgit='ringgit', sen='sen', dollar=('dollar', 'dollars'), usd='US dollars', sgd='Singapore dollars',
               euro=('euro', 'euros'), pound=('pound', 'pounds'), percent='percent', plus='plus', to='to',
               at='at', dot='dot', am='a m', pm='p m', oclock="o'clock", number='number', hash='number',
               million='million', billion='billion', thousand='thousand', minus='minus', per='per',
               slash='slash', times='times', point='point', rupee=('rupee', 'rupees'), peso=('peso', 'pesos')),
    'ms': dict(ringgit='ringgit', sen='sen', dollar='dolar', usd='dolar Amerika', sgd='dolar Singapura',
               euro='euro', pound='paun', percent='peratus', plus='tambah', to='hingga',
               at='at', dot='dot', am='pagi', pm='petang', oclock='', number='nombor', hash='nombor',
               million='juta', billion='bilion', thousand='ribu', minus='negatif', per='per',
               slash='per', times='kali', point='perpuluhan', rupee='rupee', peso='peso'),
    'zh': dict(ringgit='令吉', sen='仙', dollar='美元', usd='美元', sgd='新元', euro='欧元', pound='英镑',
               percent='百分之', plus='加', to='到', at='at', dot='dot', am='上午', pm='下午', oclock='点',
               number='号', hash='号', million='百万', billion='十亿', thousand='千', minus='负', per='每',
               slash='斜杠', times='倍', point='点', rupee='卢比', peso='比索'),
    'ta': dict(ringgit='ரிங்கிட்', sen='சென்', dollar='டாலர்', usd='அமெரிக்க டாலர்', sgd='சிங்கப்பூர் டாலர்',
               euro='யூரோ', pound='பவுண்ட்', percent='சதவீதம்', plus='பிளஸ்', to='முதல்',
               at='at', dot='dot', am='ஏ எம்', pm='பி எம்', oclock='', number='எண்', hash='எண்',
               million='மில்லியன்', billion='பில்லியன்', thousand='ஆயிரம்', minus='மைனஸ்', per='ஒரு',
               slash='ஸ்லாஷ்', times='மடங்கு', point='புள்ளி', rupee='ரூபாய்', peso='பெசோ'),
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
    '°c': {'en': ('degree Celsius', 'degrees Celsius'), 'ms': 'darjah Celsius', 'zh': '摄氏度', 'ta': 'டிகிரி செல்சியஸ்'},
    '°f': {'en': ('degree Fahrenheit', 'degrees Fahrenheit'), 'ms': 'darjah Fahrenheit', 'zh': '华氏度', 'ta': 'டிகிரி பாரன்ஹீட்'},
    '°': {'en': ('degree', 'degrees'), 'ms': 'darjah', 'zh': '度', 'ta': 'டிகிரி'},
    # abbreviations the LLM expands after a number ("6 hrs", "10 min", "45 sec", "500W", "12V", "1,200 sq ft")
    'hr': {'en': ('hour', 'hours'), 'ms': 'jam', 'zh': '小时', 'ta': 'மணி நேரம்'},
    'min': {'en': ('minute', 'minutes'), 'ms': 'minit', 'zh': '分钟', 'ta': 'நிமிடம்'},
    'sec': {'en': ('second', 'seconds'), 'ms': 'saat', 'zh': '秒', 'ta': 'வினாடி'},
    'ms': {'en': ('millisecond', 'milliseconds'), 'ms': 'milisaat', 'zh': '毫秒', 'ta': 'மில்லி வினாடி'},
    'yr': {'en': ('year', 'years'), 'ms': 'tahun', 'zh': '年', 'ta': 'ஆண்டு'},
    'mth': {'en': ('month', 'months'), 'ms': 'bulan', 'zh': '月', 'ta': 'மாதம்'},
    'w': {'en': ('watt', 'watts'), 'ms': 'watt', 'zh': '瓦', 'ta': 'வாட்'},
    'v': {'en': ('volt', 'volts'), 'ms': 'volt', 'zh': '伏', 'ta': 'வோல்ட்'},
    'hz': {'en': 'hertz', 'ms': 'hertz', 'zh': '赫兹', 'ta': 'ஹெர்ட்ஸ்'},
    'khz': {'en': 'kilohertz', 'ms': 'kilohertz', 'zh': '千赫', 'ta': 'கிலோஹெர்ட்ஸ்'},
    'mhz': {'en': 'megahertz', 'ms': 'megahertz', 'zh': '兆赫', 'ta': 'மெகாஹெர்ட்ஸ்'},
    'ghz': {'en': 'gigahertz', 'ms': 'gigahertz', 'zh': '千兆赫', 'ta': 'ஜிகாஹெர்ட்ஸ்'},
    'sqft': {'en': ('square foot', 'square feet'), 'ms': 'kaki persegi', 'zh': '平方英尺', 'ta': 'சதுர அடி'},
    'sqm': {'en': ('square meter', 'square meters'), 'ms': 'meter persegi', 'zh': '平方米', 'ta': 'சதுர மீட்டர்'},
    'ft': {'en': ('foot', 'feet'), 'ms': 'kaki', 'zh': '英尺', 'ta': 'அடி'},
    'mph': {'en': 'miles per hour', 'ms': 'batu sejam', 'zh': '英里每小时', 'ta': 'மைல் ஒரு மணி நேரத்திற்கு'},
    'lb': {'en': ('pound', 'pounds'), 'ms': 'paun', 'zh': '磅', 'ta': 'பவுண்டு'},
    'oz': {'en': ('ounce', 'ounces'), 'ms': 'auns', 'zh': '盎司', 'ta': 'அவுன்ஸ்'},
}
_UNIT_ALIASES = {'kgs': 'kg', 'kilo': 'kg', 'kilos': 'kg', '℃': '°c', '℉': '°f', 'ltr': 'l', 'litre': 'l',
                 'hrs': 'hr', 'mins': 'min', 'secs': 'sec', 'yrs': 'yr', 'mths': 'mth', 'lbs': 'lb',
                 'sqft': 'sqft', 'sq ft': 'sqft', 'sq.ft': 'sqft', 'sq. ft': 'sqft', 'sq.ft.': 'sqft',
                 'sqm': 'sqm', 'sq m': 'sqm', 'sq.m': 'sqm'}
_UNIT_ALT = (r'°\s?[CcFf]|℃|℉|°|kgs|kg|mg|g|km|cm|mm|m|ml|ltr|litre|l|gb|mbps|mb|tb|kb|kwh|khz|mhz|ghz|hz|'
             r'hrs|hr|mins|min|secs|sec|ms|yrs|yr|mths|mth|w|v|sq\.?\s?ft\.?|sqft|sq\.?\s?m|sqm|ft|mph|lbs|lb|oz')

# "/unit" after a quantity: RM5/kg, 110 km/h, 8 tablets/day, RM38/mth
_PER_UNITS = {
    'kg': ('kilogram', 'kilogram', '公斤', 'கிலோகிராமுக்கு'), 'g': ('gram', 'gram', '克', 'கிராமுக்கு'),
    'km': ('kilometer', 'kilometer', '公里', 'கிலோமீட்டருக்கு'), 'm': ('meter', 'meter', '米', 'மீட்டருக்கு'),
    'l': ('liter', 'liter', '升', 'லிட்டருக்கு'), 'h': ('hour', 'jam', '小时', 'மணி நேரத்திற்கு'),
    'day': ('day', 'hari', '天', 'நாளுக்கு'), 'week': ('week', 'minggu', '周', 'வாரத்திற்கு'),
    'month': ('month', 'bulan', '月', 'மாதத்திற்கு'), 'year': ('year', 'tahun', '年', 'ஆண்டுக்கு'),
    'min': ('minute', 'minit', '分钟', 'நிமிடத்திற்கு'), 's': ('second', 'saat', '秒', 'வினாடிக்கு'),
    'unit': ('unit', 'unit', '单位', 'யூனிட்டுக்கு'), 'person': ('person', 'orang', '人', 'நபருக்கு'),
    'piece': ('piece', 'keping', '件', 'துண்டுக்கு'), 'pax': ('pax', 'pax', '人', 'நபருக்கு'),
    'sqft': ('square foot', 'kaki persegi', '平方英尺', 'சதுர அடிக்கு'),
}
_PER_ALIASES = {'hr': 'h', 'hrs': 'h', 'hour': 'h', 'jam': 'h', 'j': 'h', '小时': 'h', 'மணி': 'h',
                'days': 'day', 'hari': 'day', '天': 'day', '日': 'day', 'நாள்': 'day',
                'wk': 'week', 'minggu': 'week', '周': 'week', 'வாரம்': 'week',
                'mth': 'month', 'months': 'month', 'bulan': 'month', '月': 'month', 'மாதம்': 'month',
                'yr': 'year', 'tahun': 'year', '年': 'year', 'ஆண்டு': 'year', 'வருடம்': 'year', 'annum': 'year',
                'minute': 'min', 'minit': 'min', '分钟': 'min', 'நிமிடம்': 'min',
                'sec': 's', 'saat': 's', '秒': 's', 'orang': 'person', 'seorang': 'person', '人': 'person',
                'நபர்': 'person', 'pc': 'piece', 'pcs': 'piece', 'keping': 'piece', 'litre': 'l', 'liter': 'l',
                '公斤': 'kg', 'கிலோ': 'kg', '公里': 'km', 'sq ft': 'sqft'}
_PER_ALT = '|'.join(sorted((re.escape(k) for k in list(_PER_UNITS) + list(_PER_ALIASES)), key=len, reverse=True))

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

_DAYS_EN = {'mon': 'Monday', 'tue': 'Tuesday', 'tues': 'Tuesday', 'wed': 'Wednesday', 'thu': 'Thursday',
            'thur': 'Thursday', 'thurs': 'Thursday', 'fri': 'Friday', 'sat': 'Saturday', 'sun': 'Sunday'}

# Words after which a following number is read digit by digit ("Room 305", "PIN 0457"). The
# LLM reads room and PIN numbers as digits, order/invoice/postcode/verification-code numbers as
# digits, but house numbers after "No." ("No. 12") and short codes as cardinals -- hence the
# length rule in `_number`. Chinese reads room numbers as cardinals (三百零五号), so its list is short.
_DIGIT_CONTEXT = {
    'en': r'room|rooms|no\.?|number|order|pin|ref|reference|id|invoice|account|acc|acct|flight|ext|extension|lot|policy|ticket|case|'
          r'code|passcode|otp|verification|postcode|postal|zip|serial|model|plate|dial|press|unit|boeing',
    'ms': r'bilik|no\.?|nombor|pesanan|pin|rujukan|invois|akaun|penerbangan|lot|blok|polisi|tiket|kes|id|order|'
          r'kod|poskod|siri|model|plat|sambungan|tekan|unit|otp',
    'zh': r'密码|编号|号码|订单|参考|账号|工单|验证码|邮编|邮政编码|序列号|型号|车牌|分机|OTP',
    'ta': r'அறை|no\.?|எண்|ஆர்டர்|pin|குறிப்பு|order|invoice|id|குறியீடு|அஞ்சல்|சீரியல்|மாடல்|நீட்டிப்பு|otp',
}
_ALWAYS_DIGITS_CONTEXT = r'room|rooms|bilik|pin|extension|ext|sambungan|அறை|密码|dial|tekan|press|postcode|poskod|zip|邮编|邮政编码|serial|siri|分机|boeing'
# short service numbers read digit by digit when the sentence is about calling ("Dial 100 ... or 999")
_EMERGENCY = {'999', '911', '112', '994', '991', '995', '108'}
_DIAL_WORDS = re.compile(r'dial|call|hubungi|dail|telefon|拨打|拨|அழைக்க|அழை', re.IGNORECASE)

# Chinese: 2 before a measure word is 两 (两箱, 两点), not 二 -- except after 第 (第二次).
_ZH_MEASURE = '个箱件位次点天人张份块条台辆间只杯瓶本层种批小周组套页颗粒根支元'

# "3-1" after these words is a score ("three one" / 三比一), not a range
_SCORE_NOUNS = r'(?<![A-Za-z])(?:score|scores|skor|keputusan|ஸ்கோர்)(?![A-Za-z])|比分|结果'
_SCORE_CTX = r'score|scores|scored|skor|result|keputusan|比分|结果|ஸ்கோர்|won|beat|lost|defeated|menang|kalah|tewas|mengalahkan|draw|seri|leading|lead|led'
# "Jalan 3/14", "Lot 5/2": a slash in an address is said, not read as a fraction
_ROAD_CTX = r'jalan|jln|lorong|lrg|lot|blok|block|unit|seksyen|section|persiaran|taman|tmn|kampung|kg|no\.?|plot|phase|fasa|ജ'
_TWENTY_FOUR_SEVEN = {'en': 'twenty-four seven', 'ms': 'dua puluh empat jam tujuh hari seminggu',
                      'zh': '二十四小时七天', 'ta': 'இருபத்து நான்கு மணி நேரம் ஏழு நாட்கள்'}

# Tamil: a case suffix written straight after a digit (2024ல், 12ஆல், RM1,000ஐ) is glued onto the
# number word with sandhi. Handlers append MARK when a Tamil letter follows the match; _ta_sandhi
# resolves it at the end (numbers.ta_attach).
MARK = ''
_TA_LETTER_RE = re.compile(r'[அ-ஔக-ஹ]')
_TA_SUFFIX_RE = re.compile(r'(\S+)' + MARK
                           + r'(இலிருந்து|லிருந்து|ஆகும்|ாகும்|க்கும்|க்குள்|க்கான|க்கு|ஆக|ாக|ஆல்|ால்|இல்|ல்|ஐ|ை|உம்|ும்|'
                             r'உடன்|ுடன்|ஓடு|ோடு|ஆய்|ாய்|ஆவது|ாவது|வது)(?![஀-௿])')


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


def _unit_key(unit):
    unit = unit.lower().replace(' ', '')
    unit = re.sub(r'^sq\.?ft\.?$', 'sqft', unit)
    unit = re.sub(r'^sq\.?m$', 'sqm', unit)
    return _UNIT_ALIASES.get(unit, unit)


def _unit_words(unit, lang, plural):
    u = UNITS[_UNIT_ALIASES.get(unit, unit)][lang]
    if isinstance(u, tuple):
        return u[1] if plural else u[0]
    return u


def _join(parts, lang):
    parts = [p for p in parts if p]
    return ''.join(parts) if lang == 'zh' else ' '.join(parts)


def _split_amt(s):
    """'1,250.50' -> ('1250', '50'); '3' -> ('3', None)."""
    whole, _, frac = s.replace(',', '').partition('.')
    return whole, (frac or None)


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
    """Letter/digit runs of an id token: letter runs kept (MH, ABC), digit runs of one or two
    digits as a cardinal ("3A-12-3" -> "three A twelve three", "COVID-19" -> "COVID nineteen"),
    round thousands as a cardinal ("XR-2000" -> "XR two thousand"), longer runs digit by digit
    ("MH370" -> "MH three seven zero", "X-12340567" -> "X one two ..."). Chinese reads the short
    runs digit by digit too (三 A 一二 三), as the LLM does."""
    out = []
    for run in re.findall(r'\d+|[A-Za-z]+|[^A-Za-z\d]+', chunk):
        if run.isdigit():
            if re.fullmatch(r'[1-9]\d?000', run):
                out.append(_card(int(run), lang))
            elif len(run) <= 2 and lang != 'zh':
                out.append(_card(int(run), lang))
            else:
                out.append(_digits(run, lang))
        elif run.isalpha():
            out.append(run)
        # separators (-, /) are dropped: they were only glue
    return _join(out, lang)


def _ctx_before(before, words, tokens=3):
    """Is one of `words` within the last `tokens` words before the match (no clause break)?"""
    return re.search(r'(?:^|(?<![A-Za-z0-9_]))(' + words + r')(?![A-Za-z0-9_])'
                     r'(?:\s*[:#]?\s*[^\s\d,.;!?]+){0,%d}\s*$' % tokens, before[-60:], re.IGNORECASE)


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


# a minus sign glued to a number / amount: -5°C, -RM250, -3.5% (a dash with a space after it is a range)
NEG_RE = re.compile(r'(?:^|(?<=[\s(:=一-鿿]))[-−](?=(?:RM|MYR|USD|US\$|SGD|S\$|AUD|GBP|EUR|\$|€|£)?\d)')


def _neg(m, lang):
    return V[lang]['minus'] + ('' if lang == 'zh' else ' ')


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
    if re.fullmatch(r'[1-9]\d{0,5}000', s):                       # 1000000 is a million, not a phone number
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


_FULL_MONTHS = {m.lower() for lang_ in ('en', 'ms', 'ta') for m in MONTHS[lang_] if m}


def _month_name_out(written, lang):
    """The month as the sentence should say it. An abbreviation ("Jan", "Sept") becomes the
    full name in the sentence language; a full name in another language is kept as written
    (the LLM says "tiga puluh June" inside Malay) -- except in English, where the Malay
    names (Mac, Jun, Ogos, Disember) become the English ones."""
    key = written.lower().rstrip('.')
    num = _MONTH_NUM.get(key)
    if num is None:
        return written, None
    if lang in MONTHS and (key not in _FULL_MONTHS or (lang == 'en' and key != MONTHS['en'][num].lower())):
        return MONTHS[lang][num], num
    return written, num


def _date_dmy(m, lang):
    d, month, y = int(m.group(1)), m.group(2), m.group(3)
    name, num = _month_name_out(month, lang)
    if not (1 <= d <= 31) or num is None:
        return m.group(0)
    # the optional dot is for "Sept."; after a full month name it is the sentence's own period
    tail = '.' if (m.group(0).endswith('.') and not y and month.lower() in _FULL_MONTHS) else ''
    if lang == 'en':
        return f'the {N.en_ordinal(d)} of {name}' + (' ' + N.year(y, 'en') if y else '') + tail
    if lang == 'zh':
        return _date_words(d, num, int(y) if y else None, 'zh') + tail
    return f'{_card(d, lang)} {name}' + (' ' + N.year(y, lang) if y else '') + tail


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
_AMPM = r'([AaPp])(?:\.\s?[Mm]\.?|\s?[Mm])(?![A-Za-z])'
TIME_AMPM_RE = re.compile(r'(?<![\d:.])(\d{1,2})(?:[:.](\d{2}))?\s?' + _AMPM)
TIME_24_RE = re.compile(r'(?<![\d:.])(\d{1,2}):(\d{2})(?![\d:])')
TIME_RANGE_RE = re.compile(
    r'(?<![\d:.])(\d{1,2})(?:[:.](\d{2}))?\s?(?:' + _AMPM + r')?\s*(?:-|–|—|to|hingga|ke|到|至)\s*'
    r'(\d{1,2})(?:[:.](\d{2}))?\s?' + _AMPM)
TIME24_RANGE_RE = re.compile(r'(?<![\d:.])(\d{1,2}):(\d{2})\s?[-–—]\s?(\d{1,2}):(\d{2})(?![\d:])')
# Malay: 9 pagi - 5 petang | 9.30 pagi hingga 5 petang
_MS_PERIOD = r'pagi|petang|malam|tengah\s?hari|tengahari'
MS_TIME_RANGE_RE = re.compile(
    r'(?<![\d:.])(\d{1,2})(?:[.:](\d{2}))?\s?(' + _MS_PERIOD + r')\s*(?:-|–|—|hingga|ke|sampai)\s*'
    r'(\d{1,2})(?:[.:](\d{2}))?\s?(' + _MS_PERIOD + r')(?![A-Za-z])', re.IGNORECASE)
# 3.30 read as a time only next to a time cue: "pukul 3.30", "at 6.30", "3.30 petang", "5.15 மணிக்கு"
TIME_DOT_RE = re.compile(r'(?<![\d.:])(\d{1,2})\.([0-5]\d)(?![\d.:])')
_TIME_CUE_BEFORE = re.compile(r'(?:^|(?<=[\s(]))(?:at|by|until|till|from|before|after|pukul|jam|pkl|sebelum|selepas|'
                              r'காலை|மாலை|இரவு|மதியம்|பிற்பகல்|முற்பகல்|நண்பகல்)\s*$', re.IGNORECASE)
_TIME_CUE_AFTER = re.compile(r'^\s?(?:' + _MS_PERIOD + r"|மணி|மணிக்கு|noon|midnight|hrs|hours|o'clock)(?![A-Za-z])", re.IGNORECASE)
_TIME_NOT_AFTER = re.compile(r'^\s?(?:%|％|per\b|percent|peratus|million|billion|juta|bilion|ribu|k\b|m\b|kg|km|cm|mm|ml|gb|mb|'
                             r'degrees?|darjah|°|out of|times|x\b|rate|kadar)', re.IGNORECASE)
_MS_CUE = re.compile(r'pukul|jam|pkl|sebelum|selepas|' + _MS_PERIOD, re.IGNORECASE)
_TA_CUE = re.compile(r'[஀-௿]')
# 0730 hours | 1900 hrs | pukul 0730 | 0730 மணிக்கு
MIL_RE = re.compile(r'(?<![\d.,:])([01]\d|2[0-3])([0-5]\d)(?![\d.,:])')
_MIL_AFTER = re.compile(r'^\s?(?:hours|hrs|h|மணிக்கு|மணி)(?![A-Za-z])', re.IGNORECASE)
_MIL_BEFORE = re.compile(r'(?:^|(?<=\s))(?:pukul|jam|pkl)\s*$', re.IGNORECASE)
_ZH_PERIOD_RE = re.compile(r'(上午|下午|中午|晚上|早上|凌晨|傍晚|夜里)\s*$')
_NOON_RE = re.compile(r'\s?(noon|midnight)(?![A-Za-z])', re.IGNORECASE)


def _time_words(h, mm, ampm, lang, before='', oclock=True):
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
    elif lang == 'en' and oclock and (mm == '00' or mm is None):
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
    oclock = not _NOON_RE.match(m.string[m.end():])        # "12:00 noon" -> twelve noon
    return _time_words(h, mm, None, lang, m.string[:m.start()][-6:], oclock)


def _time_dot(m, lang):
    h, mm = int(m.group(1)), m.group(2)
    if h > 24 or lang == 'zh':
        return m.group(0)
    before, after = m.string[:m.start()], m.string[m.end():]
    cue = _TIME_CUE_BEFORE.search(before[-16:]) or _TIME_CUE_AFTER.match(after)
    if not cue or _TIME_NOT_AFTER.match(after):
        return m.group(0)
    if lang in ('en', 'ms'):                    # "pukul 2.30 petang" inside English is still Malay
        lang = 'ms' if _MS_CUE.search(cue.group(0)) else 'en'
    return _time_words(h, mm, None, lang, before[-6:], oclock=not _NOON_RE.match(after))


def _time_range(m, lang):
    h1, m1, ap1, h2, m2, ap2 = m.groups()
    if int(h1) > 24 or int(h2) > 24:
        return m.group(0)
    a = _time_words(int(h1), m1, ap1, lang, m.string[:m.start()][-6:], oclock=False)
    b = _time_words(int(h2), m2, ap2, lang)
    if lang == 'zh':
        return a + V['zh']['to'] + b
    if lang == 'ta':
        return f'{a} {V["ta"]["to"]} {b} வரை'
    return f'{a} {V[lang]["to"]} {b}'


def _time24_range(m, lang):
    h1, m1, h2, m2 = m.groups()
    if int(h1) > 24 or int(h2) > 24 or int(m1) > 59 or int(m2) > 59:
        return m.group(0)
    a = _time_words(int(h1), m1, None, lang, m.string[:m.start()][-6:], oclock=False)
    b = _time_words(int(h2), m2, None, lang, oclock=False)
    if lang == 'zh':
        return a + V['zh']['to'] + b
    return f'{a} {V[lang]["to"]} {b}'


def _ms_time_range(m, lang):
    h1, m1, p1, h2, m2, p2 = m.groups()
    if int(h1) > 24 or int(h2) > 24:
        return m.group(0)
    lang = 'ms' if lang in ('en', 'ms') else lang       # pagi/petang make it Malay whatever the vote said
    a = _time_words(int(h1), m1, None, lang, oclock=False)
    b = _time_words(int(h2), m2, None, lang, oclock=False)
    return f'{a} {p1} {V[lang]["to"]} {b} {p2}'


def _military(m, lang):
    hh, mm = m.group(1), m.group(2)
    if not (_MIL_AFTER.match(m.string[m.end():]) or _MIL_BEFORE.search(m.string[:m.start()][-8:])):
        return m.group(0)
    if lang == 'zh':
        return N.zh_digits(hh + mm)
    hour = _digits(hh, lang) if hh[0] == '0' else _card(int(hh), lang)   # zero seven / kosong tujuh
    if mm == '00':
        minutes = 'hundred' if lang == 'en' else _digits('00', lang)       # nineteen hundred / sembilan belas kosong kosong
    elif mm[0] == '0':
        minutes = _digits(mm, lang)
    else:
        minutes = _card(int(mm), lang)
    return f'{hour} {minutes}'


# 17.4.1 | 3.12.4 : first field a cardinal, the rest digit by digit, as the LLM reads versions
VERSION_RE = re.compile(r'(?<![A-Za-z0-9_.])(\d+(?:\.\d+){2,})(?![A-Za-z0-9_])(?!\.\d)')


def _version(m, lang):
    segs = m.group(1).split('.')
    words = [_card(int(segs[0]), lang)] + [_digits(s, lang) for s in segs[1:]]
    point = V[lang]['point']
    return point.join(words) if lang == 'zh' else f' {point} '.join(words)


# RM1,250.50 | RM 3,400 | $25 | USD 10 | RM1.2 million | RM50k
_CUR = r'RM|MYR|USD|US\$|SGD|S\$|AUD|GBP|EUR|LKR|Rs\.?|ரூ\.?|₨|PHP|Php|₱|\$|€|£'
_AMT = r'\d{1,3}(?:,\d{3})+|\d+'
MONEY_RE = re.compile(
    r'(?<![A-Za-z0-9_])(' + _CUR + r')\s?(' + _AMT + r')(?:\.(\d{1,2}))?'
    r'(?:\s?(million|juta|billion|bilion|k|m|bn|百万|万|亿|மில்லியன்|பில்லியன்)(?![A-Za-z0-9_]))?(?![A-Za-z0-9_])(?!\.\d)', re.IGNORECASE)
_CURRENCY_KEY = {'rm': 'ringgit', 'myr': 'ringgit', 'usd': 'usd', 'us$': 'usd', '$': 'dollar', 'sgd': 'sgd',
                 's$': 'sgd', 'aud': 'dollar', 'gbp': 'pound', '£': 'pound', 'eur': 'euro', '€': 'euro',
                 'lkr': 'rupee', 'rs': 'rupee', 'rs.': 'rupee', 'ரூ': 'rupee', 'ரூ.': 'rupee', '₨': 'rupee',
                 'php': 'peso', '₱': 'peso'}
# minor unit of the non-ringgit currencies that have one (Sri Lankan cents are சதம் in Tamil)
_MINOR = {'rupee': {'en': 'cents', 'ms': 'sen', 'zh': '分', 'ta': 'சதம்'}, 'peso': {'en': 'centavos', 'ms': 'sen', 'zh': '分', 'ta': 'சென்'}}
_MULT_VALUE = {'million': 10 ** 6, 'juta': 10 ** 6, 'm': 10 ** 6, 'billion': 10 ** 9, 'bilion': 10 ** 9, 'bn': 10 ** 9,
               'k': 1000, '百万': 10 ** 6, '万': 10 ** 4, '亿': 10 ** 8, 'மில்லியன்': 10 ** 6, 'பில்லியன்': 10 ** 9}
_MULT_KEY = {'million': 'million', 'juta': 'million', 'm': 'million', 'மில்லியன்': 'million',
             'billion': 'billion', 'bilion': 'billion', 'bn': 'billion', 'பில்லியன்': 'billion', 'k': 'thousand'}


def money_words(cur, whole, cents, mult, lang):
    """RM1,250.50 -> 'one thousand two hundred fifty ringgit fifty sen'. RM12.5k is computed
    ('twelve thousand five hundred ringgit', as the LLM does); RM1.2 million stays 'one point two
    million ringgit'."""
    cur, whole = cur.lower(), whole.replace(',', '')
    key = _CURRENCY_KEY[cur]
    if mult and mult.lower() == 'k':
        value = float(whole + ('.' + cents if cents else '')) * 1000
        if value == int(value):
            whole, cents, mult = str(int(value)), None, None
    if key == 'ringgit':
        major, minor = V[lang]['ringgit'], V[lang]['sen']
    elif key in _MINOR:
        minor = _MINOR[key][lang]
        amount_is_one = whole == '1' and not cents and not mult
        major = _vocab(lang, key, plural=not amount_is_one)
    else:
        minor = None
        amount_is_one = whole == '1' and not cents and not mult
        major = _vocab(lang, key, plural=not amount_is_one)
    if mult:
        mv = _MULT_VALUE[mult.lower()]
        if lang == 'zh':
            value = int(round(float(whole + ('.' + cents if cents else '')) * mv))
            return _card(value, 'zh') + major
        mword = V[lang][_MULT_KEY[mult.lower()]]
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


def _money(m, lang):
    return money_words(m.group(1), m.group(2), m.group(3), m.group(4), lang)


PERCENT_RE = re.compile(r'(?<![A-Za-z0-9_.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s?(%|％|percent|peratus)(?![A-Za-z0-9_])', re.IGNORECASE)


def percent_words(whole, frac, lang):
    words = _num_words(whole, frac, lang)
    if lang == 'zh':
        return V['zh']['percent'] + words
    return f'{words} {V[lang]["percent"]}'


def _percent(m, lang):
    return percent_words(m.group(1), m.group(2), lang)


UNIT_RE = re.compile(
    r'(?<![A-Za-z0-9_.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s?(' + _UNIT_ALT + r')(?![A-Za-z0-9_])',
    re.IGNORECASE)


def unit_words(whole, frac, unit, lang, before=''):
    unit = _unit_key(unit)
    words = _num_words(whole, frac, lang)
    plural = not (whole.replace(',', '') == '1' and not frac)
    u = _unit_words(unit, lang, plural)
    if lang == 'zh':
        return words + u
    return f'{words} {u}'


def _unit(m, lang):
    return unit_words(m.group(1), m.group(2), m.group(3), lang, m.string[:m.start()])


# 10-15% | RM50-RM100 | RM50-100 | 0-100 km/h | 5-10kg : the suffix/prefix applies to both ends
_RAMT = r'\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?'
RANGE_PCT_RE = re.compile(r'(?<![A-Za-z0-9_.,])(' + _RAMT + r')\s?[-–—]\s?(' + _RAMT + r')\s?(%|％|percent|peratus)(?![A-Za-z0-9_])',
                          re.IGNORECASE)
RANGE_MONEY_RE = re.compile(r'(?<![A-Za-z0-9_])(' + _CUR + r')\s?(' + _RAMT + r')\s?[-–—]\s?(?:(' + _CUR + r')\s?)?(' + _RAMT + r')(?![A-Za-z0-9_])(?!\.\d)',
                            re.IGNORECASE)
RANGE_UNIT_RE = re.compile(r'(?<![A-Za-z0-9_.,])(' + _RAMT + r')\s?[-–—]\s?(' + _RAMT + r')\s?(' + _UNIT_ALT + r')(?![A-Za-z0-9_])',
                           re.IGNORECASE)


def _range_join(a, b, lang):
    if lang == 'zh':
        return a + V['zh']['to'] + b
    return f'{a} {V[lang]["to"]} {b}'


def _range_pct(m, lang):
    a, b = _split_amt(m.group(1)), _split_amt(m.group(2))
    if lang == 'zh':                                        # 百分之十到百分之十五, as the LLM repeats it
        return _range_join(percent_words(*a, 'zh'), percent_words(*b, 'zh'), 'zh')
    return _range_join(_num_words(*a, lang), percent_words(*b, lang), lang)


def _range_money(m, lang):
    cur1, a, cur2, b = m.groups()
    a, b = _split_amt(a), _split_amt(b)
    return _range_join(money_words(cur1, a[0], a[1], None, lang), money_words(cur2 or cur1, b[0], b[1], None, lang), lang)


def _range_unit(m, lang):
    a, b, unit = _split_amt(m.group(1)), _split_amt(m.group(2)), m.group(3)
    return _range_join(_num_words(*a, lang), unit_words(b[0], b[1], unit, lang), lang)


# "/kg", "/h", "/month" after a quantity that the earlier handlers already spelled out
PER_RE = re.compile(r'(?<=[A-Za-z一-鿿஀-௿])\s?/\s?(' + _PER_ALT + r')(?![A-Za-z0-9_])', re.IGNORECASE)


_PER_EN = {'day', 'days', 'week', 'wk', 'month', 'mth', 'months', 'year', 'yr', 'hour', 'hr', 'hrs', 'minute', 'person', 'piece', 'pc', 'pcs', 'annum'}
_PER_MS = {'hari', 'minggu', 'bulan', 'tahun', 'jam', 'j', 'minit', 'saat', 'orang', 'seorang', 'keping'}


def _per(m, lang):
    key = m.group(1).lower()
    if lang in ('en', 'ms'):
        lang = 'en' if key in _PER_EN else 'ms' if key in _PER_MS else lang
    key = _PER_ALIASES.get(key, key)
    words = _PER_UNITS[key][('en', 'ms', 'zh', 'ta').index(lang)]
    if lang == 'zh':
        return V['zh']['per'] + words
    return f' {V[lang]["per"]} {words}'


ORD_EN_RE = re.compile(r'(?<![A-Za-z0-9_.])(\d+)(st|nd|rd|th)(?![A-Za-z0-9_])')
ORD_MS_RE = re.compile(r'(?<![A-Za-z0-9_])ke-(\d+)(?![A-Za-z0-9_])')
ORD_MS_SPACE_RE = re.compile(r'(?<![A-Za-z0-9_])(kali|yang|abad|kurun|tingkat|peringkat|tahap|fasa|pusingan)\s+ke\s+(\d+)(?![A-Za-z0-9_])',
                             re.IGNORECASE)
ORD_ZH_RE = re.compile(r'第(\d+)')
ORD_TA_RE = re.compile(r'(?<![A-Za-z0-9_.])(\d+)\s?-?\s?(ஆவது|வது|ஆம்|ம்)')


def _ord_en(m, lang):
    return N.en_ordinal(int(m.group(1)))


def _ord_ms(m, lang):
    return N.ms_ordinal(int(m.group(1)))


def _ord_ms_space(m, lang):
    return f'{m.group(1)} {N.ms_ordinal(int(m.group(2)))}'


def _ord_zh(m, lang):
    return '第' + N.zh_cardinal(int(m.group(1)))


def _ord_ta(m, lang):
    suffix = 'ஆம்' if m.group(2) in ('ஆம்', 'ம்') else 'ஆவது'
    return N.ta_ordinal(int(m.group(1)), suffix)


# the 1980s | the 90s | tahun 1980-an
DECADE_RE = re.compile(r"(?<![A-Za-z0-9_])(1[89]\d0|20\d0|[1-9]0)(?:'s|s|-an)(?![A-Za-z0-9_])")


def _decade(m, lang):
    n = int(m.group(1))
    suffix = m.group(0)[len(m.group(1)):]
    if lang == 'en':
        words = N.en_year(n) if n >= 1000 else N.en_cardinal(n)
        head, _, last = words.rpartition(' ')
        plural = last[:-1] + 'ies' if last.endswith('y') else last + 's'   # eighties, thousands, tens
        return (head + ' ' if head else '') + plural
    if lang == 'ms':
        return _card(n, 'ms') + '-an'
    return _card(n, lang) + suffix


# 30-day, 24-hour, 5-star, 3-in-1, 2-for-1 : a number glued to a word keeps the hyphen
HYPHEN_RE = re.compile(r'(?<![A-Za-z0-9_])(\d+)-([A-Za-z][a-z]+(?:-[A-Za-z][a-z]+)*)(?:-(\d+))?(?![A-Za-z0-9_])')


def _hyphen(m, lang):
    a, words, c = m.groups()
    out = f'{_card(int(a), lang)}-{words}'
    if c:
        out += f'-{_card(int(c), lang)}'
    return out


# 10k, 100k, 1.5k (lower-case k: 4K is a resolution and stays "four K")
K_RE = re.compile(r'(?<![A-Za-z0-9_.])(\d+(?:\.\d+)?)\s?k(?![A-Za-z0-9_])')


def _k(m, lang):
    value = float(m.group(1)) * 1000
    if value == int(value):
        return _card(int(value), lang)
    whole, frac = m.group(1).split('.')
    return _join([N.decimal(whole, frac, lang), V[lang]['thousand']], lang)


# 2x, 3.5x : "two times" (lower-case x; "Size 2X" is not a multiplier)
X_RE = re.compile(r'(?<![A-Za-z0-9_.])(\d+(?:\.\d+)?)\s?[x×](?![A-Za-z0-9_])')


def _x(m, lang):
    whole, frac = _split_amt(m.group(1))
    if lang == 'zh':
        return ('两' if (whole == '2' and not frac) else _num_words(whole, frac, 'zh')) + V['zh']['times']
    return f'{_num_words(whole, frac, lang)} {V[lang]["times"]}'


# 12-3-5 : three or more hyphenated digit groups (an address / unit number) read digit by digit
TA_ONE_PERSON_RE = re.compile(r'(?<![\d.,])1\s+பேர்(?![஀-௿])')


def _ta_one_person(m, lang):
    return 'ஒருவர்' if lang == 'ta' else m.group(0)


GROUPS_RE = re.compile(r'(?<![A-Za-z0-9_-])(\d{1,4}(?:-\d{1,4}){2,})(?![A-Za-z0-9_-])')


def _groups(m, lang):
    return _digits(m.group(1), lang)


# 1/2 | 3/4 | 24/7 | 50/50 | 95/100 | 31/12 | Jalan 3/14
SLASH_RE = re.compile(r'(?<![A-Za-z0-9_./-])(\d{1,4})\s?/\s?(\d{1,4})(?![A-Za-z0-9_/])(?!\.\d)')
_DATE_CUE = r'on|by|before|until|till|after|dated|due|pada|sebelum|selepas|tarikh|hingga|sampai|habis|முன்|அன்று'
_EN_FRACTION = {2: ('half', 'halves'), 4: ('quarter', 'quarters')}
_TA_FRACTION = {(1, 2): 'அரை', (1, 4): 'கால்', (3, 4): 'முக்கால்'}


def _fraction(a, b, lang):
    if lang == 'en':
        if b in _EN_FRACTION:
            den = _EN_FRACTION[b][a != 1]
        elif b <= 20:
            den = N.en_ordinal(b) + ('s' if a != 1 else '')
        else:
            return f'{_card(a, "en")} out of {_card(b, "en")}'
        return f'{_card(a, "en")} {den}'
    if lang == 'ms':                                        # satu per dua, tiga per empat, satu pertiga
        return f'{_card(a, "ms")} per{"" if b == 3 else " "}{_card(b, "ms")}'
    if lang == 'zh':
        return f'{_card(b, "zh")}分之{_card(a, "zh")}'
    if (a, b) in _TA_FRACTION:
        return _TA_FRACTION[(a, b)]
    return f'{N.ta_attach(_card(b, "ta"), "இல்")} {_card(a, "ta")}'


def _slash(m, lang):
    a, b = int(m.group(1)), int(m.group(2))
    before = m.string[:m.start()]
    if (a, b) == (24, 7):
        return _TWENTY_FOUR_SEVEN[lang]
    if _ctx_before(before, _ROAD_CTX, tokens=2):
        return _join([_card(a, lang), V[lang]['slash'], _card(b, lang)], lang)
    is_date = 1 <= b <= 12 and 1 <= a <= 31
    if a < b and a >= 1 and not (is_date and _ctx_before(before, _DATE_CUE, tokens=0)):
        return _fraction(a, b, lang)
    if is_date:
        return _date_words(a, b, None, lang)
    return _join([_card(a, lang), _card(b, lang)], lang)


RANGE_RE = re.compile(r'(?<![A-Za-z0-9_.,/-])(\d{1,4})\s?[-–—]\s?(\d{1,4})(?![A-Za-z0-9_,/-])(?!\.\d)')


def _range(m, lang):
    a, b = int(m.group(1)), int(m.group(2))
    before = m.string[:m.start()]
    if _ctx_before(before, _SCORE_CTX) or (a <= 12 and b <= 12 and re.search(_SCORE_NOUNS, before, re.IGNORECASE)):
        if lang == 'zh':
            return _card(a, 'zh') + '比' + _card(b, 'zh')
        return f'{_card(a, lang)} {_card(b, lang)}'
    f = N.year if (1900 <= a <= 2099 and 1900 <= b <= 2099 and len(m.group(1)) == 4) else _card
    if lang == 'zh':
        after = m.string[m.end():m.end() + 1]
        wa = '两' if a == 2 and after in _ZH_MEASURE else f(a, 'zh')
        wb = '两' if b == 2 and after in _ZH_MEASURE else f(b, 'zh')
        return wa + V['zh']['to'] + wb
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
    hy = re.fullmatch(r'([A-Za-z]+)-(\d{1,2})', tok)      # COVID-19 -> COVID nineteen, F-16 -> F sixteen
    if hy:
        return _join([hy.group(1), _card(int(hy.group(2)), lang)], lang)
    return _spell_chunk(tok, lang)


YEAR_RE = re.compile(r'(?<![A-Za-z0-9_.,])(1[89]\d{2}|20\d{2})(?![A-Za-z0-9_])(?![.,]\d)')


def _year(m, lang):
    return N.year(int(m.group(1)), lang)


NUMBER_RE = re.compile(r'(?<![A-Za-z0-9_.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?![A-Za-z0-9_])(?!,\d)')


def _number(m, lang):
    whole, frac = m.group(1), m.group(2)
    after = m.string[m.end():m.end() + 8]
    if frac is not None:
        if lang == 'ta' and frac == '5' and re.match(r'\s?மணி', after):     # 1.5 மணி -> ஒன்றரை மணி
            return N.ta_half(int(whole.replace(',', '')))
        return _num_words(whole, frac, lang)
    digits = whole.replace(',', '')
    before = m.string[:m.start()]
    # a digit-context word within the last four tokens: "Room 305", "akaun anda ialah 1234"
    ctx = _ctx_before(before, _DIGIT_CONTEXT[lang])
    caps = re.search(r'(?<![A-Za-z])[A-Z]{2,}\s+$', before)
    round_thousand = re.fullmatch(r'[1-9]\d?000', digits)           # "the code is 1000" stays a cardinal
    if (',' not in whole) and (
            (caps and len(digits) >= 3)                                 # HTTP 404
            or (len(digits) > 1 and digits[0] == '0')                   # 0457
            or (len(digits) >= 7 and not digits.endswith('000'))        # phone-like run (1000000 is a million)
            or (ctx and not round_thousand
                and (len(digits) >= 4 or re.fullmatch(_ALWAYS_DIGITS_CONTEXT, ctx.group(1), re.IGNORECASE)))
            or (digits in _EMERGENCY and _DIAL_WORDS.search(m.string))):
        return _digits(digits, lang)
    n = int(digits)
    if lang == 'zh' and n == 2 and after[:1] in _ZH_MEASURE and not before.endswith('第'):
        return '两'
    if lang == 'ta' and n == 1 and re.match(r'\s[஀-௿]', after[:2]):
        return 'ஒரு'
    if lang == 'ms' and n == 0 and re.search(r'(?<![A-Za-z])tekan(?![A-Za-z])', m.string, re.IGNORECASE):
        return 'kosong'                                                  # "Tekan 0 untuk operator"
    return _card(n, lang)


# --------------------------------------------------------------------------- Chinese date/time particles
ZH_YEAR_RE = re.compile(r'(?<!\d)(\d{4})年')
ZH_MONTH_RE = re.compile(r'(?<![\d.])(\d{1,2})月')
ZH_DAY_RE = re.compile(r'(?<![\d.])(\d{1,2})(日|号)')
ZH_HOUR_RE = re.compile(r'(?<![\d.:])(\d{1,2})点')
ZH_MINUTE_RE = re.compile(r'(?<![\d.:])(?<!\d-)(?<!\d–)(\d{1,2})分(?!之)')


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
def _day_range(m):
    return f'{_DAYS_EN[m.group(1).lower()]} to {_DAYS_EN[m.group(2).lower()]}'


ABBREVIATIONS = {
    'en': [(r'\bDr\.(?=\s)', 'Doctor'), (r'\bProf\.(?=\s)', 'Professor'), (r'\bvs\.?(?=\s)', 'versus'),
           (r'\bapprox\.(?=\s)', 'approximately'), (r'\bNo\.(?=\s)', 'Number'),
           (r'\bhrs\b', 'hours'), (r'\bmins\b', 'minutes'),
           (r'\b(Mon|Tue|Tues|Wed|Thu|Thur|Thurs|Fri|Sat|Sun)\.?\s?[-–]\s?(Mon|Tue|Tues|Wed|Thu|Thur|Thurs|Fri|Sat|Sun)\b', _day_range)],
    'ms': [(r'\bDr\.(?=\s)', 'Doktor'), (r'\bJln\.?(?=\s)', 'Jalan'), (r'\bNo\.(?=\s)', 'Nombor'),
           (r'\bno\.(?=\s)', 'nombor'), (r'\bcth\.(?=\s)', 'contohnya'), (r'\bcth\.$', 'contohnya.'),
           (r'\bdsb\.(?=\s)', 'dan sebagainya'), (r'\bdsb\.$', 'dan sebagainya.'),
           (r'\bdll\.(?=\s)', 'dan lain-lain'), (r'\bdll\.$', 'dan lain-lain.'), (r'\bTmn\.?(?=\s)', 'Taman'), (r'\bKg\.?(?=\s[A-Z])', 'Kampung')],
    'zh': [],
    'ta': [],
}
_ABBR_COMPILED = {lang: [(re.compile(p), r) for p, r in rules] for lang, rules in ABBREVIATIONS.items()}


# --------------------------------------------------------------------------- pipeline
PIPELINE = [
    (EMAIL_RE, _email), (URL_RE, _url), (NEG_RE, _neg), (IC_RE, _ic), (DATE_NUM_RE, _date_numeric),
    (DATE_DMY_RE, _date_dmy), (DATE_MDY_RE, _date_mdy), (PHONE_RE, _phone), (MIL_RE, _military),
    (MS_TIME_RANGE_RE, _ms_time_range), (TIME_RANGE_RE, _time_range), (TIME24_RANGE_RE, _time24_range),
    (TIME_AMPM_RE, _time_ampm), (TIME_24_RE, _time_24), (TIME_DOT_RE, _time_dot), (VERSION_RE, _version),
    (RANGE_PCT_RE, _range_pct), (RANGE_MONEY_RE, _range_money), (RANGE_UNIT_RE, _range_unit),
    (MONEY_RE, _money), (PERCENT_RE, _percent), (UNIT_RE, _unit), (PER_RE, _per),
    (ORD_EN_RE, _ord_en), (ORD_MS_RE, _ord_ms), (ORD_MS_SPACE_RE, _ord_ms_space), (ORD_ZH_RE, _ord_zh), (ORD_TA_RE, _ord_ta),
    (DECADE_RE, _decade), (HYPHEN_RE, _hyphen), (K_RE, _k), (X_RE, _x), (TA_ONE_PERSON_RE, _ta_one_person), (GROUPS_RE, _groups), (SLASH_RE, _slash),
    (RANGE_RE, _range), (HASH_RE, _hash), (ALNUM_RE, _alnum), (YEAR_RE, _year), (NUMBER_RE, _number),
]


def _ta_sandhi(s):
    s = _TA_SUFFIX_RE.sub(lambda m: N.ta_attach(m.group(1), m.group(2)), s)
    return s.replace(MARK, '')


def normalize(text, lang=None):
    """Spoken form of `text`. lang: 'en' | 'ms' | 'zh' | 'ta', detected from the text when None.

    In a Malay/English code-switched sentence (markers of both present and no explicit
    lang) every number picks its own language from its neighbours (lang.local_lang)."""
    if not text or not re.search(r'\d|[@#%°℃℉]|www\.|https?://', text):
        return _abbreviations(text or '', lang or detect_lang(text or ''))
    base = lang or detect_lang(text)
    pick = None
    if lang is None and base in ('ms', 'en'):
        ms, en = latin_scores(text)
        if ms >= 1 and en >= 1:
            tie = ms == en
            pick = lambda m: local_lang(m.string, m.start(), m.end(), base, tie)   # noqa: E731
    s = text
    if base == 'zh':
        s = _zh_particles(s)
    for pattern, handler in PIPELINE:
        def repl(m, handler=handler):
            lg = base if pick is None else pick(m)
            out = handler(m, lg)
            if lg == 'ta' and out != m.group(0) and _TA_LETTER_RE.match(m.string, m.end()):
                out += MARK
            return out
        s = pattern.sub(repl, s)
    if MARK in s:
        s = _ta_sandhi(s)
    s = _abbreviations(s, base)
    if base == 'zh':
        s = re.sub(r' (?=[。，！？；：、）])', '', s)      # no space before CJK punctuation
    return re.sub(r'[ \t]{2,}', ' ', s).strip()


def _abbreviations(s, lang):
    for pattern, repl in _ABBR_COMPILED.get(lang, []):
        s = pattern.sub(repl, s)
    return s
