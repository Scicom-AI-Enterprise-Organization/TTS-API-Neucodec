"""
Regex patterns for Malaysian text normalization.
Extracted from malaya.text.regex (https://github.com/malaysia-ai/malaya)
"""

_short_date = r'\b(?:[12][0-9]{3}[-/\.](?:0?[1-9]|1[0-2])[-/\.](?:0?[1-9]|[12][0-9]|3[01])|' \
              r'(?:0?[1-9]|[12][0-9]|3[01])[-/\.](?:0?[1-9]|1[0-2])[-/\.][12][0-9]{3})\b'
_full_date_parts = [
    r'(?:(?<!:)\b\'?\d{1,4},? ?)',
    r'\b(?:[Jj]an(?:uari)?|[Ff]eb(?:ruari)?|[Mm]a(?:c)?|[Aa]pr(?:il)?|[Mm]ei|[Jj]u(?:n)?|[Jj]ula(?:i)?|[Aa]ug(?:ust)?|[Oo]gos|[Ss]ept?(?:ember)?|[Oo]kt(?:ober)?|[Nn]ov(?:ember)?|[Dd]is(?:ember)?)\b',
    r'(?:(?:,? ?\'?)?\d{1,4}(?:st|nd|rd|n?th)?\b[,\/]? ?\'?\d{2,4}[a-zA-Z]*(?: ?- ?\d{2,4}[a-zA-Z]*)?(?!:\d{1,4})\b)'
]
_fd1 = '(?:{})'.format(''.join([_full_date_parts[0] + '?', _full_date_parts[1], _full_date_parts[2]]))
_fd2 = '(?:{})'.format(''.join([_full_date_parts[0], _full_date_parts[1], _full_date_parts[2]]))

_day_month_date = r'\b[0123]?[0-9]\s+(?:[Jj]an(?:uari)?|[Ff]eb(?:ruari)?|[Mm]a(?:c)?|[Aa]pr(?:il)?|[Mm]ei|[Jj]u(?:n)?|[Jj]ula(?:i)?|[Aa]ug(?:ust)?|[Oo]gos|[Ss]ept?(?:ember)?|[Oo]kt(?:ober)?|[Nn]ov(?:ember)?|[Dd]is(?:ember)?)\b'
_month_day_date = r'\b(?:[Jj]an(?:uari)?|[Ff]eb(?:ruari)?|[Mm]a(?:c)?|[Aa]pr(?:il)?|[Mm]ei|[Jj]u(?:n)?|[Jj]ula(?:i)?|[Aa]ug(?:ust)?|[Oo]gos|[Ss]ept?(?:ember)?|[Oo]kt(?:ober)?|[Nn]ov(?:ember)?|[Dd]is(?:ember)?)\s+[0123]?[0-9]\b'
_day_month_year = (
    r'\b[0123]?[0-9]\s+'
    r'(?:[Jj]an(?:uari)?|[Ff]eb(?:ruari)?|[Mm]a(?:c)?|[Aa]pr(?:il)?|[Mm]ei|'
    r'[Jj]u(?:n)?|[Jj]ula(?:i)?|[Aa]ug(?:ust)?|[Oo]gos|[Ss]ept?(?:ember)?|'
    r'[Oo]kt(?:ober)?|[Nn]ov(?:ember)?|[Dd]is(?:ember)?)\s+'
    r'(?:\d{2,4})\b'
)
_date = '(?:' + '(?:' + _fd1 + '|' + _fd2 + ')' + '|' + _short_date + '|' + _day_month_year + '|' + _day_month_date + '|' + _month_day_date + ')'

_time = r'(?:(?:\d+)?\.?\d+\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.|pagi|pgi|morning|tengahari|tngahari|petang|ptg|malam|jam|hours|hour|hrs))|(?:(?:[0-2]?[0-9]|[2][0-3]):(?:[0-5][0-9])(?::(?:[0-5][0-9]))?(?: ?(?:AM|PM|am|pm|a\.m\.|p\.m\.|pagi|pgi|morning|tengahari|tngahari|petang|ptg|malam|hours|hrs|jam))?)'
_today_time = (
    r'(?:(?:pkul|pukul|kul)\s*'
    r'(?:[0-2]?[0-9](?::[0-5][0-9])?(?::[0-5][0-9])?|(?:\d+)?\.?\d+)'
    r'(?:\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.|pagi|pgi|morning|tengahari|tngahari|'
    r'petang|ptg|malam|hours|hrs|jam))?)'
)

_past_date_string = '(?:\\s|\\d+)\\s*(?:minggu|bulan|tahun|hari|thun|hri|mnggu|jam|minit|saat)\\s*(?:lalu|lepas|lps)\\b'
_now_date_string = '(?:sekarang|skrg|jam|tahun|thun|saat|minit) (?:ini|ni)\\b'
_yesterday_tomorrow_date_string = '(?:yesterday|semalam|kelmarin|smalam|esok|esk)\\b'
_future_date_string = '(?:dlm|dalam)\\s*\\d+(?:minggu|bulan|tahun|hari|thun|hri|mnggu|jam|minit|saat)\\b'
_depan_date_string = '(?:\\s|\\d+)\\s*(?:minggu|bulan|tahun|hari|thun|hri|mnggu|jam|minit|saat)\\s*(?:depan|dpan|dpn)\\b'

_number = r"(?<!\w)-?\d+(?:[\.,']\d+)?(?!\w)"
_number_with_shortform = r"\b(?:\d+(?:[\.,']\d+)?(?:[KkMmBbJj])|\d+(?:[\.,']\d+)?\s+(?:[Rr]ibu|[Tt]housand|[Jj]uta|[Mm]illion|[Bb]ilion|[Bb]illion))\b"
_percentage = _number + '%'
_money = r"(?:(?:[$\u20ac\u00a3\u00a2]|RM|rm)\s*\d+(?:[\.,']\d+)?\s*(?:[Rr]ibu|[Jj]uta|[Tt]housand|[Mm]illion|[MmKkBbj](?:n|(?:i(?:lion|llion)?))?)?)\b|(?:\d+(?:[\.,']\d+)?\s*(?:[MmKkBbj](?:n|(?:i(?:lion|llion)?))?|[Rr]ibu|[Jj]uta|[Tt]housand|[Mm]illion)?\s*(?:[$\u20ac\u00a3\u00a2]|sen|ringgit|cent|penny))\b"
_temperature = "-?\\d+(?:[\\.,']\\d+)?\\s*(?:K|Kelvin|kelvin|Kvin|F|f|Farenheit|farenheit|C|c|Celcius|celcius|clcius|celsius)\\b"
_distance = "-?\\d+(?:[\\.,']\\d+)?\\s*(?:kaki|mtrs|metres|meters|feet|km|m|cm|feet|feets|miles|batu|inch|inches|feets)\\b"
_volume = "-?\\d+(?:[\\.,']\\d+)?\\s*(?:ml|ML|l|L|mililiter|Mililiter|millilitre|liter|litre|litres|liters|gallon|gallons|galon)\\b"
_duration = '\\d+\\s*(?:jam|minit|hari|minggu|bulan|tahun|hours|hour|saat|second|month|months)\\b|(?:sejam|sehari|setahun|sesaat|seminit|sebulan)\\b'
_weight = "\\d+(?:[\\.,']\\d+)?\\s*(?:kg|kilo|kilogram|g|gram|KG)\\b"
_data_size = (
    r'\d+(?:\.\d+)?\s*(?:'
    r'bits?|Bits?|BITs?|bit|Bit|BIT|'
    r'bytes?|Bytes?|BYTES?|bait|Bait|BAIT|'
    r'kb|Kb|KB|kB|kilobytes?|Kilobytes?|KILOBYTES?|kilobait|Kilobait|KILOBAIT|kilobit|Kilobit|KILOBIT|'
    r'mb|Mb|MB|mB|megabytes?|Megabytes?|MEGABYTES?|megabait|Megabait|MEGABAIT|megabit|Megabit|MEGABIT|'
    r'gb|Gb|GB|gB|gigabytes?|Gigabytes?|GIGABYTES?|gigabait|Gigabait|GIGABAIT|gigabit|Gigabit|GIGABIT|'
    r'tb|Tb|TB|tB|terabytes?|Terabytes?|TERABYTES?|terabait|Terabait|TERABAIT|terabit|Terabit|TERABIT'
    r')\b'
)
_hijri_year = r'\b\d{3,4}\s*[Hh]\b'
_hari_bulan = r'\b(?:[1-9]|[12][0-9]|3[01])[Hh][Bb]\b'
_pada_tarikh = r"\b((?:pada|tarikh)\s+(?:0?[1-9]|[12][0-9]|3[01])\s(?:0?[1-9]|1[0-2]))\b"
_word_dash = r'(?:[A-Za-z0-9]+-){2,}[A-Za-z0-9]+'
_passport = r'^(?:[A-PR-WY][1-9]\d\s?\d{4}[1-9]|[A-Za-z][0-9]{8}|[A-Za-z](?=.*\d)[A-Za-z0-9]{5,19})$'

_left_datetime = '(%s) (%s)' % (_time, _date)
_right_datetime = '(%s) (%s)' % (_date, _time)
_left_datetodaytime = '(%s) (%s)' % (_today_time, _date)
_right_datetodaytime = '(%s) (%s)' % (_date, _today_time)
_left_yesterdaydatetime = '(%s) (%s)' % (_time, _yesterday_tomorrow_date_string)
_right_yesterdaydatetime = '(%s) (%s)' % (_yesterday_tomorrow_date_string, _time)
_left_yesterdaydatetodaytime = '(%s) (%s)' % (_today_time, _yesterday_tomorrow_date_string)
_right_yesterdaydatetodaytime = '(%s) (%s)' % (_yesterday_tomorrow_date_string, _today_time)

_expressions = {
    'hashtag': r'\#\b[\w\-\_]+\b',
    'cashtag': r'(?<![A-Z])\$[A-Z]+\b',
    'tag': r'<[\/]?\w+[\/]?>',
    'user': r'\@\w+',
    'emphasis': r'(?:\*\b\w+\b\*)',
    'censored': r'(?:\b\w+\*+\w+\b)',
    'acronym': r'\b(?:[A-Z]\.)(?:[A-Z]\.)+(?:\.(?!\.))?(?:[A-Z]\b)?',
    'elongated': r'\b[A-Za-z]*([a-zA-Z])\1\1[A-Za-z]*\b',
    'quotes': r'\"(\\.|[^\"]){2,}\"',
    'percent': _percentage,
    'repeat_puncts': r'([!?.]){2,}',
    'money': _money,
    'email': r'(?:^|(?<=[^\w@.)]))(?:[\w+-](?:\.(?!\.))?)*?[\w+-]@(?:\w-?)*?\w+(?:\.(?:[a-z]{2,})){1,3}(?:$|(?=\b))',
    'phone': r'(?<![0-9])(?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?![0-9])',
    'number': _number,
    'number_with_shortform': _number_with_shortform,
    'allcaps': r'(?<![#@$])\b([A-Z][A-Z ]{1,}[A-Z])\b',
    'url': r'(?:https?:\/\/(?:www\.|(?!www))[^\s\.]+\.[^\s]{2,}|www\.[^\s]+\.[^\s]{2,})',
    'url_v2': r'[(http(s)?):\/\/(www\.)?a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)',
    'url_dperini': r'^(?:(?:(?:https?|ftp):)?\/\/)(?:\S+(?::\S*)?@)?(?:(?!(?:10|127)(?:\.\d{1,3}){3})(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))|(?:(?:[a-z0-9\u00a1-\uffff][a-z0-9\u00a1-\uffff_-]{0,62})?[a-z0-9\u00a1-\uffff]\.)+(?:[a-z\u00a1-\uffff]{2,}\.?))(?::\d{2,5})?(?:[/?#]\S*)?$',
    'date': _date,
    'day_in_date': r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\b',
    'time': _time,
    'time_pukul': _today_time,
    'camel_split': r'((?<=[a-z])[A-Z]|(?<!^)[A-Z](?=[a-z])|[0-9]+|(?<=[0-9\-\_])[A-Za-z]|[\-\_])',
    'normalize_elong': r'(.)\1{2,}',
    'normalize_elong1': r'(.)\1{1,}',
    'word': r'(?:[\w_]+)',
    'hypen': r'\w+(?:-\w+)+',
    'apostrophe': r'\w+\'(?:s)?',
    'temperature': _temperature,
    'distance': _distance,
    'volume': _volume,
    'duration': _duration,
    'weight': _weight,
    'data_size': _data_size,
    'ic': r'(([[0-9]{2})(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01]))-([0-9]{2})-([0-9]{4})',
    'bracket': r'(\(.*?\))',
    'title': r'Sdn\.|Bhd\.|Corp\.|Corporation\.|corp\.|Datuk\.|datuk\.|Datin.\|datin.\|Datik\.|datik\.|dr\.|Dr\.|DR\.|yb\.|YB\.|hj\.|HJ\.|Hj\.|ybm\.|YBM\.|Ybm\.|tyt\.|TYT\.|yab\.|YAB\.|Yab\.|ybm\.|YBM\.|Ybm\.|yabhg\.|YABHG.\|Yabhg\.|ybhg\.|YBHG\.|Ybhg\.|YBhg\.|phd\.|PhD\.',
    'parliament': r'[A-Z]\.\d+',
    'hijri_year': _hijri_year,
    'hari_bulan': _hari_bulan,
    'pada_tarikh': _pada_tarikh,
    'word_dash': _word_dash,
    'passport': _passport,
}
