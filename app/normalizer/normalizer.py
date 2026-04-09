"""
Main Normalizer class for Malaysian text.
Extracted from malaya.normalizer.rules (https://github.com/malaysia-ai/malaya)
"""

import re
import itertools
import dateparser
import numpy as np
from unidecode import unidecode
from datetime import datetime
from typing import Callable, List
import logging

from .function import (
    is_laugh, is_mengeluh, multireplace, case_of, PUNCTUATION, fix_spacing,
)
from .dictionary import is_english, is_malay, is_malaysia_location
from .regex import (
    _past_date_string, _now_date_string, _future_date_string,
    _yesterday_tomorrow_date_string, _depan_date_string, _expressions,
    _left_datetime, _right_datetime, _today_time,
    _left_datetodaytime, _right_datetodaytime,
    _left_yesterdaydatetime, _right_yesterdaydatetime,
    _left_yesterdaydatetodaytime, _right_yesterdaydatetodaytime,
)
from .tatabahasa import date_replace, consonants, sounds, bulan, bulan_en
from .normalization import (
    _remove_postfix, _normalize_title, _is_number_regex, _string_to_num,
    _replace_compound, to_cardinal, to_ordinal, cardinal, digit_unit,
    rom_to_int, ordinal, fraction, money, ignore_words, digit,
    unpack_english_contractions, repeat_word, replace_laugh,
    replace_mengeluh, replace_betul, digits, normalize_numbers_with_shortform,
    parse_time_string, parse_date_string,
)
from .rules import rules_normalizer, rules_normalizer_rev
from .tokenizer import Tokenizer

logger = logging.getLogger(__name__)


def cluster_words(list_words):
    """Cluster similar words based on structure."""
    dict_words = {}
    for word in list_words:
        found = False
        for key in dict_words.keys():
            check = [word in inside for inside in dict_words[key]]
            if word in key or any(check):
                dict_words[key].append(word)
                found = True
            if key in word:
                dict_words[key].append(word)
        if not found:
            dict_words[word] = [word]
    results = []
    for key, words in dict_words.items():
        results.append(max(list(set([key] + words)), key=len))
    return list(set(results))


def normalized_entity(normalized):
    normalized = re.sub(_expressions['ic'], '', normalized)
    money_ = re.findall(_expressions['money'], normalized)
    money_ = [(s, money(s)[1]) for s in money_]
    dates_ = re.findall(_expressions['date'], normalized)

    past_date_string_ = re.findall(_past_date_string, normalized)
    now_date_string_ = re.findall(_now_date_string, normalized)
    future_date_string_ = re.findall(_future_date_string, normalized)
    yesterday_date_string_ = re.findall(_yesterday_tomorrow_date_string, normalized)
    depan_date_string_ = re.findall(_depan_date_string, normalized)
    today_time_ = re.findall(_today_time, normalized)
    time_ = re.findall(_expressions['time'], normalized)

    left_datetime_ = [f'{i[0]} {i[1]}' for i in re.findall(_left_datetime, normalized)]
    right_datetime_ = [f'{i[0]} {i[1]}' for i in re.findall(_right_datetime, normalized)]
    today_left_datetime_ = [f'{i[0]} {i[1]}' for i in re.findall(_left_datetodaytime, normalized)]
    today_right_datetime_ = [f'{i[0]} {i[1]}' for i in re.findall(_right_datetodaytime, normalized)]
    left_yesterdaydatetime_ = [f'{i[0]} {i[1]}' for i in re.findall(_left_yesterdaydatetime, normalized)]
    right_yesterdaydatetime_ = [f'{i[0]} {i[1]}' for i in re.findall(_right_yesterdaydatetime, normalized)]
    left_yesterdaydatetodaytime_ = [f'{i[0]} {i[1]}' for i in re.findall(_left_yesterdaydatetodaytime, normalized)]
    right_yesterdaydatetodaytime_ = [f'{i[0]} {i[1]}' for i in re.findall(_right_yesterdaydatetodaytime, normalized)]

    dates_ = (
        dates_ + past_date_string_ + now_date_string_ + future_date_string_
        + yesterday_date_string_ + depan_date_string_ + time_ + today_time_
        + left_datetime_ + right_datetime_ + today_left_datetime_
        + today_right_datetime_ + left_yesterdaydatetime_ + right_yesterdaydatetime_
        + left_yesterdaydatetodaytime_ + right_yesterdaydatetodaytime_
    )
    dates_ = [d.replace('.', ':') for d in dates_ if not isinstance(d, tuple)]
    dates_ = [multireplace(s, date_replace) for s in dates_]
    dates_ = [re.sub(r'[ ]+', ' ', s).strip() for s in dates_]
    dates_ = cluster_words(dates_)
    dates_ = {s: dateparser.parse(s) for s in dates_}
    money_ = {s[0]: s[1] for s in money_}

    return dates_, money_


def check_repeat(word):
    if len(word) < 2:
        return word, 1
    if word[-1].isdigit() and not word[-2].isdigit():
        try:
            repeat = int(unidecode(word[-1]))
            word = word[:-1]
        except Exception:
            repeat = 1
    else:
        repeat = 1
    if repeat < 1:
        repeat = 1
    return word, repeat


def groupby(string):
    results = []
    for word in string.split():
        if not (
            _is_number_regex(word)
            or re.findall(_expressions['url'], word)
            or re.findall(_expressions['money'], word.lower())
            or re.findall(_expressions['number'], word)
        ):
            word = ''.join([''.join(s)[:2] for _, s in itertools.groupby(word)])
        results.append(word)
    return ' '.join(results)


def put_spacing_num(string, english=False):
    string = re.sub('[A-Za-z]+', lambda ele: ' ' + ele[0] + ' ', string).split()
    for i in range(len(string)):
        if _is_number_regex(string[i]):
            string[i] = ' '.join([to_cardinal(int(n), english=english) for n in string[i]])
    string = ' '.join(string)
    return re.sub(r'[ ]+', ' ', string).strip()


def replace_multiple_cardinal(string, english=False):
    def custom_replace(match):
        number = match.group(0)
        return f' {cardinal(number, english=english)} '
    string = string.replace('-', ' ')
    string = re.sub(r'\d+', custom_replace, string)
    return re.sub(r'[ ]+', ' ', string).strip()


class Normalizer:
    def __init__(self, tokenizer, speller=None, stemmer=None):
        self._tokenizer = tokenizer
        self._speller = speller
        self._stemmer = stemmer
        self._demoji = None
        self._compiled = {
            k.lower(): re.compile(v) for k, v in _expressions.items()
        }

    def _process_multiword_dates(self, result, normalize_date, dateparser_settings, normalize_in_english):
        if not normalize_date:
            return result
        text = ' '.join(result)
        date_matches = re.findall(_expressions['date'], text)
        if date_matches:
            for match in date_matches:
                parsed_date = parse_date_string(
                    match, normalize_date=normalize_date,
                    dateparser_settings=dateparser_settings,
                    english=normalize_in_english,
                )
                text = text.replace(match, parsed_date)
            result = text.split()
        return result

    def normalize(
        self,
        string,
        normalize_text=True,
        normalize_word_rules=True,
        normalize_url=False,
        normalize_email=False,
        normalize_year=True,
        normalize_telephone=True,
        normalize_date=True,
        normalize_time=True,
        normalize_emoji=False,
        normalize_elongated=True,
        normalize_hingga=True,
        normalize_pada_hari_bulan=True,
        normalize_fraction=True,
        normalize_money=True,
        normalize_units=True,
        normalize_percent=True,
        normalize_ic=True,
        ic_dash_sempang=False,
        normalize_number=True,
        normalize_x_kali=True,
        normalize_cardinal=True,
        normalize_cardinal_title=True,
        normalize_ordinal=True,
        normalize_entity=True,
        expand_contractions=True,
        expand_units=True,
        normalize_in_english=False,
        check_english_func=None,
        check_malay_func=None,
        translator=None,
        language_detection_word=None,
        acceptable_language_detection=None,
        segmenter=None,
        text_scorer=None,
        text_scorer_window=2,
        not_a_word_threshold=1e-4,
        dateparser_settings={'TIMEZONE': 'GMT+8'},
        **kwargs,
    ):
        if check_english_func is None:
            check_english_func = is_english
        if check_malay_func is None:
            check_malay_func = is_malay
        if acceptable_language_detection is None:
            acceptable_language_detection = ['EN', 'CAPITAL', 'NOT_LANG']

        result_demoji = None

        if expand_contractions:
            string = unpack_english_contractions(string)

        tokenized = self._tokenizer(string)
        string = ' '.join(tokenized)

        if normalize_elongated:
            normalized = []
            got_speller = hasattr(self._speller, 'normalize_elongated') if self._speller else False
            for word in string.split():
                word_lower = word.lower()
                if (
                    len(re.findall(r'(.)\1{1}', word))
                    and not word[0].isupper()
                    and not word_lower.startswith('ke-')
                    and not self._compiled['email'].search(word)
                    and not self._compiled['url'].search(word)
                    and not self._compiled['hashtag'].search(word)
                    and not self._compiled['phone'].search(word)
                    and not self._compiled['money'].search(word)
                    and not self._compiled['date'].search(word)
                    and not self._compiled['ic'].search(word)
                    and not self._compiled['user'].search(word)
                    and not self._compiled['number'].search(word)
                    and not _is_number_regex(word)
                    and check_english_func is not None
                    and not check_english_func(word_lower)
                ):
                    word = self._compiled['normalize_elong'].sub(r'\1\1', groupby(word))
                    if got_speller:
                        word = self._speller.normalize_elongated(word)
                normalized.append(word)
            string = ' '.join(normalized)

        if normalize_text:
            string = replace_laugh(string)
            string = replace_mengeluh(string)
            string = replace_betul(string)
            string = _replace_compound(string)

        result, normalized = [], []
        spelling_correction = {}
        spelling_correction_condition = {}

        tokenized = self._tokenizer(string)
        index = 0
        while index < len(tokenized):
            word = tokenized[index]
            word_lower = word.lower()
            word_upper = word.upper()
            word_title = word.title()
            first_c = word[0].isupper()

            if word in PUNCTUATION:
                result.append(word)
                index += 1
                continue

            normalized.append(rules_normalizer.get(word_lower, word_lower))

            if word_lower in ignore_words:
                result.append(word)
                index += 1
                continue

            if self._compiled['ic'].search(word_lower):
                if normalize_ic:
                    splitted = word.split('-')
                    ics = [digit(s, english=normalize_in_english) for s in splitted]
                    if ic_dash_sempang:
                        join_w = ' dash ' if normalize_in_english else ' sempang '
                    else:
                        join_w = ' '
                    word = join_w.join(ics)
                result.append(word)
                index += 1
                continue

            if self._compiled['hashtag'].search(word_lower):
                result.append(word)
                index += 1
                continue

            if self._compiled['url'].search(word_lower):
                if normalize_url:
                    word = word.replace('://', ' ').replace('.', ' dot ')
                    word = put_spacing_num(word, english=normalize_in_english)
                    word = word.replace('https', 'HTTPS').replace('http', 'HTTP').replace('www', 'WWW')
                result.append(word)
                index += 1
                continue

            if self._compiled['email'].search(word_lower):
                if normalize_email:
                    at = ' at ' if normalize_in_english else ' di '
                    word = word.upper().replace('://', ' ').replace('.', ' dot ').replace('@', at)
                    word = put_spacing_num(word, english=normalize_in_english)
                result.append(word)
                index += 1
                continue

            if self._compiled['phone'].search(word_lower):
                if normalize_telephone:
                    splitted = word.split('-')
                    if len(splitted) == 2:
                        left = put_spacing_num(splitted[0], english=normalize_in_english)
                        right = put_spacing_num(splitted[1], english=normalize_in_english)
                        word = f'{left}, {right}'
                result.append(word)
                index += 1
                continue

            if self._compiled['user'].search(word_lower):
                result.append(word)
                index += 1
                continue

            if self._compiled['word_dash'].search(word_lower):
                words = []
                for c in word:
                    if c == '-':
                        w = 'dash' if normalize_in_english else 'sempang'
                    elif c in digits:
                        w = digit(c, english=normalize_in_english)
                    else:
                        w = c.upper()
                    words.append(w)
                word = ' '.join(words)
                result.append(word)
                index += 1
                continue

            if self._compiled['passport'].search(word):
                chars = []
                for c in word:
                    if c in digits:
                        c = digit(c, english=normalize_in_english)
                    chars.append(c)
                result.append(' '.join(chars))
                index += 1
                continue

            if text_scorer is not None:
                import math
                score = math.exp(text_scorer(word_lower))
                if score <= not_a_word_threshold:
                    result.append(word)
                    index += 1
                    continue

            if (
                first_c
                and not self._compiled['money'].search(word_lower)
                and not self._compiled['date'].search(word_lower)
            ):
                if normalize_word_rules and word_lower in rules_normalizer:
                    result.append(case_of(word)(rules_normalizer[word_lower]))
                    index += 1
                    continue
                elif word_upper not in ['KE', 'PADA', 'RM', 'SEN', 'HINGGA']:
                    norm_title = _normalize_title(word) if normalize_text else word
                    if norm_title != word:
                        result.append(norm_title)
                        index += 1
                        continue

                    titled = True
                    if len(word) > 1 and text_scorer is not None:
                        import math
                        l = ' '.join(result[-text_scorer_window:])
                        if len(l):
                            lower = f'{l} {word_lower}'
                            title = f'{l} {word_title}'
                            normal = f'{l} {word}'
                            upper = f'{l} {word_upper}'
                        else:
                            lower = word_lower
                            title = word_title
                            normal = word
                            upper = word_upper
                        if index + 1 < len(tokenized):
                            r = ' '.join(tokenized[index + 1: index + 1 + text_scorer_window])
                            if len(r):
                                lower = f'{lower} {r}'
                                title = f'{title} {r}'
                                normal = f'{normal} {r}'
                                upper = f'{upper} {r}'
                        lower_score = text_scorer(lower)
                        title_score = text_scorer(title)
                        normal_score = text_scorer(normal)
                        upper_score = text_scorer(upper)
                        scores = [lower_score, title_score, upper_score]
                        max_score = max(scores)
                        argmax = np.argmax(scores)
                        if max_score > normal_score:
                            if argmax == 0:
                                word = word_lower
                                titled = False
                            elif argmax == 1:
                                word = word_title
                            elif argmax == 2:
                                word = word_upper

                    if titled:
                        if normalize_cardinal_title:
                            w = replace_multiple_cardinal(word, english=normalize_in_english)
                        else:
                            w = word
                        result.append(w)
                        index += 1
                        continue

            if check_english_func is not None and len(word) > 1:
                found = False
                word_, repeat = check_repeat(word)
                word_lower_ = word_.lower()
                selected_word = word_
                if check_english_func(word_lower_):
                    found = True
                elif len(word_lower_) > 1 and len(word_) > 1 and word_lower_[-1] == word_lower_[-2] and check_english_func(word_lower_[:-1]):
                    found = True
                    selected_word = word_[:-1]
                if found:
                    if translator is not None and language_detection_word is None:
                        translated = translator(selected_word)
                        if len(translated) >= len(selected_word) * 3:
                            pass
                        elif ', United States' in translated:
                            pass
                        else:
                            selected_word = translated
                    result.append(repeat_word(case_of(word)(selected_word), repeat))
                    index += 1
                    continue

            if check_malay_func is not None and len(word) > 1:
                if word_lower not in ['pada', 'ke', 'tahun', 'thun']:
                    if check_malay_func(word_lower):
                        result.append(word)
                        index += 1
                        continue
                    elif len(word_lower) > 1 and word_lower[-1] == word_lower[-2] and check_malay_func(word_lower[:-1]):
                        result.append(word[:-1])
                        index += 1
                        continue

            if is_malaysia_location(word):
                result.append(word_lower.title())
                index += 1
                continue

            if normalize_word_rules and word_lower in rules_normalizer:
                result.append(case_of(word)(rules_normalizer[word_lower]))
                index += 1
                continue

            if len(word) > 2 and normalize_text and check_english_func is not None and not check_english_func(word):
                if word[-2] in consonants and word[-1] == 'e':
                    word = word[:-1] + 'a'

            if word[0] == 'x' and len(word) > 1 and normalize_text and check_english_func is not None and not check_english_func(word):
                result_string = 'tak '
                word = word[1:]
            else:
                result_string = ''

            if normalize_ordinal and word_lower == 'ke' and index < (len(tokenized) - 2):
                if tokenized[index + 1] == '-' and _is_number_regex(tokenized[index + 2]):
                    result.append(ordinal(word + tokenized[index + 1] + tokenized[index + 2]))
                    index += 3
                    continue
                elif tokenized[index + 1] == '-' and re.match('.*(V|X|I|L|D)', tokenized[index + 2]):
                    result.append(ordinal(word + tokenized[index + 1] + str(rom_to_int(tokenized[index + 2]))))
                    index += 3
                    continue
                else:
                    result.append('ke')
                    index += 1
                    continue

            if normalize_hingga and _is_number_regex(word) and index < (len(tokenized) - 2):
                if tokenized[index + 1] == '-' and _is_number_regex(tokenized[index + 2]):
                    until = ' until ' if normalize_in_english else ' hingga '
                    result.append(
                        to_cardinal(_string_to_num(word), english=normalize_in_english)
                        + until
                        + to_cardinal(_string_to_num(tokenized[index + 2]), english=normalize_in_english)
                    )
                    index += 3
                    continue

            if normalize_pada_hari_bulan and word_lower == 'pada' and index < (len(tokenized) - 3):
                if (
                    _is_number_regex(tokenized[index + 1])
                    and tokenized[index + 2] in '/-'
                    and _is_number_regex(tokenized[index + 3])
                ):
                    if normalize_in_english:
                        s = 'on the %s day of the %s month' % (
                            to_ordinal(_string_to_num(tokenized[index + 1]), english=True),
                            to_ordinal(_string_to_num(tokenized[index + 3]), english=True),
                        )
                    else:
                        s = 'pada %s hari bulan %s' % (
                            to_cardinal(_string_to_num(tokenized[index + 1])),
                            to_cardinal(_string_to_num(tokenized[index + 3])),
                        )
                    result.append(s)
                    index += 4
                    continue

            if word_lower in ['tahun', 'thun'] and index < (len(tokenized) - 1) and normalize_year:
                if _is_number_regex(tokenized[index + 1]) and len(tokenized[index + 1]) == 4:
                    t = tokenized[index + 1]
                    if t[1] != '0':
                        l = to_cardinal(int(t[:2]), english=normalize_in_english)
                        r = to_cardinal(int(t[2:]), english=normalize_in_english)
                        c = f'{l} {r}'
                    elif t[:2] == '20' and t[2:] != '00':
                        l = to_cardinal(int(t[:2]), english=normalize_in_english)
                        r = to_cardinal(int(t[2:]), english=normalize_in_english)
                        c = f'{l} {r}'
                    else:
                        c = to_cardinal(int(t), english=normalize_in_english)
                    if (
                        index < (len(tokenized) - 3)
                        and tokenized[index + 2] == '-'
                        and tokenized[index + 3].lower() == 'an'
                    ):
                        end = 's' if normalize_in_english else 'an'
                        plus = 4
                    else:
                        end = ''
                        plus = 2
                    start = '' if normalize_in_english else 'tahun '
                    result.append(f'{start}{c}{end}')
                    index += plus
                    continue

            if normalize_fraction and _is_number_regex(word) and index < (len(tokenized) - 2):
                if tokenized[index + 1] == '/' and _is_number_regex(tokenized[index + 2]):
                    result.append(fraction(word + tokenized[index + 1] + tokenized[index + 2]))
                    index += 3
                    continue
                if (
                    tokenized[index + 1] == '-'
                    and tokenized[index + 2].lower() == 'an'
                    and normalize_year and len(word) == 4
                ):
                    t = word
                    if t[1] != '0':
                        l = to_cardinal(int(t[:2]), english=normalize_in_english)
                        r = to_cardinal(int(t[2:]), english=normalize_in_english)
                        c = f'{l} {r}'
                    else:
                        c = to_cardinal(int(t), english=normalize_in_english)
                    result.append(f'{c}an')
                    index += 3
                    continue

            if self._compiled['money'].search(word_lower):
                if normalize_money:
                    money_, _ = money(word, english=normalize_in_english)
                    result.append(money_)
                    if index < (len(tokenized) - 1):
                        if tokenized[index + 1].lower() in ('sen', 'cent'):
                            index += 2
                        else:
                            index += 1
                    else:
                        index += 1
                else:
                    result.append(word)
                    index += 1
                continue

            if (
                self._compiled['temperature'].search(word_lower)
                or self._compiled['distance'].search(word_lower)
                or self._compiled['volume'].search(word_lower)
                or self._compiled['duration'].search(word_lower)
                or self._compiled['weight'].search(word_lower)
                or self._compiled['data_size'].search(word_lower)
            ):
                if normalize_units:
                    word = word.replace(' ', '')
                    word = digit_unit(word, expand_units=expand_units, english=normalize_in_english)
                result.append(word)
                index += 1
                continue

            if self._compiled['percent'].search(word_lower):
                if normalize_percent:
                    word = word.replace('%', '')
                    percent = ' percent' if normalize_in_english else ' peratus'
                    word = cardinal(word, english=normalize_in_english) + percent
                result.append(word)
                index += 1
                continue

            if self._compiled['date'].search(word_lower):
                word = word_lower
                word = multireplace(word, date_replace)
                word = re.sub(r'[ ]+', ' ', word).strip()
                word = parse_date_string(
                    word, normalize_date=normalize_date,
                    dateparser_settings=dateparser_settings,
                    english=normalize_in_english,
                )
                result.append(word)
                index += 1
                continue

            if self._compiled['hijri_year'].search(word):
                word = word_lower[:-1]
                word = re.sub(r'[ ]+', ' ', word).strip()
                try:
                    word = cardinal(word, english=normalize_in_english)
                except Exception as e:
                    logger.warning(str(e))
                word = word + ' Hijrah'
                result.append(word)
                index += 1
                continue

            if self._compiled['hari_bulan'].search(word):
                word = word_lower[:-2]
                word = re.sub(r'[ ]+', ' ', word).strip()
                try:
                    word = cardinal(word, english=normalize_in_english)
                except Exception as e:
                    logger.warning(str(e))
                end = ' days of the month' if normalize_in_english else ' hari bulan'
                word = word + end
                result.append(word)
                index += 1
                continue

            if self._compiled['pada_tarikh'].search(word_lower):
                _pada_tarikh = r"\b(?:pada|tarikh)\s+(0?[1-9]|[12][0-9]|3[01])\s(0?[1-9]|1[0-2])\b"
                r = re.findall(_pada_tarikh, word_lower)
                day = r[0][0]
                month = r[0][1]
                date_obj = datetime.strptime(f"{day} {month} {datetime.today().year}", "%d %m %Y")
                word = date_obj.strftime("%Y-%m-%d")
                word = parse_date_string(
                    word, normalize_date=normalize_date,
                    dateparser_settings=dateparser_settings,
                    english=normalize_in_english,
                )
                result.append(word)
                index += 1
                continue

            if (
                self._compiled['time'].search(word_lower)
                or self._compiled['time_pukul'].search(word_lower)
            ):
                word = word_lower
                word = re.sub(r'[ ]+', ' ', word).strip()
                prefix = 'at ' if normalize_in_english else 'pukul '
                try:
                    parsed = parse_time_string(word)
                    if len(parsed):
                        parsed = parsed[0]
                        word = parsed.strftime('%H:%M:%S')
                        hour, minute, second = word.split(':')
                        if normalize_time:
                            hour = parsed.strftime('%I').lstrip('0')
                            if parsed.hour < 12:
                                period = 'am' if normalize_in_english else 'pagi'
                            elif parsed.hour < 19:
                                period = 'pm' if normalize_in_english else 'petang'
                            else:
                                period = 'pm' if normalize_in_english else 'malam'
                            hour = cardinal(hour, english=normalize_in_english)
                            if int(minute) > 0:
                                minute = cardinal(minute, english=normalize_in_english)
                                end = 'minute' if normalize_in_english else 'minit'
                                minute = f'{minute} {end}'
                            else:
                                minute = ''
                            if int(second) > 0:
                                second = cardinal(second, english=normalize_in_english)
                                second = f'{second} saat'
                            else:
                                second = ''
                            word = f'{prefix}{hour} {minute} {second} {period}'
                        else:
                            pukul = f'{prefix}{hour}'
                            if int(minute) > 0:
                                pukul = f'{pukul}.{minute}'
                            if int(second) > 0:
                                pukul = f'{pukul}:{second}'
                            word = pukul
                        word = re.sub(r'[ ]+', ' ', word).strip()
                except Exception as e:
                    logger.warning(str(e))
                result.append(word)
                index += 1
                continue

            if (
                self._compiled['number'].search(word_lower)
                and word_lower[0] == '0'
                and '.' not in word_lower
            ):
                if normalize_number:
                    word = digit(word, english=normalize_in_english)
                result.append(word)
                index += 1
                continue

            if (
                normalize_x_kali
                and len(word_lower) >= 2
                and word_lower[-1] == 'x'
                and self._compiled['number'].search(word_lower[:-1])
                and '.' not in word_lower
            ):
                word = word[:-1]
                word = cardinal(word, english=normalize_in_english)
                end = 'times' if normalize_in_english else 'kali'
                word = f'{word} {end}'
                result.append(word)
                index += 1
                continue

            if normalize_date and _is_number_regex(word) and index < (len(tokenized) - 1):
                next_word = tokenized[index + 1].lower()
                if next_word in ['january', 'february', 'march', 'april', 'may', 'june',
                                 'july', 'august', 'september', 'october', 'november', 'december']:
                    result.append(to_ordinal(int(word), english=normalize_in_english))
                    index += 1
                    result.append(tokenized[index])
                    index += 1
                    continue

            if normalize_cardinal:
                cardinal_ = cardinal(word, english=normalize_in_english)
                if cardinal_ != word:
                    result.append(cardinal_)
                    index += 1
                    continue

            if normalize_ordinal:
                normalized_ke = ordinal(word, english=normalize_in_english)
                if normalized_ke != word:
                    result.append(normalized_ke)
                    index += 1
                    continue

            if self._compiled['number_with_shortform'].search(word_lower):
                if normalize_cardinal:
                    w = normalize_numbers_with_shortform(word_lower)
                    w = cardinal(w)
                else:
                    w = word
                result.append(w)
                index += 1
                continue

            if self._compiled['number'].search(word):
                if normalize_cardinal:
                    w = replace_multiple_cardinal(word, english=normalize_in_english)
                elif normalize_number:
                    w = ' '.join([digit(c, english=normalize_in_english) for c in word])
                else:
                    w = word
                result.append(w)
                index += 1
                continue

            if segmenter is not None:
                if word[-1] in digits:
                    word_ = word[:-1]
                    d = word[-1]
                else:
                    word_ = word
                    d = ''
                segmentized = segmenter(word_) + d
                words = segmentized.split()
            else:
                words = [word]

            for no_word, word in enumerate(words):
                if self._stemmer is not None:
                    word, end_result_string = _remove_postfix(
                        word, stemmer=self._stemmer, validate_word=False,
                    )
                    if len(end_result_string) and end_result_string[0] in digits:
                        word = word + end_result_string[0]
                        end_result_string = end_result_string[1:]
                else:
                    end_result_string = ''

                if normalize_text:
                    word, repeat = check_repeat(word)
                else:
                    repeat = 1

                if normalize_text:
                    if normalize_word_rules and word in sounds:
                        selected = sounds[word]
                    elif normalize_word_rules and word in rules_normalizer:
                        selected = rules_normalizer[word]
                    elif len(word) > 1 and word[-1] == word[-2]:
                        if normalize_word_rules and word[:-1] in rules_normalizer:
                            selected = rules_normalizer[word[:-1]]
                        else:
                            selected = word[:-1]
                    elif len(word) > 1 and word[-1] == word[-2] and word[:-1] in rules_normalizer_rev:
                        selected = word[:-1]
                    else:
                        selected = word
                        if translator is not None and language_detection_word is None:
                            translated = translator(word)
                            if len(translated) >= len(word) * 3:
                                pass
                            elif ', United States' in translated:
                                pass
                            elif translated in PUNCTUATION:
                                pass
                            else:
                                selected = translated
                        if selected == word and self._speller:
                            spelling_correction[len(result)] = selected
                else:
                    selected = word

                selected = repeat_word(selected, repeat)
                spelling_correction_condition[len(result)] = [repeat, result_string, end_result_string]
                result.append(result_string + selected + end_result_string)

            index += 1

        for index, selected in spelling_correction.items():
            selected = self._speller.correct(selected, string=result, index=index, **kwargs)
            repeat, result_string, end_result_string = spelling_correction_condition[index]
            selected = repeat_word(selected, repeat)
            selected = result_string + selected + end_result_string
            result[index] = selected

        result = self._process_multiword_dates(result, normalize_date, dateparser_settings, normalize_in_english)

        result = ' '.join(result)
        normalized = ' '.join(normalized)
        result = re.sub(r'[ ]+', ' ', result).strip()
        normalized = re.sub(r'[ ]+', ' ', normalized).strip()

        if translator is not None and language_detection_word is not None:
            splitted = result.split()
            result_langs = language_detection_word(splitted)
            new_result, temp, temp_lang = [], [], []
            for no_r, r in enumerate(result_langs):
                if r in acceptable_language_detection and not is_laugh(splitted[no_r]) and not is_mengeluh(splitted[no_r]):
                    temp.append(splitted[no_r])
                    temp_lang.append(r)
                else:
                    if len(temp):
                        if 'EN' in temp_lang:
                            translated = translator(' '.join(temp))
                            new_result.extend(translated.split())
                        else:
                            new_result.extend(temp)
                        temp = []
                        temp_lang = []
                    new_result.append(splitted[no_r])
            if len(temp):
                if 'EN' in temp_lang:
                    translated = translator(' '.join(temp))
                    new_result.extend(translated.split())
                else:
                    new_result.extend(temp)
            result = ' '.join(new_result)

        if normalize_entity:
            dates_, money_ = normalized_entity(normalized)
        else:
            dates_, money_ = {}, {}

        return {'normalize': fix_spacing(result), 'date': dates_, 'money': money_}


def load(speller=None, stemmer=None, **kwargs):
    """
    Load a Normalizer.

    Parameters
    ----------
    speller: Callable, optional (default=None)
    stemmer: Callable, optional (default=None)

    Returns
    -------
    result: Normalizer
    """
    tokenizer = Tokenizer(**kwargs).tokenize
    return Normalizer(tokenizer=tokenizer, speller=speller, stemmer=stemmer)
