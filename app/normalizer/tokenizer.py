"""
Tokenizer for Malaysian text normalization.
Extracted from malaya.tokenizer (https://github.com/malaysia-ai/malaya)
"""

import re
import html

from .regex import _expressions
from .function import split_into_sentences


class Tokenizer:
    def __init__(self, **kwargs):
        pipeline = []
        self.regexes = _expressions

        emojis = kwargs.get('emojis', False)
        urls = kwargs.get('urls', True)
        urls_improved = kwargs.get('urls_improved', True)
        tags = kwargs.get('tags', True)
        emails = kwargs.get('emails', True)
        users = kwargs.get('users', True)
        hashtags = kwargs.get('hashtags', True)
        cashtags = kwargs.get('cashtags', True)
        phones = kwargs.get('phones', True)
        percents = kwargs.get('percents', True)
        money = kwargs.get('money', True)
        date = kwargs.get('date', True)
        time = kwargs.get('time', True)
        time_pukul = kwargs.get('time_pukul', True)
        acronyms = kwargs.get('acronyms', True)
        emoticons = kwargs.get('emoticons', True)
        censored = kwargs.get('censored', True)
        emphasis = kwargs.get('emphasis', True)
        numbers = kwargs.get('numbers', True)
        numbers_with_shortform = kwargs.get('numbers_with_shortform', True)
        temperatures = kwargs.get('temperature', True)
        distances = kwargs.get('distance', True)
        volumes = kwargs.get('volume', True)
        durations = kwargs.get('duration', True)
        weights = kwargs.get('weight', True)
        data_sizes = kwargs.get('data_size', True)
        hypens = kwargs.get('hypen', True)
        ic = kwargs.get('ic', True)
        title = kwargs.get('title', True)
        parliament = kwargs.get('parliament', True)
        hijri_year = kwargs.get('hijri_year', True)
        hari_bulan = kwargs.get('hari_bulan', True)
        pada_tarikh = kwargs.get('pada_tarikh', True)
        word_dash = kwargs.get('word_dash', True)
        passport = kwargs.get('passport', True)

        if word_dash:
            pipeline.append(self.regexes['word_dash'])
        if title:
            pipeline.append(self.regexes['title'])
        if parliament:
            pipeline.append(self.regexes['parliament'])
        if urls:
            pipeline.append(self.regexes['url'])
        if urls_improved:
            pipeline.append(self.regexes['url_v2'])
            pipeline.append(self.regexes['url_dperini'])
        if tags:
            pipeline.append(self.regexes['tag'])
        if emails:
            pipeline.append(self.wrap_non_matching(self.regexes['email']))
        if users:
            pipeline.append(self.wrap_non_matching(self.regexes['user']))
        if hashtags:
            pipeline.append(self.wrap_non_matching(self.regexes['hashtag']))
        if cashtags:
            pipeline.append(self.wrap_non_matching(self.regexes['cashtag']))
        if phones:
            pipeline.append(self.wrap_non_matching(self.regexes['phone']))
        if percents:
            pipeline.append(self.wrap_non_matching(self.regexes['percent']))
        if money:
            pipeline.append(self.wrap_non_matching(self.regexes['money']))
        if date:
            pipeline.append(self.wrap_non_matching(self.regexes['date']))
        if time:
            pipeline.append(self.wrap_non_matching(self.regexes['time']))
        if time_pukul:
            pipeline.append(self.wrap_non_matching(self.regexes['time_pukul']))
        if acronyms:
            pipeline.append(self.wrap_non_matching(self.regexes['acronym']))
        if censored:
            pipeline.append(self.wrap_non_matching(self.regexes['censored']))
        if emphasis:
            pipeline.append(self.wrap_non_matching(self.regexes['emphasis']))
        if temperatures:
            pipeline.append(self.wrap_non_matching(self.regexes['temperature']))
        if distances:
            pipeline.append(self.wrap_non_matching(self.regexes['distance']))
        if volumes:
            pipeline.append(self.wrap_non_matching(self.regexes['volume']))
        if durations:
            pipeline.append(self.wrap_non_matching(self.regexes['duration']))
        if weights:
            pipeline.append(self.wrap_non_matching(self.regexes['weight']))
        if data_sizes:
            pipeline.append(self.wrap_non_matching(self.regexes['data_size']))
        if ic:
            pipeline.append(self.wrap_non_matching(self.regexes['ic']))
        if numbers_with_shortform:
            pipeline.append(self.regexes['number_with_shortform'])
        if numbers:
            pipeline.append(self.regexes['number'])
        if emojis:
            pipeline.append(self.regexes['emoji'])
        if hypens:
            pipeline.append(self.regexes['hypen'])
        if hijri_year:
            pipeline.append(self.wrap_non_matching(self.regexes['hijri_year']))
        if hari_bulan:
            pipeline.append(self.wrap_non_matching(self.regexes['hari_bulan']))
        if pada_tarikh:
            pipeline.append(self.wrap_non_matching(self.regexes['pada_tarikh']))
        if passport:
            pipeline.append(self.wrap_non_matching(self.regexes['passport']))

        pipeline.append(self.regexes['apostrophe'])
        pipeline.append(self.regexes['word'])
        pipeline.append('(?:\\S)')

        self.tok = re.compile(r'({})'.format('|'.join(pipeline)))

    @staticmethod
    def wrap_non_matching(exp):
        return '(?:{})'.format(exp)

    def tokenize(self, string, lowercase=False):
        escaped = html.unescape(string)
        tokenized = self.tok.findall(escaped)
        tokenized = [t[0] if isinstance(t, tuple) else t for t in tokenized]
        tokenized_all = []
        for t in tokenized:
            if len(re.findall(r'\.{2,}', t)):
                splitted = [w if len(w) else '.' for w in t.split('.')]
                tokenized_all.extend(splitted)
            else:
                tokenized_all.append(t)
        tokenized = [re.sub(r'[ ]+', ' ', t).strip() for t in tokenized_all]
        if lowercase:
            tokenized = [t.lower() for t in tokenized]
        return tokenized


class SentenceTokenizer:
    def __init__(self):
        pass

    def tokenize(self, string, minimum_length=5):
        return split_into_sentences(string, minimum_length=minimum_length)
