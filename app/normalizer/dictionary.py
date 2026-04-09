"""
Word dictionary functions for Malaysian text normalization.
Extracted from malaya.dictionary (https://github.com/malaysia-ai/malaya)
Data loaded from JSON files in data/ directory.
"""

import json
import os

from .rules import rules_normalizer

_data_dir = os.path.join(os.path.dirname(__file__), 'data')


def _load_json_set(filename):
    with open(os.path.join(_data_dir, filename), 'r', encoding='utf-8') as f:
        return set(json.load(f))


ENGLISH_WORDS = _load_json_set('english_words.json')
MALAY_WORDS = _load_json_set('malay_words.json')
CAMBRIDGE_MALAY_WORDS = _load_json_set('cambridge_malay_words.json')
KAMUS_DEWAN_WORDS = _load_json_set('kamus_dewan_words.json')
DBP_WORDS = _load_json_set('dbp_words.json')

_negeri = _load_json_set('negeri.json')
_city = _load_json_set('city.json')
_country = _load_json_set('country.json')
_daerah = _load_json_set('daerah.json')
_parlimen = _load_json_set('parlimen.json')
_adun = _load_json_set('adun.json')


def is_english(word):
    is_in = False
    if word in ENGLISH_WORDS:
        is_in = True
    elif len(word) > 1 and word[-1] in 's' and word[:-1] in ENGLISH_WORDS:
        is_in = True
    return is_in


def is_malay(word, stemmer=None):
    if word.lower() in rules_normalizer and not is_english(word.lower()):
        return True
    if stemmer is not None:
        if not hasattr(stemmer, 'stem_word'):
            raise ValueError('stemmer must have `stem_word` method')
        word = stemmer.stem_word(word)
        if word.lower() in rules_normalizer and not is_english(word.lower()):
            return True
    return word in MALAY_WORDS or word in CAMBRIDGE_MALAY_WORDS or word in KAMUS_DEWAN_WORDS or word in DBP_WORDS


def is_malaysia_location(string):
    string_lower = string.lower()
    title = string_lower.title()
    if string_lower in _negeri or title in _city or title in _country or title in _daerah or title in _parlimen or title in _adun:
        return True
    return False
