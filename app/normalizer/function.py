"""
Text utility functions for normalization.
Extracted from malaya.text.function (https://github.com/malaysia-ai/malaya)
"""

import re
import itertools
from itertools import combinations

from .tatabahasa import laughing, mengeluh
from .rules import rules_normalizer
from .regex import _expressions

PUNCTUATION = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'

BETUL = ['haah', 'aah', "ha'ah", 'haahh', 'hhaahh']
GEMBIRA = ['yeay', 'yey', 'yeey']


def generate_compound(word):
    combs = {word}
    for i in range(1, len(word) + 1):
        for c in combinations(word, i):
            cs = ''.join(c)
            r = []
            for no in range(len(word)):
                for c in cs:
                    if word[no] == c:
                        p = c + c
                    else:
                        p = word[no]
                    r.append(p)
            s = ''.join(
                ''.join(s)[i - 1: i + 1]
                for _, s in itertools.groupby(''.join(r))
            )
            combs.add(s)
    combs.add(''.join([c + c for c in word]))
    return list(combs)


MENGELUH = []
for _word in mengeluh:
    MENGELUH.extend(generate_compound(_word))
MENGELUH = set([word for word in MENGELUH if len(word) > 2])


def is_laugh(word):
    if word[0] == '#' or word[0] == '@' or word.isupper():
        return word
    word_lower = word.lower()
    return any([e in word_lower for e in laughing])


def is_mengeluh(word):
    if word[0] == '#' or word[0] == '@' or word.isupper():
        return word
    word_lower = word.lower()
    return any([e in word_lower for e in MENGELUH])


def case_of(text):
    return (
        str.title if text.istitle()
        else str.lower if text.islower()
        else str.upper if text.isupper()
        else str
    )


def multireplace(string, replacements):
    substrs = sorted(replacements, key=len, reverse=True)
    regexp = re.compile('|'.join(map(re.escape, substrs)))
    return regexp.sub(lambda match: replacements[match.group(0)], string)


def replace_any(string, lists, replace_with):
    from .dictionary import is_malay, is_english
    from .normalization import _is_number_regex

    result = []
    for word in string.split():
        word_lower = word.lower()
        if is_malay(word_lower):
            pass
        elif is_english(word_lower):
            pass
        elif (
            not len(re.findall(_expressions['email'], word))
            and not len(re.findall(_expressions['url'], word))
            and not len(re.findall(_expressions['hashtag'], word))
            and not len(re.findall(_expressions['phone'], word))
            and not len(re.findall(_expressions['money'], word))
            and not len(re.findall(_expressions['date'], word))
            and not _is_number_regex(word)
        ):
            pass
        elif any([e in word_lower for e in lists]):
            word = case_of(word)(replace_with)
        result.append(word)
    return ' '.join(result)


def fix_spacing(text):
    quote_pattern = r'"([^"]*)"'
    def fix_quotes(match):
        content = match.group(1).strip()
        return f'"{content}"'
    text = re.sub(quote_pattern, fix_quotes, text)

    paren_pattern = r'\(([^)]*)\)'
    def fix_parens(match):
        content = match.group(1).strip()
        return f'({content})'
    text = re.sub(paren_pattern, fix_parens, text)

    text = re.sub(r'\s+([,.?!])', r'\1', text)
    return text


# Sentence splitting utilities
_alphabets = '([A-Za-z])'
_prefixes = '(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt|Puan|puan|Tuan|tuan|sir|Sir)[.]'
_suffixes = '(Inc|Ltd|Jr|Sr|Co|Mo)'
_starters = '(Mr|Mrs|Ms|Dr|He\\s|She\\s|It\\s|They\\s|Their\\s|Our\\s|We\\s|But\\s|However\\s|That\\s|This\\s|Wherever|Dia|Mereka|Tetapi|Kita|Itu|Ini|Dan|Kami|Beliau|Seri|Datuk|Dato|Datin|Tuan|Puan)'
_acronyms = '([A-Z][.][A-Z][.](?:[A-Z][.])?)'
_websites = '[.](com|net|org|io|gov|me|edu|my)'
_another_websites = '(www|http|https)[.]'
_digits = '([0-9])'
_before_digits = '([Nn]o|[Nn]ombor|[Nn]umber|[Kk]e|=|al|[Pp]ukul)'
_month = '([Jj]an(?:uari)?|[Ff]eb(?:ruari)?|[Mm]a(?:c)?|[Aa]pr(?:il)?|Mei|[Jj]u(?:n)?|[Jj]ula(?:i)?|[Aa]ug(?:ust)?|[Ss]ept?(?:ember)?|[Oo]kt(?:ober)?|[Nn]ov(?:ember)?|[Dd]is(?:ember)?)'


def _replace_sub(pattern, text, replace_left='.', replace_right='<prd>'):
    alls = re.findall(pattern, text)
    for a in alls:
        text = text.replace(a, a.replace(replace_left, replace_right))
    return text


def split_into_sentences(text, minimum_length=5):
    _emails = r'(?:^|(?<=[^\w@.)]))(?:[\w+-](?:\.(?!\.))?)*?[\w+-]@(?:\w-?)*?\w+(?:\.(?:[a-z]{2,})){1,3}(?:$|(?=\b))'

    text = text.replace('\x97', '\n')
    text = '. '.join([s for s in text.split('\n') if len(s)])
    text = text + '.'
    text = ' ' + text + '  '
    text = text.replace('\n', ' ')
    text = re.sub(_prefixes, '\\1<prd>', text)
    text = _replace_sub(_emails, text)
    text = _replace_sub(_expressions['title'], text)
    text = re.sub(_websites, '<prd>\\1', text)
    text = re.sub(_another_websites, '\\1<prd>', text)
    text = re.sub('[,][.]+', '<prd>', text)
    if '...' in text:
        text = text.replace('...', '<prd><prd><prd>')
    if 'Ph.D' in text:
        text = text.replace('Ph.D.', 'Ph<prd>D<prd>')
    text = re.sub('[.]\\s*[,]', '<prd>,', text)
    text = re.sub(_before_digits + '\\s*[.]\\s*' + _digits, '\\1<prd>\\2', text)
    text = re.sub(_month + '[.]\\s*' + _digits, '\\1<prd>\\2', text)
    text = re.sub('\\s' + _alphabets + '[.][ ]+', ' \\1<prd> ', text)
    text = re.sub(_acronyms + ' ' + _starters, '\\1<stop> \\2', text)
    text = re.sub(
        _alphabets + '[.]' + _alphabets + '[.]' + _alphabets + '[.]',
        '\\1<prd>\\2<prd>\\3<prd>', text,
    )
    text = re.sub(_alphabets + '[.]' + _alphabets + '[.]', '\\1<prd>\\2<prd>', text)
    text = re.sub(' ' + _suffixes + '[.][ ]+' + _starters, ' \\1<stop> \\2', text)
    text = re.sub(' ' + _suffixes + '[.]', ' \\1<prd>', text)
    text = re.sub(' ' + _alphabets + '[.]', ' \\1<prd>', text)
    text = re.sub(_digits + '[.]' + _digits, '\\1<prd>\\2', text)
    text = re.sub(_digits + '[.]', '\\1<prd>', text)
    if '"' in text:
        text = text.replace('."', '".')
    if '\u201c' in text:
        text = text.replace('.\u201c', '\u201c.')
    if '!' in text:
        text = text.replace('!"', '"!')
    if '?' in text:
        text = text.replace('?"', '"?')
    text = text.replace('.', '.<stop>')
    text = text.replace('?', '?<stop>')
    text = text.replace('!', '!<stop>')
    text = text.replace('<prd>', '.')
    sentences = text.split('<stop>')
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences if len(s) > minimum_length]
    sentences = [s[:-1] if len(s) >= 2 and s[-2] in ';:-?!.' else s for s in sentences]
    return sentences
