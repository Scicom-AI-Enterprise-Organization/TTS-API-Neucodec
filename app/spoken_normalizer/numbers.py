"""
Number verbalization for the spoken normalizer: English, Malay, Mandarin, Tamil.

Pure Python, no dependencies. Every function takes plain ints / digit strings and returns
the words the LLM normalizer produces for the same input (bench/results/normalizer_truth.jsonl
is the reference; see app/spoken_normalizer/__init__.py for the conventions and where they
deliberately differ from the LLM).
"""

# --------------------------------------------------------------------------- English
_EN_UNITS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
             'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
             'seventeen', 'eighteen', 'nineteen']
_EN_TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
_EN_SCALES = [(10 ** 12, 'trillion'), (10 ** 9, 'billion'), (10 ** 6, 'million'), (1000, 'thousand')]
_EN_ORD_IRREGULAR = {'one': 'first', 'two': 'second', 'three': 'third', 'five': 'fifth',
                     'eight': 'eighth', 'nine': 'ninth', 'twelve': 'twelfth'}


def _en_below_1000(n):
    parts = []
    if n >= 100:
        parts.append(_EN_UNITS[n // 100] + ' hundred')
        n %= 100
    if n >= 20:
        word = _EN_TENS[n // 10]
        if n % 10:
            word += '-' + _EN_UNITS[n % 10]
        parts.append(word)
    elif n > 0 or not parts:
        parts.append(_EN_UNITS[n])
    return ' '.join(parts)


def en_cardinal(n):
    """1250 -> 'one thousand two hundred fifty' (the LLM never says 'and')."""
    n = int(n)
    if n < 0:
        return 'minus ' + en_cardinal(-n)
    if n < 1000:
        return _en_below_1000(n)
    parts = []
    for value, name in _EN_SCALES:
        if n >= value:
            parts.append(_en_below_1000(n // value) + ' ' + name)
            n %= value
    if n:
        parts.append(_en_below_1000(n))
    return ' '.join(parts)


def en_ordinal(n):
    """3 -> 'third', 21 -> 'twenty-first', 12 -> 'twelfth'."""
    words = en_cardinal(n)
    head, sep, last = words.rpartition('-') if '-' in words.split(' ')[-1] else words.rpartition(' ')
    if last in _EN_ORD_IRREGULAR:
        last = _EN_ORD_IRREGULAR[last]
    elif last.endswith('y'):
        last = last[:-1] + 'ieth'
    else:
        last += 'th'
    return head + sep + last


def en_year(n):
    """1998 -> 'nineteen ninety-eight', 2005 -> 'two thousand and five', 2024 -> 'twenty twenty-four'."""
    n = int(n)
    if 2000 <= n <= 2009:
        return 'two thousand' + (' and ' + _EN_UNITS[n - 2000] if n > 2000 else '')
    if 1000 <= n <= 9999 and n % 100 != 0 and not (2000 <= n <= 2009):
        hi, lo = divmod(n, 100)
        if lo < 10:
            return _en_below_1000(hi) + ' oh ' + _EN_UNITS[lo] if 1100 <= n <= 1999 else en_cardinal(n)
        return _en_below_1000(hi) + ' ' + _en_below_1000(lo)
    if n % 100 == 0 and 1100 <= n <= 9999 and n % 1000 != 0:
        return _en_below_1000(n // 100) + ' hundred'
    return en_cardinal(n)


def en_digits(s):
    return ' '.join(_EN_UNITS[int(c)] for c in s if c.isdigit())


# --------------------------------------------------------------------------- Malay
_MS_UNITS = ['kosong', 'satu', 'dua', 'tiga', 'empat', 'lima', 'enam', 'tujuh', 'lapan', 'sembilan']
_MS_SCALES = [(10 ** 12, 'trilion'), (10 ** 9, 'bilion'), (10 ** 6, 'juta'), (1000, 'ribu')]


def _ms_below_1000(n):
    parts = []
    if n >= 100:
        parts.append('seratus' if n // 100 == 1 else _MS_UNITS[n // 100] + ' ratus')
        n %= 100
    if n >= 20:
        parts.append(_MS_UNITS[n // 10] + ' puluh')
        if n % 10:
            parts.append(_MS_UNITS[n % 10])
    elif n >= 10:
        parts.append('sepuluh' if n == 10 else 'sebelas' if n == 11 else _MS_UNITS[n % 10] + ' belas')
    elif n > 0 or not parts:
        parts.append(_MS_UNITS[n])
    return ' '.join(parts)


def ms_cardinal(n):
    """1250 -> 'seribu dua ratus lima puluh'; 0 -> 'sifar' (the LLM's standalone zero)."""
    n = int(n)
    if n == 0:
        return 'sifar'
    if n < 0:
        return 'negatif ' + ms_cardinal(-n)
    if n < 1000:
        return _ms_below_1000(n)
    parts = []
    for value, name in _MS_SCALES:
        if n >= value:
            q = n // value
            parts.append('seribu' if (q == 1 and name == 'ribu') else _ms_below_1000(q) + ' ' + name)
            n %= value
    if n:
        parts.append(_ms_below_1000(n))
    return ' '.join(parts)


def ms_ordinal(n):
    """ke-3 -> 'ketiga', ke-1 -> 'pertama', ke-12 -> 'kedua belas'."""
    n = int(n)
    if n == 1:
        return 'pertama'
    return 'ke' + ms_cardinal(n)


def ms_digits(s):
    return ' '.join(_MS_UNITS[int(c)] for c in s if c.isdigit())


# --------------------------------------------------------------------------- Mandarin
_ZH_DIGITS = '零一二三四五六七八九'
_ZH_SMALL = ['', '十', '百', '千']
_ZH_BIG = ['', '万', '亿', '万亿']


def _zh_group(n):
    digits = str(n)
    out = ''
    zero_pending = False
    for idx, ch in enumerate(digits):
        d = int(ch)
        place = len(digits) - idx - 1
        if d == 0:
            if out:
                zero_pending = True
            continue
        if zero_pending:
            out += '零'
            zero_pending = False
        out += _ZH_DIGITS[d]
        if place:
            out += _ZH_SMALL[place]
    return out


def zh_cardinal(n):
    """1250 -> '一千二百五十', 110 -> '一百一十', 12 -> '十二', 33000000 -> '三千三百万'."""
    n = int(n)
    if n < 0:
        return '负' + zh_cardinal(-n)
    if n == 0:
        return '零'
    groups = []
    while n:
        groups.insert(0, n % 10000)
        n //= 10000
    out = ''
    prev_value = prev_gap = False
    for i, g in enumerate(groups):
        place = len(groups) - 1 - i
        if g == 0:
            prev_gap = prev_gap or prev_value
            continue
        if prev_value and (prev_gap or g < 1000):
            out += '零'
        out += _zh_group(g) + _ZH_BIG[place]
        prev_value, prev_gap = True, False
    if out.startswith('一十'):
        out = out[1:]
    return out


def zh_digits(s):
    return ''.join(_ZH_DIGITS[int(c)] for c in s if c.isdigit())


# --------------------------------------------------------------------------- Tamil
_TA_UNITS = ['பூஜ்ஜியம்', 'ஒன்று', 'இரண்டு', 'மூன்று', 'நான்கு', 'ஐந்து', 'ஆறு', 'ஏழு', 'எட்டு', 'ஒன்பது']
_TA_TEENS = ['பத்து', 'பதினொன்று', 'பன்னிரண்டு', 'பதிமூன்று', 'பதினான்கு', 'பதினைந்து', 'பதினாறு',
             'பதினேழு', 'பதினெட்டு', 'பத்தொன்பது']
_TA_TENS = {2: 'இருபது', 3: 'முப்பது', 4: 'நாற்பது', 5: 'ஐம்பது', 6: 'அறுபது', 7: 'எழுபது', 8: 'எண்பது', 9: 'தொண்ணூறு'}
_TA_TENS_OBL = {2: 'இருபத்து', 3: 'முப்பத்து', 4: 'நாற்பத்து', 5: 'ஐம்பத்து', 6: 'அறுபத்து', 7: 'எழுபத்து',
                8: 'எண்பத்து', 9: 'தொண்ணூற்று'}
_TA_HUNDREDS = {1: 'நூறு', 2: 'இருநூறு', 3: 'முந்நூறு', 4: 'நானூறு', 5: 'ஐந்நூறு', 6: 'அறுநூறு', 7: 'எழுநூறு',
                8: 'எண்ணூறு', 9: 'தொள்ளாயிரம்'}
_TA_HUNDREDS_OBL = {1: 'நூற்று', 2: 'இருநூற்று', 3: 'முந்நூற்று', 4: 'நானூற்று', 5: 'ஐந்நூற்று', 6: 'அறுநூற்று',
                    7: 'எழுநூற்று', 8: 'எண்ணூற்று', 9: 'தொள்ளாயிரத்து'}
_TA_THOUSANDS = {1: 'ஆயிரம்', 2: 'இரண்டாயிரம்', 3: 'மூவாயிரம்', 4: 'நான்காயிரம்', 5: 'ஐயாயிரம்', 6: 'ஆறாயிரம்',
                 7: 'ஏழாயிரம்', 8: 'எண்ணாயிரம்', 9: 'ஒன்பதாயிரம்'}
# independent vowel -> dependent vowel sign, for sandhi (இருபத்து + ஒன்று -> இருபத்தொன்று)
_TA_VOWEL_SIGN = {'அ': '', 'ஆ': 'ா', 'இ': 'ி', 'ஈ': 'ீ', 'உ': 'ு', 'ஊ': 'ூ', 'எ': 'ெ', 'ஏ': 'ே', 'ஐ': 'ை',
                  'ஒ': 'ொ', 'ஓ': 'ோ', 'ஔ': 'ௌ'}
_TA_U_SIGN = 'ு'


def _ta_contract(oblique, word):
    """Oblique tens 20-80 (…த்து) + vowel-initial unit fuse: இருபத்து + ஐந்து -> இருபத்தைந்து.
    Consonant-initial units (மூன்று, நான்கு) stay a separate word, and so does everything
    after தொண்ணூற்று (90): the LLM writes தொண்ணூற்று எட்டு, not தொண்ணூற்றெட்டு."""
    if word[0] in _TA_VOWEL_SIGN and oblique.endswith('த்து'):
        return oblique[:-1] + _TA_VOWEL_SIGN[word[0]] + word[1:]
    return oblique + ' ' + word


def _ta_below_100(n, oblique=False):
    if n < 10:
        return _TA_UNITS[n]
    if n < 20:
        return _TA_TEENS[n - 10]
    tens, unit = divmod(n, 10)
    if unit == 0:
        return _TA_TENS_OBL[tens] if oblique else _TA_TENS[tens]
    return _ta_contract(_TA_TENS_OBL[tens], _TA_UNITS[unit])


def _ta_below_1000(n, oblique=False):
    if n < 100:
        return _ta_below_100(n, oblique)
    hundreds, rest = divmod(n, 100)
    if rest == 0:
        return _TA_HUNDREDS_OBL[hundreds] if oblique else _TA_HUNDREDS[hundreds]
    return _TA_HUNDREDS_OBL[hundreds] + ' ' + _ta_below_100(rest, oblique)


def _ta_thousands(q, followed):
    """q thousands (1 <= q <= 999). Oblique (…த்து) when more digits follow: ஆயிரத்து ஐந்நூறு."""
    if q < 10:
        word = _TA_THOUSANDS[q]
        return word[:-2] + 'த்து' if followed else word   # …ம் -> …த்து
    word = _ta_below_1000(q)
    if word.endswith('ு'):                           # ஐம்பது + ஆயிரம் -> ஐம்பதாயிரம், as the LLM fuses it
        return word[:-1] + ('ாயிரத்து' if followed else 'ாயிரம்')
    return word + (' ஆயிரத்து' if followed else ' ஆயிரம்')


def ta_cardinal(n):
    """250 -> 'இருநூற்று ஐம்பது', 1500 -> 'ஆயிரத்து ஐந்நூறு', 2024 -> 'இரண்டாயிரத்து இருபத்து நான்கு',
    1998 -> 'ஆயிரத்து தொள்ளாயிரத்து தொண்ணூற்று எட்டு', 33000000 -> 'முப்பத்து மூன்று மில்லியன்'."""
    n = int(n)
    if n < 0:
        return 'மைனஸ் ' + ta_cardinal(-n)
    if n < 1000:
        return _ta_below_1000(n)
    parts = []
    if 10 ** 5 <= n < 10 ** 7:
        # Indian grouping below a crore, as the LLM says it: ஒரு லட்சம், பத்து லட்சம் (but
        # முப்பத்து மூன்று மில்லியன் for 33,000,000 -- it switches systems at 10^7)
        lakhs, n = divmod(n, 10 ** 5)
        head = 'ஒரு' if lakhs == 1 else _ta_below_100(lakhs)
        parts.append(head + (' லட்சத்து' if n else ' லட்சம்'))
    for value, name in ((10 ** 9, 'பில்லியன்'), (10 ** 6, 'மில்லியன்')):
        if n >= value:
            parts.append(_ta_below_1000(n // value) + ' ' + name)
            n %= value
    if n >= 1000:
        q, n = divmod(n, 1000)
        parts.append(_ta_thousands(q, followed=n > 0))
    if n:
        parts.append(_ta_below_1000(n))
    return ' '.join(parts)


def ta_ordinal(n, suffix='ஆவது'):
    """3 -> 'மூன்றாவது', 21 -> 'இருபத்தொன்றாவது'; suffix 'ஆம்' gives 'மூன்றாம்'. 1 -> 'முதலாவது' / 'முதலாம்'."""
    n = int(n)
    if n == 1:
        return 'முதல' + _TA_VOWEL_SIGN['ஆ'] + suffix[1:]   # முதலாவது / முதலாம்
    word = ta_cardinal(n)
    if word.endswith(_TA_U_SIGN):            # final -u drops before the vowel-initial suffix
        return word[:-1] + _TA_VOWEL_SIGN['ஆ'] + suffix[1:]
    if word.endswith('ம்'):                  # ஆயிரம் -> ஆயிரத்தாவது
        return word[:-2] + 'த்தாவது' if suffix == 'ஆவது' else word[:-2] + 'த்தாம்'
    return word + suffix


def ta_digits(s):
    return ' '.join(_TA_UNITS[int(c)] for c in s if c.isdigit())


_TA_SIGNS = set('ாிீுூெேைொோௌ')
_TA_CONS_SUFFIX_MA_STEM = ('ஆக', 'ாக', 'ஆகும்', 'ாகும்', 'உம்', 'ும்')


def _ta_stem(word, suffix):
    """The word ready to take a vowel-initial suffix (consonant with inherent -a at the end),
    or None when we do not know how this word inflects."""
    if word.endswith('நூறு'):                    # நூறு -> நூற்ற- (நூற்றில், நூற்றுக்கு)
        return word[:-1] + '்ற'
    if word.endswith('ு'):                       # ஐந்து -> ஐந்த- (ஐந்தில், ஐந்தை, ஐந்தால்)
        return word[:-1]
    if word.endswith('ம்'):                      # ஆயிரம் -> ஆயிரத்த- (ஆயிரத்தில்) / ஆயிரம- (ஆயிரமாக)
        return word[:-1] if suffix in _TA_CONS_SUFFIX_MA_STEM else word[:-2] + 'த்த'
    if word.endswith('ட்'):                      # ரிங்கிட் -> ரிங்கிட்ட- (ரிங்கிட்டை, ரிங்கிட்டுக்கு)
        return word + 'ட'
    if word.endswith('ன்') and len(word) <= 4:   # சென் -> சென்ன-
        return word + 'ன'
    if word.endswith('்'):                       # மில்லியன் -> மில்லியன-, டாலர் -> டாலர-
        return word[:-1]
    if word[-1] in _TA_SIGNS or word[-1] in _TA_VOWEL_SIGN:   # மணி -> மணிய- (மணியில்)
        return word + 'ய'
    return None


def ta_attach(word, suffix):
    """Glue a case suffix written directly after a digit onto its number word with sandhi:
    2024ல் -> இரண்டாயிரத்து இருபத்து நான்கில், 12ஆல் -> பன்னிரண்டால், RM1,000ஐ -> ஆயிரம் ரிங்கிட்டை,
    15ஆகும் -> பதினைந்தாகும், 2030க்குள் -> இரண்டாயிரத்து முப்பதுக்குள். Unknown shapes are
    returned glued as written."""
    if not word or not suffix:
        return word + suffix
    head, _, last = word.rpartition(' ')
    if suffix in ('ல்', 'லிருந்து'):             # bare ல் after a digit is இல்
        suffix = 'இ' + suffix
    if suffix[0] in _TA_VOWEL_SIGN:              # independent vowel -> sign
        sign, rest = _TA_VOWEL_SIGN[suffix[0]], suffix[1:]
    elif suffix[0] in _TA_SIGNS:                 # already a sign (ால், ை, ும்)
        sign, rest = suffix[0], suffix[1:]
    else:                                        # consonant-initial: க்கு, வது
        sign, rest = None, suffix
    stem = _ta_stem(last, suffix)
    if stem is None:
        return word + suffix
    if sign is None:
        if last[-1] in _TA_SIGNS or last[-1] in _TA_VOWEL_SIGN:
            joined = last + rest                 # மணி + க்கு -> மணிக்கு
        else:
            joined = stem + 'ு' + rest           # ஐந்து + க்கு -> ஐந்துக்கு, ஆயிரம் + க்கு -> ஆயிரத்துக்கு
    else:
        joined = stem + sign + rest              # ஐந்து + இல் -> ஐந்தில், ஆயிரம் + ஐ -> ஆயிரத்தை
    return (head + ' ' if head else '') + joined


def ta_half(n):
    """n.5 before மணி: 1.5 -> ஒன்றரை, 2.5 -> இரண்டரை, 0.5 -> அரை."""
    n = int(n)
    if n == 0:
        return 'அரை'
    word = ta_cardinal(n)
    stem = _ta_stem(word, 'ரை')
    return (stem if stem and word.endswith('ு') else word) + 'ரை'


# --------------------------------------------------------------------------- dispatch
CARDINAL = {'en': en_cardinal, 'ms': ms_cardinal, 'zh': zh_cardinal, 'ta': ta_cardinal}
DIGITS = {'en': en_digits, 'ms': ms_digits, 'zh': zh_digits, 'ta': ta_digits}


def cardinal(n, lang):
    return CARDINAL[lang](n)


def digits(s, lang):
    return DIGITS[lang](s)


def decimal(int_part, frac_part, lang):
    """'37', '5' -> 'thirty-seven point five' / 'tiga puluh tujuh perpuluhan lima' / '三十七点五' /
    'முப்பத்து ஏழு புள்ளி ஐந்து'. Fraction digits are read one by one, as the LLM does ('3.25' ->
    'tiga perpuluhan dua lima')."""
    whole = cardinal(int(int_part), lang) if int_part else cardinal(0, lang)
    if lang == 'ms' and int_part and int(int_part) == 0:
        whole = 'kosong'
    frac = digits(frac_part, lang)
    if lang == 'zh':
        return whole + '点' + frac
    point = {'en': 'point', 'ms': 'perpuluhan', 'ta': 'புள்ளி'}[lang]
    return f'{whole} {point} {frac}'


def year(n, lang):
    n = int(n)
    if lang == 'en':
        return en_year(n)
    if lang == 'zh':
        return zh_digits(str(n))
    return cardinal(n, lang)
