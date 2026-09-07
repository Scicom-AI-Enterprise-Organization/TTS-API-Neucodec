"""
Deterministic written -> spoken verbalizers per locale, used to fill the typed slots of the
template generator (generate.py). Every function returns the exact words a speaker would say.

  en ms zh ta ta-LK  app.spoken_normalizer (validated against the LLM; see bench/NORMALIZER.md)
  fr de es it pt nl pl ar id  num2words for the number words, locale conventions here
  tl si                        own tables (no library covers Filipino or Sinhala)

Not every slot is deterministically safe in every locale (Polish dates decline, Arabic counted
nouns change form, Filipino counts take a linker): `SAFE_SLOTS` says which ones are, and the
generator leaves the rest to the LLM-normalized pairs (llm_pairs.py). Locales flagged in
`NEEDS_NATIVE_REVIEW` were written from grammar references, not by a native speaker.
"""
import re

from num2words import num2words as _n2w

# --------------------------------------------------------------------------- locale tables
LOCALES = ['en', 'ms', 'id', 'zh', 'ta', 'ta-LK', 'si', 'tl', 'ar', 'fr', 'es', 'de', 'it', 'pt', 'nl', 'pl']
LANGUAGE_NAME = {'en': 'English', 'ms': 'Malay', 'id': 'Indonesian', 'zh': 'Mandarin Chinese', 'ta': 'Tamil',
                 'ta-LK': 'Tamil (Sri Lanka)', 'si': 'Sinhala', 'tl': 'Filipino (Tagalog)', 'ar': 'Arabic',
                 'fr': 'French', 'es': 'Spanish', 'de': 'German', 'it': 'Italian', 'pt': 'Portuguese',
                 'nl': 'Dutch', 'pl': 'Polish'}
N2W_LANG = {'fr': 'fr', 'de': 'de', 'es': 'es', 'it': 'it', 'pt': 'pt', 'nl': 'nl', 'pl': 'pl', 'ar': 'ar', 'id': 'id'}
SPOKEN_NORMALIZER_LANG = {'en': 'en', 'ms': 'ms', 'zh': 'zh', 'ta': 'ta', 'ta-LK': 'ta'}
NEEDS_NATIVE_REVIEW = {'si', 'tl', 'ar', 'pl'}

ALL_SLOTS = ['int', 'big', 'money', 'decimal', 'percent', 'phone', 'digits', 'date', 'time', 'ordinal',
             'range', 'unit', 'year', 'email', 'url', 'id']
_UNSAFE = {
    'pl': {'date', 'time', 'ordinal', 'range', 'unit', 'year'},      # genitive ordinals, noun declension
    'ar': {'date', 'time', 'ordinal', 'range', 'unit'},              # counted-noun agreement, feminine hours
    'si': {'date', 'time', 'ordinal', 'range', 'unit'},              # case suffixes on digits, -යි rules
    'tl': {'date', 'time'},                                          # Spanish-derived time/dates in speech
}
SAFE_SLOTS = {loc: [s for s in ALL_SLOTS if s not in _UNSAFE.get(loc, set())] for loc in LOCALES}

# Code-switched pairs (codeswitch.py): hand-written frames, one language per slot. They are not
# in LOCALES because nothing here verbalizes a *pair* -- every slot is filled and read as one of
# the two real languages, all of which are app.spoken_normalizer languages.
from .codeswitch import CS_LOCALES, CS_PAIRS, CS_LANGUAGE_NAME  # noqa: E402
LANGUAGE_NAME.update(CS_LANGUAGE_NAME)
ALL_LOCALES = LOCALES + CS_LOCALES

DIGIT_WORDS = {
    'fr': 'zéro un deux trois quatre cinq six sept huit neuf'.split(),
    'de': 'null eins zwei drei vier fünf sechs sieben acht neun'.split(),
    'es': 'cero uno dos tres cuatro cinco seis siete ocho nueve'.split(),
    'it': 'zero uno due tre quattro cinque sei sette otto nove'.split(),
    'pt': 'zero um dois três quatro cinco seis sete oito nove'.split(),
    'nl': 'nul een twee drie vier vijf zes zeven acht negen'.split(),
    'pl': 'zero jeden dwa trzy cztery pięć sześć siedem osiem dziewięć'.split(),
    'ar': 'صفر واحد اثنان ثلاثة أربعة خمسة ستة سبعة ثمانية تسعة'.split(),
    'id': 'nol satu dua tiga empat lima enam tujuh delapan sembilan'.split(),
    'tl': 'zero isa dalawa tatlo apat lima anim pito walo siyam'.split(),
    'si': 'බිංදුව එක දෙක තුන හතර පහ හය හත අට නවය'.split(),
}
WORDS = {   # per-locale function words used by the slots
    'fr': dict(percent='pour cent', point='virgule', to='à', at='arobase', dot='point', dash='tiret', slash='slash', plus='plus', hours='heures', hour='heure'),
    'de': dict(percent='Prozent', point='Komma', to='bis', at='at', dot='Punkt', dash='Bindestrich', slash='Schrägstrich', plus='plus', hours='Uhr'),
    'es': dict(percent='por ciento', point='coma', to='a', at='arroba', dot='punto', dash='guion', slash='barra', plus='más'),
    'it': dict(percent='per cento', point='virgola', to='a', at='chiocciola', dot='punto', dash='trattino', slash='barra', plus='più'),
    'pt': dict(percent='por cento', point='vírgula', to='a', at='arroba', dot='ponto', dash='hífen', slash='barra', plus='mais'),
    'nl': dict(percent='procent', point='komma', to='tot', at='at', dot='punt', dash='streepje', slash='slash', plus='plus', hours='uur'),
    'pl': dict(percent='procent', point='przecinek', to='do', at='małpa', dot='kropka', dash='myślnik', slash='ukośnik', plus='plus'),
    'ar': dict(percent='بالمئة', point='فاصلة', to='إلى', at='آت', dot='نقطة', dash='شرطة', slash='شرطة مائلة', plus='زائد'),
    'id': dict(percent='persen', point='koma', to='sampai', at='et', dot='titik', dash='strip', slash='garis miring', plus='plus'),
    'tl': dict(percent='porsyento', point='punto', to='hanggang', at='at', dot='dot', dash='dash', slash='slash', plus='plus'),
    'si': dict(percent='සියයට', point='දශම', to='සිට', at='ඇට්', dot='ඩොට්', dash='ඩෑෂ්', slash='ස්ලෑෂ්', plus='ප්ලස්'),
}

# currency -> (written variants with {amt}, spoken major (1, many), spoken minor (1, many))
CURRENCIES = {
    'fr': {'EUR': (['{amt} €', '{amt}€', 'EUR {amt}'], ('euro', 'euros'), ('centime', 'centimes'))},
    'de': {'EUR': (['{amt} €', '{amt}€', 'EUR {amt}'], ('Euro', 'Euro'), ('Cent', 'Cent'))},
    'es': {'EUR': (['{amt} €', '{amt}€', 'EUR {amt}'], ('euro', 'euros'), ('céntimo', 'céntimos'))},
    'it': {'EUR': (['{amt} €', '{amt}€', 'EUR {amt}'], ('euro', 'euro'), ('centesimo', 'centesimi'))},
    'pt': {'EUR': (['{amt} €', '{amt}€', 'EUR {amt}'], ('euro', 'euros'), ('cêntimo', 'cêntimos')),
           'BRL': (['R$ {amt}', 'R${amt}'], ('real', 'reais'), ('centavo', 'centavos'))},
    'nl': {'EUR': (['€ {amt}', '€{amt}', 'EUR {amt}'], ('euro', 'euro'), ('cent', 'cent'))},
    'pl': {'PLN': (['{amt} zł', '{amt}zł', 'PLN {amt}'], None, None)},          # num2words pl declines złoty/grosz
    'ar': {'SAR': (['{amt} ريال', '{amt} ر.س', 'SAR {amt}'], ('ريال', 'ريالات'), ('هللة', 'هللات')),
           'AED': (['{amt} درهم', '{amt} د.إ', 'AED {amt}'], ('درهم', 'دراهم'), ('فلس', 'فلوس')),
           'EGP': (['{amt} جنيه', '{amt} ج.م', 'EGP {amt}'], ('جنيه', 'جنيهات'), ('قرش', 'قروش')),
           'USD': (['{amt} دولار', '${amt}', 'USD {amt}'], ('دولار', 'دولارات'), ('سنت', 'سنتات'))},
    'id': {'IDR': (['Rp{amt}', 'Rp {amt}', 'IDR {amt}'], ('rupiah', 'rupiah'), ('sen', 'sen'))},
    'tl': {'PHP': (['₱{amt}', 'P{amt}', 'Php {amt}', 'PHP {amt}'], ('piso', 'piso'), ('sentimo', 'sentimo'))},
    'si': {'LKR': (['රු. {amt}', 'රු.{amt}', 'Rs. {amt}', 'LKR {amt}'], ('රුපියල්', 'රුපියල්'), ('ශත', 'ශත'))},
}
# thousands separator, decimal separator per locale (written form)
NUMBER_FORMAT = {'fr': (' ', ','), 'de': ('.', ','), 'es': ('.', ','), 'it': ('.', ','), 'pt': ('.', ','), 'nl': ('.', ','),
                 'pl': (' ', ','), 'ar': (',', '.'), 'id': ('.', ','), 'tl': (',', '.'), 'si': (',', '.'),
                 'en': (',', '.'), 'ms': (',', '.'), 'zh': (',', '.'), 'ta': (',', '.'), 'ta-LK': (',', '.')}
EASTERN_DIGITS = str.maketrans('0123456789', '٠١٢٣٤٥٦٧٨٩')

MONTHS = {
    'fr': 'janvier février mars avril mai juin juillet août septembre octobre novembre décembre'.split(),
    'de': 'Januar Februar März April Mai Juni Juli August September Oktober November Dezember'.split(),
    'es': 'enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre'.split(),
    'it': 'gennaio febbraio marzo aprile maggio giugno luglio agosto settembre ottobre novembre dicembre'.split(),
    'pt': 'janeiro fevereiro março abril maio junho julho agosto setembro outubro novembro dezembro'.split(),
    'nl': 'januari februari maart april mei juni juli augustus september oktober november december'.split(),
    'pl': 'stycznia lutego marca kwietnia maja czerwca lipca sierpnia września października listopada grudnia'.split(),
    'ar': 'يناير فبراير مارس أبريل مايو يونيو يوليو أغسطس سبتمبر أكتوبر نوفمبر ديسمبر'.split(),
    'id': 'Januari Februari Maret April Mei Juni Juli Agustus September Oktober November Desember'.split(),
    'tl': 'Enero Pebrero Marso Abril Mayo Hunyo Hulyo Agosto Setyembre Oktubre Nobyembre Disyembre'.split(),
    'si': 'ජනවාරි පෙබරවාරි මාර්තු අප්‍රේල් මැයි ජූනි ජූලි අගෝස්තු සැප්තැම්බර් ඔක්තෝබර් නොවැම්බර් දෙසැම්බර්'.split(),
}
UNITS = {   # unit -> (written variants, spoken singular, spoken plural)
    'fr': {'kg': (['kg'], 'kilogramme', 'kilogrammes'), 'km': (['km'], 'kilomètre', 'kilomètres'), 'm': (['m'], 'mètre', 'mètres'),
           'cm': (['cm'], 'centimètre', 'centimètres'), 'g': (['g'], 'gramme', 'grammes'), 'l': (['l', 'L'], 'litre', 'litres'),
           'ml': (['ml'], 'millilitre', 'millilitres'), '°C': (['°C', ' °C'], 'degré', 'degrés'), 'GB': (['Go', 'GB'], 'gigaoctet', 'gigaoctets')},
    'de': {'kg': (['kg'], 'Kilogramm', 'Kilogramm'), 'km': (['km'], 'Kilometer', 'Kilometer'), 'm': (['m'], 'Meter', 'Meter'),
           'cm': (['cm'], 'Zentimeter', 'Zentimeter'), 'g': (['g'], 'Gramm', 'Gramm'), 'l': (['l', 'Liter'], 'Liter', 'Liter'),
           'ml': (['ml'], 'Milliliter', 'Milliliter'), '°C': (['°C', ' °C', ' Grad'], 'Grad', 'Grad'), 'GB': (['GB'], 'Gigabyte', 'Gigabyte')},
    'es': {'kg': (['kg'], 'kilogramo', 'kilogramos'), 'km': (['km'], 'kilómetro', 'kilómetros'), 'm': (['m'], 'metro', 'metros'),
           'cm': (['cm'], 'centímetro', 'centímetros'), 'g': (['g'], 'gramo', 'gramos'), 'l': (['l', 'L'], 'litro', 'litros'),
           'ml': (['ml'], 'mililitro', 'mililitros'), '°C': (['°C', ' °C', ' grados'], 'grado', 'grados'), 'GB': (['GB'], 'gigabyte', 'gigabytes')},
    'it': {'kg': (['kg'], 'chilogrammo', 'chilogrammi'), 'km': (['km'], 'chilometro', 'chilometri'), 'm': (['m'], 'metro', 'metri'),
           'cm': (['cm'], 'centimetro', 'centimetri'), 'g': (['g'], 'grammo', 'grammi'), 'l': (['l', 'L'], 'litro', 'litri'),
           'ml': (['ml'], 'millilitro', 'millilitri'), '°C': (['°C', ' °C', ' gradi'], 'grado', 'gradi'), 'GB': (['GB'], 'gigabyte', 'gigabyte')},
    'pt': {'kg': (['kg'], 'quilograma', 'quilogramas'), 'km': (['km'], 'quilómetro', 'quilómetros'), 'm': (['m'], 'metro', 'metros'),
           'cm': (['cm'], 'centímetro', 'centímetros'), 'g': (['g'], 'grama', 'gramas'), 'l': (['l', 'L'], 'litro', 'litros'),
           'ml': (['ml'], 'mililitro', 'mililitros'), '°C': (['°C', ' °C', ' graus'], 'grau', 'graus'), 'GB': (['GB'], 'gigabyte', 'gigabytes')},
    'nl': {'kg': (['kg'], 'kilo', 'kilo'), 'km': (['km'], 'kilometer', 'kilometer'), 'm': (['m'], 'meter', 'meter'),
           'cm': (['cm'], 'centimeter', 'centimeter'), 'g': (['g'], 'gram', 'gram'), 'l': (['l', 'L'], 'liter', 'liter'),
           'ml': (['ml'], 'milliliter', 'milliliter'), '°C': (['°C', ' °C', ' graden'], 'graad', 'graden'), 'GB': (['GB'], 'gigabyte', 'gigabyte')},
    'id': {'kg': (['kg'], 'kilogram', 'kilogram'), 'km': (['km'], 'kilometer', 'kilometer'), 'm': (['m'], 'meter', 'meter'),
           'cm': (['cm'], 'sentimeter', 'sentimeter'), 'g': (['g', 'gr'], 'gram', 'gram'), 'l': (['l', 'L'], 'liter', 'liter'),
           'ml': (['ml'], 'mililiter', 'mililiter'), '°C': (['°C', ' °C', ' derajat'], 'derajat', 'derajat'), 'GB': (['GB'], 'gigabyte', 'gigabyte')},
    'tl': {'kg': (['kg', ' kilo'], 'kilo', 'kilo'), 'km': (['km'], 'kilometro', 'kilometro'), 'm': (['m'], 'metro', 'metro'),
           'g': (['g'], 'gramo', 'gramo'), 'l': (['L'], 'litro', 'litro'), 'GB': (['GB'], 'gigabyte', 'gigabyte')},
}
FR_PLURAL_UNITS = {'fr', 'es', 'it', 'pt'}


# --------------------------------------------------------------------------- number words
def n2w(n, lang, **kw):
    out = _n2w(n, lang=N2W_LANG[lang], **kw)
    if lang == 'ar':
        out = out.replace(' و ', ' و')          # attach و to the next word, as Arabic is written
    return out


def _tl_linker(word):
    """Filipino linker: lima -> limang, apat -> apat na, sampu -> sampung."""
    if word[-1] in 'aeiou':
        return word + 'ng'
    if word.endswith('n'):
        return word[:-1] + 'ng'
    return word + ' na'


_TL_UNITS = ['zero', 'isa', 'dalawa', 'tatlo', 'apat', 'lima', 'anim', 'pito', 'walo', 'siyam']
_TL_TEENS = ['sampu', 'labing-isa', 'labindalawa', 'labintatlo', 'labing-apat', 'labinlima', 'labing-anim', 'labimpito', 'labingwalo', 'labinsiyam']
_TL_TENS = ['', '', 'dalawampu', 'tatlumpu', 'apatnapu', 'limampu', 'animnapu', 'pitumpu', 'walumpu', 'siyamnapu']


def _tl_below_100(n):
    if n < 10:
        return _TL_UNITS[n]
    if n < 20:
        return _TL_TEENS[n - 10]
    tens, unit = divmod(n, 10)
    return _TL_TENS[tens] + (f"'t {_TL_UNITS[unit]}" if unit else '')


def _tl_hundreds(h):
    word = _tl_linker(_TL_UNITS[h])
    return word + (' raan' if word.endswith(' na') else ' daan')


def tl_cardinal(n):
    """1250 -> 'isang libo dalawang daan at limampu'."""
    n = int(n)
    if n < 100:
        return _tl_below_100(n)
    parts = []
    if n >= 10 ** 6:
        parts.append(_tl_linker(tl_cardinal(n // 10 ** 6)) + ' milyon')
        n %= 10 ** 6
    if n >= 1000:
        parts.append(_tl_linker(tl_cardinal(n // 1000)) + ' libo')
        n %= 1000
    if n >= 100:
        parts.append(_tl_hundreds(n // 100))
        n %= 100
    if n:
        parts.append('at ' + _tl_below_100(n))
    return ' '.join(parts)


_TL_ORD_SPECIAL = {1: 'una', 2: 'ikalawa', 3: 'ikatlo'}


def tl_ordinal(n):
    n = int(n)
    if n in _TL_ORD_SPECIAL:
        return _TL_ORD_SPECIAL[n]
    return 'ika' + tl_cardinal(n)


_SI_UNITS = ['බිංදුව', 'එක', 'දෙක', 'තුන', 'හතර', 'පහ', 'හය', 'හත', 'අට', 'නවය']
_SI_TEENS = ['දහය', 'එකොළහ', 'දොළහ', 'දහතුන', 'දාහතර', 'පහළොව', 'දහසය', 'දහහත', 'දහඅට', 'දහනවය']
_SI_TENS = ['', '', 'විස්ස', 'තිහ', 'හතළිහ', 'පනහ', 'හැට', 'හැත්තෑව', 'අසූව', 'අනූව']
_SI_TENS_ATTR = ['', '', 'විසි', 'තිස්', 'හතළිස්', 'පනස්', 'හැට', 'හැත්තෑ', 'අසූ', 'අනූ']
_SI_HUNDREDS = ['', 'සියය', 'දෙසියය', 'තුන්සියය', 'හාරසියය', 'පන්සියය', 'හයසියය', 'හත්සියය', 'අටසියය', 'නවසියය']
_SI_HUNDREDS_ATTR = ['', 'එකසිය', 'දෙසිය', 'තුන්සිය', 'හාරසිය', 'පන්සිය', 'හයසිය', 'හත්සිය', 'අටසිය', 'නවසිය']
_SI_THOUSANDS = ['', 'දහස', 'දෙදහස', 'තුන්දහස', 'හාරදහස', 'පන්දහස', 'හයදහස', 'හත්දහස', 'අටදහස', 'නවදහස']
_SI_THOUSANDS_ATTR = ['', 'එක්දහස්', 'දෙදහස්', 'තුන්දහස්', 'හාරදහස්', 'පන්දහස්', 'හයදහස්', 'හත්දහස්', 'අටදහස්', 'නවදහස්']
_SI_TENK = {10: ('දහදහස', 'දහදහස්'), 20: ('විසිදහස', 'විසිදහස්'), 30: ('තිස්දහස', 'තිස්දහස්'), 40: ('හතළිස්දහස', 'හතළිස්දහස්'),
            50: ('පනස්දහස', 'පනස්දහස්'), 60: ('හැටදහස', 'හැටදහස්'), 70: ('හැත්තෑදහස', 'හැත්තෑදහස්'), 80: ('අසූදහස', 'අසූදහස්'),
            90: ('අනූදහස', 'අනූදහස්')}


def _si_below_100(n):
    if n < 10:
        return _SI_UNITS[n]
    if n < 20:
        return _SI_TEENS[n - 10]
    tens, unit = divmod(n, 10)
    return _SI_TENS[tens] if unit == 0 else _SI_TENS_ATTR[tens] + _SI_UNITS[unit]


def _si_below_1000(n):
    if n < 100:
        return _si_below_100(n)
    h, rest = divmod(n, 100)
    return _SI_HUNDREDS[h] if rest == 0 else _SI_HUNDREDS_ATTR[h] + ' ' + _si_below_100(rest)


def si_cardinal(n):
    """250 -> 'දෙසිය පනහ', 1250 -> 'එක්දහස් දෙසිය පනහ', 2024 -> 'දෙදහස් විසිහතර'. Supports < 100,000
    with round tens of thousands (the generator keeps Sinhala amounts in that range)."""
    n = int(n)
    if n < 1000:
        return _si_below_1000(n)
    q, rest = divmod(n, 1000)
    if q < 10:
        head = _SI_THOUSANDS_ATTR[q] if rest else _SI_THOUSANDS[q]
    elif q in _SI_TENK:
        head = _SI_TENK[q][1 if rest else 0]
    elif 20 < q < 100:
        tens, unit = divmod(q, 10)
        head = _SI_TENS_ATTR[tens] + _SI_THOUSANDS_ATTR[unit] if rest else _SI_TENS_ATTR[tens] + _SI_THOUSANDS[unit]
    else:
        raise ValueError('si_cardinal: thousands 11-19 and n >= 100000 are left to the LLM pairs')
    return head + (' ' + _si_below_1000(rest) if rest else '')


def si_count(n):
    """A completed count takes -යි: රුපියල් පනහයි."""
    return si_cardinal(n) + 'යි'


def cardinal(n, lang):
    n = int(n)
    if lang == 'tl':
        return tl_cardinal(n)
    if lang == 'si':
        return si_cardinal(n)
    return n2w(n, lang)


def digits(s, lang):
    words = DIGIT_WORDS[lang]
    return ' '.join(words[int(c)] for c in s if c.isdigit())


def decimal(whole, frac, lang):
    """'3', '5' -> 'trois virgule cinq'; fraction digits one by one."""
    if lang == 'si':
        return f'{si_cardinal(int(whole))} {WORDS["si"]["point"]} {digits(frac, "si")}'
    return f'{cardinal(int(whole), lang)} {WORDS[lang]["point"]} {digits(frac, lang)}'


def year(n, lang):
    n = int(n)
    if lang in ('fr', 'de', 'es', 'it', 'pt', 'nl'):
        return n2w(n, lang, to='year')
    return cardinal(n, lang)


def ordinal(n, lang):
    n = int(n)
    if lang == 'tl':
        return tl_ordinal(n)
    if lang == 'fr' and n == 1:
        return 'premier'
    return n2w(n, lang, to='ordinal')


def percent(whole, frac, lang):
    num = decimal(whole, frac, lang) if frac else cardinal(int(whole), lang)
    if lang == 'si':
        return f'{WORDS["si"]["percent"]} {num}යි'
    if lang == 'tl':
        return f'{_tl_linker(num)} {WORDS["tl"]["percent"]}'
    return f'{num} {WORDS[lang]["percent"]}'


def _ar_counted(n, singular, plural):
    """Arabic counted noun: 1 ريال واحد, 2 ريالان, 3-10 ريالات, 11-99 ريالاً (accusative), x00 ريال."""
    if n == 1:
        return f'{singular} واحد'
    if n == 2:
        return f'{singular}ان'
    last2 = n % 100
    if 3 <= last2 <= 10:
        return f'{cardinal(n, "ar")} {plural}'
    if last2 == 0 or singular.endswith('و'):            # مئة ريال / يورو (indeclinable)
        return f'{cardinal(n, "ar")} {singular}'
    return f'{cardinal(n, "ar")} {singular}' + ('ً' if singular.endswith('ة') else 'اً')


def money(whole, cents, code, lang):
    """1250, '50', 'EUR' -> 'mille deux cent cinquante euros et cinquante centimes'."""
    whole = int(whole)
    cents = int(cents) if cents else 0
    if lang == 'pl':
        out = n2w(float(whole) + cents / 100, 'pl', to='currency', currency='PLN').replace(', ', ' ')
        return out.replace(' zero groszy', '') if not cents else (out[len('zero złotych '):] if not whole else out)
    _, major, minor = CURRENCIES[lang][code]
    if lang == 'ar':
        out = _ar_counted(whole, *major) if whole else ''
        if cents:
            out += (' و' if out else '') + _ar_counted(cents, *minor)
        return out
    if lang == 'si':
        out = f'{major[0]} {si_count(whole)}' if whole or not cents else ''
        if cents:
            out += (' ' if out else '') + f'{minor[0]} {si_count(cents)}'
        return out
    if lang == 'tl':
        out = f'{_tl_linker(tl_cardinal(whole))} {major[0]}' if whole or not cents else ''
        if cents:
            out += (' at ' if out else '') + f'{_tl_linker(tl_cardinal(cents))} {minor[0]}'
        return out
    joiner = {'fr': ' et ', 'de': ' und ', 'es': ' con ', 'it': ' e ', 'pt': ' e ', 'nl': ' ', 'id': ' '}[lang]
    out = f'{count_word(whole, lang)} {major[0] if whole == 1 else major[1]}' if whole or not cents else ''
    if cents:
        out += (joiner if out else '') + f'{cardinal(cents, lang)} {minor[0] if cents == 1 else minor[1]}'
    return out


_ONE_BEFORE_NOUN = {'de': 'ein', 'es': 'un', 'it': 'un', 'pt': 'um', 'nl': 'één', 'fr': 'un'}


def count_word(n, lang):
    """A cardinal that stands before a noun: 1 -> ein/un (num2words says eins/uno)."""
    if int(n) == 1 and lang in _ONE_BEFORE_NOUN:
        return _ONE_BEFORE_NOUN[lang]
    return cardinal(n, lang)


def unit(whole, frac, u, lang):
    variants, sing, plur = UNITS[lang][u]
    num = decimal(whole, frac, lang) if frac else count_word(int(whole), lang)
    if lang == 'tl':
        return f'{_tl_linker(num)} {sing}'
    plural = not (int(whole) == 1 and not frac)
    word = plur if plural else sing
    if u == '°C':
        word += {'fr': ' Celsius', 'de': ' Celsius', 'es': ' Celsius', 'it': ' Celsius', 'pt': ' Celsius', 'nl': ' Celsius', 'id': ' Celsius'}[lang]
    return f'{num} {word}'


def date_words(d, m, y, lang):
    """Spoken date, day-month(-year), in the locale's running-text form."""
    month = MONTHS[lang][m - 1]
    if lang == 'fr':
        day = 'premier' if d == 1 else cardinal(d, 'fr')
        return f'{day} {month}' + (f' {year(y, "fr")}' if y else '')
    if lang == 'de':                                   # dative, as after "am": am fünfzehnten März
        day = n2w(d, 'de', to='ordinal') + 'n'
        return f'{day} {month}' + (f' {year(y, "de")}' if y else '')
    if lang == 'es':
        return f'{cardinal(d, "es")} de {month}' + (f' de {year(y, "es")}' if y else '')
    if lang == 'it':
        day = 'primo' if d == 1 else cardinal(d, 'it')
        return f'{day} {month}' + (f' {year(y, "it")}' if y else '')
    if lang == 'pt':
        return f'{cardinal(d, "pt")} de {month}' + (f' de {year(y, "pt")}' if y else '')
    if lang == 'nl':
        return f'{cardinal(d, "nl")} {month}' + (f' {year(y, "nl")}' if y else '')
    if lang == 'id':
        return f'{cardinal(d, "id")} {month}' + (f' {year(y, "id")}' if y else '')
    raise ValueError(f'date not deterministic for {lang}')


def time_words(h, mm, lang):
    """15, '30' -> 'quinze heures trente' / 'fünfzehn Uhr dreißig' / 'quindici e trenta'."""
    hour = cardinal(h, lang)
    minutes = None if mm in (None, '00') else (digits(mm, lang) if mm[0] == '0' and lang in ('es', 'nl', 'id') else cardinal(int(mm), lang))
    if lang == 'fr':
        out = f'{hour} {"heure" if h == 1 else "heures"}'
        return out + (f' {cardinal(int(mm), "fr")}' if minutes else '')
    if lang == 'de':
        return f'{hour} Uhr' + (f' {cardinal(int(mm), "de")}' if minutes else '')
    if lang == 'nl':
        return f'{hour} uur' + (f' {minutes}' if minutes else '')
    if lang == 'it':
        return hour + (f' e {minutes}' if minutes else '')
    if lang == 'pt':
        return f'{hour} {"hora" if h == 1 else "horas"}' + (f' e {minutes}' if minutes else '')
    if lang in ('es', 'id'):
        return hour + (f' {minutes}' if minutes else '')
    raise ValueError(f'time not deterministic for {lang}')


def phone_words(s, lang):
    """French reads pairs (zéro six douze trente-quatre ...), everyone else digit by digit."""
    ds = re.sub(r'\D', '', s)
    plus = s.strip().startswith('+')
    if lang == 'fr' and not plus and len(ds) == 10:
        words = ' '.join(('zéro ' + DIGIT_WORDS['fr'][int(ds[i + 1])]) if ds[i] == '0' else cardinal(int(ds[i:i + 2]), 'fr')
                         for i in range(0, 10, 2))
    else:
        words = digits(ds, lang)
    return (WORDS[lang]['plus'] + ' ' if plus else '') + words


def range_words(a, b, lang):
    if lang in ('es', 'pt', 'it'):
        return f'{cardinal(a, lang)} {WORDS[lang]["to"]} {cardinal(b, lang)}'
    return f'{cardinal(a, lang)} {WORDS[lang]["to"]} {cardinal(b, lang)}'


def spell_id(tok, lang):
    """MH370 -> 'MH trois sept zéro' (letters kept, digits one by one)."""
    out = []
    for run in re.findall(r'\d+|[A-Za-z]+', tok):
        out.append(digits(run, lang) if run.isdigit() else run)
    return ' '.join(out)


def email_words(local, domain, lang):
    w = WORDS[lang]

    def part(s):
        s = re.sub(r'\d+', lambda m: ' ' + digits(m.group(0), lang) + ' ', s)
        return re.sub(r'\s+', ' ', s.replace('.', f' {w["dot"]} ').replace('-', f' {w["dash"]} ').replace('_', ' underscore ')).strip()
    return f'{part(local)} {w["at"]} {part(domain)}'


def url_words(url, lang):
    w = WORDS[lang]
    out = re.sub(r'^https?://', '', url)
    out = re.sub(r'^www\.', 'w w w . ', out)
    out = out.replace('/', f' {w["slash"]} ').replace('.', f' {w["dot"]} ').replace('-', f' {w["dash"]} ')
    out = re.sub(r'\d+', lambda m: ' ' + digits(m.group(0), lang) + ' ', out)
    return re.sub(r'\s+', ' ', out).strip()


def format_number(n, lang, frac=None):
    """Written form with the locale's separators: 1250 -> '1 250' (fr) / '1.250' (de) / '1,250' (en)."""
    thou, dec = NUMBER_FORMAT[lang]
    s = f'{int(n):,}'.replace(',', '\x00').replace('.', dec).replace('\x00', thou)
    if frac is not None:
        s += dec + frac
    return s
