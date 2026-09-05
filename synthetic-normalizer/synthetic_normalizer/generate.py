#!/usr/bin/env python3
"""
Template pairs: fill the seed + LLM-written templates with random locale-formatted values and
produce the spoken form deterministically (verbalize.py; app.spoken_normalizer for en/ms/zh/ta).

    uv run --with num2words python -m synthetic_normalizer.generate --per-locale 2500

Writes data/template_pairs.jsonl, one
{"id","lang","language","source":"template","template_id","slots","text","normalized"} per line.
Every row is digit-free on the normalized side and differs from its text (checked).
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys

sys.path.insert(0, os.environ.get('SN_TTS_REPO') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from app.spoken_normalizer import normalize as my_normalize  # noqa: E402
from app.spoken_normalizer.core import MONTHS as MY_MONTHS  # noqa: E402
from . import verbalize as V  # noqa: E402
from .locales import (PHONE_FORMATS, DATE_FORMATS, TIME_FORMATS, ORDINAL_FORMATS, MY_UNITS, MY_CURRENCIES,  # noqa: E402
                      SEED_TEMPLATES)
from .llm_common import RESULTS  # noqa: E402

MY = set(V.SPOKEN_NORMALIZER_LANG)
SLOT_RE = re.compile(r'\{(' + '|'.join(V.ALL_SLOTS) + r')\}')
NAMES = ['ali', 'siti', 'mei.ling', 'kumar', 'nurul_h', 'ahmad.b', 'sofia', 'jean.dupont', 'anna', 'luca', 'pedro', 'jan.devries',
         'kasia', 'omar', 'fatima', 'maria', 'juan', 'ravi', 'dilani', 'chen88', 'support', 'info', 'billing', 'help']
DOMAINS = ['gmail.com', 'yahoo.com', 'scicom.com.my', 'example.com', 'company.co', 'bank.my', 'mail.fr', 'web.de', 'correo.es',
           'posta.it', 'sapo.pt', 'ziggo.nl', 'onet.pl', 'gmail.co.id', 'yahoo.com.ph', 'sltnet.lk', 'outlook.sa']
URLS = ['www.{d}', 'https://{d}', 'https://{d}/help', 'www.{d}/login', 'https://{d}/faq', 'https://www.{d}/support']
# an LLM-written template may put a unit or currency after a slot ("{decimal} kg"); the pair is then only half normalized
LEFTOVER_RE = re.compile(r'(?<![A-Za-z])(kg|kgs|km|cm|mm|ml|gb|mb|tb|kb|mbps|kwh|hrs?|mins?|secs?)(?![A-Za-z])|[%€$£₱₨¥٪]|(?<![A-Za-z])(Rp|RM|Rs|LKR|PHP|USD|EUR|SAR|AED|EGP|IDR|PLN|zł|ر\.س|د\.إ|ج\.م)(?![A-Za-z])|රු(?![඀-෿])', re.I)
IDS = ['MH370', 'AK6120', 'ABC1234', 'X-12340567', 'WXY 1234', 'A12', 'REF-2024-0113', 'INV0472', 'TK-88213', 'B2-14', 'QZ8501', 'PO 55219']
TL_PARTICLES = {'na', 'ng', 'ang', 'ay', 'sa', 'at', 'o', 'pa', 'lang', 'ba', 'din', 'rin', 'daw', 'raw', 'po', 'ho', 'kaya', 'pala',
                'naman', 'muna', 'hanggang', 'para', 'kung', 'kapag', 'dahil', 'pero', 'kasi'}
EN_MONTHS = MY_MONTHS['en'][1:]
MY_MONTH_NAMES = {'en': EN_MONTHS, 'ms': MY_MONTHS['ms'][1:], 'ta': MY_MONTHS['ta'][1:], 'ta-LK': MY_MONTHS['ta'][1:], 'zh': EN_MONTHS}


# --------------------------------------------------------------------------- slot fillers -> (written, spoken|None)
def f_int(loc, rng):
    n = rng.choice([rng.randint(2, 9), rng.randint(10, 99), rng.randint(100, 999)])
    if loc not in ('fr', 'de', 'es', 'it', 'pt', 'nl') and rng.random() < 0.08:
        n = 1
    return str(n), None if loc in MY else V.cardinal(n, loc)


def f_big(loc, rng):
    if loc == 'si':
        n = rng.choice([rng.randint(1000, 9999), rng.choice(range(20000, 100000, 10000)) + rng.randint(0, 999)])
    else:
        n = rng.choice([rng.randint(1000, 9999), rng.randint(10000, 99999), rng.randint(100000, 999999), rng.randint(1000000, 99999999)])
    written = V.format_number(n, loc) if (n >= 10000 or rng.random() < 0.5) else str(n)
    return written, None if loc in MY else V.cardinal(n, loc)


def f_money(loc, rng):
    if loc in MY:
        whole = rng.choice([rng.randint(1, 99), rng.randint(100, 999), rng.randint(1000, 99999)])
        cents = rng.choice([None, None, '50', '90', f'{rng.randint(1, 99):02d}'])
        amt = V.format_number(whole, loc, cents) if whole >= 1000 or rng.random() < 0.5 else (str(whole) + ('.' + cents if cents else ''))
        return rng.choice(MY_CURRENCIES[loc]).format(amt=amt), None
    code = rng.choice(list(V.CURRENCIES[loc]))
    variants, _, _ = V.CURRENCIES[loc][code]
    if loc == 'id':
        whole, cents = rng.choice([rng.randint(1, 99) * 1000, rng.randint(100, 999) * 1000, rng.randint(1, 50) * 100000, rng.randint(1000, 9999) * 1000]), None
    elif loc == 'si':
        whole, cents = rng.choice([rng.randint(1, 99), rng.randint(100, 9999), rng.choice(range(10000, 100000, 10000)) + rng.randint(0, 999)]), rng.choice([None, None, '50'])
    else:
        whole, cents = rng.choice([rng.randint(1, 99), rng.randint(100, 999), rng.randint(1000, 99999)]), rng.choice([None, None, '50', '90', f'{rng.randint(1, 99):02d}'])
    amt = V.format_number(whole, loc, cents)
    return variants[rng.randrange(len(variants))].format(amt=amt), V.money(whole, cents, code, loc)


def f_decimal(loc, rng):
    whole, frac = rng.randint(0, 99), rng.choice([str(rng.randint(1, 9)), f'{rng.randint(1, 99):02d}'.rstrip('0') or '5', str(rng.randint(10, 99))])
    _, dec = V.NUMBER_FORMAT[loc]
    return f'{whole}{dec}{frac}', None if loc in MY else V.decimal(str(whole), frac, loc)


def f_percent(loc, rng):
    whole, frac = rng.randint(1, 100), rng.choice([None, None, str(rng.randint(1, 9))])
    _, dec = V.NUMBER_FORMAT[loc]
    written = f'{whole}{dec + frac if frac else ""}' + rng.choice(['%', '%', ' %'] if loc in ('fr', 'de', 'pl', 'es') else ['%', '%'])
    return written, None if loc in MY else V.percent(str(whole), frac, loc)


def f_phone(loc, rng):
    pat = rng.choice(PHONE_FORMATS[loc])
    written = ''.join(str(rng.randint(0, 9)) if c == 'd' else c for c in pat)
    return written, None if loc in MY else V.phone_words(written, loc)


def f_digits(loc, rng):
    n = rng.randint(4, 6)
    written = ''.join(str(rng.randint(0, 9)) for _ in range(n))
    return written, None if loc in MY else V.digits(written, loc)


def f_date(loc, rng):
    d, m, y = rng.randint(1, 28), rng.randint(1, 12), rng.randint(1995, 2032)
    fmt = rng.choice(DATE_FORMATS[loc])
    month = (MY_MONTH_NAMES[loc] if loc in MY else V.MONTHS[loc])[m - 1]
    written = fmt.format(d=d, m=m, dd=f'{d:02d}', mm=f'{m:02d}', y=y, Month=month)
    if loc in MY:
        return written, None
    return written, V.date_words(d, m, y if '{y}' in fmt else None, loc)


def f_time(loc, rng):
    H, MM = rng.randint(0, 23), rng.choice(['00', '00', '05', '10', '15', '20', '30', '40', '45', '50'])
    fmt = rng.choice(TIME_FORMATS[loc])
    h = H % 12 or 12
    ap = 'am' if H < 12 else 'pm'
    ms_period = 'pagi' if 5 <= H < 12 else 'petang' if 12 <= H < 19 else 'malam'
    zh_period = '上午' if H < 12 else '下午'
    if '{h}' in fmt and loc in ('ms', 'zh'):
        pass
    if fmt.endswith('h') or fmt.endswith(' uur') or fmt == '{H}' or fmt == '{h}{ap}' or fmt == '{h} {ms_period}' or fmt == '{h}点':
        MM = '00'
    written = fmt.format(H=H, h=h, MM=MM, ap=rng.choice([ap, ap.upper(), ap[0] + '.' + ap[1] + '.']) if '{ap}' in fmt else '',
                         ms_period=ms_period, zh_period=zh_period).strip()
    if loc in MY:
        return written, None
    return written, V.time_words(H, MM, loc)


def f_ordinal(loc, rng):
    n = rng.randint(1, 31)
    fmt = rng.choice(ORDINAL_FORMATS[loc])
    sfx = 'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    written = fmt.format(n=n, sfx=sfx)
    if fmt == '{n}e' and loc == 'fr' and n == 1:
        written = '1er'
    return written, None if loc in MY else V.ordinal(n, loc)


def f_range(loc, rng):
    a = rng.randint(1, 30)
    b = a + rng.randint(1, 10)
    return f'{a}{rng.choice(["-", "-", "–", " - "])}{b}', None if loc in MY else V.range_words(a, b, loc)


def f_unit(loc, rng):
    if loc in MY:
        u = rng.choice(MY_UNITS)
        n = rng.choice([str(rng.randint(1, 500)), f'{rng.randint(0, 99)}.{rng.randint(1, 9)}'])
        return f'{n}{rng.choice(["", " "])}{u}', None
    u = rng.choice(list(V.UNITS[loc]))
    variants, _, _ = V.UNITS[loc][u]
    whole = rng.choice([1, rng.randint(2, 9), rng.randint(10, 500)])
    frac = rng.choice([None, None, str(rng.randint(1, 9))])
    _, dec = V.NUMBER_FORMAT[loc]
    num = f'{whole}{dec + frac if frac else ""}'
    variant = rng.choice(variants)
    written = num + (variant if variant.startswith(' ') else rng.choice(['', ' ']) + variant)
    return written, V.unit(str(whole), frac, u, loc)


def f_year(loc, rng):
    y = rng.randint(1950, 2035)
    return str(y), None if loc in MY else V.year(y, loc)


def f_email(loc, rng):
    local, dom = rng.choice(NAMES), rng.choice(DOMAINS)
    if rng.random() < 0.3:
        local += str(rng.randint(1, 99))
    return f'{local}@{dom}', None if loc in MY else V.email_words(local, dom, loc)


def f_url(loc, rng):
    url = rng.choice(URLS).format(d=rng.choice(DOMAINS))
    return url, None if loc in MY else V.url_words(url, loc)


def f_id(loc, rng):
    tok = rng.choice(IDS)
    return tok, None if loc in MY else V.spell_id(tok, loc)


FILLERS = {'int': f_int, 'big': f_big, 'money': f_money, 'decimal': f_decimal, 'percent': f_percent, 'phone': f_phone,
           'digits': f_digits, 'date': f_date, 'time': f_time, 'ordinal': f_ordinal, 'range': f_range, 'unit': f_unit,
           'year': f_year, 'email': f_email, 'url': f_url, 'id': f_id}


# --------------------------------------------------------------------------- templates
def load_templates(loc, llm_path=None):
    tpls = list(SEED_TEMPLATES.get(loc, []))
    llm_path = llm_path or os.path.join(RESULTS, 'templates_llm.jsonl')
    if os.path.exists(llm_path):
        for line in open(llm_path):
            if line.strip():
                r = json.loads(line)
                if r.get('lang') == loc and r.get('template'):
                    tpls.append(r['template'])
    from .templates_llm import valid, clean          # re-validate cached templates with the current rules
    out, seen = [], set()
    for t in tpls:
        t = clean(t)
        if t in seen or not valid(t, loc):
            continue
        seen.add(t)
        out.append(t)
    return out


def fill_template(tpl, loc, rng):
    """-> (written sentence, spoken sentence, slots) or None when the pair fails a check."""
    slots = []
    written_parts, spoken_parts = [], []
    pos = 0
    for m in SLOT_RE.finditer(tpl):
        slot = m.group(1)
        w, s = FILLERS[slot](loc, rng)
        slots.append(slot)
        written_parts.append(tpl[pos:m.start()] + w)
        if s is not None and loc == 'tl' and slot in ('int', 'big', 'range', 'decimal'):
            nxt = re.match(r'\s+([A-Za-zÀ-ɏ]+)', tpl[m.end():])
            if nxt and nxt.group(1).lower() not in TL_PARTICLES:
                head, _, last = s.rpartition(' ')
                s = (head + ' ' if head else '') + V._tl_linker(last)
        spoken_parts.append(tpl[pos:m.start()] + (s if s is not None else w))
        pos = m.end()
    written = ''.join(written_parts) + tpl[pos:]
    if loc in MY:
        spoken = my_normalize(written, lang=V.SPOKEN_NORMALIZER_LANG[loc])
    else:
        spoken = ''.join(spoken_parts) + tpl[pos:]
        spoken = re.sub(r'[ \t]{2,}', ' ', spoken).strip()
    if loc == 'ar' and rng.random() < 0.3:
        written = written.translate(V.EASTERN_DIGITS)
    if re.search(r'[0-9٠-٩]', spoken) or spoken == written or not spoken or LEFTOVER_RE.search(spoken):
        return None
    return written, spoken, slots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-locale', type=int, default=2500)
    ap.add_argument('--locales', default=','.join(V.LOCALES))
    ap.add_argument('--seed', type=int, default=20260905)
    ap.add_argument('--out', default=os.path.join(RESULTS, 'template_pairs.jsonl'))
    args = ap.parse_args()
    rng = random.Random(args.seed)
    n_written = 0
    with open(args.out, 'w') as f:
        for loc in args.locales.split(','):
            tpls = load_templates(loc)
            if not tpls:
                print(f'{loc}: no templates'); continue
            made, tries, seen = 0, 0, set()
            while made < args.per_locale and tries < args.per_locale * 4:
                tries += 1
                tpl = rng.choice(tpls)
                res = fill_template(tpl, loc, rng)
                if res is None or res[0] in seen:
                    continue
                seen.add(res[0])
                tid = hashlib.sha1(tpl.encode()).hexdigest()[:10]
                row = {'id': f'{loc}-t-{made:06d}', 'lang': loc, 'language': V.LANGUAGE_NAME[loc], 'source': 'template',
                       'template_id': tid, 'slots': res[2], 'text': res[0], 'normalized': res[1]}
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
                made += 1
            n_written += made
            print(f'{loc:<6} templates {len(tpls):>3}  pairs {made:>6}  (rejected {tries - made})')
    print(f'wrote {n_written} rows to {args.out}')


if __name__ == '__main__':
    main()
