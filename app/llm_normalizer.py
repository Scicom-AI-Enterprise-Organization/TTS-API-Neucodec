"""
LLM-based text normalizer: sends text to any OpenAI-compatible /chat/completions
endpoint (OPENAI_BASE_URL) with the few-shot prompt in app/prompt.py, constrained
to `{"normalized": "..."}` via `response_format` json_schema so the reply is always
just the normalized text, ready to feed the TTS LM.

This module must stay importable without the GPU stack (torch/NeuCodec) so its
unit tests run anywhere; only aiohttp + app.env + app.prompt are allowed here.
"""

import asyncio
import json
import os
import re
import logging
from enum import Enum

import aiohttp

from app.env import OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL_NAME, OPENAI_TIMEOUT
from app.prompt import SYSTEM_PROMPT, EXAMPLES, JSON_SCHEMA, CALLER_PROMPT_HEADER, CLOCK_HINT_HEADER
from app.spoken_normalizer.lang import latin_scores
from app.spoken_normalizer.numbers import en_cardinal, ms_cardinal, zh_cardinal


class NormalizerMode(str, Enum):
    rule = 'rule'        # legacy malaya-derived pipeline (app/normalizer)
    llm = 'llm'          # OpenAI-compatible LLM with the prompt in app/prompt.py
    spoken = 'spoken'    # rule-based replica of the LLM (app/spoken_normalizer), no network


class LLMNormalizerError(Exception):
    """LLM endpoint unreachable, non-200, or returned an unusable reply."""


_CODE_FENCE = re.compile(r'^```(?:json)?\s*(.*?)\s*```$', re.DOTALL)


# How each clock time is read, worked out here and handed to the LLM with the text
# (CLOCK_HINT_HEADER), so a time is read the same way every time. The rule and examples alone
# left the LLM inconsistent on 6 of 61 time sentences over 20 runs (2026-11-16): "at 8:00" ->
# "eight p m" 20/20, "14:05" -> "dua empat lima minit" 1/20. The reading (the user's): a
# written time is 24-hour unless am/pm or a period word is written, said on the 12-hour clock
# with its period; Malay "pukul" + hour + minutes "minit" + period, English hour + minutes +
# a m / p m, Chinese period + hour 点 + minutes 分. A whole hour is the hour alone.
_MS_PERIODS = r'pagi|petang|malam|tengah\s?hari|tengahari|tengah\s+malam'
_ZH_PERIODS = '凌晨|早上|上午|中午|下午|晚上|傍晚'
_CLOCK = re.compile(
    r'(?P<cue>(?<![A-Za-z])(?:pukul|jam|pkl)\s+)?'
    rf'(?P<pre>(?<![A-Za-z])(?:{_MS_PERIODS})\s+|{_ZH_PERIODS})?'
    r'(?<![\d:.,/])(?P<h>\d{1,2})(?:(?P<sep>[:.])(?P<m>[0-5]\d))?(?![\d])(?![.:]\d)'
    r'(?:\s?(?P<ap>[AaPp](?:\.\s?[Mm]\.|\s?[Mm]))(?![A-Za-z])'
    rf'|\s?(?P<post>{_MS_PERIODS})(?![A-Za-z]))?', re.IGNORECASE)
_EN_PERIOD_AFTER = re.compile(r'\s?(?:noon|midnight|in\s+the\s+(?:morning|afternoon|evening)|at\s+night)(?![A-Za-z])',
                              re.IGNORECASE)
_EN_CUE_BEFORE = re.compile(r'(?<![A-Za-z])(?:at|by|from|until|till|to|before|after|around|since)\s*$', re.IGNORECASE)
_MS_CUE_BEFORE = re.compile(r'(?<![A-Za-z])(?:pada|sebelum|selepas|dari|hingga|sehingga|sampai|antara|dan)\s*$',
                            re.IGNORECASE)


def _ms_period(h24):
    if h24 == 0:
        return 'tengah malam'
    if h24 < 12:
        return 'pagi'
    if h24 == 12:
        return 'tengah hari'
    return 'petang' if h24 <= 18 else 'malam'


def _zh_period(h24):
    if h24 < 6:
        return '凌晨'
    if h24 < 12:
        return '上午'
    if h24 == 12:
        return '中午'
    return '下午' if h24 < 18 else '晚上'


def _clock_lang(text, m):
    if _HAN_CHAR.search(text):
        return 'zh'
    if m.group('cue') or (m.group('pre') and not re.match(_ZH_PERIODS, m.group('pre'))) or m.group('post'):
        return 'ms'
    before = text[:m.start()]
    if _EN_CUE_BEFORE.search(before):
        return 'en'
    if _MS_CUE_BEFORE.search(before):
        return 'ms'
    ms, en = latin_scores(text)
    if ms == en:            # no marker word either way: "bas pertama 7:05"
        return _lexicon_lang(text)
    return 'ms' if ms > en else 'en'


# When the marker words tie, the dictionaries bundled with the legacy normalizer decide: a word
# only in the Malay ones (DBP, Kamus Dewan, malay_words) votes Malay, a word only in the English
# one votes English. Marker words alone sent "bas pertama 7:05" to English (neither word is a
# marker), read "seven zero five a m" 10 of 10.
_LEXICONS = None


def _lexicon_lang(text):
    global _LEXICONS
    if _LEXICONS is None:
        data = os.path.join(os.path.dirname(__file__), 'normalizer', 'data')

        def words(*names):
            out = set()
            for n in names:
                try:
                    with open(os.path.join(data, f'{n}.json')) as f:
                        out.update(w.lower() for w in json.load(f))
                except (OSError, ValueError):
                    pass
            return out

        _LEXICONS = (words('malay_words', 'dbp_words', 'kamus_dewan_words'), words('english_words'))
    ms_words, en_words = _LEXICONS
    ms = en = 0
    for tok in re.findall(r'[a-z]{2,}', text.lower()):
        ms += tok in ms_words and tok not in en_words
        en += tok in en_words and tok not in ms_words
    return 'ms' if ms > en else 'en'


def clock_readings(text):
    """[(written, spoken)] for every clock time in `text`, in order. Empty for Tamil text."""
    if not text or re.search('[஀-௿]', text):
        return []
    out = []
    for m in _CLOCK.finditer(text):
        h, sep, mm, ap = int(m.group('h')), m.group('sep'), m.group('m'), m.group('ap')
        pre, post, cue = m.group('pre'), m.group('post'), m.group('cue')
        if sep == ':':
            ok = True
        elif sep == '.':                    # 8.00 only beside a time word: RM8.00, 37.50 are not times
            ok = bool(ap or pre or post or cue)
        else:                               # a bare hour only with am/pm: 9am
            ok = bool(ap)
        if not ok or h > 24 or (ap and not 1 <= h <= 12):
            continue
        h24 = (h % 12) + (12 if ap and ap[0] in 'Pp' else 0) if ap else h % 24
        h12 = h24 % 12 or 12
        minutes = mm if mm and mm != '00' else None
        lang = _clock_lang(text, m)
        after = text[m.end():]
        if lang == 'zh':
            period = pre if pre and re.match(_ZH_PERIODS, pre) else _zh_period(h24)
            hour = '零' if h24 == 0 else ('两' if h12 == 2 else zh_cardinal(h12))
            spoken = f'{period}{hour}点'
            if minutes:
                spoken += ('零' if minutes[0] == '0' else '') + zh_cardinal(int(minutes)) + '分'
        elif lang == 'ms':
            words = [cue.strip() if cue else 'pukul']
            if pre:
                words.append(' '.join(pre.split()))
            words.append(ms_cardinal(h12))
            if minutes:
                words += [ms_cardinal(int(minutes)), 'minit']
            if post:
                words.append(' '.join(post.split()))
            elif not pre:
                words.append(_ms_period(h24))
            spoken = ' '.join(words)
        else:
            words = [en_cardinal(h12)]
            if minutes:
                words.append(('zero ' if minutes[0] == '0' else '') + en_cardinal(int(minutes)))
            if ap or not _EN_PERIOD_AFTER.match(after):
                period = 'a m' if h24 < 12 else 'p m'
                words.append(period.upper() if ap and ap[0].isupper() and ap[-1].isupper() else period)
            spoken = ' '.join(words)
        written = m.group(0).strip()
        out.append((written, spoken))
    return out


def clock_hint(text):
    """The CLOCK_HINT_HEADER block for `text`, or None when it has no clock time."""
    readings = clock_readings(text)
    if not readings:
        return None
    return CLOCK_HINT_HEADER + '\n' + '\n'.join(f'{w} -> {s}' for w, s in readings)


def build_messages(text, prompt=None):
    """`prompt` is the caller's extra context (the request's `prompt` field), put both at
    the end of the system prompt and in brackets above the text (why both: app/prompt.py);
    blank or None leaves the messages exactly as without it. A clock time in `text` adds its
    reading (clock_hint) the same two ways; text without one gets the messages unchanged."""
    prompt = (prompt or '').strip()
    hint = clock_hint(text)
    extra = [x for x in (prompt, hint) if x]
    system = SYSTEM_PROMPT
    if prompt:
        system = f'{system}\n\n{CALLER_PROMPT_HEADER}\n{prompt}'
    if hint:
        system = f'{system}\n\n{hint}'
    messages = [{'role': 'system', 'content': system}]
    for before, after in EXAMPLES:
        messages.append({'role': 'user', 'content': before})
        messages.append({
            'role': 'assistant',
            'content': json.dumps({'normalized': after}, ensure_ascii=False),
        })
    user = text
    if extra:
        user = '\n'.join(f'[{x}]' for x in extra) + '\n' + text
    messages.append({'role': 'user', 'content': user})
    return messages


def parse_normalized(content):
    """Extract the normalized text from a model reply. Guided decoding guarantees
    clean JSON, but stay defensive for backends that ignore response_format:
    unwrap code fences, accept a bare JSON string, else use the raw text."""
    content = (content or '').strip()
    fenced = _CODE_FENCE.match(content)
    if fenced:
        content = fenced.group(1).strip()
    try:
        out = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return content or None
    if isinstance(out, dict) and isinstance(out.get('normalized'), str):
        return out['normalized'].strip() or None
    if isinstance(out, str):
        return out.strip() or None
    return None


# ---------------------------------------------------------------------------
# "Nothing to normalize" fast path
# ---------------------------------------------------------------------------
# One LLM round trip is ~0.55 s of TTFB (gemma-4-31b behind the serverless proxy, measured
# from tm-h20 on 2026-09-04: mean 0.56 s, max 0.67 s over 21 sentences), and on plain
# conversational text it hands the input back unchanged (19/19 plain evalset sentences
# came back byte-identical). Everything SYSTEM_PROMPT asks the model to rewrite -- numbers,
# money, IC/phone numbers, dates, times, percentages, decimals, ordinals, units, emails,
# URLs -- plus what it expands in practice beyond that ("Dr." -> "Doctor", acronyms) leaves
# a lexical trace: a digit, a symbol, a dotted or ALL-CAPS token, or a known abbreviation.
# Text with none of those skips the call and gets the same pre/post cleanup the LLM path
# applies, so the output is identical to a no-op LLM reply. Conservative on purpose: any
# doubt still goes to the LLM (a wasted call costs latency, a wrong skip costs correctness).
# main.py gates this behind LLM_NORMALIZER_SKIP_PLAIN; bench/normalizer_gate_eval.py
# checks the assumption against the live LLM.

_PLAIN_PUNCT = ".,!?;:'’‘\"“”()\\-–—…、。，！？；：「」『』（）"
# Tamil vowel signs and virama (U+0BBE..U+0BCD) are combining marks, which Python's \w does
# not match -- without this every Tamil sentence looked like it contained symbols and went
# to the LLM regardless of the gate. Whole Tamil block treated as letters.
_TAMIL = '஀-௿'
_NEEDS_LLM = re.compile(
    r'\d'                                   # any digit (Unicode-aware)
    rf'|[^\w\s{_TAMIL}{_PLAIN_PUNCT}]'      # any symbol beyond plain punctuation: % $ @ & / + = # * ...
    r'|_'                                   # \w admits underscore (markdown, identifiers)
    r'|\w\.\w'                              # dotted tokens: site.com, e.g., U.S.
    r'|\b[A-Z]{2,}\b'                       # acronyms / all caps: IC, TNB, OTP, RM
    # abbreviations that are rarely ordinary words: match bare
    r'|(?i:\b(?:dr|mr|mrs|ms|prof|sdn|bhd|jln|tmn|hj|etc|eg|ie|vs|approx|dept|govt|tel|fax|'
    r'ext|acct|amt|mth|yr|wk|sept|km|cm|mm|ml|kg|mg|gb|mb|kb|tb|hz|khz|mhz|ghz|mph|kph|kmh|'
    r'lbs|oz|sq)\b)'
    # ... and ones that are ordinary words too ("no", "am", "sat"): only when written with a period
    r'|(?i:\b(?:no|st|rd|co|inc|ltd|corp|min|max|sec|hrs?|mins?|mon|tues?|wed|thur?s?|fri|sat|'
    r'sun|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec|am|pm|en|pn|tn|cik|ir|ref|acc|bal|ave|'
    r'blvd|ft)\.)'
)


def needs_normalization(text):
    """True if `text` may contain something the LLM normalizer would rewrite.

    False means the text is plain words and punctuation only, for which the LLM returns
    its input unchanged -- so the caller can skip the call. See the note above.
    """
    return bool(_NEEDS_LLM.search(text or ''))


_UNSPOKEN = re.compile(r'\d|[^\w\s' + _TAMIL + re.escape(_PLAIN_PUNCT) + r']|\w\.\w')


def has_unspoken(text):
    """Narrower than needs_normalization(): only what is unspeakable as written (a digit, a
    symbol, a dotted token). Acronyms and abbreviations do not count -- the TTS LM reads those
    and the LLM leaves most of them alone. Used by LLM_NORMALIZER_RULE_FIRST to decide whether
    the spoken normalizer left anything for the LLM to do."""
    return bool(_UNSPOKEN.search(text or ''))


# ---------------------------------------------------------------------------
# Replies that must not be spoken
# ---------------------------------------------------------------------------
# The LLM sometimes hands back a reply that reads well and is wrong in a way no listener can
# recover from: a number dropped, the digits of a read-back dropped or left as digits ("D 3 6 7 6"
# came back as written), or English text read with Chinese numerals ("8 8 0 1" -> "八 八 零 一"),
# which the TTS then speaks in Mandarin. Each leaves a trace that needs no second model to see. rejected_reason() names it; llm_normalize() raises
# on it, so the TTS path falls back to the spoken rules, which read every digit.
_HAN = '㐀-䶿一-鿿豈-﫿'
_CJK_TEXT = re.compile(f'[{_HAN}　-〿！-？]')
_HAN_CHAR = re.compile(f'[{_HAN}]')
_DIGIT_WORD = re.compile(
    r'(?i)(?<![a-z])(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|double|triple|'
    r'kosong|sifar|satu|dua|tiga|empat|lima|enam|tujuh|lapan|sembilan)(?![a-z])|[零〇一二两三四五六七八九]')
_NUMBER_WORD = re.compile(
    r'(?i)(?<![a-z])(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|'
    r'(?:thir|four|fif|six|seven|eigh|nine)teen|(?:twen|thir|for|fif|six|seven|eigh|nine)ty|'
    r'hundred|thousand|million|billion|half|quarter|first|second|third|'
    r'(?:four|fif|six|seven|eigh|nin|ten|eleven|twelf|(?:\w+teen)|(?:\w+tie)|hundred|thousand)th|'
    r'kosong|sifar|satu|dua|tiga|empat|lima|enam|tujuh|lapan|sembilan|sepuluh|sebelas|belas|'
    r'puluh|ratus|ribu|juta|bilion|seratus|seribu|sejuta|separuh|pertama|'
    r'ke(?:dua|tiga|empat|lima|enam|tujuh|lapan|sembilan|sepuluh|sebelas))(?![a-z])'
    r'|[零〇一二两三四五六七八九十百千万亿半]')
# Digits that are read one at a time: a 7+ digit run (not a round amount), single digits
# spaced out, or dash-joined groups of 9+ digits (IC, phone; a dashed date has 8).
_READ_BACK = re.compile(r'(?<![\d,.])(?:\d(?: \d(?!\d)){6,}|\d+(?:-\d+)+|\d{7,})(?![\d,.]\d)')


def _read_back_digits(text):
    need = 0
    for m in _READ_BACK.finditer(text):
        run = re.sub(r'\D', '', m.group(0))
        if '-' in m.group(0) and len(run) < 9:
            continue
        if ' ' not in m.group(0) and '-' not in m.group(0) and run.endswith('000'):
            continue
        need += len(run)
    return need


def _spoken_digits(out):
    words = [w.lower() for w in _DIGIT_WORD.findall(out)]
    return len(words) + words.count('triple')       # "double eight" is two words already


_ORDINAL_WORD = re.compile(r'(?i)pertama|ke\w+|first|second|third|\w+th')


def _runs_on(rest):
    """`rest` opens with a number word that would continue a clock reading. An ordinal does
    not: "pukul lapan malam, pertama pukul tujuh ..." is two readings."""
    w = _NUMBER_WORD.match(rest)
    return bool(w) and not _ORDINAL_WORD.fullmatch(w.group(0))


def _clock_kept(normalized, spoken):
    """`spoken` is in the reply as a whole reading: not "pukul pagi lapan" inside "pukul pagi
    lapan sifar sifar", i.e. no number word runs on after it."""
    squashed = ' '.join(normalized.split())
    for m in re.finditer(re.escape(' '.join(spoken.split())), squashed, re.IGNORECASE):
        rest = squashed[m.end():]
        if _HAN_CHAR.match(spoken[-1:] or ' ') or spoken.endswith('点') or spoken.endswith('分'):
            if not re.match(r'[零〇一二两三四五六七八九十百半整分]', rest):
                return True
        elif not re.match(r'[A-Za-z]', rest) and not _runs_on(rest.lstrip(' ,')) \
                and not re.match(r'(?i)\s*(?:minit|minutes?|o.?clock|[ap]\.?\s?m\b)', rest):
            return True
    return False


# Malay says one hundred and one thousand with the prefix se-: "seratus", "seribu". The LLM
# wrote "satu ratus" for RM120, 110 km/j, RM180,000 (10 of 20 shapes, many 10/10), never for
# RM100 or RM199.90. "satu ribu" stays when it ends a larger group: 21,000 is "dua puluh satu
# ribu", 201,000 "dua ratus satu ribu".
_SATU_RATUS = re.compile(r'\b([Ss])atu ratus\b')
_SATU_RIBU = re.compile(r'(?<![A-Za-z])(?<!puluh )(?<!ratus )(?<!belas )([Ss])atu ribu\b')


def malay_hundreds(text, normalized):
    """"satu ratus" -> "seratus" (and a leading "satu ribu" -> "seribu") in the reply, unless
    the text itself wrote that phrase, which is the caller's to keep."""
    if not re.search(r'(?i)\bsatu r(?:atus|ibu)\b', text):
        normalized = _SATU_RATUS.sub(lambda m: ('S' if m.group(1) == 'S' else 's') + 'eratus', normalized)
        normalized = _SATU_RIBU.sub(lambda m: ('S' if m.group(1) == 'S' else 's') + 'eribu', normalized)
    return normalized


def rejected_reason(text, normalized, prompt=None, keep=(), clocks=()):
    """Why the LLM's `normalized` reply to `text` must not be spoken, or None if it may.

    `clocks`: the clock_readings() handed to the LLM, each to come back whole.
    `keep`: phrases already spelled out (rules.product_names) that the reply
    must hand back unchanged. Skipped for Tamil (no digit-word list here). With a caller
    `prompt` the keep, Chinese and leftover-digit checks are off, since a prompt may ask for
    exactly that."""
    if not prompt:
        squashed = ' '.join(normalized.lower().split())
        for k in keep:
            if ' '.join(k.lower().split()) not in squashed:
                return 'spelled-out phrase changed'
        for spoken in clocks:
            if not _clock_kept(normalized, spoken):
                return 'clock time changed'
    if not re.search(r'\d', text) or re.search('[஀-௿]', text):
        return None
    if not re.search(r'\d', normalized) and not _NUMBER_WORD.search(normalized):
        return 'number dropped'
    if _spoken_digits(normalized) + len(re.findall(r'\d', normalized)) < _read_back_digits(text):
        return 'read-back digits dropped'
    if prompt:
        return None
    if re.search(r'\d', normalized):
        return 'digits left'
    if _HAN_CHAR.search(normalized) and not _CJK_TEXT.search(text):
        return 'Chinese numerals in non-Chinese text'
    return None


async def llm_normalize(text, base_url=None, api_key=None, model=None, timeout=None,
                        prompt=None, keep=()):
    """Normalize `text` via the OpenAI-compatible endpoint, with the caller's `prompt`
    (see build_messages) if given. Raises LLMNormalizerError on any failure, including a
    reply rejected_reason() refuses (`keep` goes there); callers decide whether to fall back
    to the rule-based path."""
    base_url = (OPENAI_BASE_URL if base_url is None else base_url).rstrip('/')
    api_key = OPENAI_API_KEY if api_key is None else api_key
    model = OPENAI_MODEL_NAME if model is None else model
    timeout = OPENAI_TIMEOUT if timeout is None else timeout

    if not base_url or not model:
        raise LLMNormalizerError('OPENAI_BASE_URL / OPENAI_MODEL_NAME not configured')

    payload = {
        'model': model,
        'messages': build_messages(text, prompt),
        'temperature': 0,
        # verbalization expands the text (digits -> words); chars overestimate tokens,
        # so this cap is safe.
        'max_tokens': min(4096, 256 + 2 * len(text)),
        'response_format': {'type': 'json_schema', 'json_schema': JSON_SCHEMA},
    }
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'

    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as session:

            async def post(p):
                async with session.post(
                    f'{base_url}/chat/completions', json=p, headers=headers
                ) as r:
                    return r.status, await r.text()

            status, body = await post(payload)
            if status == 400:
                # backend may not support response_format json_schema; retry unconstrained
                logging.warning(
                    f'llm normalizer got 400, retrying without response_format: {body[:200]}'
                )
                payload.pop('response_format')
                status, body = await post(payload)
            if status != 200:
                raise LLMNormalizerError(f'{base_url} returned {status}: {body[:500]}')
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        raise LLMNormalizerError(f'request to {base_url} failed: {e!r}') from e

    try:
        content = json.loads(body)['choices'][0]['message']['content']
    except (ValueError, LookupError, TypeError) as e:
        raise LLMNormalizerError(f'malformed completion response: {body[:500]}') from e

    normalized = parse_normalized(content)
    if normalized is None:
        raise LLMNormalizerError(f'unusable reply from model: {content[:500]!r}')
    normalized = malay_hundreds(text, normalized)
    # a clock reading handed to the LLM must come back as given (clock_hint)
    reason = rejected_reason(text, normalized, prompt, keep, [s for _, s in clock_readings(text)])
    if reason:
        raise LLMNormalizerError(f'rejected reply ({reason}): {normalized[:500]!r}')
    return normalized
