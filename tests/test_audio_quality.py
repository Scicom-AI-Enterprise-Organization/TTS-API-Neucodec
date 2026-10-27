"""
Audio quality of what the API speaks: is it intelligible, at a normal pace, at a steady pitch,
at an even level, natural, and does it say the numbers and nothing else.

The text tests stop at the text the LM is given. These start one step later, at the audio, and
the audio is sampled (temperature 0.6): the same text never gives the same waveform, so nothing
here compares against a golden file. Each text is rendered N times and every check is a RATE
over takes, held to a threshold measured on a baseline run (THRESHOLDS says which). They catch a
checkpoint, decode, stitcher or normalizer change that makes the voice worse; they do not prove
a voice is good, and a listen still decides that.

The ear is software, so each check names what it trusts:
  words        Whisper large-v3, twice per take: language left to it (a Mandarin read of an
               English chunk comes back as Chinese text and `zh`), and forced to the case's
               language, with a verbatim prompt on read-backs so digit by digit stays digits
  pitch        librosa pyin (0.11.0; 1.0.0 segfaults in pyin), in semitones against the median
               of the same language's statements
  pace         syllables (vowel groups, Han characters) per second of the speech span
  level        RMS of the speech frames (STREAM_NORMALIZE's job) and clipped samples
  naturalness  the MOS API (MOS_API + MOS_API_KEY from .env, POST /predict?reps=16 with a wav;
               skipped without it). 16 reps, never 1: the score is stochastic, ±0.17 MOS at
               reps=1 on a bit-identical file, which is bigger than the effects worth catching

Run against a live stack with production's env. For the read-back checks the app's OPENAI_*
should point at the normalizer model production runs, not a stand-in:
    set -a; source .env; set +a      # MOS_API, MOS_API_KEY
    TTS_TEST_URL=http://127.0.0.1:9187 python -m pytest tests/test_audio_quality.py -v
Needs torch, transformers, librosa==0.11.0, requests. ~10 minutes on one H20 at 4 takes.
Env: AUDIO_TEST_TAKES (4), AUDIO_TEST_VOICE (TM_English_Normal), AUDIO_TEST_WHISPER
(openai/whisper-large-v3), AUDIO_TEST_REPORT (per-take metrics as JSON, the way to record a
new baseline), AUDIO_TEST_DIR (keep the wavs).
"""

import io
import json
import os
import re
import statistics
import tempfile
import uuid
import wave
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pytest

BASE_URL = os.environ.get('TTS_TEST_URL', '').rstrip('/')
MOS_URL = os.environ.get('MOS_API', '').rstrip('/')
MOS_KEY = os.environ.get('MOS_API_KEY', '')
TAKES = int(os.environ.get('AUDIO_TEST_TAKES', '4'))
VOICE = os.environ.get('AUDIO_TEST_VOICE', 'TM_English_Normal')
WHISPER = os.environ.get('AUDIO_TEST_WHISPER', 'openai/whisper-large-v3')
REPORT = os.environ.get('AUDIO_TEST_REPORT', '')
WAV_DIR = os.environ.get('AUDIO_TEST_DIR', '')

# Explicit opt-in: this loads Whisper and renders a few hundred clips.
pytestmark = pytest.mark.skipif(not BASE_URL, reason='live audio suite: set TTS_TEST_URL')

SR = 24000
HOP = 240                      # 10 ms frames
HAN = re.compile('[一-鿿]')

# Baseline 2026-10-27: production's checkpoint (v5-recut) and env on an H20, 4 takes, four runs
# (two code versions twice) that send the LM identical text for every plain sentence, so they
# differ only by sampling. Their range is in the comments; a threshold leaves room for it, so a
# pass means "no worse than production", not "good". Nothing is zero-tolerance over ~150 takes:
# a single odd take turned up in the fourth run that the first three never had. The spread is
# wide at 4 takes for anything counted over 8 to 16 takes (questions: 1/16 to 5/16 off).
THRESHOLDS = {
    'cer_median': 0.05,         # per language, plain sentences (0 / 0 in all three)
    'cer_bad': 0.30,            # a take this far off is garbled or cut
    'cer_bad_share': 0.10,      # sentences of 6+ words (zh 0 to 0.06); one misheard word in
                                # "Okay, noted." is 50% CER already, so short lines are left out
    'rate_band': {'en': (4.0, 6.0), 'ms': (5.0, 7.2), 'zh': (4.3, 6.3)},   # syllables/s median:
                                # en 4.87 to 4.98, ms 5.96 to 6.15, zh 5.20 to 5.28
    'rate_spread': 1.5,         # a take vs the median of its own text, either way
    'rate_spread_share': 0.03,  # 0, 0, 0, 2 of 152 takes past 1.5x
    'rate_spread_max': 2.0,     # none past 2x in any run: past it is a cut-off or babble
    'long_pause_s': 0.8,
    'long_pause_share': 0.05,   # 0 in all four
    'lead_s': 0.5,
    'trail_s': 0.8,
    'edge_share': 0.05,         # 0 in all four
    'register_st': 3.0,         # a take's median pitch this far from its language's statements
    'register_share': {'statement': 0.08, 'question': 0.40, 'short': 0.30, 'long': 0.40},
                                # 0 to 0.03, 0.06 to 0.31, 0.09 to 0.21, 0.13 to 0.25 (8 takes)
    'monotone_iqr_st': 1.0,     # pitch spread of a flat, robotic read
    'monotone_share': 0.05,     # 0 in all four
    'jumps_per_100': 4.5,       # level +3 dB and pitch +1.5 st across adjacent 0.5 s windows
                                # (1.52, 2.47, 3.08, 3.13 per 100 window pairs)
    'join_mean_st': 0.3,        # interleaved chunk N+1's median pitch minus chunk N's: -0.31 to -0.60.
                                # Continuous speech drifts down; a cold chunk resets up (+0.53 st
                                # in INTERLEAVE.md), so a mean above +0.3 means interleave broke
    'join_p90_st': 4.0,         # 2.09 to 3.24
    'level_db': 2.5,            # a take's speech level vs the run median
    'level_share': 0.06,        # 0 to 0.033
    'clip_frac': 0.001,         # a take with this share of samples at full scale is clipped
    'clip_share': 0.02,         # 0, 0, 0, 1 of 152 (0.12%; the limiter keeps the rest under 0.03%)
    'first_word_share': 0.93,   # onset murmur swallows the first word (0.956 to 0.985)
    'mos_mean': 2.7,            # en 3.02 to 3.03, ms 2.83 to 2.88, zh 2.92 to 2.97
    'mos_low': 2.0,
    'mos_low_share': 0.05,      # 0 in all four
    'readback_share': 0.9,      # 1.0 with the leading-number fix; 0.60 before it
}

# Synthetic voicebot lines (made-up values). kind decides which checks a take feeds.
CASES = [
    ('st_en_1', 'statement', 'en', 'Thank you for calling, I can help you with your bill today.'),
    ('st_en_2', 'statement', 'en', 'Your payment has been received and your account is up to date.'),
    ('st_en_3', 'statement', 'en', 'I will send the details to your registered email address shortly.'),
    ('st_en_4', 'statement', 'en', 'The technician will contact you before arriving at your location.'),
    ('st_en_5', 'statement', 'en', 'Please keep your modem switched on while we run the line test.'),
    ('st_en_6', 'statement', 'en', 'Your service request has been created and our team will follow up.'),
    ('st_ms_1', 'statement', 'ms', 'Terima kasih kerana menghubungi kami, saya boleh bantu anda hari ini.'),
    ('st_ms_2', 'statement', 'ms', 'Bayaran anda telah diterima dan akaun anda sudah dikemas kini.'),
    ('st_ms_3', 'statement', 'ms', 'Saya akan hantar butiran ke alamat emel anda sebentar lagi.'),
    ('st_ms_4', 'statement', 'ms', 'Juruteknik kami akan menghubungi anda sebelum tiba di lokasi.'),
    ('st_ms_5', 'statement', 'ms', 'Sila pastikan modem anda sentiasa dihidupkan semasa ujian talian.'),
    ('st_ms_6', 'statement', 'ms', 'Permintaan anda telah direkodkan dan pasukan kami akan membuat susulan.'),
    ('st_zh_1', 'statement', 'zh', '感谢您的来电，我今天可以帮您处理账单问题。'),
    ('st_zh_2', 'statement', 'zh', '您的付款已经收到，账户状态已经更新。'),
    ('st_zh_3', 'statement', 'zh', '我们会把详细信息发送到您登记的电子邮箱。'),
    ('st_zh_4', 'statement', 'zh', '技术人员到达之前会先联系您。'),
    ('q_en_1', 'question', 'en', 'Is there anything else I can help you with today?'),
    ('q_en_2', 'question', 'en', 'Could you please confirm your full name?'),
    ('q_ms_1', 'question', 'ms', 'Ada apa-apa lagi yang boleh saya bantu?'),
    ('q_ms_2', 'question', 'ms', 'Boleh sahkan nama penuh anda?'),
    ('sh_en_1', 'short', 'en', 'Okay, noted.'),
    ('sh_en_2', 'short', 'en', 'Sure, let me check.'),
    ('sh_en_3', 'short', 'en', 'One moment please.'),
    ('sh_en_4', 'short', 'en', 'Thank you.'),
    ('sh_ms_1', 'short', 'ms', 'Baik, sebentar ya.'),
    ('sh_ms_2', 'short', 'ms', 'Terima kasih.'),
    ('lg_en_1', 'long', 'en', 'To restore your connection, please switch off the modem, wait for about thirty '
                              'seconds, then switch it back on and wait until the internet light turns green '
                              'before you try browsing again.'),
    ('lg_ms_1', 'long', 'ms', 'Untuk memulihkan sambungan anda, sila matikan modem, tunggu kira-kira tiga puluh '
                              'saat, kemudian hidupkan semula dan tunggu sehingga lampu internet bertukar hijau '
                              'sebelum mencuba semula.'),
    # through the normalizer: digits must be heard in order, in the call's language
    ('rb_en_1', 'readback', 'en', '750412105678. Did we get that right?'),
    ('rb_en_2', 'readback', 'en', 'Your IC number is 750412-10-5678. Is that correct?'),
    ('rb_ms_1', 'readback', 'ms', 'Nombor telefon anda 019-277 3451, betul?'),
    ('rb_en_3', 'readback', 'en', '6 4 0 9 2 3 0 8 7 7 6 6.'),
    ('rb_en_4', 'readback', 'en', 'Your number is 6711 [pause] 0514 [pause] 4262.'),
    # an agent's tool call that reached the TTS on its own: nothing should be heard
    ('tool_1', 'tool', 'en', '<|tool_call>call:end_call{}<tool_call|>'),
]
DIGITS = {'rb_en_1': '750412105678', 'rb_en_2': '750412105678', 'rb_ms_1': '0192773451',
          'rb_en_3': '640923087766', 'rb_en_4': '671105144262'}
# one reply, cut the way LiveKit's StreamAdapter sends it, under one interleave id
REPLIES = [
    ('rep_en', 'en', ['Thank you for waiting.', 'I have checked your account,', 'and the outage in your area',
                      'was fixed this morning.', 'Please restart your modem now.']),
    ('rep_ms', 'ms', ['Terima kasih kerana menunggu.', 'Saya sudah semak akaun anda,',
                      'dan gangguan di kawasan anda', 'telah dibaiki pagi tadi.', 'Sila hidupkan semula modem anda.']),
]
PLAIN = ('statement', 'question', 'short', 'long')
PROMPTS = {'en': 'Okay, so your number is seven five zero four, one two one zero. Did we get that right?',
           'ms': 'Okey, nombor anda tujuh lima kosong empat, satu dua satu kosong. Betul ya?'}
WLANG = {'en': 'english', 'ms': 'malay', 'zh': 'chinese'}


# ---------------------------------------------------------------- measuring one take

def analyze(wav_bytes):
    """Level, silence and pitch tracks of one wav. Top level so a process pool can run it."""
    import librosa
    if not wav_bytes:
        return {'dur': 0.0, 'clip_frac': 0.0, 'silent': True}
    with wave.open(io.BytesIO(wav_bytes)) as w:
        y = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    out = {'dur': len(y) / SR, 'clip_frac': float(np.mean(np.abs(y) >= 0.999)) if len(y) else 0.0}
    db = 20 * np.log10(librosa.feature.rms(y=y, frame_length=600, hop_length=HOP)[0] + 1e-9) \
        if len(y) >= 600 else np.array([-120.0])
    if db.max() < -50:
        return {**out, 'silent': True}
    act = db > max(db.max() - 35, -60)
    idx = np.flatnonzero(act)
    gaps = np.diff(idx) - 1
    f0, vflag, _ = librosa.pyin(y, fmin=65, fmax=450, sr=SR, frame_length=2048, hop_length=HOP)
    n = min(len(f0), len(db))
    voiced = vflag[:n] & act[:n] & ~np.isnan(f0[:n])
    st = np.where(voiced, 12 * np.log2(np.where(voiced, f0[:n], 100.0) / 100.0), np.nan)
    power = 10 ** (db[act] / 10)
    return {**out, 'silent': False,
            'lead_s': idx[0] * HOP / SR, 'trail_s': max(0.0, out['dur'] - (idx[-1] + 1) * HOP / SR),
            'span_s': (idx[-1] - idx[0] + 1) * HOP / SR,
            'max_pause_s': float(gaps.max()) * HOP / SR if len(gaps) else 0.0,
            'speech_db': float(10 * np.log10(power.mean())),
            'db': db[:n].round(2).tolist(), 'st': [None if np.isnan(v) else round(float(v), 3) for v in st]}


def syllables(text, lang):
    if lang == 'zh':
        return len(HAN.findall(text))
    n = 0
    for w in re.findall(r"[a-z']+", text.lower()):
        k = len(re.findall(r'[aeiouy]+' if lang == 'en' else r'[aeiou]+', w))
        if lang == 'en' and w.endswith('e') and not w.endswith(('le', 'ee')) and k > 1:
            k -= 1                                        # silent e
        n += max(k, 1)
    return n


def cer(ref, hyp, lang):
    def norm(s):
        s = re.sub(r"[^\w\s']", ' ', s.lower())
        return re.sub(r'\s+', '' if lang == 'zh' else ' ', s).strip()
    a, b = norm(ref), norm(hyp)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(len(a), 1)


_W = {'zero': '0', 'oh': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6',
      'seven': '7', 'eight': '8', 'nine': '9', 'kosong': '0', 'sifar': '0', 'satu': '1', 'dua': '2',
      'tiga': '3', 'empat': '4', 'lima': '5', 'enam': '6', 'tujuh': '7', 'lapan': '8', 'sembilan': '9',
      '零': '0', '〇': '0', '一': '1', '二': '2', '两': '2', '三': '3', '四': '4', '五': '5', '六': '6',
      '七': '7', '八': '8', '九': '9'}


def heard_digits(t):
    seq, rep = [], 1
    for tok in re.findall(r'\d|[A-Za-z]+|[零〇一二两三四五六七八九]', t):
        low = tok.lower()
        if low in ('double', 'triple'):
            rep = 2 if low == 'double' else 3
        elif tok.isdigit() or low in _W:
            seq.append((tok if tok.isdigit() else _W[low]) * rep)
            rep = 1
    return ''.join(seq)


def _voiced_st(t, sl):
    st = t['st'][sl]
    return [v for v in st if v is not None]


def jumps(t):
    """Adjacent 0.5 s windows, both >= 70% voiced, where level rises >= 3 dB AND pitch >= 1.5 st."""
    db, st, W = t['db'], t['st'], 50
    ev = pairs = 0
    for s in range(0, len(db) - 2 * W + 1, 25):
        a, b = range(s, s + W), range(s + W, s + 2 * W)
        va, vb = [i for i in a if st[i] is not None], [i for i in b if st[i] is not None]
        if len(va) < 0.7 * W or len(vb) < 0.7 * W:
            continue
        pairs += 1
        dl = np.mean([db[i] for i in vb]) - np.mean([db[i] for i in va])
        dr = np.median([st[i] for i in vb]) - np.median([st[i] for i in va])
        ev += int(dl >= 3 and dr >= 1.5)
    return ev, pairs


# ---------------------------------------------------------------- the run (rendered once)

def _speak(text, interleave_id=None):
    import requests
    r = requests.post(f'{BASE_URL}/v1/audio/speech', timeout=180,
                      headers={'X-Interleave-Id': interleave_id} if interleave_id else {},
                      json={'input': text, 'voice': VOICE, 'model': 'TTS-model',
                            'response_format': 'wav', 'stream': False})
    return r.status_code, r.content


def _render():
    jobs = [(cid, kind, lang, text, k) for cid, kind, lang, text in CASES for k in range(TAKES)]

    def one(job):
        cid, kind, lang, text, k = job
        status, wav = _speak(text)
        return [{'id': cid, 'kind': kind, 'lang': lang, 'text': text, 'take': k, 'status': status, 'wav': wav}]

    def reply(job):
        rid, lang, chunks, k = job
        iid = f'audio-test-{uuid.uuid4()}'
        out = []
        for i, text in enumerate(chunks):
            status, wav = _speak(text, iid)
            out.append({'id': f'{rid}_c{i}', 'reply': rid, 'chunk': i, 'kind': 'chunk', 'lang': lang,
                        'text': text, 'take': k, 'status': status, 'wav': wav})
        return out

    with ThreadPoolExecutor(4) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        futs += [ex.submit(reply, (rid, lang, chunks, k)) for rid, lang, chunks in REPLIES for k in range(TAKES)]
        return [t for f in futs for t in f.result()]


def _ear():
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    proc = WhisperProcessor.from_pretrained(WHISPER)
    model = WhisperForConditionalGeneration.from_pretrained(
        WHISPER, dtype=torch.float16 if dev == 'cuda' else torch.float32).to(dev).eval()
    tok = proc.tokenizer

    def hear(y16, lang=None, prompt=''):
        f = proc(y16, sampling_rate=16000, return_tensors='pt').input_features.to(dev, model.dtype)
        kw = {'task': 'transcribe', 'max_new_tokens': 200}
        if lang:
            kw['language'] = WLANG[lang]
        if prompt:
            kw['prompt_ids'] = proc.get_prompt_ids(prompt, return_tensors='pt').to(dev)
        with torch.no_grad():
            detected = None
            if not lang:
                detected = tok.convert_ids_to_tokens(int(model.detect_language(f)[0]))[2:-2]
            ids = model.generate(f, **kw)
        text = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        if prompt and text.startswith(prompt):
            text = text[len(prompt):].strip()
        return text, detected
    return hear


def _mos(wav):
    import requests
    r = requests.post(f'{MOS_URL}/predict', params={'reps': 16}, timeout=120,
                      headers={'Authorization': f'Bearer {MOS_KEY}'} if MOS_KEY else {},
                      files={'file': ('take.wav', wav, 'audio/wav')})
    r.raise_for_status()
    return float(r.json()['mos'])


def _tone_wav(seconds=1.0, hz=150.0):
    t = np.arange(int(SR * seconds)) / SR
    bio = io.BytesIO()
    with wave.open(bio, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((0.2 * np.sin(2 * np.pi * hz * t) * 32767).astype(np.int16).tobytes())
    return bio.getvalue()


@pytest.fixture(scope='module')
def run():
    # pyin is numba code, cached next to the package by default. Two runs at once, 8 workers
    # each, wrote that cache together once; every pyin after it segfaulted ("A process in the
    # process pool was terminated abruptly"). So: a private cache, filled by one call in this
    # process before the pool forks, and the workers inherit the compiled code instead of
    # writing it. NUMBA_CACHE_DIR only counts if set before numba is first imported.
    os.environ.setdefault('NUMBA_CACHE_DIR', tempfile.mkdtemp(prefix='audio-test-numba-'))
    librosa = pytest.importorskip('librosa')
    analyze(_tone_wav())
    pytest.importorskip('transformers')
    pytest.importorskip('requests')
    takes = _render()
    with ProcessPoolExecutor(8) as ex:
        for t, m in zip(takes, ex.map(analyze, [t['wav'] if t['status'] == 200 else b'' for t in takes])):
            t.update(m if t['status'] == 200 else {'silent': True})
    hear = _ear()
    for t in takes:
        if t['status'] != 200 or t['dur'] < 0.1:
            t['heard'], t['heard_auto'], t['heard_lang'] = '', '', None
            continue
        with wave.open(io.BytesIO(t['wav'])) as w:
            y = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        y16 = librosa.resample(y, orig_sr=SR, target_sr=16000)
        t['heard_auto'], t['heard_lang'] = hear(y16)
        t['heard'], _ = hear(y16, t['lang'], PROMPTS.get(t['lang'], '') if t['kind'] == 'readback' else '')
    if MOS_URL:
        with ThreadPoolExecutor(8) as ex:
            for t, m in zip(takes, ex.map(lambda t: _mos(t['wav']) if not t.get('silent') else None, takes)):
                t['mos'] = m
    # pitch reference: the median of each language's statements
    ref = {lang: statistics.median(v for t in takes if t['lang'] == lang and t['kind'] == 'statement'
                                   and not t.get('silent') for v in _voiced_st(t, slice(None)))
           for lang in {t['lang'] for t in takes if t['kind'] == 'statement'}}
    for t in takes:
        v = _voiced_st(t, slice(None)) if not t.get('silent') else []
        if len(v) >= 30 and t['lang'] in ref:
            t['register_st'] = statistics.median(v) - ref[t['lang']]
            q = np.percentile(v, [25, 75])
            t['iqr_st'] = float(q[1] - q[0])
            t['voiced_s'] = len(v) * HOP / SR
        if not t.get('silent') and t['kind'] in PLAIN + ('chunk',):
            t['rate'] = syllables(t['text'], t['lang']) / max(t['span_s'], 1e-3)
    state = {'takes': takes, 'ref': ref, 'agg': {}}
    if WAV_DIR:
        os.makedirs(WAV_DIR, exist_ok=True)
        for t in takes:
            if t['status'] == 200:
                open(os.path.join(WAV_DIR, f"{t['id']}_t{t['take']}.wav"), 'wb').write(t['wav'])
    yield state
    if REPORT:
        rows = [{k: v for k, v in t.items() if k not in ('wav', 'db', 'st')} for t in takes]
        json.dump({'voice': VOICE, 'takes_per_text': TAKES, 'ref_st_re_100hz': ref, 'agg': state['agg'],
                   'takes': rows}, open(REPORT, 'w'), ensure_ascii=False, indent=1,
                  default=lambda o: o.item() if hasattr(o, 'item') else str(o))


def _pick(run, kinds):
    return [t for t in run['takes'] if t['kind'] in kinds]


def _share(xs):
    return sum(xs) / max(len(xs), 1)


# ---------------------------------------------------------------- the checks

def test_every_request_succeeds(run):
    bad = [(t['id'], t['status']) for t in run['takes'] if t['status'] != 200]
    assert not bad, f'HTTP errors: {bad[:10]}'


def test_intelligible(run):
    """Whisper hears the words that were sent: per-language median CER, and few garbled takes."""
    agg = {}
    for lang in ('en', 'ms', 'zh'):
        ts = [t for t in _pick(run, PLAIN) if t['lang'] == lang]
        c = [cer(t['text'], t['heard'], lang) for t in ts]
        # garbled is judged on sentences of 6+ words (or 8+ Han characters): on a two-word line
        # one misheard word is 50% CER whether the voice slipped or Whisper did
        long_ = [x for x, t in zip(c, ts) if len(HAN.findall(t['text'])) >= 8 or len(t['text'].split()) >= 6]
        agg[lang] = {'median': statistics.median(c), 'bad_share': _share([x > THRESHOLDS['cer_bad'] for x in long_]),
                     'worst': sorted(((round(x, 2), t['id'], t['heard']) for x, t in zip(c, ts)), reverse=True)[:3]}
    run['agg']['intelligible'] = agg
    for lang, a in agg.items():
        assert a['median'] <= THRESHOLDS['cer_median'], f'{lang}: median CER {a["median"]:.3f} ({a["worst"]})'
        assert a['bad_share'] <= THRESHOLDS['cer_bad_share'], f'{lang}: garbled share {a["bad_share"]:.2f} ({a["worst"]})'


def test_pace(run):
    """Syllables per second: each language in a normal band, and no take far off its own text's median."""
    ts = [t for t in _pick(run, PLAIN + ('chunk',)) if 'rate' in t]
    agg = {'median': {}, 'off': []}
    for lang, (lo, hi) in THRESHOLDS['rate_band'].items():
        r = [t['rate'] for t in ts if t['lang'] == lang]
        agg['median'][lang] = statistics.median(r)
        assert lo <= agg['median'][lang] <= hi, f'{lang}: {agg["median"][lang]:.2f} syllables/s outside [{lo}, {hi}]'
    by = {}
    for t in ts:
        by.setdefault(t['id'], []).append(t)
    k, kmax = THRESHOLDS['rate_spread'], THRESHOLDS['rate_spread_max']
    for cid, group in by.items():
        m = statistics.median(t['rate'] for t in group)
        agg['off'] += [(cid, t['take'], round(float(t['rate'] / m), 2)) for t in group if not 1 / k <= t['rate'] / m <= k]
    agg['share_off'] = len(agg['off']) / len(ts)
    run['agg']['pace'] = agg
    far = [x for x in agg['off'] if not 1 / kmax <= x[2] <= kmax]
    assert not far, f'takes {kmax}x off their text\'s pace (cut short, or babbling): {far}'
    assert agg['share_off'] <= THRESHOLDS['rate_spread_share'], f'{agg["share_off"]:.3f} of takes {k}x off: {agg["off"]}'


def test_no_long_pauses(run):
    ts = [t for t in _pick(run, PLAIN + ('chunk',)) if not t.get('silent')]
    long_ = [(t['id'], t['take'], round(t['max_pause_s'], 2)) for t in ts if t['max_pause_s'] > THRESHOLDS['long_pause_s']]
    run['agg']['long_pauses'] = {'share': _share([1] * len(long_) + [0] * (len(ts) - len(long_))), 'takes': long_}
    assert run['agg']['long_pauses']['share'] <= THRESHOLDS['long_pause_share'], f'long pauses: {long_}'


def test_starts_and_ends_promptly(run):
    """Leading silence is heard as lag; trailing silence stacks up between an agent's chunks."""
    ts = [t for t in _pick(run, PLAIN + ('chunk',)) if not t.get('silent')]
    late = [(t['id'], t['take'], round(t['lead_s'], 2), round(t['trail_s'], 2)) for t in ts
            if t['lead_s'] > THRESHOLDS['lead_s'] or t['trail_s'] > THRESHOLDS['trail_s']]
    run['agg']['edges'] = {'share': len(late) / len(ts), 'takes': late}
    assert len(late) / len(ts) <= THRESHOLDS['edge_share'], f'slow starts / long tails: {late}'


def test_register_stable(run):
    """'Calm, then suddenly high and excited': a take whose median pitch sits 3+ st off its language."""
    agg = {}
    for kind, limit in THRESHOLDS['register_share'].items():
        ts = [t for t in _pick(run, (kind,)) if 'register_st' in t]
        off = [(t['id'], t['take'], round(t['register_st'], 1)) for t in ts if abs(t['register_st']) >= THRESHOLDS['register_st']]
        agg[kind] = {'share': len(off) / max(len(ts), 1), 'n': len(ts), 'takes': off}
    run['agg']['register'] = agg
    for kind, a in agg.items():
        assert a['share'] <= THRESHOLDS['register_share'][kind], f'{kind}: {a["share"]:.2f} of takes 3+ st off: {a["takes"]}'


def test_not_monotone(run):
    ts = [t for t in _pick(run, ('statement', 'question', 'long')) if t.get('voiced_s', 0) >= 1.5]
    flat = [(t['id'], t['take'], round(t['iqr_st'], 2)) for t in ts if t['iqr_st'] < THRESHOLDS['monotone_iqr_st']]
    run['agg']['monotone'] = {'share': len(flat) / max(len(ts), 1), 'takes': flat}
    assert len(flat) / max(len(ts), 1) <= THRESHOLDS['monotone_share'], f'flat reads: {flat}'


def test_no_sudden_jumps(run):
    ev = pairs = 0
    where = []
    for t in _pick(run, PLAIN + ('chunk',)):
        if t.get('silent'):
            continue
        e, p = jumps(t)
        ev, pairs = ev + e, pairs + p
        if e:
            where.append((t['id'], t['take'], e))
    rate = 100 * ev / max(pairs, 1)
    run['agg']['jumps'] = {'per_100_pairs': rate, 'pairs': pairs, 'takes': where}
    assert rate <= THRESHOLDS['jumps_per_100'], f'{rate:.2f} jumps per 100 window pairs: {where}'


def test_chunk_joins_continue(run):
    """Interleaved chunks of one reply: the next chunk picks up where the last left off. Measured
    as in INTERLEAVE.md, chunk N+1's median pitch minus chunk N's: continuous speech drifts down,
    a cold chunk resets up (+0.53 st there). Not the last second of N against the first of N+1:
    that compares a phrase's fall with the next phrase's onset and is positive in natural speech."""
    steps = []
    for rid, _, chunks in REPLIES:
        for k in range(TAKES):
            seq = sorted((t for t in run['takes'] if t.get('reply') == rid and t['take'] == k), key=lambda t: t['chunk'])
            steps += [b['register_st'] - a['register_st'] for a, b in zip(seq, seq[1:])
                      if 'register_st' in a and 'register_st' in b]
    assert len(steps) >= len(REPLIES) * TAKES * 2, f'too few measurable joins ({len(steps)})'
    mean, p90 = float(np.mean(steps)), float(np.percentile(np.abs(steps), 90))
    run['agg']['joins'] = {'n': len(steps), 'mean_st': mean, 'p90_abs_st': p90}
    assert mean <= THRESHOLDS['join_mean_st'], f'chunks restart higher: mean step {mean:+.2f} st'
    assert p90 <= THRESHOLDS['join_p90_st'], f'p90 |step| {p90:.2f} st'


def test_even_level(run):
    ts = [t for t in _pick(run, PLAIN + ('chunk',)) if not t.get('silent')]
    med = statistics.median(t['speech_db'] for t in ts)
    off = [(t['id'], t['take'], round(t['speech_db'] - med, 1)) for t in ts if abs(t['speech_db'] - med) > THRESHOLDS['level_db']]
    clipped = [(t['id'], t['take']) for t in ts if t['clip_frac'] > THRESHOLDS['clip_frac']]
    run['agg']['level'] = {'median_db': med, 'share_off': len(off) / len(ts), 'off': off, 'clipped': clipped}
    assert len(off) / len(ts) <= THRESHOLDS['level_share'], f'level off the median by > {THRESHOLDS["level_db"]} dB: {off}'
    assert len(clipped) / len(ts) <= THRESHOLDS['clip_share'], f'clipped: {clipped}'


def test_first_word_heard(run):
    """The v4-calm murmur ('herm' before the first word) showed up as a lost first word."""
    ts = [t for t in _pick(run, PLAIN + ('chunk',)) if t['lang'] != 'zh']
    def ok(t):
        first = re.findall(r'[a-z]+', t['text'].lower())[0]
        return any(w.startswith(first[:3]) for w in re.findall(r'[a-z]+', t['heard'].lower())[:3])
    miss = [(t['id'], t['take'], t['heard'][:40]) for t in ts if not ok(t)]
    share = 1 - len(miss) / len(ts)
    run['agg']['first_word'] = {'share': share, 'missed': miss}
    assert share >= THRESHOLDS['first_word_share'], f'first word heard in {share:.2f}: {miss}'


@pytest.mark.skipif(not MOS_URL, reason='MOS_API not set')
def test_natural(run):
    agg = {}
    for lang in ('en', 'ms', 'zh'):
        m = [t['mos'] for t in _pick(run, PLAIN) if t['lang'] == lang and t.get('mos') is not None]
        agg[lang] = {'mean': float(np.mean(m)), 'low_share': _share([x < THRESHOLDS['mos_low'] for x in m])}
    run['agg']['mos'] = agg
    for lang, a in agg.items():
        assert a['mean'] >= THRESHOLDS['mos_mean'], f'{lang}: MOS {a["mean"]:.2f}'
        assert a['low_share'] <= THRESHOLDS['mos_low_share'], f'{lang}: {a["low_share"]:.2f} of takes under {THRESHOLDS["mos_low"]}'


def test_readbacks_heard_in_order(run):
    """Every digit, in order, in the call's language: not dropped, not read as Chinese numerals."""
    ts = _pick(run, ('readback',))
    def ok(t):
        if HAN.search(t['heard_auto']) or t['heard_lang'] == 'zh':
            return False
        return heard_digits(t['heard']) == DIGITS[t['id']] and 'pause' not in t['heard'].lower()
    miss = [(t['id'], t['take'], t['heard'], t['heard_lang']) for t in ts if not ok(t)]
    share = 1 - len(miss) / len(ts)
    run['agg']['readbacks'] = {'share': share, 'missed': miss}
    assert share >= THRESHOLDS['readback_share'], f'read-backs right in {share:.2f}: {miss}'


def test_tool_call_is_not_spoken(run):
    ts = _pick(run, ('tool',))
    spoken = [(t['take'], t['heard'], round(t['dur'], 2)) for t in ts
              if len(re.findall(r'\w+', t['heard'])) > 1 or re.search(r'\bcall\b', t['heard'], re.I)]
    run['agg']['tool'] = {'spoken': spoken, 'durations': [round(t['dur'], 2) for t in ts]}
    assert not spoken, f'a tool call was spoken: {spoken}'
