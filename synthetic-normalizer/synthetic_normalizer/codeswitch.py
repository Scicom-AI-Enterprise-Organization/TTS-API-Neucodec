#!/usr/bin/env python3
"""
Code-switched templates for the Malaysian language mix (Malay, English, Mandarin, Tamil).

Why a separate module
---------------------
Every other row in this dataset is ONE language. Malaysian speech is not: a sentence carries a
matrix language and drops words, whole phrases and often the number itself into another
("bil bulan ini RM250, due date 12/3/2024"). A normalizer trained only on monolingual rows has
to guess which language reads the digits, and that guess is where it breaks -- the same problem
`app/spoken_normalizer/lang.py` solves at runtime with a per-number neighbour vote.

These templates are **hand-written**, not LLM-written, and the spoken side is produced
**deterministically**. The normalizer LLM was tried first and is not a usable teacher here: asked
for the spoken form of "bil anda RM250 due 12/3/2024" it answered
"bil anda ringgit malaysia dua ratus lima puluh due …" -- currency before the amount, which no
Malay speaker says. So the written side comes from these frames and the spoken side from
`app.spoken_normalizer` with the language **forced per slot**, which is exactly the judgement a
code-switch corpus has to encode and the one thing an LLM cannot be trusted to make.

The `{slot:lang}` tag
---------------------
A slot names the language it is READ in, not the language of the sentence::

    'Encik, bil bulan ini {money:ms} dan due date pada {date:en}.'
        RM250          -> 'dua ratus lima puluh ringgit'     (Malay: it sits in the Malay clause)
        12 March 2024  -> 'the twelfth of March …'           (English: it sits in the English clause)

The tag is written by hand for every slot, per fragment, because that is the label being taught.
The written value is also formatted in that language (`f_date('ms', ...)` gives a Malay month
name, `f_time('en', ...)` an am/pm clock), which is what the mixed text actually looks like.

Reading a slot with its own context
-----------------------------------
`generate.fill_template` normalizes each slot together with the few carrier words on either side
of it (same fragment, same language, digit-free) and strips them off again, because the cue is
what decides the reading:

    '704251'                      -> 'tujuh ratus empat ribu …'        (a quantity)
    'nombor rujukan anda 704251'  -> '… tujuh kosong empat dua lima satu' (digit by digit)
    '9.50'                        -> 'ஒன்பது புள்ளி ஐந்து பூஜ்ஜியம்'    (a decimal)
    '9.50 மணிக்கு'                -> 'ஒன்பது ஐம்பது மணிக்கு'            (a time)

So write the cue words into the frame next to the slot ('nombor rujukan', 'OTP', 'மணிக்கு',
'参考号码') and the reading follows. Nothing else in a frame may contain digits or a currency
symbol -- `generate.fill_template` rejects a pair whose spoken side still has one.

Pairs
-----
Ordered (matrix, embedded): the matrix language is the grammar of the sentence, the embedded one
is what gets mixed in. `ms-en` and `en-ms` are both there because Bahasa rojak runs in both
directions and the number reading follows the fragment, not the pair.

⚠ Tamil rows (`ta-en`, `ta-ms`) have not been checked by a native speaker -- same caveat the
README carries for the deterministic Sinhala, Filipino, Arabic and Polish rows.
"""
import os
import re
import sys

# app.spoken_normalizer is the verbalizer for every language in these pairs (and its marker-word
# lists are what "is this really mixed?" means here), so the repo has to be importable even when
# this module is used on its own.
sys.path.insert(0, os.environ.get('SN_TTS_REPO') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

# (matrix, embedded) -- both are languages app.spoken_normalizer verbalizes (en/ms/zh/ta), which
# is what makes the deterministic per-slot reading possible.
CS_PAIRS = {
    'ms-en': ('ms', 'en'),
    'en-ms': ('en', 'ms'),
    'zh-en': ('zh', 'en'),
    'zh-ms': ('zh', 'ms'),
    'ta-en': ('ta', 'en'),
    'ta-ms': ('ta', 'ms'),
}
CS_LOCALES = list(CS_PAIRS)
CS_LANGUAGE_NAME = {
    'ms-en': 'Malay-English code-switching (Malaysia)',
    'en-ms': 'English-Malay code-switching (Malaysia)',
    'zh-en': 'Mandarin-English code-switching (Malaysia)',
    'zh-ms': 'Mandarin-Malay code-switching (Malaysia)',
    'ta-en': 'Tamil-English code-switching (Malaysia)',
    'ta-ms': 'Tamil-Malay code-switching (Malaysia)',
}
# A short note for the SFT system prompt, so the fine-tuned model is told what it is looking at.
CS_NOTE = {
    'ms-en': 'Malaysian Malay-English code-switching (Bahasa rojak). RM is ringgit, sen for cents.',
    'en-ms': 'Malaysian English with Malay mixed in. RM is ringgit, sen for cents.',
    'zh-en': 'Malaysian Mandarin with English mixed in. RM is 令吉, cents 仙.',
    'zh-ms': 'Malaysian Mandarin with Malay mixed in. RM is 令吉, cents 仙.',
    'ta-en': 'Malaysian Tamil with English mixed in. RM is ரிங்கிட், cents சென்.',
    'ta-ms': 'Malaysian Tamil with Malay mixed in. RM is ரிங்கிட், cents சென்.',
}

SLOT_TAG_RE = re.compile(r'\{([a-z_]+):([a-z-]+)\}')
# Slots only these frames use: the generic {unit} picks any unit, which turns a "data plan" into
# kilograms and a body temperature into kilometres. Filled by generate.CS_FILLERS.
CS_EXTRA_SLOTS = ['unit_data', 'unit_temp', 'int_small', 'time_plain']
# Slots that are Latin tokens wherever they appear -- a tracking id, an email address, a URL is
# spelled the same way in a Malay, Chinese or Tamil sentence -- so they may be tagged `en` even
# in a pair that has no English side (zh-ms, ta-ms).
LATIN_TOKEN_SLOTS = {'email', 'url', 'id'}


def cs_safe_slots(safe_slots):
    return {lang: list(slots) + CS_EXTRA_SLOTS for lang, slots in safe_slots.items()}

# --------------------------------------------------------------------------- templates
# Service-domain sentences (bank/e-wallet, telco, clinic, delivery, government, school, airline,
# shop, food delivery, insurance, ride-hailing, hotel) in the register these actually arrive in:
# an agent speaking, an SMS, an app notification.
CS_TEMPLATES = {
    'ms-en': [
        'Encik, bil bulan ini {money:ms} dan due date pada {date:ms}.',
        'Payment sebanyak {money:ms} dah kami terima pada {date:ms}, terima kasih.',
        'Sila top up sekurang-kurangnya {money:ms} sebelum pukul {time:ms} hari ini ya.',
        'Tracking number parcel anda ialah {id:en}, boleh check kat {url:en}.',
        'Appointment doktor pada {date:ms} pukul {time:ms}, please datang awal sikit.',
        'Nombor rujukan anda {digits:ms}, simpan untuk reference ya.',
        'Baki akaun anda tinggal {money:ms} je, top up dulu sebelum guna.',
        'Kami akan call balik di {phone:ms} dalam masa {int_small:ms} minit.',
        'Diskaun {percent:ms} untuk semua item hari ini, jangan miss out.',
        'Order anda akan sampai dalam {range:en} working days.',
        'OTP anda {digits:ms}, jangan share dengan sesiapa.',
        'Meeting kita reschedule ke {date:ms} pukul {time:ms} kat office.',
        'Invoice {id:en} berjumlah {money:ms} masih outstanding.',
        'Anda nombor {ordinal:ms} dalam queue, please hold sekejap.',
        'Data plan anda tinggal {unit_data:ms} sahaja untuk bulan ni.',
        'Sila email dokumen ke {email:en} sebelum {date:ms}.',
        'Kami dah refund {money:ms}, take {range:en} working days untuk masuk.',
        'Flight {id:en} berlepas pukul {time:ms}, please check in awal.',
        'Bilik ready selepas pukul {time:ms}, check out pula sebelum {time:ms}.',
        'Sejak tahun {year:ms} kami dah serve lebih {big:ms} pelanggan.',
        'Yuran pendaftaran {money:ms}, payment boleh buat online kat {url:en}.',
        'Suhu badan anak encik {unit_temp:ms}, doktor cadang monitor dulu.',
        'Polisi insurans expire pada {date:ms}, renew sekarang dapat {percent:ms} off.',
        'Driver sampai dalam {int_small:ms} minit, plate number {id:en}.',
        'Points anda ada {big:ms}, boleh redeem voucher {money:ms}.',
        'Cash back {percent:ms} untuk spending lebih {money:ms} sebulan.',
        'Kelas ganti pada {date:ms}, please confirm attendance sebelum {time:ms}.',
        'Bil naik {percent:ms} sebab extra usage {unit_data:ms} bulan lepas.',
        'Sila isi survey kat {url:en}, ambil masa {int_small:ms} minit je.',
        'Loan approve {money:ms}, tenure {int:ms} tahun, interest {percent:ms} setahun.',
        'Stok tinggal {int:ms} unit je, order sekarang sebelum sold out.',
        'Nombor akaun anda {digits:ms}, please verify dengan customer service.',
        'Delivery fee {money:ms}, free kalau order lebih {money:ms}.',
        'Slot vaksin available pada {date:ms}, walk in pun boleh.',
        'Kad anda akan expire {date:ms}, replacement fee {money:ms}.',
        'Kami buka dari pukul {time:ms} sampai {time:ms}, close on public holiday.',
    ],
    'en-ms': [
        'Your bill this month is {money:en} and the due date is {date:en}, ya.',
        'Please bayar before {date:en} to avoid late charges.',
        'Sorry encik, your appointment is on {date:en} at {time:en}.',
        'Boleh check your balance? It is {money:en} only now.',
        'Your parcel {id:en} sudah sampai at the hub, delivery on {date:en}.',
        'The promo is {percent:en} off, tapi only until {date:en}.',
        'Please hold sekejap, you are number {ordinal:en} in the queue.',
        'Your OTP is {digits:en}, jangan share with anyone ya.',
        'We already refund {money:en}, tunggu {range:en} working days.',
        'Call me back at {phone:en} kalau the line drop.',
        'Your reference number is {digits:en}, simpan for follow up.',
        'The clinic buka from {time:en} until {time:en} every weekday.',
        'Your data usage is {unit_data:en} already, lebih than the free quota.',
        'Total is {money:en}, boleh pay by card or e-wallet.',
        'Sila email the form to {email:en} before {date:en}.',
        'The driver will arrive in {int_small:en} minutes, plate {id:en}.',
        'Your flight {id:en} departs at {time:en}, please check in awal.',
        'Registration fee is {money:en}, payment online kat {url:en}.',
        'We give {percent:en} discount for members, tapi terms apply.',
        'Since {year:en} we already served more than {big:en} customers.',
        'Your policy expires on {date:en}, renew dulu before it lapse.',
        'The room is ready after {time:en}, check out sebelum {time:en}.',
        'You have {big:en} points, boleh redeem a {money:en} voucher.',
        'Delivery takes {range:en} days, tapi faster kalau you top up.',
        'Your loan of {money:en} is approved, interest {percent:en} per year.',
        'Please fill the survey at {url:en}, ambil {int_small:en} minutes only.',
        'The class is on {date:en}, tolong confirm your attendance.',
        'Temperature is {unit_temp:en}, doktor said monitor for now.',
        'Stock tinggal {int:en} units, order now before habis.',
        'Late payment charge is {percent:en} of the outstanding amount, ya.',
    ],
    'zh-en': [
        '您好，您本月的账单是{money:zh}，due date 是{date:zh}。',
        '您的parcel已经到了，tracking number 是{id:en}。',
        '请在{time:zh}之前完成payment，谢谢您。',
        '您的OTP是{digits:zh}，请不要share给别人。',
        '现在有{percent:zh}的discount，只到{date:zh}。',
        '您的户口余额是{money:zh}，请先top up。',
        '我们会在{int_small:zh}分钟内call您，号码是{phone:zh}。',
        '您的参考号码是{digits:zh}，请save起来。',
        '医生的appointment安排在{date:zh}{time:zh}。',
        '您排在{ordinal:zh}位，请hold一下。',
        '这个月的data用了{unit_data:en}，已经超过free quota。',
        '请把文件email到{email:en}，deadline 是{date:zh}。',
        'Refund的{money:zh}已经处理，需要{range:zh}个working days。',
        '您的flight {id:en} 在{time:zh}起飞，请早点check in。',
        '房间{time:zh}之后ready，check out 是{time:zh}之前。',
        '我们从{year:zh}年开始，已经serve超过{big:zh}位客户。',
        '报名费是{money:zh}，可以online 在{url:en}付款。',
        '孩子的体温是{unit_temp:zh}，医生建议先monitor。',
        '您的保单在{date:zh}到期，现在renew有{percent:zh}折扣。',
        'Driver {int_small:zh}分钟后到，车牌是{id:en}。',
        '您有{big:zh}积分，可以redeem {money:zh}的voucher。',
        '消费满{money:zh}就有{percent:zh}的cash back。',
        '补课安排在{date:zh}，请在{time:zh}之前confirm。',
        '账单增加了{percent:zh}，因为上个月多用了{unit_data:en}。',
        '请到{url:en}填survey，只需要{int_small:zh}分钟。',
        '贷款{money:zh}已经approve，年利率{percent:zh}。',
        '库存只剩{int:zh}个，请尽快order。',
        '运费{money:zh}，满{money:zh}就free delivery。',
        '您的卡{date:zh}过期，补办fee 是{money:zh}。',
        '我们的营业时间是{time:zh}到{time:zh}，public holiday休息。',
    ],
    'zh-ms': [
        '您的bil这个月是{money:zh}，请在{date:zh}之前bayar。',
        '您的户口balance只剩{money:zh}，请先top up。',
        '参考号码是{digits:zh}，请simpan好。',
        '请在{time:zh}之前完成pembayaran，谢谢。',
        '现在有{percent:zh}的diskaun，只到{date:zh}。',
        '您的parcel在{date:zh}送到，nombor tracking 是{id:en}。',
        '我们会call您的nombor {phone:zh}，请保持畅通。',
        '医生的temujanji在{date:zh}{time:zh}。',
        '这个月的data用了{unit_data:ms}，超过了kuota。',
        '您排在{ordinal:zh}位，请tunggu一下。',
        '您的OTP是{digits:zh}，请不要kongsi给别人。',
        'Yuran是{money:zh}，可以在{url:en}付款。',
        '退款{money:zh}已经处理，需要{range:zh}个hari bekerja。',
        '孩子的suhu是{unit_temp:zh}，请先observe。',
        '您的insurans在{date:zh}到期，renew有{percent:zh}折扣。',
        '从{year:zh}年开始，我们已经服务超过{big:zh}位pelanggan。',
        '您有{big:zh}点mata ganjaran，可以换{money:zh}的baucar。',
        '库存只剩{int:zh}个，请尽快pesan。',
        '运费{money:zh}，满{money:zh}就percuma。',
        '房间在{time:zh}之后sedia，check out 是{time:zh}之前。',
        '罚款是outstanding的{percent:zh}，请准时bayar。',
        '请把dokumen email到{email:en}，最迟{date:zh}。',
        '补课在{date:zh}，请在{time:zh}之前sahkan。',
        '我们的waktu operasi是{time:zh}到{time:zh}。',
    ],
    'ta-en': [
        'உங்கள் bill {money:ta}, due date {date:en}.',
        'உங்கள் parcel-ன் tracking number {id:en}, delivery {date:ta} அன்று.',
        'உங்கள் OTP {digits:ta}, யாருடனும் share செய்ய வேண்டாம்.',
        'குறியீட்டு எண் {digits:ta}, reference-க்காக save செய்யுங்கள்.',
        'உங்கள் account-ல் {money:ta} மட்டுமே உள்ளது, top up செய்யுங்கள்.',
        'Doctor appointment {date:ta} அன்று {time_plain:ta} மணிக்கு.',
        'நாங்கள் {int_small:ta} நிமிடத்தில் {phone:ta} என்ற எண்ணுக்கு call செய்வோம்.',
        'இன்று அனைத்து item-க்கும் {percent:ta} discount உண்டு.',
        'உங்கள் order {range:en} working days-ல் வந்துவிடும்.',
        'நீங்கள் queue-ல் {ordinal:ta} இடத்தில் இருக்கிறீர்கள், please hold.',
        'இந்த மாதம் data {unit_data:ta} பயன்படுத்தி விட்டீர்கள், free quota முடிந்தது.',
        'ஆவணங்களை {email:en} க்கு email செய்யுங்கள், deadline {date:en}.',
        '{money:ta} refund process ஆகிவிட்டது, {range:en} working days ஆகும்.',
        'உங்கள் flight {id:en} {time_plain:ta} மணிக்கு புறப்படும், early-ஆக check in செய்யுங்கள்.',
        'Room {time_plain:ta} மணிக்கு பிறகு ready, check out {time_plain:ta} மணிக்கு முன்.',
        '{year:ta} முதல் நாங்கள் {big:ta} customers-க்கு சேவை செய்கிறோம்.',
        'பதிவு கட்டணம் {money:ta}, payment-ஐ {url:en} இல் செய்யலாம்.',
        'குழந்தையின் உடல் வெப்பநிலை {unit_temp:ta}, doctor monitor செய்ய சொன்னார்.',
        'உங்கள் policy {date:ta} அன்று expire ஆகும், இப்போது renew செய்தால் {percent:ta} தள்ளுபடி.',
        'Driver {int_small:ta} நிமிடத்தில் வருவார், plate number {id:en}.',
        'உங்களிடம் {big:ta} points உள்ளன, {money:ta} voucher redeem செய்யலாம்.',
        '{money:ta}க்கு மேல் செலவு செய்தால் {percent:ta} cash back கிடைக்கும்.',
        'Extra class {date:ta} அன்று, {time_plain:ta} மணிக்கு முன் confirm செய்யுங்கள்.',
        'கடந்த மாதம் {unit_data:ta} அதிகமாக பயன்படுத்தியதால் bill {percent:ta} அதிகரித்தது.',
        '{url:en} இல் survey நிரப்புங்கள், {int_small:ta} நிமிடம் மட்டுமே ஆகும்.',
        '{money:ta} loan approve ஆகிவிட்டது, வட்டி ஆண்டுக்கு {percent:ta}.',
        'Stock-ல் {int:ta} unit மட்டுமே உள்ளது, விரைவில் order செய்யுங்கள்.',
        'Delivery fee {money:ta}, {money:ta}க்கு மேல் order செய்தால் free.',
        'உங்கள் card {date:ta} அன்று expire ஆகும், replacement fee {money:ta}.',
        'எங்கள் office {time_plain:ta} மணி முதல் {time_plain:ta} மணி வரை open.',
    ],
    'ta-ms': [
        'உங்கள் bil {money:ta}, {date:ta}க்கு முன் bayar செய்யவும்.',
        'உங்கள் akaun-ல் {money:ta} மட்டுமே உள்ளது, tolong top up.',
        'Nombor rujukan {digits:ta}, simpan செய்து வையுங்கள்.',
        'உங்கள் OTP {digits:ta}, யாருடனும் kongsi செய்ய வேண்டாம்.',
        'இன்று அனைத்து barang-க்கும் {percent:ta} diskaun உண்டு.',
        'Temujanji {date:ta} அன்று {time_plain:ta} மணிக்கு.',
        'நாங்கள் {int_small:ta} நிமிடத்தில் {phone:ta}க்கு telefon செய்வோம்.',
        'உங்கள் bungkusan {date:ta} அன்று வரும், nombor tracking {id:en}.',
        'Yuran pendaftaran {money:ta}, bayaran-ஐ {url:en} இல் செய்யலாம்.',
        'இந்த மாதம் data {unit_data:ta} பயன்படுத்தி விட்டீர்கள், kuota முடிந்தது.',
        'நீங்கள் giliran-ல் {ordinal:ta} இடத்தில் இருக்கிறீர்கள், tunggu sekejap.',
        '{money:ta} wang dikembalikan process ஆகிவிட்டது.',
        'Bilik {time_plain:ta} மணிக்கு பிறகு sedia ஆகும்.',
        '{year:ta} முதல் {big:ta} pelanggan-க்கு சேவை செய்கிறோம்.',
        'குழந்தையின் suhu {unit_temp:ta}, doktor monitor செய்ய சொன்னார்.',
        'உங்கள் insurans {date:ta} அன்று tamat, இப்போது renew செய்தால் {percent:ta} diskaun.',
        'Pemandu {int_small:ta} நிமிடத்தில் வருவார், nombor plat {id:en}.',
        'உங்களிடம் {big:ta} mata உள்ளன, {money:ta} baucar redeem செய்யலாம்.',
        'Kelas ganti {date:ta} அன்று, {time_plain:ta} மணிக்கு முன் sahkan செய்யுங்கள்.',
        'Stok-ல் {int:ta} unit மட்டுமே உள்ளது, cepat pesan செய்யுங்கள்.',
        'Caj penghantaran {money:ta}, {money:ta}க்கு மேல் என்றால் percuma.',
        'உங்கள் kad {date:ta} அன்று tamat tempoh, caj gantian {money:ta}.',
        'Pejabat {time_plain:ta} மணி முதல் {time_plain:ta} மணி வரை buka.',
        'Denda outstanding-ன் {percent:ta}, tolong bayar tepat pada masa.',
    ],
}


def slot_langs(tpl):
    """[(slot, lang), ...] in order of appearance."""
    return [(m.group(1), m.group(2)) for m in SLOT_TAG_RE.finditer(tpl)]


def validate(tpl, loc, safe_slots):
    """A frame is usable when: it has 1-4 tagged slots, every tag names one of the pair's two
    languages and a slot that language can fill, no untagged brace is left, and the carrier
    itself carries no digit (the value comes from the slot, so a digit here would be one the
    spoken side never reads)."""
    pair = CS_PAIRS[loc]
    tags = slot_langs(tpl)
    if not 1 <= len(tags) <= 4:
        return False, 'slot count'
    for slot, lang in tags:
        if lang not in pair and not (lang == 'en' and slot in LATIN_TOKEN_SLOTS):
            return False, f'{lang} not in {loc}'
        if slot not in safe_slots[lang]:
            return False, f'{slot} unsafe for {lang}'
    if re.search(r'[0-9]', tpl):
        return False, 'digit in the frame'
    if _TA_MERGING_SUFFIX.search(tpl):
        return False, 'Tamil suffix merges into the number word'
    if re.search(r'\{[^}]*\}', SLOT_TAG_RE.sub('', tpl)):
        return False, 'untagged brace'
    return True, ''


# A Tamil suffix starting with an independent vowel merges INTO the number word by sandhi
# (1970 + ஆம் -> '... எழுபதாம்', not '... எழுபது ஆம்'), so a frame that puts one after a slot
# cannot be reassembled from a separately-read slot and a copied carrier -- the spoken side comes
# out as 'எழுபதுஆம்'. Suffixes that only concatenate are fine: 'RM27க்கு' -> 'ரிங்கிட்டுக்கு'.
# Rephrase instead ('{year:ta} முதல்').
_TA_MERGING_SUFFIX = re.compile(r'\{[a-z_]+:ta\}\s*(ஆம்|ஆக|ஆன|ஆவது)')

_SCRIPT = {'zh': re.compile(r'[一-鿿]'), 'ta': re.compile(r'[஀-௿]')}


def has_both_languages(tpl, loc):
    """The frame really mixes: for zh/ta pairs that is script (CJK/Tamil *and* Latin letters);
    for ms-en/en-ms, where both sides are Latin, it is marker words from both languages
    (app/spoken_normalizer/lang.py's own lists, so 'the sentence is code-switched' means here
    what it means at runtime)."""
    matrix, embedded = CS_PAIRS[loc]
    text = SLOT_TAG_RE.sub(' ', tpl)
    if matrix in _SCRIPT:
        return bool(_SCRIPT[matrix].search(text)) and bool(re.search(r'[A-Za-z]', text))
    from app.spoken_normalizer.lang import MS_WORDS, EN_WORDS
    words = {w.strip('.,!?()').lower() for w in re.findall(r"[A-Za-z']+", text)}
    return bool(words & MS_WORDS) and bool(words & EN_WORDS)
