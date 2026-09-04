"""
Unit tests for the rule-based spoken normalizer (app/spoken_normalizer). No GPU, no network:

    uv run --with pytest pytest tests/test_spoken_normalizer.py -v

Expected strings are the LLM normalizer's outputs on the same inputs (bench/results/
normalizer_truth.jsonl) wherever the two agree; where they deliberately differ the docstring
says why. The corpus-wide properties at the bottom are the real guardrail: every digit gets
read, plain text is untouched, and the output is a fixed point.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bench'))

import pytest

from app.spoken_normalizer import normalize, detect_lang
from app.spoken_normalizer import numbers as N
from normalizer_corpus import CORPUS


def canon(s):
    s = s.lower().replace('-', ' ')
    return re.sub(r'[\s,.;:!?。，！？、]+', ' ', s).strip()


# --------------------------------------------------------------------------- numbers
class TestNumbers:
    @pytest.mark.parametrize('n,en,ms,zh,ta', [
        (0, 'zero', 'sifar', '零', 'பூஜ்ஜியம்'),
        (3, 'three', 'tiga', '三', 'மூன்று'),
        (12, 'twelve', 'dua belas', '十二', 'பன்னிரண்டு'),
        (21, 'twenty-one', 'dua puluh satu', '二十一', 'இருபத்தொன்று'),
        (110, 'one hundred ten', 'seratus sepuluh', '一百一十', 'நூற்று பத்து'),
        (250, 'two hundred fifty', 'dua ratus lima puluh', '二百五十', 'இருநூற்று ஐம்பது'),
        (305, 'three hundred five', 'tiga ratus lima', '三百零五', 'முந்நூற்று ஐந்து'),
        (1000, 'one thousand', 'seribu', '一千', 'ஆயிரம்'),
        (1250, 'one thousand two hundred fifty', 'seribu dua ratus lima puluh', '一千二百五十', 'ஆயிரத்து இருநூற்று ஐம்பது'),
        (1500, 'one thousand five hundred', 'seribu lima ratus', '一千五百', 'ஆயிரத்து ஐந்நூறு'),
        (1998, 'one thousand nine hundred ninety-eight', 'seribu sembilan ratus sembilan puluh lapan', '一千九百九十八',
         'ஆயிரத்து தொள்ளாயிரத்து தொண்ணூற்று எட்டு'),
        (2024, 'two thousand twenty-four', 'dua ribu dua puluh empat', '二千零二十四', 'இரண்டாயிரத்து இருபத்து நான்கு'),
        (33000000, 'thirty-three million', 'tiga puluh tiga juta', '三千三百万', 'முப்பத்து மூன்று மில்லியன்'),
    ])
    def test_cardinal(self, n, en, ms, zh, ta):
        assert N.cardinal(n, 'en') == en
        assert N.cardinal(n, 'ms') == ms
        assert N.cardinal(n, 'zh') == zh
        assert N.cardinal(n, 'ta') == ta

    def test_english_ordinals_and_years(self):
        assert [N.en_ordinal(n) for n in (1, 2, 3, 12, 21)] == ['first', 'second', 'third', 'twelfth', 'twenty-first']
        assert N.en_year(1998) == 'nineteen ninety-eight'
        assert N.en_year(2005) == 'two thousand and five'
        assert N.en_year(2024) == 'twenty twenty-four'
        assert N.en_year(2030) == 'twenty thirty'

    def test_malay_ordinals(self):
        assert [N.ms_ordinal(n) for n in (1, 2, 3, 12)] == ['pertama', 'kedua', 'ketiga', 'kedua belas']

    def test_tamil_sandhi(self):
        # oblique tens fuse with vowel-initial units, stay apart from consonant-initial ones
        assert N.ta_cardinal(25) == 'இருபத்தைந்து'
        assert N.ta_cardinal(24) == 'இருபத்து நான்கு'
        assert N.ta_cardinal(98) == 'தொண்ணூற்று எட்டு'      # no fusion after 90, as the LLM writes it
        assert N.ta_ordinal(3) == 'மூன்றாவது'
        assert N.ta_ordinal(21, 'ஆம்') == 'இருபத்தொன்றாம்'
        assert N.ta_ordinal(1) == 'முதலாவது'

    def test_digits_and_decimals(self):
        assert N.digits('0457', 'en') == 'zero four five seven'
        assert N.digits('03-12', 'ms') == 'kosong tiga satu dua'
        assert N.digits('960', 'zh') == '九六零'
        assert N.decimal('3', '25', 'ms') == 'tiga perpuluhan dua lima'
        assert N.decimal('37', '5', 'en') == 'thirty-seven point five'
        assert N.decimal('2', '5', 'zh') == '二点五'
        assert N.decimal('4', '8', 'ta') == 'நான்கு புள்ளி எட்டு'


# --------------------------------------------------------------------------- language
class TestLanguage:
    @pytest.mark.parametrize('text,lang', [
        ('Hello there, how can I help you today?', 'en'),
        ('Selamat pagi, apa yang boleh saya bantu encik hari ini?', 'ms'),
        ('您的余额是RM1,250.50。', 'zh'),
        ('உங்கள் இருப்பு RM1,250.50.', 'ta'),
        ('Nanti saya call balik at 3pm ya, nombor encik 012-3456789 kan?', 'ms'),
        ('Okay encik, your booking is confirmed, total RM120 for 2 nights.', 'en'),
        ('Jalan Tun Razak 12号', 'zh'),
    ])
    def test_detect(self, text, lang):
        assert detect_lang(text) == lang


# --------------------------------------------------------------------------- categories, per language
LLM_AGREED = [
    # English
    ('Your balance is RM1,250.50 as of today.', 'Your balance is one thousand two hundred fifty ringgit fifty sen as of today.'),
    ('It costs RM12.05 including tax.', 'It costs twelve ringgit five sen including tax.'),
    ('The house is listed at RM1.2 million.', 'The house is listed at one point two million ringgit.'),
    ('Minimum top-up is USD 10.', 'Minimum top-up is ten US dollars.'),
    ('That will be $25 or about RM110.', 'That will be twenty-five dollars or about one hundred ten ringgit.'),
    ('Interest is charged at 3.25 per annum.', 'Interest is charged at three point two five per annum.'),
    ('Battery is at 100% and the discount is 12.5%.', 'Battery is at one hundred percent and the discount is twelve point five percent.'),
    ('Reach us on +60 12-345 6789 or 1-300-88-1234.',
     'Reach us on plus six zero one two three four five six seven eight nine or one three zero zero eight eight one two three four.'),
    ('My IC is 960314875079.', 'My IC is nine six zero three one four eight seven five zero seven nine.'),
    ('The meeting is on 15/3/2024 at the main office.', 'The meeting is on the fifteenth of March twenty twenty-four at the main office.'),
    ('Your delivery is scheduled for 2024-03-15.', 'Your delivery is scheduled for March fifteenth twenty twenty-four.'),
    ('She was born on March 15, 1998.', 'She was born on March fifteenth, nineteen ninety-eight.'),
    ('The meeting starts at 10:45 AM sharp.', 'The meeting starts at ten forty-five A M sharp.'),
    ('The train departs at 14:05.', 'The train departs at fourteen zero five.'),
    ('The office opens at 8:00 and closes at 17:00.', "The office opens at eight o'clock and closes at seventeen o'clock."),
    ('Please arrive by 9.30am.', 'Please arrive by nine thirty a m.'),
    ("You're 3rd in the queue.", "You're third in the queue."),
    ('Happy 21st birthday!', 'Happy twenty-first birthday!'),
    ('We are open 9am-5pm from Monday to Friday.', 'We are open nine a m to five p m from Monday to Friday.'),
    ('Delivery takes 3-5 working days.', 'Delivery takes three to five working days.'),
    ('The bag weighs 5kg and the box is 30 cm wide.', 'The bag weighs five kilograms and the box is thirty centimeters wide.'),
    ('It is 32°C outside today.', 'It is thirty-two degrees Celsius outside today.'),
    ('The plan includes 50GB of data.', 'The plan includes fifty gigabytes of data.'),
    ('Send the form to support.team@scicom.com.my today.', 'Send the form to support dot team at scicom dot com dot my today.'),
    ('Go to https://example.com/help for the guide.', 'Go to h t t p s colon slash slash example dot com slash help for the guide.'),
    ('Your report id is X-12340567.', 'Your report id is X one two three four zero five six seven.'),
    ('Order #4471 has been shipped.', 'Order number four four seven one has been shipped.'),
    ('Flight MH370 was rescheduled, and your seat is 24A.', 'Flight MH three seven zero was rescheduled, and your seat is twenty-four A.'),
    ('Room 305 is on the third floor, next to room 310.', 'Room three zero five is on the third floor, next to room three one zero.'),
    ('The code is 1000 and the backup is 2000.', 'The code is one thousand and the backup is two thousand.'),
    ('The company was founded in 1998 and expanded in 2024.', 'The company was founded in nineteen ninety-eight and expanded in twenty twenty-four.'),
    ('Dr. Lim will call you back.', 'Doctor Lim will call you back.'),
    ('Prof. Ahmad vs. the committee, approx. 20 people.', 'Professor Ahmad versus the committee, approximately twenty people.'),
    ('Order 2 units at RM199 each, delivered in 3-5 days to No. 12, Jalan Tun Razak.',
     'Order two units at one hundred ninety-nine ringgit each, delivered in three to five days to Number twelve, Jalan Tun Razak.'),
    # Malay
    ('Baki anda RM1,250.50 setakat hari ini.', 'Baki anda seribu dua ratus lima puluh ringgit lima puluh sen setakat hari ini.'),
    ('Rumah itu berharga RM1.2 juta.', 'Rumah itu berharga satu perpuluhan dua juta ringgit.'),
    ('Anda jimat RM0.50 untuk pesanan ini.', 'Anda jimat lima puluh sen untuk pesanan ini.'),
    ('Kadar faedah 3.25 setahun.', 'Kadar faedah tiga perpuluhan dua lima setahun.'),
    ('Sila hubungi 03-12345678 untuk bantuan.', 'Sila hubungi kosong tiga satu dua tiga empat lima enam tujuh lapan untuk bantuan.'),
    ('Talian kami +60 3-1234 5678 atau 1-300-88-1234.',
     'Talian kami tambah enam kosong tiga satu dua tiga empat lima enam tujuh lapan atau satu tiga kosong kosong lapan lapan satu dua tiga empat.'),
    ('Mesyuarat pada 15/3/2024 di pejabat utama.', 'Mesyuarat pada lima belas Mac dua ribu dua puluh empat di pejabat utama.'),
    ('Tarikh akhir ialah 5 Jan 2026.', 'Tarikh akhir ialah lima Januari dua ribu dua puluh enam.'),
    ('Mesyuarat bermula pukul 10:45 pagi.', 'Mesyuarat bermula pukul sepuluh empat puluh lima pagi.'),
    ('Sila tiba sebelum 9.30am.', 'Sila tiba sebelum sembilan tiga puluh pagi.'),
    ('Pejabat dibuka jam 8:00 dan ditutup jam 17:00.', 'Pejabat dibuka jam lapan dan ditutup jam tujuh belas.'),
    ('Anda yang ke-3 dalam barisan.', 'Anda yang ketiga dalam barisan.'),
    ('Kami dibuka 9am-5pm dari Isnin hingga Jumaat.', 'Kami dibuka sembilan pagi hingga lima petang dari Isnin hingga Jumaat.'),
    ('Bilik 305 di tingkat 3, sebelah bilik 310.', 'Bilik tiga kosong lima di tingkat tiga, sebelah bilik tiga satu kosong.'),
    ('Tiada tiket tertunggak, jumlahnya 0.', 'Tiada tiket tertunggak, jumlahnya sifar.'),
    ('Pelan ini termasuk 50GB data.', 'Pelan ini termasuk lima puluh gigabait data.'),
    ('Bawa dokumen anda, cth. invois, dsb.', 'Bawa dokumen anda, contohnya invois, dan sebagainya.'),
    ('Pesan 2 unit pada RM199 setiap satu, dihantar dalam 3-5 hari ke No. 12, Jalan Tun Razak.',
     'Pesan dua unit pada seratus sembilan puluh sembilan ringgit setiap satu, dihantar dalam tiga hingga lima hari ke Nombor dua belas, Jalan Tun Razak.'),
    # Mandarin
    ('您的余额是RM1,250.50。', '您的余额是一千二百五十令吉五十仙。'),
    ('价格是RM12.05，包括税。', '价格是十二令吉五仙，包括税。'),
    ('这间房子售价RM1.2百万。', '这间房子售价一百二十万令吉。'),
    ('这个套餐是$25，大约RM110。', '这个套餐是二十五美元，大约一百一十令吉。'),
    ('电池是100%，折扣是12.5%。', '电池是百分之一百，折扣是百分之十二点五。'),
    ('请拨打03-12345678。', '请拨打零三一二三四五六七八。'),
    ('会议在2024年3月15日举行。', '会议在二零二四年三月十五日举行。'),
    ('送货日期是15/3/2024。', '送货日期是二零二四年三月十五日。'),
    ('您的预约在明天下午2点30分。', '您的预约在明天下午两点三十分。'),
    ('火车在14:05出发。', '火车在十四点零五分出发。'),
    ('会议在上午10:45开始。', '会议在上午十点四十五分开始。'),
    ('这批货有2箱，每箱200个。', '这批货有两箱，每箱二百个。'),
    ('您的房间是305号，在3楼。', '您的房间是三百零五号，在三楼。'),
    ('您是排队的第3位。', '您是排队的第三位。'),
    ('送货需要3-5个工作日。', '送货需要三到五个工作日。'),
    ('商店离这里2.5 km。', '商店离这里二点五公里。'),
    ('订单#4471已经发出。', '订单号四四七一已经发出。'),
    ('公司成立于1998年，并在2024年扩展。', '公司成立于一九九八年，并在二零二四年扩展。'),
    # Tamil
    ('மாதக் கட்டணம் RM50.', 'மாதக் கட்டணம் ஐம்பது ரிங்கிட்.'),
    ('விலை RM12.05, வரி உட்பட.', 'விலை பன்னிரண்டு ரிங்கிட் ஐந்து சென், வரி உட்பட.'),
    ('விற்பனை 25% உயர்ந்தது.', 'விற்பனை இருபத்தைந்து சதவீதம் உயர்ந்தது.'),
    ('என் தொலைபேசி எண் 012-345 6789.', 'என் தொலைபேசி எண் பூஜ்ஜியம் ஒன்று இரண்டு மூன்று நான்கு ஐந்து ஆறு ஏழு எட்டு ஒன்பது.'),
    ('கடந்த மாதம் 1,500 வாடிக்கையாளர்கள் இணைந்தனர்.', 'கடந்த மாதம் ஆயிரத்து ஐந்நூறு வாடிக்கையாளர்கள் இணைந்தனர்.'),
    ('கூட்டம் காலை 10:45 மணிக்கு தொடங்கும்.', 'கூட்டம் காலை பத்து நாற்பத்தைந்து மணிக்கு தொடங்கும்.'),
    ('நாங்கள் மாலை 5pm மணிக்கு மூடுகிறோம்.', 'நாங்கள் மாலை ஐந்து பி எம் மணிக்கு மூடுகிறோம்.'),
    ('நீங்கள் வரிசையில் 3வது.', 'நீங்கள் வரிசையில் மூன்றாவது.'),
    ('21ஆம் பிறந்தநாள் வாழ்த்துக்கள்!', 'இருபத்தொன்றாம் பிறந்தநாள் வாழ்த்துக்கள்!'),
    ('டெலிவரிக்கு 3-5 வேலை நாட்கள் ஆகும்.', 'டெலிவரிக்கு மூன்று முதல் ஐந்து வேலை நாட்கள் ஆகும்.'),
    ('இன்று வெளியே 32°C.', 'இன்று வெளியே முப்பத்திரண்டு டிகிரி செல்சியஸ்.'),
    ('நிறுவனம் 1998 இல் நிறுவப்பட்டு 2024 இல் விரிவடைந்தது.',
     'நிறுவனம் ஆயிரத்து தொள்ளாயிரத்து தொண்ணூற்று எட்டு இல் நிறுவப்பட்டு இரண்டாயிரத்து இருபத்து நான்கு இல் விரிவடைந்தது.'),
    ('இந்த வீட்டின் விலை RM1.2 மில்லியன்.', 'இந்த வீட்டின் விலை ஒரு புள்ளி இரண்டு மில்லியன் ரிங்கிட்.'),
]


@pytest.mark.parametrize('text,expected', LLM_AGREED, ids=[t[:28] for t, _ in LLM_AGREED])
def test_matches_llm(text, expected):
    assert canon(normalize(text)) == canon(expected)


class TestCodeSwitching:
    """Malay/English mixed sentences: each number picks its language from its neighbours,
    with the sentence language as prior (lang.local_lang). Expected strings are the LLM's."""

    @pytest.mark.parametrize('text,expected', [
        ('Nombor akaun anda ialah 1234, and your balance is RM50 today.',
         'Nombor akaun anda ialah satu dua tiga empat, and your balance is fifty ringgit today.'),
        ('Please pay RM120 before 5pm, kalau tidak akaun anda akan digantung selama 3 hari.',
         'Please pay one hundred twenty ringgit before five p m, kalau tidak akaun anda akan digantung selama tiga hari.'),
        ('Total 3 items, harga RM45.90 semuanya, delivered in 2 days.',
         'Total three items, harga empat puluh lima ringgit sembilan puluh sen semuanya, delivered in two days.'),
        ('Meeting pukul 10 pagi esok at level 12, jangan lupa bring 2 copies.',
         'Meeting pukul sepuluh pagi esok at level dua belas, jangan lupa bring dua copies.'),
        ('Your appointment on 5 Jun 2025 at 2.30pm, sila datang 15 minit awal.',
         'Your appointment on the fifth of June twenty twenty-five at two thirty p m, sila datang lima belas minit awal.'),
        ('Sila tunggu, I will transfer you to extension 305 in 2 minutes.',
         'Sila tunggu, I will transfer you to extension three zero five in two minutes.'),
        ('Okay encik, your booking is confirmed, total RM120 for 2 nights.',
         'Okay encik, your booking is confirmed, total one hundred twenty ringgit for two nights.'),
        ('Encik boleh dapat 25% discount kalau bayar before 30 June.',
         'Encik boleh dapat dua puluh lima peratus discount kalau bayar before tiga puluh June.'),
    ])
    def test_matches_llm(self, text, expected):
        assert canon(normalize(text)) == canon(expected)

    def test_local_vote(self):
        from app.spoken_normalizer.lang import local_lang, is_code_switched
        t = 'Nombor akaun anda ialah 1234, and your balance is RM50 today.'
        assert is_code_switched(t)
        i = t.index('1234')
        assert local_lang(t, i, i + 4, 'en') == 'ms'          # neighbours beat the sentence prior
        j = t.index('50')
        assert local_lang(t, j, j + 2, 'ms') == 'en'

    def test_monolingual_sentences_are_not_split(self):
        from app.spoken_normalizer.lang import is_code_switched
        assert not is_code_switched('Your balance is RM1,250.50 as of today.')
        assert not is_code_switched('Baki anda RM1,250.50 setakat hari ini.')

    def test_generated_words_do_not_vote(self):
        # "one hundred twenty ringgit" from the first number must not pull the second one
        out = normalize('Bayar RM120 sekarang, then 3 more instalments follow.')
        assert out.endswith('then three more instalments follow.')


class TestDeliberateDifferences:
    """Where the rules and the LLM part ways on purpose."""

    def test_tamil_range_is_read(self):
        # the LLM left this one untouched
        assert normalize('10 முதல் 15 நிமிடங்கள் காத்திருக்கவும்.') == 'பத்து முதல் பதினைந்து நிமிடங்கள் காத்திருக்கவும்.'

    def test_tamil_sentence_stays_tamil(self):
        # the LLM answered this one in English
        out = normalize('5/6/2025 க்கு முன் RM89.90 செலுத்துங்கள், இல்லையெனில் 10% தாமதக் கட்டணம்.')
        assert out == 'ஐந்து ஜூன் இரண்டாயிரத்து இருபத்தைந்து க்கு முன் எண்பத்தொன்பது ரிங்கிட் தொண்ணூறு சென் செலுத்துங்கள், இல்லையெனில் பத்து சதவீதம் தாமதக் கட்டணம்.'

    def test_iso_date_in_malay_is_day_first(self):
        # the LLM read 2024-03-15 in written order ("dua ribu ... Mac lima belas")
        assert normalize('Penghantaran dijadualkan pada 2024-03-15.') == 'Penghantaran dijadualkan pada lima belas Mac dua ribu dua puluh empat.'

    def test_celsius_in_mandarin(self):
        # the LLM produced 三十二度C
        assert normalize('今天外面32°C。') == '今天外面摄氏三十二度。'

    def test_ic_digits_are_exact(self):
        # the LLM inserted an extra zero here
        assert normalize('Please confirm your IC number 960314-08-5079.') == \
            'Please confirm your IC number nine six zero three one four zero eight five zero seven nine.'


class TestEdges:
    def test_sentence_final_period_does_not_break_a_match(self):
        assert normalize('The total comes to RM 3,400.') == 'The total comes to three thousand four hundred ringgit.'
        assert normalize('Please pay before 30-06-2025.') == 'Please pay before the thirtieth of June twenty twenty-five.'

    def test_trailing_comma_is_not_part_of_the_number(self):
        assert normalize('உங்கள் அறை 305, 3ஆம் மாடியில் உள்ளது.') == 'உங்கள் அறை மூன்று பூஜ்ஜியம் ஐந்து, மூன்றாம் மாடியில் உள்ளது.'

    def test_year_range(self):
        assert normalize('from 2020-2024 we grew') == 'from twenty twenty to twenty twenty-four we grew'

    def test_pin_and_leading_zero(self):
        assert normalize('The reference is ABC1234 and the PIN is 0457.') == \
            'The reference is ABC one two three four and the PIN is zero four five seven.'

    def test_lang_override(self):
        assert normalize('3 items', lang='ms') == 'tiga items'

    def test_empty(self):
        assert normalize('') == ''


# --------------------------------------------------------------------------- corpus-wide properties
@pytest.mark.parametrize('cid,lang,cat,text', CORPUS, ids=[c[0] for c in CORPUS])
def test_every_digit_is_read(cid, lang, cat, text):
    out = normalize(text)
    assert not re.search(r'\d', out), out
    assert not re.search(r'[%@]', out), out


@pytest.mark.parametrize('cid,lang,cat,text', [c for c in CORPUS if c[2] == 'plain'], ids=[c[0] for c in CORPUS if c[2] == 'plain'])
def test_plain_text_is_untouched(cid, lang, cat, text):
    assert normalize(text) == text


@pytest.mark.parametrize('cid,lang,cat,text', CORPUS, ids=[c[0] for c in CORPUS])
def test_idempotent(cid, lang, cat, text):
    out = normalize(text)
    assert normalize(out) == out


@pytest.mark.parametrize('cid,lang,cat,text', [c for c in CORPUS if c[1] != 'cs'], ids=[c[0] for c in CORPUS if c[1] != 'cs'])
def test_language_detection_on_corpus(cid, lang, cat, text):
    assert detect_lang(text) == lang
