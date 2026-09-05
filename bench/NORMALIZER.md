# Spoken normalizer: rule-based replica of the LLM normalizer

`app/spoken_normalizer/` — `mode: "spoken"` on `/v1/audio/normalize` and TTS requests. Rewrites
everything that is not speakable as written into words, in the language of the surrounding text,
for **English, Malay, Mandarin and Tamil**, exactly as the LLM normalizer (`app/prompt.py`) is asked
to. Pure Python, no model, no network, **~45 µs** per sentence (1.2 ms through the API, against
**491 ms** for the LLM call, measured on tm-h20 on 2026-09-04).

## Why

`mode=llm` costs one round trip to gemma-4-31b through the serverless proxy, ~0.5 s, and on a TTS
request all of it sits in front of the first audio byte: TTFB on a sentence with a number went from
680–718 ms (`llm`) to 103 ms (`spoken`) on the same instance. The LLM's output is also
non-deterministic in style (see *Where the LLM contradicts itself*). Rules are instant and
repeatable, and the LLM stays available where rules are not enough.

## Method

The LLM's own behaviour is the specification, so it was captured first and the rules were written
to reproduce it:

1. `bench/normalizer_corpus.py` — 497 inputs: 152 English, 114 Malay, 107 Mandarin, 102 Tamil, 22
   Malay/English code-switch; every category the prompt names plus abbreviations, acronyms,
   ids, years and plain sentences that must come back untouched. The first 300 are the everyday
   call-centre shapes; the 197 added on 2026-09-04 are the *harder cases* below (hyphenated
   compounds, fractions and slashes, versions, decades, negatives, per-units, ranges with a shared
   suffix, scores, verification codes and postcodes, dotted and military times, Tamil case
   suffixes glued to digits).
2. `bench/normalizer_truth.py` — runs them through the live LLM (temperature 0) and stores the
   pairs in `bench/results/normalizer_truth.jsonl`. Incremental: only new ids are queried.
3. `bench/normalizer_agreement.py` — scores `normalize()` against those pairs: two outputs
   *agree* when identical after lower-casing and collapsing hyphens/whitespace/punctuation
   (none of which is audible; the app's own post-processing strips hyphens anyway). Also reports
   CER against the LLM text, per language and per category, and prints every disagreement.
4. `bench/normalizer_api_agreement.py --url` — the same comparison through a running app, so
   the pre/post cleanup (markdown, replace mappings, trailing period) is included.
5. `tests/test_spoken_normalizer.py` — 1,726 tests: number words in all four languages, one
   LLM-agreed example per category and language, the deliberate differences, and three
   corpus-wide properties: **no digit survives in any output**, plain text is byte-identical,
   and the output is a fixed point (`normalize(normalize(x)) == normalize(x)`).

To extend: add sentences to the corpus **first** (stable ids), regenerate the truth, then add
or change rules until `normalizer_agreement.py` is happy.

## Agreement with the LLM

| Language | All 497 (offline) | Original 300 | 197 harder cases | Through the API, original 300 (instance C, box 1024) |
|---|---|---|---|---|
| English | 88.8% (135/152) | 90.7% | 86.4% (57/66) | 90.7% |
| Malay | 86.8% (99/114) | 90.5% | 80.0% (32/40) | 89.2% |
| Mandarin | 87.9% (94/107) | 90.3% | 84.4% (38/45) | 90.3% |
| Tamil | 66.7% (68/102) | 71.7% | 59.5% (25/42) | 73.3% |
| Malay/English code-switch | 59.1% (13/22) | 66.7% | 25.0% (1/4) | 61.1% |
| **All** | **82.3% (409/497)** | **85.3%** | **77.7% (153/197)** | **85.0%** |

Mean CER of the rule output against the LLM output is 0.033 overall: even the disagreements are
usually one word. The 88 remaining disagreements split into **28 outright LLM errors** (it returned
8 Tamil sentences with the digits still in them, misread 10.45 as முப்பத்தைந்து, wrote "three point
thirty", "dua pukul petang", "satu ratus", 九十五分之一百, and left "2 hours 30 minutes" as digits),
**50 cases of the LLM contradicting its own style elsewhere in the corpus** (spelled acronyms,
"and" in hundreds, 两百 vs 二百, computing RM12.5k in English but not in Malay, colloquial Tamil
numerals) and **10 code-switch hybrids** ("tiga tiga puluh p m"). No disagreement leaves a digit unread.

By category (offline): decade, email, neg, percent, plain, score, year 100%; abbr 93%, int 92%,
phone 92%, decimal 91%, money 89%, unit 87%, hyphen 86%, ordinal 83%, range 82%, acro 82%, version 80%,
time 79%, code 78%, id 78%, date 76%, frac 67%, ic 67%, mixed 63%, per 62%, suffix 56%, url 33%. The low
ones are explained below.

## Coverage matrix

One example per category and language, input → rule output. "same" = identical to the LLM after
canonicalization; where it is not, the LLM's text is shown.

### English

| Category | Input | Output | vs LLM |
|---|---|---|---|
| int | The venue can seat 250 people comfortably. | The venue can seat two hundred fifty people comfortably. | LLM said "two hundred **and** fifty" here but never elsewhere |
| money | Your balance is RM1,250.50 as of today. | Your balance is one thousand two hundred fifty ringgit fifty sen as of today. | same |
| decimal | Interest is charged at 3.25 per annum. | Interest is charged at three point two five per annum. | same |
| percent | Battery is at 100% and the discount is 12.5%. | Battery is at one hundred percent and the discount is twelve point five percent. | same |
| phone | Reach us on +60 12-345 6789 or 1-300-88-1234. | Reach us on plus six zero one two three four five six seven eight nine or one three zero zero eight eight one two three four. | same |
| IC | My IC is 960314875079. | My IC is nine six zero three one four eight seven five zero seven nine. | same |
| date | The meeting is on 15/3/2024 at the main office. | The meeting is on the fifteenth of March twenty twenty-four at the main office. | same |
| time | The meeting starts at 10:45 AM sharp. | The meeting starts at ten forty-five A M sharp. | same |
| ordinal | Happy 21st birthday! | Happy twenty-first birthday! | same |
| range | We are open 9am-5pm from Monday to Friday. | We are open nine a m to five p m from Monday to Friday. | same |
| unit | It is 32°C outside today. | It is thirty-two degrees Celsius outside today. | same |
| email | Send the form to support.team@scicom.com.my today. | Send the form to support dot team at scicom dot com dot my today. | same |
| url | Go to https://example.com/help for the guide. | Go to h t t p s colon slash slash example dot com slash help for the guide. | same |
| id | Flight MH370 was rescheduled, and your seat is 24A. | Flight MH three seven zero was rescheduled, and your seat is twenty-four A. | same |
| year | The company was founded in 1998 and expanded in 2024. | The company was founded in nineteen ninety-eight and expanded in twenty twenty-four. | same |
| titles / abbreviations | Prof. Ahmad vs. the committee, approx. 20 people. | Professor Ahmad versus the committee, approximately twenty people. | same |
| acronyms | Please enter the OTP sent by SMS. | (unchanged) | same |
| plain | Hello there, how can I help you today? | (unchanged) | same |

### Malay

| Category | Input | Output | vs LLM |
|---|---|---|---|
| int | Bilik 305 di tingkat 3, sebelah bilik 310. | Bilik tiga kosong lima di tingkat tiga, sebelah bilik tiga satu kosong. | same |
| money | Baki anda RM1,250.50 setakat hari ini. | Baki anda seribu dua ratus lima puluh ringgit lima puluh sen setakat hari ini. | same |
| decimal | Kadar faedah 3.25 setahun. | Kadar faedah tiga perpuluhan dua lima setahun. | same |
| percent | Bateri pada 100% dan diskaun 12.5%. | Bateri pada seratus peratus dan diskaun dua belas perpuluhan lima peratus. | same |
| phone | Talian kami +60 3-1234 5678 atau 1-300-88-1234. | Talian kami tambah enam kosong tiga satu dua tiga empat lima enam tujuh lapan atau satu tiga kosong kosong lapan lapan satu dua tiga empat. | same |
| IC | Nombor IC saya 960314875079. | Nombor IC saya sembilan enam kosong tiga satu empat lapan tujuh lima kosong tujuh sembilan. | same |
| date | Mesyuarat pada 15/3/2024 di pejabat utama. | Mesyuarat pada lima belas Mac dua ribu dua puluh empat di pejabat utama. | same |
| time | Mesyuarat bermula pukul 10:45 pagi. | Mesyuarat bermula pukul sepuluh empat puluh lima pagi. | same |
| ordinal | Anda yang ke-3 dalam barisan. | Anda yang ketiga dalam barisan. | same |
| range | Kami dibuka 9am-5pm dari Isnin hingga Jumaat. | Kami dibuka sembilan pagi hingga lima petang dari Isnin hingga Jumaat. | same |
| unit | Suhu di luar 32°C hari ini. | Suhu di luar tiga puluh dua darjah Celsius hari ini. | same |
| email | Emel saya di husein@site.com untuk maklumat lanjut. | Emel saya di husein at site dot com untuk maklumat lanjut. | same |
| url | Pergi ke https://example.com/bantuan untuk panduan. | Pergi ke h t t p s colon slash slash example dot com slash bantuan untuk panduan. | same |
| id | Penerbangan MH370 dijadualkan semula, tempat duduk anda 24A. | Penerbangan MH tiga tujuh kosong dijadualkan semula, tempat duduk anda dua puluh empat A. | same |
| year | Syarikat ditubuhkan pada 1998 dan berkembang pada 2024. | Syarikat ditubuhkan pada seribu sembilan ratus sembilan puluh lapan dan berkembang pada dua ribu dua puluh empat. | same |
| titles / abbreviations | Bawa dokumen anda, cth. invois, dsb. | Bawa dokumen anda, contohnya invois, dan sebagainya. | same |
| acronyms | Bawa IC anda ke kaunter TNB di KL. | (unchanged) | same |
| plain | Selamat pagi, apa yang boleh saya bantu encik hari ini? | (unchanged) | same |

### Mandarin

| Category | Input | Output | vs LLM |
|---|---|---|---|
| int | 这批货有2箱，每箱200个。 | 这批货有两箱，每箱二百个。 | same (两 before a measure word, 二 elsewhere) |
| money | 您的余额是RM1,250.50。 | 您的余额是一千二百五十令吉五十仙。 | same |
| decimal | 您的体温是37.5度。 | 您的体温是三十七点五度。 | same |
| percent | 电池是100%，折扣是12.5%。 | 电池是百分之一百，折扣是百分之十二点五。 | same |
| phone | 请拨打03-12345678。 | 请拨打零三一二三四五六七八。 | same |
| IC | 我的身份证号是960314875079。 | 我的身份证号是九六零三一四八七五零七九。 | same |
| date | 送货日期是15/3/2024。 | 送货日期是二零二四年三月十五日。 | same |
| time | 您的预约在明天下午2点30分。 | 您的预约在明天下午两点三十分。 | same |
| ordinal | 您是排队的第3位。 | 您是排队的第三位。 | same |
| range | 送货需要3-5个工作日。 | 送货需要三到五个工作日。 | same |
| unit | 商店离这里2.5 km。 | 商店离这里二点五公里。 | same |
| email | 请发邮件到husein@site.com。 | 请发邮件到 husein at site dot com。 | same |
| url | 请浏览www.scicom.com.my了解更多。 | 请浏览 w w w dot scicom dot com dot my 了解更多。 | LLM spelled the TLD "m y" |
| id | 订单#4471已经发出。 | 订单号四四七一已经发出。 | same |
| year | 公司成立于1998年，并在2024年扩展。 | 公司成立于一九九八年，并在二零二四年扩展。 | same |
| titles / abbreviations | Dr. Lim会给您回电。 | (unchanged) | same |
| acronyms | 请带您的IC到TNB柜台。 | (unchanged) | same |
| plain | 你好，请问有什么可以帮您？ | (unchanged) | same |

### Tamil

| Category | Input | Output | vs LLM |
|---|---|---|---|
| int | கடந்த மாதம் 1,500 வாடிக்கையாளர்கள் இணைந்தனர். | கடந்த மாதம் ஆயிரத்து ஐந்நூறு வாடிக்கையாளர்கள் இணைந்தனர். | same |
| money | விலை RM12.05, வரி உட்பட. | விலை பன்னிரண்டு ரிங்கிட் ஐந்து சென், வரி உட்பட. | same |
| decimal | சராசரி மதிப்பீடு 5க்கு 4.8. | சராசரி மதிப்பீடு ஐந்துக்கு நான்கு புள்ளி எட்டு. | same |
| percent | விற்பனை 25% உயர்ந்தது. | விற்பனை இருபத்தைந்து சதவீதம் உயர்ந்தது. | same |
| phone | என் தொலைபேசி எண் 012-345 6789. | என் தொலைபேசி எண் பூஜ்ஜியம் ஒன்று இரண்டு மூன்று நான்கு ஐந்து ஆறு ஏழு எட்டு ஒன்பது. | same |
| IC | என் அடையாள அட்டை எண் 960314875079. | என் அடையாள அட்டை எண் ஒன்பது ஆறு பூஜ்ஜியம் மூன்று ஒன்று நான்கு எட்டு ஏழு ஐந்து பூஜ்ஜியம் ஏழு ஒன்பது. | same |
| date | கூட்டம் 15/3/2024 அன்று நடைபெறும். | கூட்டம் பதினைந்து மார்ச் இரண்டாயிரத்து இருபத்து நான்கு அன்று நடைபெறும். | LLM wrote the colloquial "இருபத்தி நான்கு" here and "இருபத்து நான்கு" elsewhere |
| time | நாங்கள் மாலை 5pm மணிக்கு மூடுகிறோம். | நாங்கள் மாலை ஐந்து பி எம் மணிக்கு மூடுகிறோம். | same |
| ordinal | 21ஆம் பிறந்தநாள் வாழ்த்துக்கள்! | இருபத்தொன்றாம் பிறந்தநாள் வாழ்த்துக்கள்! | same |
| range | டெலிவரிக்கு 3-5 வேலை நாட்கள் ஆகும். | டெலிவரிக்கு மூன்று முதல் ஐந்து வேலை நாட்கள் ஆகும். | same |
| unit | இன்று வெளியே 32°C. | இன்று வெளியே முப்பத்திரண்டு டிகிரி செல்சியஸ். | same |
| email | விவரங்களுக்கு husein@site.com க்கு மின்னஞ்சல் அனுப்புங்கள். | விவரங்களுக்கு husein at site dot com க்கு மின்னஞ்சல் அனுப்புங்கள். | same |
| url | மேலும் அறிய www.scicom.com.my ஐ பார்வையிடுங்கள். | மேலும் அறிய w w w dot scicom dot com dot my ஐ பார்வையிடுங்கள். | LLM kept "www" as a word |
| id | உங்கள் அறிக்கை எண் X-12340567. | உங்கள் அறிக்கை எண் X ஒன்று இரண்டு மூன்று நான்கு பூஜ்ஜியம் ஐந்து ஆறு ஏழு. | same |
| year | 2030 க்குள் இரட்டிப்பாகும் என எதிர்பார்க்கிறோம். | இரண்டாயிரத்து முப்பது க்குள் இரட்டிப்பாகும் என எதிர்பார்க்கிறோம். | same |
| titles / abbreviations | Dr. Lim உங்களை மீண்டும் அழைப்பார். | (unchanged) | same |
| acronyms | SMS மூலம் அனுப்பப்பட்ட OTP ஐ உள்ளிடுங்கள். | (unchanged) | same |
| plain | வணக்கம், நான் உங்களுக்கு எப்படி உதவ முடியும்? | (unchanged) | same |

### Titles, honorifics and abbreviations

Only what the LLM expands is expanded; everything else is left for the TTS LM, which reads it
fine and which the LLM also leaves alone.

| Written | English | Malay | Mandarin / Tamil |
|---|---|---|---|
| Dr. | Doctor | Doktor | unchanged (as the LLM) |
| Prof. | Professor | unchanged | unchanged |
| vs. / approx. | versus / approximately | — | — |
| No. (before a number) | Number | Nombor | unchanged |
| Jln / Tmn / Kg | — | Jalan / Taman / Kampung | — |
| cth. / dsb. / dll. | — | contohnya / dan sebagainya / dan lain-lain | — |
| Mr. Mrs. Ms. En. Pn. Cik Encik Puan Tan Sri Dato' Datuk Sdn Bhd e.g. etc. | unchanged | unchanged | unchanged |
| Acronyms (IC, OTP, SMS, TNB, KL, NASA, API) | unchanged | unchanged | unchanged |

### Mixed sentences

Several categories in one sentence work because each handler claims its own digits in order
(email, URL, IC, numeric date, month-name date, phone, time range, time, money, percent, unit,
ordinal, range, `#N`, alphanumeric id, year, plain number) and nothing is left for the generic
number rule to misread:

- *Your total is RM1,250.50 and the meeting is at 3pm on 12/9/2026.* → Your total is one thousand two hundred fifty ringgit fifty sen and the meeting is at three p m on the twelfth of September twenty twenty-six.
- *Order 2 units at RM199 each, delivered in 3-5 days to No. 12, Jalan Tun Razak.* → Order two units at one hundred ninety-nine ringgit each, delivered in three to five days to Number twelve, Jalan Tun Razak.
- *您的总额是RM1,250.50，会议在12/9/2026下午3点。* → 您的总额是一千二百五十令吉五十仙，会议在二零二六年九月十二日下午三点。

## Harder cases (2026-09-04 extension)

Shapes the first corpus did not contain, each broken on the old rules (the probe read "30-day" as
"three zero day", "1/2" as "one/two", "17.4.1" left a digit, "pukul 3.30 petang" became "tiga
perpuluhan tiga kosong petang", "2024ல்" became "நான்குல்"). Ground truth was generated first, then the
rules were written to it; where the four languages' LLM outputs disagree the note says which was kept.

| Shape | English | Malay | Mandarin | Tamil |
|---|---|---|---|---|
| hyphenated compound `30-day`, `3-in-1` | thirty-day, three-in-one | tiga puluh hari (written spaced) | 三十天 | முப்பது நாள் |
| fraction `1/2`, `3/4`, `1/3`, `95/100` | one half, three quarters, one third, ninety-five out of one hundred | satu per dua, tiga per empat, satu pertiga | 二分之一, 四分之三, 一百分之九十五 | அரை, முக்கால், மூன்றில் ஒன்று, நூற்றில் தொண்ணூற்று ஐந்து |
| `24/7`, `50/50` | twenty-four seven, fifty fifty | dua puluh empat jam tujuh hari seminggu | 二十四小时七天 | இருபத்து நான்கு மணி நேரம் ஏழு நாட்கள் |
| slash in an address `Jalan 3/14` | Jalan three slash fourteen | Jalan tiga per empat belas | Jalan 三斜杠十四 (LLM said "slash") | ஸ்லாஷ் |
| day/month without a year `31/12`, `1/1` | the thirty-first of December, the first of January | tiga puluh satu Disember | 十二月三十一日 | முப்பத்தொன்று டிசம்பர் |
| version `17.4.1`, `3.12.4` | seventeen point four point one, three point one two point four | tujuh belas perpuluhan empat perpuluhan satu | 十七点四点一 | பதினேழு புள்ளி நான்கு புள்ளி ஒன்று |
| decade `1980s`, `90s`, `1980-an` | nineteen eighties, nineties, two thousands, twenty tens | seribu sembilan ratus lapan puluh-an | 九十年代 (already a particle) | — |
| negative `-5°C`, `-RM250`, `-3.5%` | minus five degrees Celsius | negatif lima darjah Celsius | 负五摄氏度, 负百分之三点五 | மைனஸ் ஐந்து டிகிரி செல்சியஸ் |
| per-unit `RM5/kg`, `110 km/h`, `8 tablets/day`, `RM38/mth` | five ringgit per kilogram, kilometers per hour, tablets per day, per month | lima ringgit per kilogram, kilometer per jam, per bulan (LLM also said "sebulan") | 五令吉每公斤, 公里每小时 | ஐந்து ரிங்கிட் ஒரு கிலோகிராமுக்கு (LLM: ஒரு கிலோகிராம், no dative) |
| range with a shared suffix/prefix `10-15%`, `RM50-RM100`, `0-100 km/h` | ten to fifteen percent, fifty ringgit to one hundred ringgit, zero to one hundred kilometers per hour | sepuluh hingga lima belas peratus | 百分之十到百分之十五 (repeated, as the LLM), 五十令吉到一百令吉, 一到两天 | பத்து முதல் பதினைந்து சதவீதம் |
| score `3-1` after score/skor/比分/ஸ்கோர்/won/beat | three one | tiga satu | 三比一 | மூன்று ஒன்று |
| `10k`, `100k`, `1.5k` (lower-case k) | ten thousand, one hundred thousand, one thousand five hundred | sepuluh ribu | 一万, 十万 | பத்தாயிரம், ஒரு லட்சம் |
| `2x`, `3.5x` (lower-case x) | two times, three point five times | dua kali | 两倍 | இரண்டு மடங்கு |
| `1000000` (no commas) | one million (was a "phone number") | satu juta | 一百万 | பத்து லட்சம் |
| codes: verification code, postcode, serial, plate, `unit 12-3-5`, `Dial 100 / 999` | four eight two nine, five zero four five zero, one two three five, one zero zero / nine nine nine; "the code is 1000" stays one thousand | Kod pengesahan … empat lapan dua …, Poskod lima kosong …, Tekan … atau kosong | 验证码四八二九一三, 邮编五零四五零 | குறியீடு நான்கு எட்டு …; 12-3-5 as digits (LLM: cardinals) |
| ids `COVID-19`, `Boeing 737`, `gate A12`, `3A-12-3`, `XR-2000` | COVID nineteen, Boeing seven three seven, A twelve, three A twelve three, XR two thousand | same | COVID十九, 三A一二三 | same as English |
| dotted time with a cue `pukul 3.30 petang`, `at 6.30`, `5.15 மணிக்கு` (no cue ⇒ decimal: "3.25 per annum") | six thirty | tiga tiga puluh petang, dua petang (2.00) | — (Mandarin writes 点) | ஐந்து பதினைந்து |
| Malay period range `9 pagi - 5 petang` | — | sembilan pagi hingga lima petang | — | — |
| military time `0730 hours`, `pukul 0730`, `1900 hrs` | zero seven thirty hours, nineteen hundred hours | kosong tujuh tiga puluh | 零七三零 | பூஜ்ஜியம் ஏழு முப்பது |
| `3 to 4pm`, `2.30 to 4.30pm`, `12:00 noon`, `Mon-Fri` | three to four p m (period not copied to the first time), two thirty to four thirty p m, twelve noon, Monday to Friday | — | — | — |
| units `6 hrs`, `10 min`, `45 sec`, `500W`, `12V`, `1,200 sq ft` | six hours, ten minutes, forty-five seconds, five hundred watts, twelve volts, square feet | jam, minit, saat, watt, volt, kaki persegi | 小时, 分钟, 秒, 瓦, 伏, 平方英尺 | மணி நேரம், நிமிடம், வினாடி, வாட், வோல்ட், சதுர அடி |
| Malay `kali ke 3` | — | kali ketiga | — | — |
| Tamil suffix glued to a digit `2024ல்`, `12ஆல்`, `RM1,000ஐ`, `15ஆகும்`, `2030க்குள்` | — | — | — | இரண்டாயிரத்து இருபத்து நான்கில், பன்னிரண்டால், ஆயிரம் ரிங்கிட்டை, பதினைந்தாகும், முப்பதுக்குள் |
| Tamil `1.5 மணி`, `1/2 மணி`, `1 பேர்`, `100,000` | — | — | — | ஒன்றரை மணி, அரை மணி, ஒருவர், ஒரு லட்சம் (lakhs below 10⁷, மில்லியன் above, as the LLM) |

Tamil suffixes work through a marker: a handler whose match is followed directly by a Tamil letter
appends a private-use character, and a final pass joins the number word and the suffix with sandhi
(`numbers.ta_attach`: ு-final words take the vowel sign, ம்-final words take த்த-, ட் doubles, நூறு →
நூற்ற-). A suffix written with a space (`2030 க்குள்`) is left alone, as before.

## Code-switching (Malay/English)

Tamil and Mandarin decide themselves by script (the whole sentence, so "Jalan Tun Razak 12号" is
十二号, as the LLM does). Latin text is Malay or English by a marker-word majority over the
sentence — function words plus the call-centre vocabulary these requests are made of, with
everyday borrowings such as *order*, *check*, *call*, *total* deliberately in neither list.

When a sentence carries markers of **both** languages, every number picks its own language
(`lang.local_lang`): the four nearest real words on each side vote with weight 1/distance, a
comma between them and the number halves the weight, a clause end stops the window, words the
normalizer itself produced ("one hundred twenty ringgit", "pagi") do not vote, and the sentence
language gets a prior of 0.5 unless the sentence was itself a tie. That is what the LLM does:

- *Nombor akaun anda ialah 1234, and your balance is RM50 today.* → Nombor akaun anda ialah **satu dua tiga empat**, and your balance is **fifty ringgit** today.
- *Please pay RM120 before 5pm, kalau tidak akaun anda akan digantung selama 3 hari.* → Please pay **one hundred twenty ringgit** before **five p m**, kalau tidak akaun anda akan digantung selama **tiga** hari.
- *Meeting pukul 10 pagi esok at level 12, jangan lupa bring 2 copies.* → Meeting pukul **sepuluh** pagi esok at level **dua belas**, jangan lupa bring **dua** copies.

12 of the 18 code-switched sentences match the LLM verbatim. The other six are the LLM
contradicting itself: it read "Discount 20%" in English but "The delivery fee is RM8" in Malay,
both with an English head word, and produced hybrids like "tiga p m" and "enam pm". There is no
rule that reproduces that, and these sentences have no single right answer.

## Language detection: marker words vs fastText vs the LLM

`bench/langid_compare.py` scores four detectors on the 497 corpus sentences plus 40 extra
code-switched sentences written with a designed majority language (20 Malay-majority with English
insertions, 20 the reverse; the LLM agreed with the designed label on all 40). Truth for the 22
corpus code-switched rows is the LLM's own majority call. Measured 2026-09-05:

| Set | marker words (`lang.py`) | fastText lid.176.ftz (1 MB) | Mesolitica bahasa/english fastText (331 MB) | LLM (gemma-4-31b, one round trip) |
|---|---|---|---|---|
| English (152) | 100% | 99% | 95% | 100% |
| Malay (114) | 100% | 89% | 89% | 99% |
| Mandarin (107) | 100% | 85% | 0% (no class) | 100% |
| Tamil (102) | 100% | 100% | 0% (no class) | 100% |
| Code-switch, corpus 22, LLM majority | 64% | 68% | 73% | 100% (by definition) |
| Code-switch, 40 designed majority | 100% | 90% | 100% | 100% |
| Latency per sentence | 2 µs | 7 µs | 12 µs | ~500 ms |

Reading it honestly:

- The marker-word 100% on the monolingual rows is **inflated**: the word lists were extended
  while building this corpus. The 40 designed code-switched rows were written afterwards and are
  the fairer number.
- **lid.176** fails exactly where we need it: short Malay sentences that are mostly numbers go to
  English, Italian, Spanish or Uzbek at confidence 0.1–0.5 ("Jualan naik 25%", "Sila tiba sebelum
  9.30am", "Had laju 110 km/j"), and 16 of 107 Mandarin sentences with Latin ids or units inside
  are labelled Japanese. Tamil is perfect (script). It is fine as a low-confidence-gated prior on
  long prose, useless on the numeric fragments the normalizer sees.
- **Mesolitica's model** has no Mandarin or Tamil class, and its "other" class swallows
  number-heavy text at confidence 1.0 ("Nombor IC saya 960314875079", "Pesanan #4471 telah
  dihantar", "Yurannya RM50 sebulan"). Trained on prose, wrong tool for this input.
- The **LLM** is the only detector that is right everywhere, at the cost of the ~0.5 s round trip
  the rule normalizer exists to remove. Its majority call on Manglish leans Malay whenever Malay
  function words are present ("Saya dah email invoice no. 4471 to you, please check by 15/3/2024"
  → malay), which is where the markers' tie-to-English policy loses its 8 corpus rows; 4 of those
  8 are exact ties.
- For the normalizer the sentence majority is only the prior: in a code-switched sentence each
  number is decided by the words next to it (`local_lang`), which no sentence-level model can do.

Conclusion: keep the script check + marker vote; a `language` field on the request (the upstream
agent knows the conversation language) is the robust fix; use the LLM offline to label real
traffic for a fitted detector, not at request time.

## Where the LLM contradicts itself, and what the rules do

| Topic | LLM | Rules |
|---|---|---|
| "and" in English hundreds | "two hundred and fifty" once, no "and" in 27 other numbers | never "and" |
| Letter runs in ids | "A B C one two three four" (en) but "ABC satu dua tiga empat" (ms), "MH three seven zero" | letter runs kept as written, single letters and digits read one by one |
| www / TLD | "w w w … dot m y" (en, zh), "double u double u double u" (ms), "www dot … dot my" (ta) | "w w w dot … dot my" everywhere |
| Malay ordinals | "ketiga" but "ke dua belas", "ke empat" | standard "kedua belas", "keempat" |
| Malay am/pm | "sembilan tiga puluh pagi" and "tiga p m" | pagi / petang |
| Tamil numerals | "இருபத்தைந்து" fused, "முப்பத்து ஏழு" unfused, "இருபத்தி நான்கு" colloquial | fuse vowel-initial units after 20–80 (இருபத்தைந்து, முப்பத்தேழு), space after consonant-initial ones and after 90 (இருபத்து நான்கு, தொண்ணூற்று எட்டு) |
| Tamil thousands | "ஒரு ஆயிரம் இருநூற்று ஐம்பது", "மூன்று ஆயிரம் நானூறு" | standard "ஆயிரத்து இருநூற்று ஐம்பது", "மூவாயிரத்து நானூறு" |
| ISO dates in Malay | written order "dua ribu dua puluh empat Mac lima belas" | day first "lima belas Mac dua ribu dua puluh empat" |
| RM0.50 | "zero ringgit fifty sen" (en), "lima puluh sen" (ms) | "fifty sen" |
| Sentence-final "5 Jan 2026" | "fifth of January" (no "the") | "the fifth of January" |
| RM12.5k / RM1.5k | "twelve thousand five hundred ringgit" (en) but "dua belas perpuluhan lima ribu ringgit" (ms) and "one point five thousand ringgit" | computed: "twelve thousand five hundred", "one thousand five hundred" |
| "/bulan" | "per bulan" and "sebulan" | "per bulan" |
| 200 before a measure word | 二百 (每箱200个) and 两百 (200多页) | 二百 |
| 9:15 in Mandarin | 九点十五 and 九点十五分 | 九点十五分 |
| Acronyms | K L I A, W X Y, X R, A P I spelled; OTP, SMS, IC, TNB not | never spelled |
| Tamil "12-3-5" | "பன்னிரண்டு மூன்று ஐந்து" (cardinals) while en/ms/zh got digits | digits everywhere |
| Tamil "RM5/kg" | "ஐந்து ரிங்கிட் ஒரு கிலோகிராம்" (no case) but "ஒரு மணி நேரத்திற்கு" for km/h | dative on the unit: ஒரு கிலோகிராமுக்கு |
| Tamil "1900 hrs" | digit by digit, but "0730" as பூஜ்ஜியம் ஏழு முப்பது | hour + minutes: பத்தொன்பது பூஜ்ஜியம் பூஜ்ஜியம் |
| Tamil 24 | இருபத்து நான்கு and colloquial இருபத்தி நான்கு | இருபத்து நான்கு |
| Malay 2.0 | "dua perpuluhan sifar" here, kosong for other decimal zeros | kosong |

## Where the LLM was wrong, and the rules are right

- Tamil range left as digits: *10 முதல் 15 நிமிடங்கள்* came back unchanged; rules: பத்து முதல் பதினைந்து.
- A Tamil sentence answered in English: *5/6/2025 க்கு முன் RM89.90 …* → "five June twenty twenty-five … eighty-nine ringgit ninety sen"; rules: ஐந்து ஜூன் இரண்டாயிரத்து இருபத்தைந்து … எண்பத்தொன்பது ரிங்கிட் தொண்ணூறு சென்.
- A Tamil price mangled into "நூற்றொன்பதுபத்தொன்பது" (RM199); rules: நூற்று தொண்ணூற்று ஒன்பது ரிங்கிட்.
- An IC digit invented: 960314-08-5079 → "… zero eight **zero** five zero seven nine" (13 digits); rules read the 12 that are there.
- 32°C → "三十二度C" (and 180°C → 一百八十度); rules: 三十二摄氏度, the form the LLM itself used for 负五摄氏度.
- "9.30am" in Mandarin → "九点三十分am"; rules: 上午九点三十分.
- "30-06-2025" in Tamil → "முப்பத்து நாள் ஜூன் …"; rules: முப்பது ஜூன் இரண்டாயிரத்து இருபத்தைந்து.
- Eight of the 42 harder Tamil sentences came back **unchanged, digits included** (`30 நாள் … 24 மணி`,
  `1/2 கப் … 3/4 கப்`, `100க்கு 95`, `2010 முதல் 2020 வரை … 2.3%`, `1ஆம் நாள் முதல் 15ஆம் நாள்`,
  `5ஐ 3ஆல் பெருக்கினால் 15ஆகும்`, `5 தவணைகளில் 3வது 15ஆம் தேதி`); one more was answered in English
  (`RM1,000ஐ` → "one thousand ringgit ஐ"). The rules read all of them.
- "10.45 மணிக்கு" → "பத்து முப்பத்தைந்து" (35 for 45); "2024ஆம்" → "இருபத்தி நான்குஆம்" (no sandhi);
  "50450" → "ஐம்பதாயிரத்து நானூற்று ஐபது" (typo, and a postcode read as a quantity).
- "2 hrs 30 mins" → "2 hours 30 minutes" (digits left); "not 3.30" → "three point thirty".
- "buka semula 2.00 petang" → "dua pukul petang"; "110 km/j" and "180°C" → "satu ratus …" (seratus).
- "95/100分" → "九十五分之一百分" (numerator and denominator swapped); "2块钱 … 2元" left unchanged;
  "9am-5pm" → "九点到五点" (periods dropped).

These account for most of the Tamil gap: on Tamil the rules are the more reliable of the two.

## Known limits

- Malay/English detection is by marker words. A sentence made only of words in neither list
  defaults to English. Pass `lang=` to force a language. Letters glued to a digit ("3A") do not vote.
- `a/b` is a fraction when a < b, a date when it could be one and a ≥ b (31/12, 1/1) or a date
  cue precedes it (on 5/6), "a b" otherwise (50/50); after Jalan/Lot/Blok/Unit it is said with
  "slash" / "per". `24/7` is special-cased per language.
- A dotted time (`3.30`) is only a time next to a cue (at/by/from/until/pukul/jam, or
  pagi/petang/malam/மணி after it) and never before per/percent/units; `3.25 per annum` stays a
  decimal. A negative sign must be glued to the number (`-5°C`); `9 - 5` is a range.
- `k` and `x` multipliers are lower-case only: `10k` is ten thousand but `4K` is "four K"; `2x` is
  two times but `2X` is a size.
- "/unit" after a quantity is limited to the units in `_PER_UNITS` (kg g km m l h day week month
  year min s unit person piece pax sqft and their Malay/Chinese/Tamil names).
- Tamil case suffixes handled: இல்/ல், ஆல், ஐ, உம், ஆக, ஆகும், க்கு, க்கும், க்குள், க்கான,
  இலிருந்து, உடன், ஓடு, ஆய், ஆவது/வது. Anything else is left glued as written.
- Which bare numbers are read digit by digit (rooms, PINs, order and invoice numbers, extensions,
  verification codes, postcodes, serials, plates, Boeing models, emergency numbers in a sentence
  about dialling) versus as a quantity is decided by nearby context words (`_DIGIT_CONTEXT`), by a
  leading zero, or by length (≥7 digits unless it is a round number, or ≥4 after a context word
  unless it is a round thousand: "the code is 1000"). New kinds of identifiers may need a word
  added there.
- Acronyms are never spelled out; "OTP" stays "OTP". The LLM spelled "A P I" and "H T T P" but not
  "OTP", "SMS", "IC" — inconsistent, so the rules keep them and let the TTS LM read them.
- Currencies: RM/MYR, $, USD, SGD, AUD, GBP/£, EUR/€. Units: kg g mg km m cm mm ml l GB MB TB KB
  Mbps kWh °C °F. Anything else is left as written (and, in `mode=llm` with
  `LLM_NORMALIZER_RULE_FIRST`, that leftover is exactly what still triggers the LLM call).
- Native-speaker review of the Tamil style choices (fused vs spaced numerals, literary thousands)
  is worth a pass; they are the standard forms, not the LLM's colloquial ones.

## Using it

```bash
# request field
{"input": "...", "mode": "spoken"}
# or as the default
DEFAULT_NORMALIZER_MODE=spoken
# or keep mode=llm and let the rules go first (LLM only for what they leave unspeakable)
LLM_NORMALIZER_RULE_FIRST=true
```

`mode=llm` falls back to `spoken` (not the legacy `rule` pipeline) when the LLM fails or is not
configured, so an outage degrades to this output.
