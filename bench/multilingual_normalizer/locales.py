"""
Per-locale written formats and seed templates for the generator. Templates carry typed slots
in braces; generate.py fills each with a (written, spoken) pair from verbalize.py. LLM-written
templates (templates_llm.py) are appended to these seeds at run time.
"""
from .verbalize import LOCALES  # noqa: F401

# phone number shapes: d = random digit, fixed digits kept
PHONE_FORMATS = {
    'en': ['01d-ddd dddd', '01d-dddddddd', '03-dddd dddd', '+60 1d-ddd dddd', '1-300-88-dddd'],
    'ms': ['01d-ddd dddd', '01d-dddddddd', '03-dddd dddd', '+60 1d-ddd dddd', '1-300-88-dddd'],
    'zh': ['01d-ddd dddd', '01d-dddddddd', '03-dddd dddd', '+60 1d-ddd dddd'],
    'ta': ['01d-ddd dddd', '01d-dddddddd', '03-dddd dddd', '+60 1d-ddd dddd'],
    'ta-LK': ['07d ddd dddd', '07d-dddddddd', '+94 7d ddd dddd', '011 ddd dddd'],
    'si': ['07d ddd dddd', '07d-dddddddd', '+94 7d ddd dddd', '011 ddd dddd'],
    'tl': ['09dd ddd dddd', '09dd-ddd-dddd', '+63 9dd ddd dddd', '(02) dddd dddd'],
    'id': ['08dd-dddd-dddd', '08dddddddddd', '+62 8dd-dddd-dddd', '(021) ddd-dddd'],
    'ar': ['05d ddd dddd', '05dddddddd', '+966 5d ddd dddd', '01d dddd dddd'],
    'fr': ['0d dd dd dd dd', '+33 d dd dd dd dd'],
    'de': ['017d dddddddd', '015d dddddddd', '+49 17d dddddddd', '030 ddddddd'],
    'es': ['6dd ddd ddd', '7dd ddd ddd', '9dd ddd ddd', '+34 6dd ddd ddd'],
    'it': ['3dd ddd dddd', '+39 3dd ddd dddd', '06 dddd dddd'],
    'pt': ['9dd ddd ddd', '+351 9dd ddd ddd', '21 ddd dddd'],
    'nl': ['06-dddddddd', '06 dddd dddd', '+31 6 dddd dddd', '020 ddd dddd'],
    'pl': ['5dd ddd ddd', '6dd ddd ddd', '+48 5dd ddd ddd', '22 ddd dd dd'],
}
# date shapes; {d} {m} day/month without padding, {dd} {mm} padded, {y} year, {Month} name
DATE_FORMATS = {
    'en': ['{d}/{m}/{y}', '{d} {Month} {y}', '{Month} {d}, {y}', '{dd}-{mm}-{y}', '{d} {Month}'],
    'ms': ['{d}/{m}/{y}', '{d} {Month} {y}', '{dd}-{mm}-{y}', '{d} {Month}'],
    'zh': ['{y}年{m}月{d}日', '{d}/{m}/{y}', '{y}-{mm}-{dd}', '{m}月{d}日'],
    'ta': ['{d}/{m}/{y}', '{d} {Month} {y}', '{y}-{mm}-{dd}', '{Month} {d}'],
    'ta-LK': ['{d}/{m}/{y}', '{d} {Month} {y}', '{y}-{mm}-{dd}', '{Month} {d}'],
    'id': ['{d}/{mm}/{y}', '{d} {Month} {y}', '{dd}-{mm}-{y}', '{d} {Month}'],
    'fr': ['{dd}/{mm}/{y}', '{d} {Month} {y}', '{d}/{m}/{y}', '{d} {Month}'],
    'de': ['{dd}.{mm}.{y}', '{d}. {Month} {y}', '{d}.{m}.{y}', '{d}. {Month}'],
    'es': ['{dd}/{mm}/{y}', '{d} de {Month} de {y}', '{d}/{m}/{y}', '{d} de {Month}'],
    'it': ['{dd}/{mm}/{y}', '{d} {Month} {y}', '{d}/{m}/{y}', '{d} {Month}'],
    'pt': ['{dd}/{mm}/{y}', '{d} de {Month} de {y}', '{d}/{m}/{y}', '{d} de {Month}'],
    'nl': ['{dd}-{mm}-{y}', '{d} {Month} {y}', '{d}-{m}-{y}', '{d} {Month}'],
}
# time shapes; {H} 24h hour, {h} 12h hour, {MM} minutes, {ap} am/pm
TIME_FORMATS = {
    'en': ['{h}:{MM} {ap}', '{h}.{MM}{ap}', '{H}:{MM}', '{h}{ap}'],
    'ms': ['{H}:{MM}', '{h}.{MM} {ms_period}', '{h}:{MM} {ms_period}', '{h} {ms_period}'],
    'zh': ['{H}:{MM}', '{zh_period}{h}点{MM}分', '{zh_period}{h}:{MM}', '{h}点'],
    'ta': ['{h}:{MM} {ap}', '{H}:{MM}', '{h}.{MM}', '{h}{ap}'],
    'ta-LK': ['{h}:{MM} {ap}', '{H}:{MM}', '{h}.{MM}', '{h}{ap}'],
    'id': ['{H}.{MM}', '{H}:{MM}', '{H}'],
    'fr': ['{H}h{MM}', '{H} h {MM}', '{H}h', '{H}:{MM}'],
    'de': ['{H}:{MM} Uhr', '{H}.{MM} Uhr', '{H} Uhr', '{H}:{MM}'],
    'es': ['{H}:{MM}', '{H}:{MM} h', '{H}.{MM}'],
    'it': ['{H}:{MM}', '{H}.{MM}'],
    'pt': ['{H}h{MM}', '{H}:{MM}', '{H}h'],
    'nl': ['{H}:{MM}', '{H}.{MM} uur', '{H} uur'],
}
ORDINAL_FORMATS = {
    'en': ['{n}{sfx}'], 'ms': ['ke-{n}'], 'id': ['ke-{n}'], 'zh': ['第{n}'], 'ta': ['{n}ஆவது', '{n}வது'], 'ta-LK': ['{n}ஆவது', '{n}வது'],
    'tl': ['ika-{n}'], 'fr': ['{n}e', '{n}ème'], 'de': ['{n}.'], 'es': ['{n}.º', '{n}º'], 'it': ['{n}º', '{n}°'], 'pt': ['{n}.º', '{n}º'],
    'nl': ['{n}e'],
}
# units the app.spoken_normalizer handles, for the MY locales (others use verbalize.UNITS)
MY_UNITS = ['kg', 'km', 'cm', 'g', 'ml', 'l', 'GB', '°C']
MY_CURRENCIES = {'en': ['RM{amt}', 'RM {amt}', 'USD {amt}', '${amt}'], 'ms': ['RM{amt}', 'RM {amt}'], 'zh': ['RM{amt}', 'RM {amt}'],
                 'ta': ['RM{amt}', 'RM {amt}'], 'ta-LK': ['Rs. {amt}', 'Rs.{amt}', 'LKR {amt}', 'ரூ. {amt}']}

SEED_TEMPLATES = {
    'en': [
        'Your balance is {money} as of {date}.', 'Please call us at {phone} before {time}.',
        'The delivery will take {range} working days.', 'Your OTP is {digits}, valid for {int} minutes.',
        'Order {id} was shipped on {date} and weighs {unit}.', 'Sales grew {percent} in {year}.',
        'The meeting is at {time} on {date}, room {digits}.', 'Send the form to {email} or visit {url}.',
        'You are {ordinal} in the queue, about {int} minutes to go.', 'The rating is {decimal} out of 5 from {big} reviews.',
    ],
    'ms': [
        'Baki anda ialah {money} setakat {date}.', 'Sila hubungi kami di {phone} sebelum pukul {time}.',
        'Penghantaran mengambil masa {range} hari bekerja.', 'Kod OTP anda ialah {digits}, sah selama {int} minit.',
        'Pesanan {id} telah dihantar pada {date} dan seberat {unit}.', 'Jualan meningkat {percent} pada tahun {year}.',
        'Mesyuarat pada pukul {time}, {date}, di bilik {digits}.', 'Hantar borang ke {email} atau layari {url}.',
        'Anda yang {ordinal} dalam barisan, lebih kurang {int} minit lagi.', 'Penilaian purata {decimal} daripada 5 daripada {big} ulasan.',
    ],
    'id': [
        'Saldo Anda saat ini {money} per tanggal {date}.', 'Silakan hubungi kami di {phone} sebelum pukul {time}.',
        'Pengiriman membutuhkan {range} hari kerja.', 'Kode OTP Anda {digits}, berlaku selama {int} menit.',
        'Pesanan {id} dikirim pada {date} dengan berat {unit}.', 'Penjualan naik {percent} pada tahun {year}.',
        'Rapat dimulai pukul {time} tanggal {date} di ruang {digits}.', 'Kirim formulir ke {email} atau kunjungi {url}.',
        'Anda berada di antrean {ordinal}, sekitar {int} menit lagi.', 'Rating rata-rata {decimal} dari 5 berdasarkan {big} ulasan.',
    ],
    'zh': [
        '您的余额为{money}，截至{date}。', '请在{time}之前拨打{phone}联系我们。', '送货需要{range}个工作日。',
        '您的验证码是{digits}，{int}分钟内有效。', '订单{id}已于{date}发出，重量{unit}。', '销售额在{year}增长了{percent}。',
        '会议于{date}{time}在{digits}号房举行。', '请将表格发送至{email}或访问{url}。', '您排在{ordinal}位，还需等待约{int}分钟。',
        '平均评分{decimal}分，来自{big}条评价。',
    ],
    'ta': [
        'உங்கள் இருப்பு {date} நிலவரப்படி {money}.', '{time} மணிக்கு முன் {phone} என்ற எண்ணில் எங்களை அழைக்கவும்.',
        'டெலிவரிக்கு {range} வேலை நாட்கள் ஆகும்.', 'உங்கள் OTP {digits}, {int} நிமிடங்களுக்கு செல்லுபடியாகும்.',
        'ஆர்டர் {id} {date} அன்று அனுப்பப்பட்டது, எடை {unit}.', '{year} இல் விற்பனை {percent} உயர்ந்தது.',
        'கூட்டம் {date} அன்று {time} மணிக்கு, அறை {digits}.', 'படிவத்தை {email} க்கு அனுப்புங்கள் அல்லது {url} ஐ பார்வையிடுங்கள்.',
        'நீங்கள் வரிசையில் {ordinal}, இன்னும் {int} நிமிடங்கள்.', 'சராசரி மதிப்பீடு 5க்கு {decimal}, {big} மதிப்புரைகளில்.',
    ],
    'si': [
        'ඔබේ ශේෂය {money} වේ.', 'කරුණාකර {phone} අංකයට අමතන්න.', 'ඔබේ OTP අංකය {digits} වේ, එය මිනිත්තු {int} සඳහා වලංගුයි.',
        'ඇණවුම {id} යවා ඇත.', 'විකුණුම් වර්ධනය {percent} වේ.', 'පෝරමය {email} වෙත එවන්න නැතහොත් {url} වෙත පිවිසෙන්න.',
        'සාමාන්‍ය ලකුණු {decimal} වේ.', 'පාරිභෝගිකයින් {big} දෙනෙක් ලියාපදිංචි විය.', '{year} වර්ෂයේ විකුණුම් ඉහළ ගියේය.',
        'ගාස්තුව {money} සහ බදු {percent} වේ.',
    ],
    'tl': [
        'Ang balanse mo ay {money}.', 'Tumawag sa {phone} para sa tulong.', 'Aabutin ng {range} araw ang delivery.',
        'Ang OTP mo ay {digits}, valid sa loob ng {int} minuto.', 'Naipadala na ang order {id} at tumitimbang ito ng {unit}.',
        'Tumaas ng {percent} ang benta noong {year}.', 'Ikaw ang {ordinal} sa pila, mga {int} minuto pa.',
        'Ipadala ang form sa {email} o bisitahin ang {url}.', 'Ang average rating ay {decimal} mula sa {big} na review.',
        'May {int} opsyon ka ngayon.',
    ],
    'ar': [
        'رصيدك الحالي هو {money}.', 'يرجى الاتصال بنا على الرقم {phone}.', 'رمز التحقق الخاص بك هو {digits}.',
        'تم شحن الطلب رقم {id}.', 'ارتفعت المبيعات بنسبة {percent} في عام {year}.', 'أرسل النموذج إلى {email} أو تفضل بزيارة {url}.',
        'متوسط التقييم {decimal} من خمسة.', 'عدد المشتركين بلغ {big}.', 'الرسوم {money} شاملة الضريبة.', 'صلاحية الرمز {int} دقائق.',
    ],
    'fr': [
        'Votre solde est de {money} au {date}.', 'Appelez-nous au {phone} avant {time}.', 'La livraison prend {range} jours ouvrés.',
        'Votre code est le {digits}, valable {int} minutes.', 'La commande {id} a été expédiée le {date} et pèse {unit}.',
        'Les ventes ont augmenté de {percent} en {year}.', 'La réunion est à {time} le {date}, salle {digits}.',
        'Envoyez le formulaire à {email} ou consultez {url}.', 'Vous êtes {ordinal} dans la file, encore {int} minutes environ.',
        'La note moyenne est de {decimal} sur 5 pour {big} avis.',
    ],
    'es': [
        'Su saldo es de {money} a fecha de {date}.', 'Llámenos al {phone} antes de las {time}.', 'La entrega tarda {range} días laborables.',
        'Su código es {digits}, válido durante {int} minutos.', 'El pedido {id} se envió el {date} y pesa {unit}.',
        'Las ventas subieron un {percent} en {year}.', 'La reunión es a las {time} del {date}, en la sala {digits}.',
        'Envíe el formulario a {email} o visite {url}.', 'Es el {ordinal} en la cola, unos {int} minutos más.',
        'La valoración media es {decimal} sobre 5 con {big} opiniones.',
    ],
    'de': [
        'Ihr Kontostand beträgt {money} zum {date}.', 'Rufen Sie uns vor {time} unter {phone} an.', 'Die Lieferung dauert {range} Werktage.',
        'Ihr Code lautet {digits} und ist {int} Minuten gültig.', 'Die Bestellung {id} wurde am {date} versandt und wiegt {unit}.',
        'Der Umsatz stieg {year} um {percent}.', 'Das Treffen ist um {time} am {date} in Raum {digits}.',
        'Senden Sie das Formular an {email} oder besuchen Sie {url}.', 'Sie sind der {ordinal} in der Warteschlange, noch etwa {int} Minuten.',
        'Die durchschnittliche Bewertung liegt bei {decimal} von 5 aus {big} Bewertungen.',
    ],
    'it': [
        'Il suo saldo è di {money} al {date}.', 'Ci chiami al {phone} prima delle {time}.', 'La consegna richiede {range} giorni lavorativi.',
        'Il suo codice è {digits}, valido per {int} minuti.', "L'ordine {id} è stato spedito il {date} e pesa {unit}.",
        'Le vendite sono cresciute del {percent} nel {year}.', 'La riunione è alle {time} del {date}, in sala {digits}.',
        'Invii il modulo a {email} oppure visiti {url}.', 'È il {ordinal} in coda, ancora circa {int} minuti.',
        'La valutazione media è {decimal} su 5 con {big} recensioni.',
    ],
    'pt': [
        'O seu saldo é de {money} em {date}.', 'Ligue-nos para o {phone} antes das {time}.', 'A entrega demora {range} dias úteis.',
        'O seu código é {digits}, válido por {int} minutos.', 'A encomenda {id} foi enviada em {date} e pesa {unit}.',
        'As vendas subiram {percent} em {year}.', 'A reunião é às {time} de {date}, na sala {digits}.',
        'Envie o formulário para {email} ou visite {url}.', 'É o {ordinal} na fila, faltam cerca de {int} minutos.',
        'A avaliação média é {decimal} em 5 com {big} opiniões.',
    ],
    'nl': [
        'Uw saldo is {money} per {date}.', 'Bel ons op {phone} vóór {time}.', 'De levering duurt {range} werkdagen.',
        'Uw code is {digits}, {int} minuten geldig.', 'Bestelling {id} is op {date} verzonden en weegt {unit}.',
        'De omzet steeg in {year} met {percent}.', 'De vergadering is om {time} op {date} in kamer {digits}.',
        'Stuur het formulier naar {email} of bezoek {url}.', 'U bent de {ordinal} in de rij, nog ongeveer {int} minuten.',
        'De gemiddelde beoordeling is {decimal} van 5 op basis van {big} reviews.',
    ],
    'pl': [
        'Twoje saldo wynosi {money}.', 'Zadzwoń do nas pod numer {phone}.', 'Twój kod to {digits}.', 'Zamówienie {id} zostało wysłane.',
        'Sprzedaż wzrosła o {percent}.', 'Wyślij formularz na {email} lub odwiedź {url}.', 'Średnia ocena to {decimal} na 5.',
        'Liczba opcji: {int}.', 'Liczba użytkowników wynosi {big}.', 'Opłata wynosi {money} brutto.',
    ],
}
SEED_TEMPLATES['ta-LK'] = SEED_TEMPLATES['ta']

# what the LLM is told when writing sentences for each category (llm_pairs.py)
CATEGORY_HINTS = {
    'int': 'plain counts and quantities written as digits (3, 12, 250, 1,500)',
    'big': 'large numbers with thousands separators (33,000,000; 1.250.000; 2 500 000)',
    'money': 'amounts of money with the local currency symbol or code, sometimes with cents',
    'decimal': 'decimal numbers (3.5, 0.75, 12,5) as written locally',
    'percent': 'percentages (25%, 12.5 %, 100%)',
    'phone': 'phone numbers in local formats, including one with a country code',
    'digits': 'codes read digit by digit: OTP, PIN, postcode, account or order numbers',
    'date': 'dates in several local written forms (numeric, month name, with and without year)',
    'time': 'clock times in several local written forms (24h, 12h, with am/pm or local words)',
    'ordinal': 'ordinal numbers as written locally (1st, 15., 3º, ke-3, 第3)',
    'range': 'numeric ranges with a hyphen (3-5 days, 10-15%, 9am-5pm)',
    'unit': 'measurements with unit abbreviations (5 kg, 2.5 km, 32°C, 500 ml, 50 GB)',
    'year': 'years and decades (1998, 2024, the 1980s)',
    'email': 'email addresses',
    'url': 'web addresses (www..., https://...)',
    'id': 'alphanumeric identifiers (flight MH370, reference ABC1234, seat 24A, plate WXY 1234)',
    'mixed': 'two or three of: money, dates, times, phone numbers, percentages, codes in one sentence',
    'fraction': 'fractions and slashes (1/2, 3/4, 24/7, 95/100)',
    'negative': 'negative numbers and temperatures (-5°C, -3.5%)',
    'plain': 'NO digits, symbols or abbreviations at all: ordinary conversational sentences that need no normalization',
}
