"""
Which language should numbers be read in? Script first, then Malay-vs-English by word lists.

The LLM prompt's rule is "verbalize in the language of the surrounding text". For Tamil and
Mandarin the script settles it; the whole sentence is treated as that language even where a
Latin name or acronym sits next to the number (the LLM does the same: "Jalan Tun Razak 12号"
-> "十二号"). Malay vs English is decided per sentence by counting marker words -- function
words plus the call-centre vocabulary these requests are made of -- with ties going to
English. In a code-switched sentence ("Nombor akaun anda ialah 1234, and your balance is
RM50") each number is decided on its own by `local_lang`: a distance-weighted vote of the
words around it, damped across a comma, with the sentence language as a prior -- which is
how the LLM behaves on such sentences (it read that example as "satu dua tiga empat" and
"fifty ringgit"). No fasttext, no model download: this has to import anywhere.
"""
import re

TAMIL_RE = re.compile(r'[஀-௿]')
CJK_RE = re.compile(r'[一-鿿㐀-䶿豈-﫿]')
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")

MS_WORDS = set('''
saya anda awak kamu kami kita dia mereka beliau ini itu yang dan atau untuk dengan pada di ke
dari daripada kepada adalah ialah akan sudah telah belum sedang boleh tidak tak bukan ada ya lah
kan nanti sekejap dah encik puan cik tuan terima kasih sila tolong hari esok semalam pagi petang
malam tengah jam pukul minit hingga sehingga sampai dalam luar atas bawah sebelum selepas lepas
semua setiap sebulan setahun seorang orang nombor jumlah bayar bayaran harga baki
akaun pesanan penghantaran tarikh masa tempat bilik tingkat hubungi menghubungi semak sahkan kad
pengenalan rumah kereta kerja bekerja pejabat syarikat pelanggan maklumat sistem borang emel talian
bantuan tempahan temu janji mesyuarat bulan tahun minggu kalau jika kerana sebab macam mana apa
siapa bila berapa bagaimana kenapa mengapa seperti juga sahaja saja lagi masih pun tu ni nak hendak
mahu ingin perlu mesti harus dapat buat beri berikan ambil hantar terus balik semula cuba tunggu
tengok dengar cakap kata tanya jawab faham tahu kenal guna pakai sebut dulu sekarang kemudian
seterusnya akhir awal mula bermula tamat tutup buka dibuka ditutup secara antara bagi oleh tentang
mengenai berkenaan sini sana situ tersebut berikut boleh bolehkah adakah tidakkah sudahkah lupa tapi
sahaja semuanya pelajar resit digantung selama minit awal lewat cepat lambat esok pagi tengahari
selamat datang jumpa baik baiklah okey ok ye betul salah lain lagi sangat amat paling lebih kurang
kurangkan tambah tambahan sila mohon harap sekiranya sebarang pertanyaan lanjut jangan teragak
kemaskini setakat termasuk cukai jimat berharga juta ribu deposit sewa suhu badan darjah penilaian
purata kadar faedah jualan naik turun bateri diskaun tiada tertunggak dewan memuatkan penduduk
mendaftar pilihan setiap dijadualkan program berlangsung dilahirkan denda dikenakan unit dihantar
pesan pesanan laporan rujukan penerbangan tempat duduk ditubuhkan berkembang berubah polisi
menjelang jangka berganda dokumen invois kaunter masukkan dihantar melalui layari pergi ke
'''.split())

EN_WORDS = set('''
the a an is are was were be been being am to of for with on at by from in into and or but if then
than your you yours we our ours they their them he she it its this that these those please thank
thanks will would can could should may might shall must have has had do does did not yes hello
hi sorry today tomorrow yesterday morning afternoon evening night week month year day days hour hours
minute minutes before after until between about as so very just only also more most some any all each
every other another new first last next here there where when what which who how why
booking confirmed amount payment pay price balance account meeting appointment office delivery
customer customers support service help information hotline please let me know thank you
welcome again sure right wrong good great fine still already yet soon later now once twice per each
including tax saved listed house fee costs comes charged interest rating average temperature degrees
package long went up sales users saw error battery discount reach us dialled dialed number service
confirm passport verified scheduled runs event deadline renewal falls born sharp close closes weekdays
weekends train departs arrive opens queue reminder follow happy birthday take exit roundabout open
expect wait working weighs box wide store away outside water tablets plan includes data details form
send visit learn report shipped reference tracking flight seat rescheduled company founded
expanded policy changed double waiting lobby located documents invoice committee people enter
items item copies level extension transfer received ticket tickets minutes seconds units
sent counter returned statement joint issued patience shortly worry sort quietly meet half past
'''.split())


def script_lang(text):
    """'ta' / 'zh' when that script is present (Tamil wins a tie), else None."""
    ta = len(TAMIL_RE.findall(text))
    zh = len(CJK_RE.findall(text))
    if ta and ta >= zh:
        return 'ta'
    if zh:
        return 'zh'
    return None


# Words the normalizer itself emits. They must not vote when a later number in the same
# sentence looks at its neighbours, or "one hundred twenty ringgit" would pull the next
# number towards English (or Malay, via "pagi"/"petang").
_GENERATED = set('''
zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen
sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred
thousand million billion point percent ringgit sen dollars dollar cents plus minus
kosong sifar satu dua tiga empat lima enam tujuh lapan sembilan sepuluh sebelas belas puluh seratus
ratus seribu ribu juta bilion perpuluhan peratus pagi petang tambah hingga dolar
'''.split())
_CLAUSE_END = re.compile(r'[.!?;:]')


def _score(tok):
    """(ms, en) vote of one lower-cased token."""
    if tok in MS_WORDS:
        return 1.0, 0.0
    if tok in EN_WORDS:
        return 0.0, 1.0
    if len(tok) > 5 and tok.endswith('nya'):
        return 1.0, 0.0
    if len(tok) > 6 and tok.startswith(('meng', 'peng', 'menge', 'penge', 'ber', 'ter')) \
            and tok.endswith(('kan', 'an', 'i')):
        return 0.5, 0.0
    return 0.0, 0.0


def latin_scores(text):
    """Sentence-level (ms, en) marker counts."""
    ms = en = 0.0
    for tok in _WORD_RE.findall(text.lower()):
        a, b = _score(tok)
        ms += a
        en += b
    return ms, en


def latin_lang(text):
    """'ms' or 'en' for Latin-script text, by marker-word majority (ties -> 'en')."""
    ms, en = latin_scores(text)
    return 'ms' if ms > en else 'en'


def is_code_switched(text):
    ms, en = latin_scores(text)
    return ms >= 1 and en >= 1


def local_lang(text, start, end, default, tie=False, window=4, prior=0.5, comma_damp=0.5):
    """Language for the number at text[start:end] in a code-switched sentence.

    The `window` nearest real words on each side vote with weight 1/distance (the nearest
    word 1, the next 1/2, ...); words the normalizer itself produced and single letters do
    not vote or count as distance; a comma between the word and the number halves its
    weight; a sentence/clause end (. ! ? ; :) stops the window. `default` (the sentence
    language) gets `prior` unless the sentence itself was a tie."""
    ms = en = 0.0
    for side in (_CLAUSE_END.split(text[:start])[-1][::-1], _CLAUSE_END.split(text[end:])[0]):
        d = 0
        damp = 1.0
        for chunk in re.finditer(r"[A-Za-z][A-Za-z']*|,", side):
            tok = chunk.group(0)
            if tok == ',':
                damp *= comma_damp
                continue
            tok = tok[::-1].lower() if side is not None and text[:start][::-1].startswith(side) else tok.lower()
            if len(tok) < 2 or tok in _GENERATED:
                continue
            d += 1
            if d > window:
                break
            a, b = _score(tok)
            ms += a * damp / d
            en += b * damp / d
    if not tie:
        if default == 'ms':
            ms += prior
        else:
            en += prior
    if ms > en:
        return 'ms'
    if en > ms:
        return 'en'
    return default


def detect_lang(text):
    return script_lang(text) or latin_lang(text)
