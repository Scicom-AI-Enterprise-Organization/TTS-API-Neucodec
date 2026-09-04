"""
Which language should numbers be read in? Script first, then Malay-vs-English by word lists.

The LLM prompt's rule is "verbalize in the language of the surrounding text". For Tamil and
Mandarin the script settles it; the whole sentence is treated as that language even where a
Latin name or acronym sits next to the number (the LLM does the same: "Jalan Tun Razak 12号"
-> "十二号"). Malay vs English is decided per sentence by counting marker words -- function
words plus the call-centre vocabulary these requests are made of -- with ties going to
English. Code-switched sentences ("your bill RM85.50 dah overdue") are inherently ambiguous
and the LLM itself flips on them; the sentence-level majority is the predictable choice.
No fasttext, no model download: this has to import anywhere.
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
semua setiap sebulan setahun seorang orang ringgit sen nombor jumlah bayar bayaran harga baki
akaun pesanan penghantaran tarikh masa tempat bilik tingkat hubungi menghubungi semak sahkan kad
pengenalan rumah kereta kerja bekerja pejabat syarikat pelanggan maklumat sistem borang emel talian
bantuan tempahan temu janji mesyuarat bulan tahun minggu kalau jika kerana sebab macam mana apa
siapa bila berapa bagaimana kenapa mengapa seperti juga sahaja saja lagi masih pun tu ni nak hendak
mahu ingin perlu mesti harus dapat buat beri berikan ambil hantar terus balik semula cuba tunggu
tengok dengar cakap kata tanya jawab faham tahu kenal guna pakai sebut dulu sekarang kemudian
seterusnya akhir awal mula bermula tamat tutup buka dibuka ditutup secara antara bagi oleh tentang
mengenai berkenaan sini sana situ tersebut berikut boleh bolehkah adakah tidakkah sudahkah
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
send visit learn report order shipped reference tracking flight seat rescheduled company founded
expanded policy changed double waiting lobby located bring documents invoice committee people enter
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


def latin_lang(text):
    """'ms' or 'en' for Latin-script text, by marker-word majority (ties -> 'en')."""
    ms = en = 0.0
    for tok in _WORD_RE.findall(text.lower()):
        if tok in MS_WORDS:
            ms += 1
        elif tok in EN_WORDS:
            en += 1
        elif len(tok) > 5 and tok.endswith('nya'):
            ms += 1
        elif len(tok) > 6 and tok.startswith(('meng', 'peng', 'menge', 'penge', 'ber', 'ter')) \
                and tok.endswith(('kan', 'an', 'i')):
            ms += 0.5
    return 'ms' if ms > en else 'en'


def detect_lang(text):
    return script_lang(text) or latin_lang(text)
