"""
Malay linguistic data.
Extracted from malaya.text.tatabahasa (https://github.com/malaysia-ai/malaya)
"""

consonants = 'bcdfghjklmnpqrstvwxyz'

sounds = {
    'x': 'tidak', 'y': 'kenapa', 'n': 'dan', 'g': 'pergi',
    's': 'seperti', 'd': 'di', 'k': 'ok', 'u': 'awak',
    't': 'nanti', 'p': 'pergi', 'wai': 'kenapa', 'i': 'saya',
}

bulan = {
    1: 'Januari', 2: 'Februari', 3: 'Mac', 4: 'April',
    5: 'Mei', 6: 'Jun', 7: 'Julai', 8: 'Ogos',
    9: 'September', 10: 'Oktober', 11: 'November', 12: 'Disember',
}

bulan_en = {
    1: 'January', 2: 'February', 3: 'March', 4: 'April',
    5: 'May', 6: 'June', 7: 'July', 8: 'August',
    9: 'September', 10: 'October', 11: 'November', 12: 'December',
}

date_replace = {
    'awk': 'awak', 'ngkau': 'engkau', 'lps': 'lalu', 'lepas': 'lalu',
    'mnggu': 'minggu', 'bln': 'bulan', 'skrg': 'sekarang', 'thun': 'tahun',
    'hri': 'hari', 'minute': 'minit', 'mnit': 'minit', 'sec': 'saat',
    'second': 'saat', 'yesterday': 'semalam', 'kelmarin': 'kelmarin',
    'smalam': 'semalam', 'dpan': 'depan', 'dpn': 'depan', 'esk': 'esok',
    'pagi': 'AM', 'pgi': 'AM', 'morning': 'AM',
    'tengahari': 'PM', 'tngahari': 'PM', 'petang': 'PM', 'ptg': 'PM',
    'malam': 'PM', 'semalam': 'semalam', 'pkul': 'pukul',
}

hujung_malaysian = ['lah', 'la', 'ler', 'lh']

hujung = {
    'kn': 'kan', 'knn': 'kan', 'kknn': 'kan', 'kkn': 'kan',
    'kan': 'kan', 'kann': 'kan', 'kkann': 'kan', 'kaan': 'kan',
    'kaann': 'kan', 'kah': 'kah', 'kahh': 'kah',
    'lah': 'lah', 'lahh': 'lah', 'loh': 'lah', 'lohh': 'lah',
    'lh': 'lah', 'lhh': 'lah', 'ler': 'lah',
    'tah': 'tah', 'tahh': 'tah',
    'nya': 'nya', 'nyaa': 'nya', 'nye': 'nya', 'nyee': 'nya',
    'nyo': 'nya', 'nyoo': 'nya', 'ny': 'nya',
    'an': 'an', 'ann': 'an',
    'wan': 'wan', 'wann': 'wan',
    'wati': 'wati', 'watii': 'wati',
    'ita': 'ita', 'itaa': 'ita',
}

permulaan = {
    'bel': 'bel', 'se': 'se', 'see': 'se',
    'ter': 'ter', 'terr': 'ter',
    'men': 'men', 'menn': 'men', 'meng': 'meng', 'mengg': 'meng',
    'mem': 'mem', 'mm': 'mem', 'memper': 'memper',
    'di': 'di', 'ddi': 'di',
    'pe': 'pe', 'ppe': 'pe', 'ppee': 'pe',
    'me': 'me', 'mme': 'mme',
    'ke': 'ke', 'kee': 'ke',
    'ber': 'ber', 'berr': 'ber',
    'pen': 'pen', 'penn': 'pen',
    'per': 'per', 'perr': 'perr',
}

calon_dictionary = {
    'dr': 'Doktor', 'yb': 'Yang Berhormat', 'hj': 'Haji',
    'ybm': 'Yang Berhormat Mulia', 'tyt': 'Tuan Yang Terutama',
    'yab': 'Yang Berhormat', 'yabhg': 'Yang Amat Berbahagia',
    'ybhg': 'Yang Berbahagia', 'miss': 'Cik', 'ydh': 'Yang Dihormati',
}

laughing = {
    'huhu', 'hahha', 'gagaga', 'sksk', 'haha', 'wkwk', 'hshs', 'uwu',
    'hihi', 'hoho', 'huehue', 'ksks', 'hehe', 'hewhew', 'jahagaha',
    'wkawka', 'keke', 'ahksk', 'ahakss', 'kiki',
}

mengeluh = {'argh', 'hais', 'adoi', 'aduh', 'haih', 'hoih'}
