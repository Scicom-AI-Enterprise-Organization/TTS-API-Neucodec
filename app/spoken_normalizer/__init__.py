"""
Spoken-form text normalizer: a rule-based replica of the LLM normalizer (app/prompt.py) for
English, Malay, Mandarin and Tamil. `mode="spoken"` on /v1/audio/normalize and TTS requests.

    from app.spoken_normalizer import normalize
    normalize('Your total is RM1,250.50 and the meeting is at 3pm on 12/9/2026.')
    # 'Your total is one thousand two hundred fifty ringgit fifty sen and the meeting is at
    #  three p m on the twelfth of September twenty twenty-six.'

Why it exists: the LLM round trip is ~0.55 s and sits entirely in front of the first audio
byte; this runs in microseconds, deterministically, offline. The LLM's own outputs on
bench/normalizer_corpus.py (bench/results/normalizer_truth.jsonl) are the specification, and
bench/normalizer_agreement.py measures how closely each language matches them. Where the
LLM was inconsistent (spaced vs fused Tamil numerals, "A B C" vs "ABC", "w w w dot ... dot
m y") one convention was picked; where it was wrong (Tamil ranges left as digits, Tamil
sentences answered in English) the rules do the right thing instead.

Language: Tamil and CJK script decide themselves; Latin text is Malay or English by marker
words (lang.py). Pass lang= to override. Importable without torch/GPU.
"""
from .core import normalize, V, UNITS, MONTHS
from .lang import detect_lang

__all__ = ['normalize', 'detect_lang', 'V', 'UNITS', 'MONTHS']
