"""
Prompt for the LLM-based TTS text normalizer (`mode="llm"` on /v1/audio/normalize).

The LLM rewrites written text into the exact words a speaker would say out loud
(numbers, money, IC/phone numbers, dates, ...) so the result can be fed directly
to the TTS LM. The reply is constrained to `{"normalized": "..."}` with an OpenAI
`response_format` json_schema (vLLM guided decoding), so the model physically
cannot return anything except the normalized text.

EXAMPLES are sent as real few-shot chat turns: each `before` becomes a user
message and each `after` an assistant message already wrapped in the JSON shape.
"""

SYSTEM_PROMPT = """\
You are the text normalizer of a multilingual Malaysian text-to-speech (TTS) system.
Rewrite the user's text into its exact spoken form, ready to be synthesized.

Rules:
- Expand everything that is not speakable as written: numbers, money (RM -> ringgit/sen \
in Malay and English, 令吉/仙 in Chinese), IC numbers digit by digit, phone numbers digit \
by digit, dates, times, percentages, decimals, ordinals, units, emails and URLs.
- Verbalize in the language of the surrounding text: Malay numbers for Malay text, \
English for English, Chinese for Chinese. Never translate the text itself.
- Never answer questions, never add, drop, reorder or translate words. Only rewrite \
non-speakable tokens into words; everything else stays exactly as written.
- If nothing needs normalizing, return the text unchanged.
- Reply with JSON only: {"normalized": "<spoken form>"}\
"""

# (before, after) few-shot pairs. Keep them short and per-language; they are sent
# with every request, so every extra pair costs prompt tokens.
EXAMPLES = [
    # Malay
    ("helo nama saya husein, IC 960314875079",
     "helo nama saya husein, IC sembilan enam kosong tiga satu empat lapan tujuh lima kosong tujuh sembilan"),
    ("baki saya tinggal RM50",
     "baki saya tinggal lima puluh ringgit"),
    ("jumlah RM1,250.50",
     "jumlah seribu dua ratus lima puluh ringgit lima puluh sen"),
    ("sila hubungi 03-12345678",
     "sila hubungi kosong tiga satu dua tiga empat lima enam tujuh lapan"),
    ("mesyuarat pada 15/3/2024 pukul 3:30 petang",
     "mesyuarat pada lima belas Mac dua ribu dua puluh empat pukul tiga tiga puluh petang"),
    ("diskaun 25%",
     "diskaun dua puluh lima peratus"),
    ("suhu 37.5 darjah",
     "suhu tiga puluh tujuh perpuluhan lima darjah"),
    ("anda yang ke-3 dalam barisan",
     "anda yang ketiga dalam barisan"),
    ("apa khabar semua?",
     "apa khabar semua?"),
    # English
    ("my name is husein, IC 960314875079",
     "my name is husein, IC nine six zero three one four eight seven five zero seven nine"),
    ("report id X-12340567",
     "report id X one two three four zero five six seven"),
    ("call me at 012-3456789",
     "call me at zero one two three four five six seven eight nine"),
    ("balance is RM1,250.50",
     "balance is one thousand two hundred fifty ringgit fifty sen"),
    ("meeting on 15/3/2024 at 10:45 AM",
     "meeting on the fifteenth of March twenty twenty-four at ten forty-five A M"),
    ("up by 25%",
     "up by twenty-five percent"),
    ("you're 3rd in the queue",
     "you're third in the queue"),
    ("email me at husein@site.com",
     "email me at husein at site dot com"),
    ("Thank you for calling, how can I help you today?",
     "Thank you for calling, how can I help you today?"),
    # Chinese
    ("我叫侯赛因，身份证号是960314875079",
     "我叫侯赛因，身份证号是九六零三一四八七五零七九"),
    ("请拨打03-12345678",
     "请拨打零三一二三四五六七八"),
    ("余额还剩RM50",
     "余额还剩五十令吉"),
    ("总共RM1,250.50",
     "总共一千二百五十令吉五十仙"),
    ("会议在2024年3月15日下午2点30分",
     "会议在二零二四年三月十五日下午两点三十分"),
    ("折扣25%",
     "折扣百分之二十五"),
]

# The request's `prompt` field (caller context the text alone does not carry, e.g.
# "normalize this in Malay"; with none the LLM read a bare "RM50" as "五十令吉") goes in two
# places: under this heading at the end of SYSTEM_PROMPT, and in brackets above the text in
# the final user turn. Measured on the live LLM (2026-10-07, 10 cases with one right answer
# + 18 plain sentences under language prompts):
#   system prompt only  5/10, 18/18 plain unchanged -- the 23 few-shots outvote it:
#                       "1500" + "spell one by one in en" -> "一千五百"
#   user turn only     10/10, 15/18 -- began translating plain sentences
#   both               10/10, 18/18, and the instruction never leaked into the output
CALLER_PROMPT_HEADER = "Additional instructions from the caller for this text:"

# Clock times: llm_normalizer.clock_hint() works out how each time in the text is read and
# puts it under this heading, at the end of the system prompt and in brackets above the text
# (the two places the caller `prompt` goes, for the same reason). The reply must keep it.
CLOCK_HINT_HEADER = "Read the clock times in this text exactly as given here (written -> spoken):"

# Number language: llm_normalizer.number_language() decides from the text's words that its
# numbers are read in Malay, and this line goes with the text the same two ways as the clock
# hint. Without it a short Malay reply had its amount read in English: "Okay, 100 ringgit,
# betul?" -> "one hundred ringgit" (gemma-4-31b, 10 of 10, 2026-11-24). The wording matters:
# "write every number in it in Malay words" fixed the numbers but narrowed the job to numbers
# ("5 GB" stayed "GB", "no." and "WXY" were left, 78/96 hinted truth rows against 83/82);
# this one read all 18 hinted test lines in Malay 10 of 10 and kept the hinted rows at 82/84.
NUMBER_LANGUAGE_HINT = "This text is Malay: normalize it as Malay text."

# Postcodes: llm_normalizer.postcode_hint() writes one of these per postcode, with its
# digit-by-digit reading, the same two places as the clock hint; the reply must keep it.
# Why: a round postcode in Malay was read as an amount ("60000 Kuala Lumpur" -> "enam puluh
# ribu", "80000 Johor Bahru" -> "lapan puluh ribu" 14 of 20). The wording matters: the
# clock hint's "written -> spoken" list got 2 of 20 on the prod address (the house number
# ran into it, "dua satu kosong kosong kosong") and 7 of 20 on another; this sentence got
# 140 of 140 over 7 Malay addresses (gemma-4-31b, concurrency 20, 2026-11-25).
POSTCODE_HINT = '{written} is a postcode: read it digit by digit, "{spoken}".'
POSTCODE_HINT_TAIL = 'Read the other numbers as usual.'

# Strict schema for OpenAI `response_format`; guided decoding guarantees the reply
# parses as {"normalized": str}, so the output can be fed directly to the TTS LM.
JSON_SCHEMA = {
    "name": "normalized_text",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"normalized": {"type": "string"}},
        "required": ["normalized"],
        "additionalProperties": False,
    },
}
