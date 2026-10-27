import re
import json

from app.normalizer.regex import _expressions as _malay_expressions

contractions = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "we'd": "we would",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who're": "who are",
    "who's": "who is",
    "who've": "who have",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}

before_replace_mapping = {
    '–': ' ',
    ';': ',',
    '’': "'",
    '!': ',',
    '！': '，'
}

replace_mapping = {
    '–': ' ',
    ';': ',',
    '-': ' ',
    '(': '',
    ')': '',
    '[': '',
    ']': '',
    '’': "'",
}

pattern_range = re.compile(
    r'(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)*)'
)

_phone_pattern = re.compile(_malay_expressions['phone'], re.IGNORECASE)
_PHONE_PLACEHOLDER = '\x00PHONE{}\x00'


def protect_phone_numbers(text):
    """Mask phone-number-shaped substrings (e.g. '012-1234567') so `pattern_range`
    doesn't misparse them as a numeric range ('012' to '1234567')."""
    phones = []

    def _mask(match):
        phones.append(match.group(0))
        return _PHONE_PLACEHOLDER.format(len(phones) - 1)

    return _phone_pattern.sub(_mask, text), phones


def restore_phone_numbers(text, phones):
    for i, phone in enumerate(phones):
        text = text.replace(_PHONE_PLACEHOLDER.format(i), phone)
    return text

with open('app/pronunciation.json', 'r', encoding='utf-8') as f:
    pronunciation_dict = json.load(f)

sorted_keys = sorted(pronunciation_dict.keys(), key=len, reverse=True)

def apply_pronunciation_replacements(text):
    """Apply pronunciation replacements from the dictionary"""
    if not pronunciation_dict:
        return text
    
    try:
        for key in sorted_keys:
            pattern = r'\b' + re.escape(key) + r'\b'
            text = re.sub(pattern, pronunciation_dict[key], text)
        
        return text
    except Exception as e:
        logging.error(f"Error applying pronunciation replacements: {e}")
        return text

# Tool calls an agent's LLM wrote into its reply as text: gemma's
# `<|tool_call>call:name{key:<|"|>value<|"|>}<tool_call|>`, `<call:name{...}/>`, and the python
# shape `name_with_underscores(key="value")`. The HTML-tag rule in sanitize_markdown drops the
# tags and keeps what is between them, which the voice then reads out: "call colon transfer
# underscore to...". The agent splits a reply into sentences before it reaches us, so a call
# can arrive cut in two: an opening with no close loses everything after it, a close with no
# opening everything before it, and a chunk holding only gemma's quote token is the middle of
# a call. `<tool_call>...</tool_call>` is the same in the Hermes/Qwen shape.
_TOOL_MARK = re.compile(r'<\|tool_call>|<tool_call\|>|<\|"\|>|</?call:|</?tool_call>')
_TOOL_OPEN = re.compile(r'<\|tool_call>|<call:|<tool_call>')
_TOOL_CLOSE = re.compile(r'<tool_call\|>>?|</call:[\w.-]*>|</tool_call>|/>|\}>')
_PY_CALL = re.compile(r'\b[a-z]+(?:_[a-z]+)+\(\s*\w+\s*=.*?\)(?=\s*(?:$|[.!?]))', re.DOTALL)


def _without_tool_calls(text):
    text = _PY_CALL.sub('', text)
    if not _TOOL_MARK.search(text):
        return text
    opened, closed = _TOOL_OPEN.search(text), _TOOL_CLOSE.search(text)
    if closed and (opened is None or closed.start() < opened.start()):
        text = text[closed.end():]
    elif opened is None:
        return ''
    kept = []
    while (opened := _TOOL_OPEN.search(text)) is not None:
        kept.append(text[:opened.start()])
        closed = _TOOL_CLOSE.search(text, opened.end())
        text = '' if closed is None else text[closed.end():]
    return ''.join(kept) + text


# Stage directions for the voice ("6711 [pause] 4051"). They are read out ("pause", "pos") and
# they broke the digit runs they sat in; a comma is the pause they asked for. Only these words:
# a bracketed tag the model was trained on must still reach it.
_STAGE_DIRECTION = re.compile(r'\s*\[\s*(?:(?:short|long|brief)\s+)?(?:pause|beat|silence|jeda)\s*\]\s*',
                              re.IGNORECASE)
_SENTENCE_PUNCT = '.,!?;:。，！？；：'


def _stage_direction(m):
    before, after = m.string[:m.start()].rstrip(), m.string[m.end():].lstrip()
    if not before or not after or after[0] in _SENTENCE_PUNCT:
        return ''
    return ' ' if before[-1] in _SENTENCE_PUNCT else ', '


def sanitize_markdown(text: str) -> str:
    """Remove markdown formatting and HTML tags from text so TTS doesn't speak markup."""
    text = _without_tool_calls(text)
    text = _STAGE_DIRECTION.sub(_stage_direction, text)
    # code blocks (fenced)
    text = re.sub(r'```[\s\S]*?```', '', text)
    # inline code
    text = re.sub(r'`([^`]*)`', r'\1', text)
    # images
    text = re.sub(r'!\[([^\]]*)\]\([^)]*\)', r'\1', text)
    # links
    text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)
    # headings
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # horizontal rules (must be before bold/italic to avoid partial matching)
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    # bold/italic (order matters: bold before italic)
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', text)
    text = re.sub(r'___(.+?)___', r'\1', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'(?<![a-zA-Z0-9])_(.+?)_(?![a-zA-Z0-9])', r'\1', text)
    # strikethrough
    text = re.sub(r'~~(.+?)~~', r'\1', text)
    # blockquotes
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    # unordered list markers
    text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)
    # ordered list markers: one or two digits. A longer number at the start of a line is
    # content, e.g. a read-back "880101145678. Did we get that right?", which lost its number here.
    text = re.sub(r'^[\s]*\d{1,2}\.\s+', '', text, flags=re.MULTILINE)
    # HTML tags (with or without attributes, self-closing)
    text = re.sub(r'<[^>]+>', '', text)
    return text


def expand_contractions(text: str) -> str:
    pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in contractions.keys()) + r')\b', flags=re.IGNORECASE)
    
    def replace(match):
        word = match.group(0)
        expanded = contractions.get(word.lower())
        if word[0].isupper():
            expanded = expanded.capitalize()
        return expanded
    
    return pattern.sub(replace, text)

def split_alpha_num(text: str) -> str:
    text = re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', text)
    text = re.sub(r'(?<=\d)(?=[A-Za-z])', ' ', text)
    return text