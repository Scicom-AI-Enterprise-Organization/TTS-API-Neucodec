import re
import json

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