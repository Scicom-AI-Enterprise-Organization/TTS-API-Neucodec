"""
Normalization rule dictionaries.
Extracted from malaya.text.rules (https://github.com/malaysia-ai/malaya)
Data loaded from JSON files in data/ directory.
"""

import json
import os

_data_dir = os.path.join(os.path.dirname(__file__), 'data')


def _load_json(filename):
    with open(os.path.join(_data_dir, filename), 'r', encoding='utf-8') as f:
        return json.load(f)


rules_normalizer = _load_json('rules_normalizer.json')
rules_compound_normalizer = _load_json('rules_compound_normalizer.json')
rules_normalizer_rev = {v: k for k, v in rules_normalizer.items()}

# Unicode character normalization mapping
normalized_chars = {
    8210: '-', 8211: '-', 8213: '-', 8208: '-', 8212: '-', 9473: '-',
    45: '-', 9644: '-',
    171: '"', 187: '"', 8220: '"', 8221: '"', 168: '"', 34: '"',
    8217: "'", 39: "'", 699: "'", 712: "'", 180: "'", 96: "'",
    8242: "'", 8216: "'", 146: "'",
    818: '_', 95: '_',
    173: '', 127: '',
    10: ' ', 13: ' ', 9: ' ', 8203: ' ', 150: ' ',
}
