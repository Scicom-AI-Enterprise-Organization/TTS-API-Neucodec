"""
Malaysian text normalizer.
Extracted from malaya (https://github.com/malaysia-ai/malaya) for independent maintenance.
"""

from .normalizer import Normalizer, load
from .normalization import to_cardinal, to_ordinal

__all__ = ['Normalizer', 'load', 'to_cardinal', 'to_ordinal']
