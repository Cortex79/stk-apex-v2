"""Magyar-angol BPE tokenizáló (24 576 token).

Mért fertilitás: HU 1,96 tok/szó, EN 1,21 — a GPT-4 cl100k-hoz képest
2,18× tömörebb magyaron, a szótár negyedével.
"""
from .stk_tokenizer import STKTokenizer

__all__ = ["STKTokenizer"]
