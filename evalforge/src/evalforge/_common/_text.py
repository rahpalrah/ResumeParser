# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""Tokenisation, sentence splitting and claim segmentation.

Deliberately dependency free: no NLTK, no spaCy. The tokenizer is a Unicode
aware regex tokenizer and the sentence splitter handles the abbreviation cases
that matter for evaluation corpora (titles, initials, decimals, ellipses).
"""

import re
import unicodedata
from typing import List, Sequence, Set

__all__ = [
    "normalize",
    "tokenize",
    "ngrams",
    "split_sentences",
    "split_claims",
    "STOPWORDS",
    "content_tokens",
    "jaccard",
    "lcs_length",
]

_WORD_RE = re.compile(r"[\w'’-]+", re.UNICODE)
_SENT_END_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+")
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
    "inc", "ltd", "co", "corp", "dept", "fig", "no", "approx", "ph.d", "u.s",
}

#: Minimal English stop list used for content-word overlap metrics.
STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "so", "because",
    "as", "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from", "up",
    "down", "in", "out", "on", "off", "over", "under", "again", "further", "once",
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did",
    "doing", "have", "has", "had", "having", "i", "me", "my", "we", "our", "you",
    "your", "he", "him", "his", "she", "her", "it", "its", "they", "them", "their",
    "this", "that", "these", "those", "what", "which", "who", "whom", "how", "why",
    "when", "where", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "too", "very", "can",
    "will", "just", "should", "would", "could", "may", "might", "must", "s", "t",
}


def normalize(text: str, *, lower: bool = True, strip_accents: bool = False) -> str:
    """Normalise whitespace and (optionally) case and accents."""
    if text is None:
        return ""
    out = unicodedata.normalize("NFKC", str(text))
    if strip_accents:
        out = "".join(c for c in unicodedata.normalize("NFD", out) if not unicodedata.combining(c))
    out = re.sub(r"\s+", " ", out).strip()
    return out.lower() if lower else out


def tokenize(text: str, *, lower: bool = True) -> List[str]:
    """Split text into word tokens."""
    if not text:
        return []
    source = str(text).lower() if lower else str(text)
    return _WORD_RE.findall(unicodedata.normalize("NFKC", source))


def content_tokens(text: str) -> List[str]:
    """Tokens with stopwords and single characters removed."""
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def ngrams(tokens: Sequence[str], n: int) -> List[tuple]:
    """All contiguous ``n``-grams of ``tokens``."""
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def split_sentences(text: str) -> List[str]:
    """Split ``text`` into sentences, guarding common abbreviations."""
    if not text:
        return []
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    if not cleaned:
        return []
    pieces = _SENT_END_RE.split(cleaned)
    merged: List[str] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        if merged:
            prev = merged[-1]
            tail = prev.rstrip(".").split()
            last_word = tail[-1].lower().strip("\"'([") if tail else ""
            # Re-attach if the previous fragment ended on an abbreviation or a
            # single initial such as "J." in "J. Smith".
            if last_word in _ABBREVIATIONS or (len(last_word) == 1 and last_word.isalpha()):
                merged[-1] = prev + " " + piece
                continue
        merged.append(piece)
    return merged


def split_claims(text: str, *, max_claims: int = 64) -> List[str]:
    """Segment a response into atomic, independently verifiable claims.

    Sentences are the base unit; sentences that coordinate several independent
    predications (``A, and B``; ``A; B``) are split further, because attribution
    is only meaningful at the level of a single assertion.
    """
    claims: List[str] = []
    for sentence in split_sentences(text):
        parts = re.split(r";\s+|\s+,?\s*(?:and also|and then)\s+", sentence)
        for part in parts:
            part = part.strip(" ;,")
            if len(tokenize(part)) >= 3:
                claims.append(part)
            elif part and not claims:
                claims.append(part)
        if len(claims) >= max_claims:
            break
    return claims[:max_claims]


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    """Jaccard index of two token sequences treated as sets."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence (used by ROUGE-L)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        cur = [0]
        for j, token_b in enumerate(b):
            if token_a == token_b:
                cur.append(prev[j] + 1)
            else:
                cur.append(max(prev[j + 1], cur[j]))
        prev = cur
    return prev[len(b)]
