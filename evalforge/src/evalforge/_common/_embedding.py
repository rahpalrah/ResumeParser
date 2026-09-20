# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""Text encoders.

The default :class:`HashingEncoder` is a deterministic, dependency-free
signed-hashing embedder. It exists so every EvalForge algorithm -- including the
geometric ones -- runs offline and reproducibly. It is *not* a claim of
semantic parity with a neural encoder: for production accuracy, inject a real
encoder through the :class:`Encoder` protocol, which every algorithm accepts.

Reproducibility: hashing uses ``blake2b`` rather than :func:`hash`, whose seed
is randomised per interpreter process. Two runs on two machines produce
identical vectors.
"""

import hashlib
import math
import struct
from typing import Dict, Iterable, List, Optional, Sequence

from ._linalg import l2_normalize
from ._text import ngrams, tokenize

__all__ = ["Encoder", "HashingEncoder", "get_default_encoder", "set_default_encoder"]


class Encoder:
    """Protocol for pluggable text encoders.

    Implementations must return a list of equal-length float vectors, one per
    input string, and must be deterministic for a given input.
    """

    dimension: int = 0

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed a batch of strings."""
        raise NotImplementedError

    def encode_one(self, text: str) -> List[float]:
        """Embed a single string."""
        return self.encode([text])[0]


def _hash_index(token: str, dimension: int) -> "tuple[int, float]":
    """Map a token to a bucket and a sign via blake2b."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = struct.unpack("<Q", digest)[0]
    bucket = value % dimension
    sign = 1.0 if (value >> 63) & 1 else -1.0
    return bucket, sign


class HashingEncoder(Encoder):
    """Deterministic signed-hashing encoder over word and character n-grams.

    Features are word unigrams/bigrams plus character n-grams (which give the
    representation robustness to morphology and typos). Term frequencies are
    dampened sub-linearly (``1 + log tf``) and optionally re-weighted by inverse
    document frequency learned from a corpus via :meth:`fit`.

    :param dimension: Size of the output vector.
    :param word_ngram_max: Largest word n-gram used as a feature.
    :param char_ngram_range: Inclusive range of character n-gram sizes.
    """

    def __init__(
        self,
        dimension: int = 256,
        *,
        word_ngram_max: int = 2,
        char_ngram_range: "tuple[int, int]" = (3, 5),
        use_char_ngrams: bool = True,
    ) -> None:
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self.dimension = dimension
        self.word_ngram_max = max(1, word_ngram_max)
        self.char_ngram_range = char_ngram_range
        self.use_char_ngrams = use_char_ngrams
        self._idf: Dict[str, float] = {}
        self._default_idf = 1.0

    # -- feature extraction -------------------------------------------------

    def _features(self, text: str) -> List[str]:
        tokens = tokenize(text)
        feats: List[str] = list(tokens)
        for n in range(2, self.word_ngram_max + 1):
            feats.extend("w%d:%s" % (n, "_".join(g)) for g in ngrams(tokens, n))
        if self.use_char_ngrams:
            padded = " " + " ".join(tokens) + " "
            lo, hi = self.char_ngram_range
            for n in range(lo, hi + 1):
                if len(padded) >= n:
                    feats.extend("c%d:%s" % (n, padded[i : i + n]) for i in range(len(padded) - n + 1))
        return feats

    # -- public API ---------------------------------------------------------

    def fit(self, corpus: Iterable[str]) -> "HashingEncoder":
        """Learn IDF weights from ``corpus``. Optional; improves discrimination."""
        docs = [set(self._features(text)) for text in corpus]
        n_docs = len(docs)
        if n_docs == 0:
            return self
        counts: Dict[str, int] = {}
        for doc in docs:
            for feat in doc:
                counts[feat] = counts.get(feat, 0) + 1
        self._idf = {
            feat: math.log((n_docs + 1.0) / (count + 1.0)) + 1.0 for feat, count in counts.items()
        }
        self._default_idf = math.log(n_docs + 1.0) + 1.0
        return self

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed a batch of strings into L2-normalised vectors."""
        return [self._encode_one(t) for t in texts]

    def _encode_one(self, text: str) -> List[float]:
        vector = [0.0] * self.dimension
        if not text:
            return vector
        counts: Dict[str, int] = {}
        for feat in self._features(text):
            counts[feat] = counts.get(feat, 0) + 1
        for feat, tf in counts.items():
            weight = (1.0 + math.log(tf)) * self._idf.get(feat, self._default_idf if self._idf else 1.0)
            bucket, sign = _hash_index(feat, self.dimension)
            vector[bucket] += sign * weight
        return l2_normalize(vector)


_DEFAULT_ENCODER: Optional[Encoder] = None


def get_default_encoder() -> Encoder:
    """Return the process-wide default encoder, creating it on first use."""
    global _DEFAULT_ENCODER
    if _DEFAULT_ENCODER is None:
        _DEFAULT_ENCODER = HashingEncoder()
    return _DEFAULT_ENCODER


def set_default_encoder(encoder: Optional[Encoder]) -> None:
    """Override the process-wide default encoder (pass ``None`` to reset)."""
    global _DEFAULT_ENCODER
    _DEFAULT_ENCODER = encoder
