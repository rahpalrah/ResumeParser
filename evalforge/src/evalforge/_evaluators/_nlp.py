# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""Reference-based n-gram overlap evaluators.

These are deterministic, need no model, and are exact implementations of the
published metric definitions:

* BLEU  -- Papineni et al., 2002 (sentence level, +1 smoothing).
* GLEU  -- Wu et al., 2016 (the symmetric min(precision, recall) variant).
* ROUGE -- Lin, 2004 (ROUGE-N and ROUGE-L, precision/recall/F1).
* METEOR-- Banerjee & Lavie, 2005 (exact + stem matching, fragmentation penalty).
* F1    -- token-level F1 as used by SQuAD.
"""

import math
from collections import Counter
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from .._common._text import lcs_length, ngrams, tokenize
from ._common._base import EvaluatorBase

__all__ = [
    "F1ScoreEvaluator",
    "BleuScoreEvaluator",
    "GleuScoreEvaluator",
    "RougeScoreEvaluator",
    "RougeType",
    "MeteorScoreEvaluator",
]


class RougeType(str, Enum):
    """ROUGE variants supported by :class:`RougeScoreEvaluator`."""

    ROUGE_1 = "rouge1"
    ROUGE_2 = "rouge2"
    ROUGE_3 = "rouge3"
    ROUGE_4 = "rouge4"
    ROUGE_5 = "rouge5"
    ROUGE_L = "rougeL"


def _modified_precision(candidate: Sequence[str], reference: Sequence[str], n: int) -> "tuple[int, int]":
    """Clipped n-gram match count and total, per Papineni et al."""
    cand_ngrams = Counter(ngrams(candidate, n))
    ref_ngrams = Counter(ngrams(reference, n))
    total = max(sum(cand_ngrams.values()), 0)
    matched = sum(min(count, ref_ngrams[gram]) for gram, count in cand_ngrams.items())
    return matched, total


class _ReferenceEvaluator(EvaluatorBase):
    """Shared plumbing for response/ground-truth metrics."""

    _singleton_inputs = ["response", "ground_truth"]

    def __init__(self, *, threshold: float = 0.5, **kwargs: Any) -> None:
        super().__init__(threshold=threshold, higher_is_better=True, **kwargs)


class F1ScoreEvaluator(_ReferenceEvaluator):
    """Token-level F1 between a response and a ground truth.

    .. code-block:: python

        F1ScoreEvaluator()(response="Paris is in France", ground_truth="Paris is in France")
        # {'f1_score': 1.0, 'f1_result': 'pass', 'f1_threshold': 0.5}

    :param threshold: Minimum passing F1. Default 0.5.
    """

    id = "evalforge.evaluators.f1_score"

    async def _do_eval(self, eval_input: Dict[str, Any]) -> Dict[str, Any]:
        candidate = tokenize(eval_input.get("response", ""))
        reference = tokenize(eval_input.get("ground_truth", ""))
        common = Counter(candidate) & Counter(reference)
        overlap = sum(common.values())
        if overlap == 0 or not candidate or not reference:
            score = 0.0
        else:
            precision = overlap / len(candidate)
            recall = overlap / len(reference)
            score = 2 * precision * recall / (precision + recall)
        return {
            "f1_score": score,
            "f1_result": self._passed(score),
            "f1_threshold": self._threshold,
        }


class BleuScoreEvaluator(_ReferenceEvaluator):
    """Sentence-level BLEU with add-one smoothing on higher-order n-grams.

    :param threshold: Minimum passing BLEU. Default 0.5.
    :param max_order: Highest n-gram order used. Default 4.
    """

    id = "evalforge.evaluators.bleu_score"

    def __init__(self, *, threshold: float = 0.5, max_order: int = 4, **kwargs: Any) -> None:
        super().__init__(threshold=threshold, **kwargs)
        self._max_order = max(1, max_order)

    async def _do_eval(self, eval_input: Dict[str, Any]) -> Dict[str, Any]:
        candidate = tokenize(eval_input.get("response", ""))
        reference = tokenize(eval_input.get("ground_truth", ""))
        score = 0.0
        if candidate and reference:
            log_precisions: List[float] = []
            for n in range(1, self._max_order + 1):
                matched, total = _modified_precision(candidate, reference, n)
                if total == 0:
                    log_precisions.append(float("-inf"))
                    continue
                # Add-one smoothing above unigrams keeps a single higher-order
                # miss from collapsing the geometric mean to zero.
                if n > 1:
                    precision = (matched + 1.0) / (total + 1.0)
                else:
                    precision = matched / total if matched else 0.0
                log_precisions.append(math.log(precision) if precision > 0 else float("-inf"))

            if all(lp != float("-inf") for lp in log_precisions) and log_precisions:
                geometric_mean = math.exp(sum(log_precisions) / len(log_precisions))
                brevity = 1.0 if len(candidate) > len(reference) else math.exp(
                    1.0 - len(reference) / max(len(candidate), 1)
                )
                score = brevity * geometric_mean
        return {
            "bleu_score": score,
            "bleu_result": self._passed(score),
            "bleu_threshold": self._threshold,
        }


class GleuScoreEvaluator(_ReferenceEvaluator):
    """Google-BLEU: the minimum of n-gram precision and recall over orders 1-4.

    Unlike BLEU it needs no brevity penalty and behaves well on single
    sentences, which makes it the better default for per-row evaluation.

    :param threshold: Minimum passing GLEU. Default 0.5.
    """

    id = "evalforge.evaluators.gleu_score"

    async def _do_eval(self, eval_input: Dict[str, Any]) -> Dict[str, Any]:
        candidate = tokenize(eval_input.get("response", ""))
        reference = tokenize(eval_input.get("ground_truth", ""))
        cand_total = ref_total = matched = 0
        for n in range(1, 5):
            cand_ngrams = Counter(ngrams(candidate, n))
            ref_ngrams = Counter(ngrams(reference, n))
            overlap = cand_ngrams & ref_ngrams
            matched += sum(overlap.values())
            cand_total += sum(cand_ngrams.values())
            ref_total += sum(ref_ngrams.values())
        if cand_total == 0 or ref_total == 0:
            score = 0.0
        else:
            score = min(matched / cand_total, matched / ref_total)
        return {
            "gleu_score": score,
            "gleu_result": self._passed(score),
            "gleu_threshold": self._threshold,
        }


class RougeScoreEvaluator(_ReferenceEvaluator):
    """ROUGE precision, recall and F1.

    .. code-block:: python

        RougeScoreEvaluator(RougeType.ROUGE_L)(
            response="the cat sat on the mat", ground_truth="the cat was on the mat"
        )

    :param rouge_type: Which ROUGE variant to compute.
    :param precision_threshold: Minimum passing precision. Default 0.5.
    :param recall_threshold: Minimum passing recall. Default 0.5.
    :param f1_score_threshold: Minimum passing F1. Default 0.5.
    """

    id = "evalforge.evaluators.rouge_score"

    def __init__(
        self,
        rouge_type: RougeType = RougeType.ROUGE_L,
        *,
        precision_threshold: float = 0.5,
        recall_threshold: float = 0.5,
        f1_score_threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=f1_score_threshold, **kwargs)
        self._rouge_type = RougeType(rouge_type)
        self._precision_threshold = precision_threshold
        self._recall_threshold = recall_threshold
        self._f1_threshold = f1_score_threshold

    async def _do_eval(self, eval_input: Dict[str, Any]) -> Dict[str, Any]:
        candidate = tokenize(eval_input.get("response", ""))
        reference = tokenize(eval_input.get("ground_truth", ""))

        if self._rouge_type is RougeType.ROUGE_L:
            matched = lcs_length(candidate, reference)
            cand_total, ref_total = len(candidate), len(reference)
        else:
            n = int(self._rouge_type.value[-1])
            cand_ngrams = Counter(ngrams(candidate, n))
            ref_ngrams = Counter(ngrams(reference, n))
            matched = sum((cand_ngrams & ref_ngrams).values())
            cand_total = sum(cand_ngrams.values())
            ref_total = sum(ref_ngrams.values())

        precision = matched / cand_total if cand_total else 0.0
        recall = matched / ref_total if ref_total else 0.0
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

        from .._constants import EVALUATION_PASS_FAIL_MAPPING

        return {
            "rouge_precision": precision,
            "rouge_recall": recall,
            "rouge_f1_score": f1,
            "rouge_precision_result": EVALUATION_PASS_FAIL_MAPPING[precision >= self._precision_threshold],
            "rouge_recall_result": EVALUATION_PASS_FAIL_MAPPING[recall >= self._recall_threshold],
            "rouge_f1_score_result": EVALUATION_PASS_FAIL_MAPPING[f1 >= self._f1_threshold],
            "rouge_precision_threshold": self._precision_threshold,
            "rouge_recall_threshold": self._recall_threshold,
            "rouge_f1_score_threshold": self._f1_threshold,
        }


def _stem(token: str) -> str:
    """Light suffix stripping used for METEOR's stem-match stage."""
    for suffix in ("ational", "ization", "iveness", "fulness", "ousness", "ing", "edly", "ed", "es", "s", "ly", "ion"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


class MeteorScoreEvaluator(_ReferenceEvaluator):
    """METEOR: harmonic mean of unigram precision/recall with a fragmentation penalty.

    Matching proceeds in two stages -- exact, then stem -- and the penalty
    grows with the number of contiguous chunks the alignment breaks into, so a
    correctly ordered response outscores a bag of the same words.

    :param alpha: Recall weight in the harmonic mean. Default 0.9.
    :param beta: Fragmentation penalty exponent. Default 3.0.
    :param gamma: Fragmentation penalty weight. Default 0.5.
    :param threshold: Minimum passing METEOR. Default 0.5.
    """

    id = "evalforge.evaluators.meteor_score"

    def __init__(
        self,
        *,
        alpha: float = 0.9,
        beta: float = 3.0,
        gamma: float = 0.5,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, **kwargs)
        self._alpha, self._beta, self._gamma = alpha, beta, gamma

    def _align(self, candidate: Sequence[str], reference: Sequence[str]) -> List["tuple[int, int]"]:
        """Greedy two-stage alignment returning (candidate_index, reference_index) pairs."""
        used_ref: set = set()
        alignment: List["tuple[int, int]"] = []
        for stage in ("exact", "stem"):
            for i, token in enumerate(candidate):
                if any(pair[0] == i for pair in alignment):
                    continue
                for j, ref_token in enumerate(reference):
                    if j in used_ref:
                        continue
                    matches = token == ref_token if stage == "exact" else _stem(token) == _stem(ref_token)
                    if matches:
                        alignment.append((i, j))
                        used_ref.add(j)
                        break
        return sorted(alignment)

    async def _do_eval(self, eval_input: Dict[str, Any]) -> Dict[str, Any]:
        candidate = tokenize(eval_input.get("response", ""))
        reference = tokenize(eval_input.get("ground_truth", ""))
        score = 0.0
        if candidate and reference:
            alignment = self._align(candidate, reference)
            matches = len(alignment)
            if matches:
                precision = matches / len(candidate)
                recall = matches / len(reference)
                f_mean = precision * recall / (self._alpha * precision + (1 - self._alpha) * recall)
                # Count maximal runs that are contiguous in both sequences.
                chunks = 1
                for k in range(1, len(alignment)):
                    prev_c, prev_r = alignment[k - 1]
                    cur_c, cur_r = alignment[k]
                    if not (cur_c == prev_c + 1 and cur_r == prev_r + 1):
                        chunks += 1
                penalty = self._gamma * (chunks / matches) ** self._beta
                score = f_mean * (1 - penalty)
        return {
            "meteor_score": score,
            "meteor_result": self._passed(score),
            "meteor_threshold": self._threshold,
        }
