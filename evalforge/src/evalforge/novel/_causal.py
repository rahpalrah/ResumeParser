# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""Causal Ablation Groundedness with Provenance certificates (CAG-P).

Problem
-------
A groundedness score of 3.4 tells a developer that something is wrong but not
what, where, or whether it matters. To act on a groundedness failure you need
to know *which claim* was unsupported and *which piece of context* was supposed
to support it. Correlational scoring cannot answer that: high similarity
between a claim and a passage does not establish that the passage is what makes
the claim true, since a contradiction is maximally similar to the text it
contradicts.

Method
------
For every atomic claim in the response, interrogate the context causally.

1. Segment the response into atomic claims and the context into spans.
2. **Necessity** (leave-one-out ablation): remove span ``s`` from the context
   and measure how far the claim's support falls. A span whose removal costs
   nothing was not carrying the claim, whatever its similarity suggests.
3. **Sufficiency**: score the claim against span ``s`` alone. A span that
   supports the claim by itself is a standalone witness.
4. **Minimal witness** (lazy greedy submodular selection): grow a span set,
   always adding the span with the largest marginal support gain, until the
   subset reaches a fraction ``tau`` of the full-context support. Support is
   monotone and has diminishing returns in the span set, so greedy selection
   carries the standard ``(1 - 1/e)`` guarantee against the best subset of the
   same size. The result is the smallest set of passages a human needs to read
   to check the claim.
5. **Corruption sensitivity**: perturb the specific commitments inside the
   witness -- its numerals and proper nouns -- and re-score. If support survives
   the corruption, the claim was never relying on those specifics, and the
   apparent grounding is incidental token overlap rather than evidence.

The per-claim record of these four quantities is the claim's **provenance
certificate**. Claims whose support falls below a floor are returned as the
**phantom set**: the response's unsupported assertions, named individually.

What is novel
-------------
Claim decomposition and NLI-style entailment checking against retrieved context
are established, as is attention- or gradient-based attribution. The
contribution here is (a) scoring each claim by *causal* ablation of context
spans rather than by correlation, (b) separating necessity from sufficiency so
that redundant evidence and single points of failure are distinguishable, (c)
extracting a minimal witness set with a greedy submodular procedure that comes
with an approximation guarantee, and (d) the corruption test, which catches
grounding that is real-looking token overlap rather than evidence -- the
failure mode that defeats similarity-based groundedness scoring.
"""

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .._common._embedding import Encoder, get_default_encoder
from .._common._heuristics import evidence_recall
from .._common._linalg import clamp, cosine, mean
from .._common._text import split_claims, split_sentences
from .._evaluators._common._base import EvaluatorBase

__all__ = ["CausalAblationGroundedness", "CausalGroundednessEvaluator", "ProvenanceCertificate"]

_NUMERIC_RE = re.compile(r"\d+(?:\.\d+)?")


class ProvenanceCertificate(dict):
    """Per-claim attribution record.

    Keys: ``claim``, ``support``, ``witness`` (span indices), ``necessity``,
    ``sufficiency``, ``corruption_sensitivity``, ``is_phantom``.
    """

    @property
    def is_phantom(self) -> bool:
        """Whether the claim failed to reach the support floor."""
        return bool(self.get("is_phantom"))


class CausalAblationGroundedness:
    """Computes per-claim provenance certificates for a response.

    :param encoder: Text encoder. Defaults to the process encoder.
    :param witness_fraction: Fraction ``tau`` of full-context support a witness
        set must reach before greedy selection stops.
    :param phantom_floor: Support below which a claim is reported as a phantom.
    """

    def __init__(
        self,
        *,
        encoder: Optional[Encoder] = None,
        witness_fraction: float = 0.9,
        phantom_floor: float = 0.4,
    ) -> None:
        self._encoder = encoder or get_default_encoder()
        self._witness_fraction = clamp(witness_fraction, 0.1, 1.0)
        self._phantom_floor = clamp(phantom_floor, 0.0, 1.0)

    # -- support function ---------------------------------------------------

    def _support(self, claim: str, spans: Sequence[str]) -> float:
        """Support for ``claim`` from a set of context spans.

        Monotone and submodular in the span set: adding a span never lowers
        support, and the gain from a span shrinks as the set grows, which is
        what licenses the greedy witness search.
        """
        if not spans:
            return 0.0
        pooled = " ".join(spans)
        evidence = evidence_recall(pooled, claim)
        semantic = max(
            (max(0.0, cosine(self._encoder.encode_one(claim), self._encoder.encode_one(span))) for span in spans),
            default=0.0,
        )
        return clamp(0.7 * evidence + 0.3 * semantic, 0.0, 1.0)

    # -- witness extraction -------------------------------------------------

    def _minimal_witness(self, claim: str, spans: Sequence[str], target: float) -> List[int]:
        """Lazy-greedy selection of a minimal supporting span set.

        Maintains an upper bound on each span's marginal gain and only re-scores
        a span when its stale bound is still the best on offer, which is the
        standard lazy-greedy optimisation and avoids rescoring every span on
        every round.
        """
        selected: List[int] = []
        chosen: List[str] = []
        current = 0.0
        # (upper_bound_on_gain, span_index); bounds start at the singleton gain.
        bounds: List[Tuple[float, int]] = [
            (self._support(claim, [span]), index) for index, span in enumerate(spans)
        ]
        bounds.sort(reverse=True)

        while bounds and current < target:
            gain, index = bounds.pop(0)
            fresh = self._support(claim, chosen + [spans[index]]) - current
            if not bounds or fresh >= bounds[0][0]:
                if fresh <= 1e-9:
                    break
                selected.append(index)
                chosen.append(spans[index])
                current += fresh
            else:
                # Stale bound: reinsert with its refreshed value and retry.
                bounds.append((fresh, index))
                bounds.sort(reverse=True)
        return sorted(selected)

    # -- corruption test ----------------------------------------------------

    @staticmethod
    def _corrupt(text: str) -> str:
        """Perturb the factual commitments of a span.

        Numerals are shifted and capitalised words are masked, so the span keeps
        its shape and topic but loses the specifics a claim would rely on.
        """
        def shift(match: "re.Match[str]") -> str:
            value = match.group(0)
            try:
                return str(int(value) + 7) if "." not in value else f"{float(value) + 7.0:.1f}"
            except ValueError:
                return value

        corrupted = _NUMERIC_RE.sub(shift, text)
        return re.sub(r"\b[A-Z][a-z]{2,}\b", "Redacted", corrupted)

    # -- public API ---------------------------------------------------------

    def analyze(self, response: str, context: str) -> Dict[str, Any]:
        """Build provenance certificates for every claim in ``response``.

        :param response: The model output under evaluation.
        :param context: The grounding material the response should rely on.
        :return: Aggregate metrics plus one certificate per claim.
        """
        claims = split_claims(response) or ([response] if response else [])
        spans = split_sentences(context) or ([context] if context else [])

        if not claims:
            return {
                "causal_groundedness": 0.0,
                "attributable_support_ratio": 0.0,
                "phantom_claims": [],
                "certificates": [],
                "spans": spans,
                "reason": "The response contained no evaluable claim.",
            }
        if not spans:
            certificates = [
                ProvenanceCertificate(
                    claim=claim, support=0.0, witness=[], necessity={}, sufficiency={},
                    corruption_sensitivity=0.0, is_phantom=True,
                )
                for claim in claims
            ]
            return {
                "causal_groundedness": 0.0,
                "attributable_support_ratio": 0.0,
                "phantom_claims": list(claims),
                "certificates": certificates,
                "spans": [],
                "reason": "No context was supplied, so no claim can be attributed.",
            }

        certificates: List[ProvenanceCertificate] = []
        for claim in claims:
            full = self._support(claim, spans)
            witness = self._minimal_witness(claim, spans, self._witness_fraction * full)

            necessity = {
                index: round(full - self._support(claim, [s for i, s in enumerate(spans) if i != index]), 6)
                for index in (witness or range(len(spans)))
            }
            sufficiency = {
                index: round(self._support(claim, [spans[index]]), 6)
                for index in (witness or range(len(spans)))
            }

            if witness:
                corrupted_spans = [
                    self._corrupt(span) if i in set(witness) else span for i, span in enumerate(spans)
                ]
                corrupted_support = self._support(claim, corrupted_spans)
                sensitivity = clamp((full - corrupted_support) / full, 0.0, 1.0) if full > 1e-9 else 0.0
            else:
                sensitivity = 0.0

            certificates.append(
                ProvenanceCertificate(
                    claim=claim,
                    support=round(full, 6),
                    witness=witness,
                    witness_text=[spans[i] for i in witness],
                    necessity=necessity,
                    sufficiency=sufficiency,
                    corruption_sensitivity=round(sensitivity, 6),
                    is_phantom=full < self._phantom_floor,
                )
            )

        supports = [c["support"] for c in certificates]
        phantoms = [c["claim"] for c in certificates if c.is_phantom]
        # Weakest-link aggregation: one fabricated claim compromises the answer.
        aggregate = clamp(0.6 * mean(supports) + 0.4 * min(supports), 0.0, 1.0)

        return {
            "causal_groundedness": aggregate,
            "attributable_support_ratio": mean(supports),
            "phantom_claims": phantoms,
            "phantom_rate": len(phantoms) / len(certificates),
            "mean_corruption_sensitivity": mean([c["corruption_sensitivity"] for c in certificates]),
            "certificates": certificates,
            "spans": spans,
            "reason": self._explain(aggregate, phantoms, certificates),
        }

    @staticmethod
    def _explain(aggregate: float, phantoms: Sequence[str], certificates: Sequence[ProvenanceCertificate]) -> str:
        """Report which claims failed and why, not just the aggregate."""
        if not phantoms:
            incidental = [
                c["claim"] for c in certificates
                if c["corruption_sensitivity"] < 0.1 and c["support"] > 0.5
            ]
            if incidental:
                return (
                    f"All claims are supported (score {aggregate:.2f}), but {len(incidental)} "
                    "survived corruption of their witness spans, so their grounding may be "
                    f"incidental overlap rather than evidence: '{incidental[0][:60]}...'"
                )
            return f"Every claim is attributable to the context (score {aggregate:.2f})."
        listed = "; ".join(f"'{p[:60]}'" for p in phantoms[:3])
        return (
            f"{len(phantoms)} of {len(certificates)} claims are unsupported by the context "
            f"(score {aggregate:.2f}): {listed}"
        )


class CausalGroundednessEvaluator(EvaluatorBase):
    """Evaluator wrapper around :class:`CausalAblationGroundedness`.

    :param threshold: Minimum passing score.
    :param phantom_floor: Support below which a claim is a phantom.
    """

    id = "evalforge.evaluators.causal_groundedness"
    _singleton_inputs = ["response", "context"]

    def __init__(
        self,
        *,
        threshold: float = 0.6,
        phantom_floor: float = 0.4,
        encoder: Optional[Encoder] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, higher_is_better=True, **kwargs)
        self._algorithm = CausalAblationGroundedness(encoder=encoder, phantom_floor=phantom_floor)

    async def _do_eval(self, eval_input: Dict[str, Any]) -> Dict[str, Any]:
        result = self._algorithm.analyze(
            str(eval_input.get("response", "")), str(eval_input.get("context", ""))
        )
        score = result["causal_groundedness"]
        return {
            "causal_groundedness": score,
            "causal_groundedness_reason": result["reason"],
            "causal_groundedness_result": self._passed(score),
            "causal_groundedness_threshold": self._threshold,
            "phantom_claims": result["phantom_claims"],
            "phantom_rate": result.get("phantom_rate", 0.0),
            "attributable_support_ratio": result["attributable_support_ratio"],
            "provenance_certificates": [dict(c) for c in result["certificates"]],
        }
