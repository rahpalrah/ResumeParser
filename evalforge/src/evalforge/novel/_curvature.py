# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""Semantic Curvature Probe (SCP).

Problem
-------
A model that *knows* something answers the same question the same way however
you phrase it. A model that is confabulating produces answers that swing with
irrelevant changes in phrasing. Existing self-consistency detectors sample the
model several times and measure the **variance** of the answers. Variance is a
first-order quantity: it says the answers are spread out, but not whether the
spread is an orderly response to the input changing or a sign that the model is
on unstable ground.

Method
------
Probe the *geometry* of the map from question to answer, not just its spread.

1. Build a **perturbation ladder**: phrasings ``q_0, q_1, q_2, ...`` of one
   question, where each rung applies exactly one more meaning-preserving
   operator than the last. The rungs are therefore evenly spaced by
   construction.
2. Obtain the model's answer at each rung and embed both sides, giving input
   points ``x_k`` and output points ``y_k``.
3. Along each consecutive triple, form the **discrete second difference**
   ``D2y = y_{k+1} - 2 y_k + y_{k-1}``. For a locally affine, stable map the
   answer at the middle rung sits at the midpoint of its neighbours and
   ``||D2y||`` vanishes. Normalise it by the local first-difference scale --
   ``||y_{k+1} - y_k|| + ||y_k - y_{k-1}||`` -- giving the discrete analogue of
   ``|y''| / |y'|``. The result is dimensionless and lands in ``[0, 1]``: zero
   when successive answers advance in a consistent direction, one when the
   answer reverses direction completely at that rung.

   Normalising by the *input* second difference instead is the obvious choice
   and is wrong in practice: meaning-preserving rephrasings barely move the
   query embedding, so that denominator is dominated by numerical noise and the
   ratio explodes for stable and unstable models alike.
4. Report curvature alongside the first-order **Lipschitz ratio**
   ``||y_k - y_0|| / ||x_k - x_0||`` and the raw dispersion of the answers.

Curvature and the Lipschitz ratio answer different questions. A model that
rewrites its answer in proportion to the rephrasing has a high Lipschitz ratio
but low curvature: it is sensitive but *orderly*. A model that returns the same
answer twice and then something unrelated has a modest Lipschitz ratio and high
curvature. The second pattern is the one that indicates fabrication.

Encoder sensitivity
-------------------
Curvature is a geometric statement about the embedding space, so its
discriminative power depends on the encoder more than the other metrics here
do. The built-in hashing encoder sends unrelated n-grams to near-orthogonal
directions, so a progressively *elaborated* answer registers more curvature
than it should. Ordering is preserved, but for this probe in particular,
injecting a real sentence encoder materially sharpens the separation between
orderly sensitivity and genuine instability.

What is novel
-------------
Self-consistency sampling, paraphrase-invariance testing and embedding-variance
hallucination scores are established. The contribution here is measuring the
**second-order** structure of the phrasing-to-answer map along an evenly spaced
perturbation ladder, which separates orderly sensitivity from instability -- a
distinction first-order variance cannot make -- and reporting the two
components separately so the failure mode is identifiable, not just flagged.
"""

import math
from typing import Any, Callable, Dict, List, Optional, Sequence

from .._common._embedding import Encoder, get_default_encoder
from .._common._linalg import centroid, clamp, cosine, mean, norm, sub
from .._evaluators._common._base import EvaluatorBase
from ._paraphrase import perturbation_ladder

__all__ = ["SemanticCurvatureProbe", "SemanticCurvatureEvaluator"]

_EPS = 1e-9


class SemanticCurvatureProbe:
    """Estimates the curvature of a model's phrasing-to-answer map.

    :param encoder: Text encoder. Defaults to the process encoder.
    :param rungs: Number of ladder rungs. At least three are needed for a
        second difference.
    """

    def __init__(self, *, encoder: Optional[Encoder] = None, rungs: int = 4) -> None:
        self._encoder = encoder or get_default_encoder()
        self._rungs = max(3, rungs)

    def probe(
        self,
        query: str,
        generate: Callable[[str], str],
    ) -> Dict[str, Any]:
        """Run the ladder against a generator callable.

        :param query: The question to probe.
        :param generate: Callable mapping a phrasing to the model's answer.
        """
        phrasings = perturbation_ladder(query, self._rungs)
        responses = [generate(phrasing) for phrasing in phrasings]
        return self.analyze(phrasings, responses)

    def analyze(self, phrasings: Sequence[str], responses: Sequence[str]) -> Dict[str, Any]:
        """Compute the decomposition from pre-generated ladder responses.

        :param phrasings: The ladder phrasings, evenly spaced by construction.
        :param responses: The model's answer to each phrasing, same order.
        """
        if len(phrasings) != len(responses):
            raise ValueError("phrasings and responses must be the same length")
        if len(responses) < 3:
            return {
                "semantic_curvature": 0.0,
                "lipschitz_ratio": 0.0,
                "answer_dispersion": 0.0,
                "instability_index": 0.0,
                "rungs": len(responses),
                "reason": "At least three ladder rungs are required for a second difference.",
            }

        x = self._encoder.encode(list(phrasings))
        y = self._encoder.encode(list(responses))

        # --- second-order term -------------------------------------------
        curvatures: List[float] = []
        for k in range(1, len(y) - 1):
            d2y = norm([y[k + 1][i] - 2 * y[k][i] + y[k - 1][i] for i in range(len(y[k]))])
            first_order = norm(sub(y[k + 1], y[k])) + norm(sub(y[k], y[k - 1]))
            if first_order <= _EPS:
                # Successive answers are identical: no motion, hence no bend.
                curvatures.append(0.0)
            else:
                curvatures.append(clamp(d2y / first_order, 0.0, 1.0))
        curvature = mean(curvatures)

        # --- first-order term ---------------------------------------------
        ratios: List[float] = []
        for k in range(1, len(y)):
            input_step = norm(sub(x[k], x[0]))
            output_step = norm(sub(y[k], y[0]))
            if input_step > _EPS:
                ratios.append(output_step / input_step)
        lipschitz = mean(ratios)

        # --- zeroth-order term --------------------------------------------
        center = centroid(y)
        dispersion = mean([1.0 - max(0.0, cosine(v, center)) for v in y])

        # Curvature dominates the instability index: orderly sensitivity (high
        # Lipschitz, low curvature) should not be flagged as fabrication.
        instability = clamp(
            0.60 * curvature
            + 0.15 * clamp(lipschitz / 2.0, 0.0, 1.0)
            + 0.25 * clamp(dispersion / 0.5, 0.0, 1.0),
            0.0,
            1.0,
        )

        return {
            "semantic_curvature": curvature,
            "lipschitz_ratio": lipschitz,
            "answer_dispersion": dispersion,
            "instability_index": instability,
            "rungs": len(responses),
            "per_triple_curvature": curvatures,
            "reason": self._explain(curvature, lipschitz, dispersion, instability),
        }

    @staticmethod
    def _explain(curvature: float, lipschitz: float, dispersion: float, instability: float) -> str:
        """Name the failure mode rather than just reporting the number."""
        if instability < 0.25:
            return (
                f"Answers are stable under rephrasing (curvature {curvature:.2f}, "
                f"dispersion {dispersion:.2f})."
            )
        if curvature < 0.45 and lipschitz > 0.5:
            return (
                f"Answers change with phrasing but in an orderly way (curvature {curvature:.2f}, "
                f"Lipschitz {lipschitz:.2f}): sensitive, not unstable."
            )
        return (
            f"Answers are unstable under meaning-preserving rephrasing (curvature "
            f"{curvature:.2f}, dispersion {dispersion:.2f}). The model is likely on "
            "ground it does not actually know."
        )


class SemanticCurvatureEvaluator(EvaluatorBase):
    """Evaluator wrapper around :class:`SemanticCurvatureProbe`.

    Accepts a pre-generated ladder via ``ladder_phrasings``/``ladder_responses``,
    which keeps the evaluator free of any model-calling responsibility.

    :param threshold: Maximum passing instability index. Lower is better.
    """

    id = "evalforge.evaluators.semantic_curvature"
    _singleton_inputs = ["ladder_phrasings", "ladder_responses"]

    def __init__(self, *, threshold: float = 0.5, encoder: Optional[Encoder] = None, **kwargs: Any) -> None:
        super().__init__(threshold=threshold, higher_is_better=False, **kwargs)
        self._probe = SemanticCurvatureProbe(encoder=encoder)

    async def _do_eval(self, eval_input: Dict[str, Any]) -> Dict[str, Any]:
        result = self._probe.analyze(
            list(eval_input["ladder_phrasings"]), list(eval_input["ladder_responses"])
        )
        score = result["instability_index"]
        return {
            "semantic_curvature": result["semantic_curvature"],
            "lipschitz_ratio": result["lipschitz_ratio"],
            "answer_dispersion": result["answer_dispersion"],
            "instability_index": score,
            "instability_reason": result["reason"],
            "instability_result": self._passed(score),
            "instability_threshold": self._threshold,
        }
