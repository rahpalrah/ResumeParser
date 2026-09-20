# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""Judge Item-Response Calibration (JIRC).

Problem
-------
Panels of LLM judges are usually combined by averaging, or by majority vote.
Both treat every judge as equally trustworthy, which they are not. Judges
differ in **severity** (one marks everything a 3, another everything a 5), in
**discrimination** (one separates good from bad outputs, another returns nearly
the same score regardless), and in **noise**. Averaging lets a severe judge
depress an entire leaderboard and lets an uninformative judge dilute the
signal, and it offers no way to tell which case you are in.

Method
------
Fit a graded item-response model to the judge-by-item score matrix. Each
observed score is modelled as

    x_ij = b_j + a_j * theta_i + noise(0, sigma_j^2)

where ``theta_i`` is the latent quality of item ``i`` and each judge ``j`` has a
severity ``b_j``, a discrimination ``a_j``, and a noise level ``sigma_j``.
Parameters are fitted by alternating conditional estimation:

* **Judge step** -- with the current ``theta`` held fixed, each judge's
  ``(a_j, b_j)`` is a least-squares regression of its scores on ``theta``, and
  ``sigma_j^2`` is its residual variance.
* **Item step** -- with judge parameters held fixed, ``theta_i`` is the mean of
  the Gaussian posterior under a standard normal prior:
  ``theta_i = sum_j a_j (x_ij - b_j) / sigma_j^2 / (sum_j a_j^2 / sigma_j^2 + 1)``.
  Each judge contributes in **inverse proportion to its own noise**, so a noisy
  judge is down-weighted automatically rather than by hand.

The model is invariant to shifting and scaling ``theta``, and to flipping its
sign together with every ``a_j``. Fitting is therefore anchored each round:
``theta`` is standardised, and the sign is fixed so that the average
discrimination is positive (higher ``theta`` means better output, not worse).

The posterior variance of ``theta_i`` falls out of the same expression. It is
small when discriminating, low-noise judges agree and large when they do not,
which makes it a principled basis for routing an item to a human -- and it is
the input the conformal controller in :mod:`evalforge.novel._conformal` uses.

What is novel
-------------
Item-response theory is long established in psychometrics, and LLM-as-judge
ensembling is established practice. The contribution is the combination: latent
quality estimation over an **LLM jury**, with per-judge noise estimated from
the panel itself rather than assumed, sign/scale anchoring that makes the fit
reproducible without gold labels, and a posterior variance exported as a
calibrated confidence signal for selective human review.
"""

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .._common._linalg import clamp, mean, stdev, variance

__all__ = ["JudgeCalibration", "JudgeItemResponseCalibrator"]

_EPS = 1e-9


class JudgeCalibration(dict):
    """Fitted calibration.

    Keys: ``latent_quality``, ``posterior_sd``, ``judges`` (per-judge
    ``discrimination``/``severity``/``noise``/``reliability``),
    ``consensus_scores``, ``iterations``, ``converged``.
    """

    @property
    def latent_quality(self) -> List[float]:
        """Posterior mean latent quality per item, standardised."""
        return list(self.get("latent_quality", []))

    @property
    def consensus_scores(self) -> List[float]:
        """Latent quality mapped back onto the observed score scale."""
        return list(self.get("consensus_scores", []))

    def most_reliable_judge(self) -> Optional[str]:
        """Name of the judge with the highest discrimination-to-noise ratio."""
        judges: Mapping[str, Mapping[str, float]] = self.get("judges", {})
        if not judges:
            return None
        return max(judges, key=lambda name: judges[name]["reliability"])


class JudgeItemResponseCalibrator:
    """Fits the graded item-response model to a judge-by-item score matrix.

    :param max_iterations: Maximum alternating passes.
    :param tolerance: Convergence tolerance on the latent quality vector.
    :param prior_precision: Precision of the standard normal prior on latent
        quality. Larger values shrink estimates harder toward the mean.
    """

    def __init__(
        self,
        *,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        prior_precision: float = 1.0,
    ) -> None:
        self._max_iterations = max(1, max_iterations)
        self._tolerance = tolerance
        self._prior_precision = max(_EPS, prior_precision)

    def fit(self, scores: Mapping[str, Sequence[float]]) -> JudgeCalibration:
        """Fit the model.

        :param scores: ``{judge_name: [score_per_item, ...]}``. Every judge must
            score every item, in the same order.
        :return: The fitted :class:`JudgeCalibration`.
        :raises ValueError: If fewer than two judges are supplied, or the score
            vectors have differing lengths.
        """
        names = list(scores)
        if len(names) < 2:
            raise ValueError("Judge calibration requires at least two judges.")
        matrix = [[float(v) for v in scores[name]] for name in names]
        n_items = len(matrix[0])
        if any(len(row) != n_items for row in matrix):
            raise ValueError("Every judge must score the same number of items.")
        if n_items < 2:
            raise ValueError("Judge calibration requires at least two items.")

        observed_mean = mean([v for row in matrix for v in row])
        observed_sd = stdev([v for row in matrix for v in row]) or 1.0

        # Initialise latent quality from the standardised item means.
        theta = [
            (mean([row[i] for row in matrix]) - observed_mean) / observed_sd for i in range(n_items)
        ]
        theta = self._standardise(theta)

        discrimination = [1.0] * len(names)
        severity = [0.0] * len(names)
        noise = [1.0] * len(names)
        converged = False
        iterations = 0

        for iterations in range(1, self._max_iterations + 1):
            # --- judge step: regress each judge's scores on theta ---------
            theta_mean = mean(theta)
            theta_var = variance(theta)
            for j, row in enumerate(matrix):
                if theta_var <= _EPS:
                    a = 0.0
                else:
                    covariance = mean([(row[i] - mean(row)) * (theta[i] - theta_mean) for i in range(n_items)])
                    a = covariance / theta_var
                b = mean(row) - a * theta_mean
                residuals = [row[i] - (b + a * theta[i]) for i in range(n_items)]
                discrimination[j] = a
                severity[j] = b
                # Floor the noise so a judge that happens to fit perfectly on a
                # small panel cannot take infinite weight.
                noise[j] = max(variance(residuals), 1e-4)

            # --- item step: Gaussian posterior mean under a N(0,1) prior ---
            updated: List[float] = []
            for i in range(n_items):
                precision = self._prior_precision
                weighted = 0.0
                for j, row in enumerate(matrix):
                    a, b, s2 = discrimination[j], severity[j], noise[j]
                    precision += (a * a) / s2
                    weighted += a * (row[i] - b) / s2
                updated.append(weighted / precision if precision > _EPS else 0.0)

            updated = self._standardise(updated)
            # Anchor the sign: higher latent quality must mean higher scores.
            if mean(discrimination) < 0:
                updated = [-t for t in updated]

            shift = max((abs(updated[i] - theta[i]) for i in range(n_items)), default=0.0)
            theta = updated
            if shift < self._tolerance:
                converged = True
                break

        # Posterior standard deviation from the final precision.
        posterior_sd: List[float] = []
        for _ in range(n_items):
            precision = self._prior_precision + sum(
                (discrimination[j] ** 2) / noise[j] for j in range(len(names))
            )
            posterior_sd.append(math.sqrt(1.0 / precision) if precision > _EPS else float("inf"))

        # Map latent quality back onto the observed scale for reporting.
        consensus = [observed_mean + observed_sd * t for t in theta]

        judges = {
            name: {
                "discrimination": discrimination[j],
                "severity": severity[j],
                "noise": noise[j],
                # Discrimination per unit of noise: how much usable signal the
                # judge contributes, which is exactly its posterior weight.
                "reliability": abs(discrimination[j]) / math.sqrt(noise[j]),
            }
            for j, name in enumerate(names)
        }

        return JudgeCalibration(
            latent_quality=theta,
            posterior_sd=posterior_sd,
            consensus_scores=consensus,
            judges=judges,
            iterations=iterations,
            converged=converged,
            observed_mean=observed_mean,
            observed_sd=observed_sd,
        )

    @staticmethod
    def _standardise(values: Sequence[float]) -> List[float]:
        """Centre and scale to unit variance, resolving the model's scale invariance."""
        mu = mean(values)
        sd = stdev(values)
        if sd <= _EPS:
            return [0.0] * len(values)
        return [(v - mu) / sd for v in values]
