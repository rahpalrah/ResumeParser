# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""Conformal Risk-Calibrated Evaluation (CRCE).

Problem
-------
Automated evaluation is used to decide whether a release ships, but an
evaluation score carries no guarantee. "Mean groundedness 4.1" says nothing
about how many bad outputs slipped past the judge, and a team that routes
uncertain rows to human review has no principled way to pick the cut-off or to
state what the remaining automated decisions are worth.

Method
------
Treat automated acceptance as a prediction problem with a distribution-free
guarantee, using a labelled calibration set of ``n`` rows.

1. Assign each row a **nonconformity score** -- how unusual or hard to judge it
   is. The composite used here fuses three signals that are available from the
   rest of this package:

   * judge disagreement (the posterior standard deviation from
     :mod:`evalforge.novel._irt`),
   * the margin between the consensus score and the pass threshold, since rows
     near the boundary are where judges are least trustworthy,
   * answer instability under rephrasing (from :mod:`evalforge.novel._curvature`).

2. Define the risk of a cut-off ``lambda`` as the fraction of rows the system
   would **auto-accept** that are in fact failures:
   ``R(lambda) = (1/n) * #{i : s_i <= lambda and y_i is a failure}``.
   ``R`` is non-decreasing in ``lambda``, which is what makes a one-dimensional
   search valid.

3. Choose the **largest** ``lambda`` whose risk still satisfies a finite-sample
   upper confidence bound, ``R_hat(lambda) + sqrt(log(1/delta) / (2n)) <=
   alpha`` (Hoeffding). Largest, because every smaller cut-off sends more rows
   to humans than necessary.

The result is an operating point with a statement attached: *with probability at
least 1 - delta, at most alpha of all rows are failures that were accepted
automatically* -- assuming only that calibration and production rows are
exchangeable. No assumption is made about the judge, the model, or the score
distribution. Rows above ``lambda`` are abstentions, routed to human review.

What is novel
-------------
Split conformal prediction and risk-controlling prediction sets are established
statistics. The contribution here is applying them to **LLM-judge evaluation**
with a nonconformity score built from judge-panel disagreement, decision margin
and rephrasing instability -- quantities specific to this setting and produced
by the other algorithms in this module -- so that an evaluation run yields a
certified bound on undetected failures and a defensible human-review budget
rather than an uncalibrated average.
"""

import math
from typing import Any, Dict, List, Optional, Sequence

from .._common._linalg import clamp, mean, quantile

__all__ = ["ConformalRiskController", "composite_nonconformity", "RiskCertificate"]


class RiskCertificate(dict):
    """The outcome of calibration.

    Keys: ``lambda_hat``, ``certified_risk_bound``, ``alpha``, ``delta``,
    ``abstention_rate``, ``calibration_size``, ``feasible``.
    """

    @property
    def feasible(self) -> bool:
        """Whether any cut-off met the risk target on this calibration set."""
        return bool(self.get("feasible"))


def composite_nonconformity(
    *,
    posterior_sd: Optional[Sequence[float]] = None,
    consensus_scores: Optional[Sequence[float]] = None,
    threshold: Optional[float] = None,
    instability: Optional[Sequence[float]] = None,
    weights: "tuple[float, float, float]" = (0.45, 0.35, 0.20),
) -> List[float]:
    """Build composite nonconformity scores from the available signals.

    Each component is normalised to ``[0, 1]`` across the batch before being
    combined, so the weights mean what they say regardless of the natural scale
    of each input.

    :param posterior_sd: Judge-panel posterior standard deviations.
    :param consensus_scores: Consensus quality scores.
    :param threshold: Pass threshold; the margin term is the *inverse* distance
        from it, so boundary rows score as more nonconforming.
    :param instability: Rephrasing-instability indices.
    :param weights: Relative weight of (disagreement, margin, instability).
    :return: One nonconformity score per row, higher meaning less trustworthy.
    :raises ValueError: If no component is supplied, or lengths disagree.
    """
    components: List[List[float]] = []
    active_weights: List[float] = []

    def normalise(values: Sequence[float]) -> List[float]:
        lo, hi = min(values), max(values)
        if hi - lo < 1e-12:
            return [0.0] * len(values)
        return [(v - lo) / (hi - lo) for v in values]

    if posterior_sd:
        components.append(normalise(list(posterior_sd)))
        active_weights.append(weights[0])
    if consensus_scores and threshold is not None:
        margins = [abs(float(s) - float(threshold)) for s in consensus_scores]
        # Small margin -> high nonconformity, hence the inversion.
        components.append([1.0 - m for m in normalise(margins)])
        active_weights.append(weights[1])
    if instability:
        components.append(normalise(list(instability)))
        active_weights.append(weights[2])

    if not components:
        raise ValueError(
            "composite_nonconformity needs at least one of posterior_sd, "
            "(consensus_scores + threshold), or instability."
        )
    size = len(components[0])
    if any(len(c) != size for c in components):
        raise ValueError("All nonconformity components must have the same length.")

    total = sum(active_weights)
    return [
        clamp(sum(w * c[i] for w, c in zip(active_weights, components)) / total, 0.0, 1.0)
        for i in range(size)
    ]


class ConformalRiskController:
    """Selects an abstention cut-off with a finite-sample risk guarantee.

    :param alpha: Target risk -- the maximum tolerated fraction of rows that are
        failures accepted automatically.
    :param delta: Confidence parameter; the guarantee holds with probability at
        least ``1 - delta`` over the draw of the calibration set.
    """

    def __init__(self, *, alpha: float = 0.1, delta: float = 0.05) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1")
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must lie strictly between 0 and 1")
        self.alpha = alpha
        self.delta = delta
        self._lambda: Optional[float] = None

    def calibrate(
        self, nonconformity: Sequence[float], failures: Sequence[bool]
    ) -> RiskCertificate:
        """Fit the cut-off on a labelled calibration set.

        :param nonconformity: Nonconformity score per calibration row.
        :param failures: ``True`` where the row is genuinely a failure.
        :return: The :class:`RiskCertificate` for the selected cut-off.
        :raises ValueError: If the inputs are empty or of differing length.
        """
        if len(nonconformity) != len(failures):
            raise ValueError("nonconformity and failures must be the same length")
        n = len(nonconformity)
        if n == 0:
            raise ValueError("calibration set must be non-empty")

        # Hoeffding slack: the price of not knowing the true risk from n samples.
        slack = math.sqrt(math.log(1.0 / self.delta) / (2.0 * n))

        pairs = sorted(zip(nonconformity, failures), key=lambda p: p[0])
        candidates = sorted({float(s) for s in nonconformity})

        best_lambda: Optional[float] = None
        best_risk = 0.0
        # R(lambda) is non-decreasing, so scanning upward and keeping the last
        # feasible cut-off yields the largest one, i.e. the least abstention.
        accepted_failures = 0
        index = 0
        for candidate in candidates:
            while index < n and pairs[index][0] <= candidate:
                if pairs[index][1]:
                    accepted_failures += 1
                index += 1
            empirical = accepted_failures / n
            if empirical + slack <= self.alpha:
                best_lambda = candidate
                best_risk = empirical
            else:
                break

        if best_lambda is None:
            # Even accepting nothing cannot meet the target, which happens when
            # the slack alone exceeds alpha (too few calibration rows).
            self._lambda = float("-inf")
            return RiskCertificate(
                lambda_hat=float("-inf"),
                certified_risk_bound=slack,
                empirical_risk=0.0,
                alpha=self.alpha,
                delta=self.delta,
                abstention_rate=1.0,
                calibration_size=n,
                hoeffding_slack=slack,
                feasible=False,
                reason=(
                    f"No cut-off meets risk {self.alpha} at confidence {1 - self.delta}: the "
                    f"finite-sample slack alone is {slack:.4f} with n={n}. Collect at least "
                    f"{math.ceil(math.log(1.0 / self.delta) / (2.0 * self.alpha ** 2))} "
                    "calibration rows."
                ),
            )

        self._lambda = best_lambda
        abstention = sum(1 for s in nonconformity if s > best_lambda) / n
        return RiskCertificate(
            lambda_hat=best_lambda,
            certified_risk_bound=best_risk + slack,
            empirical_risk=best_risk,
            alpha=self.alpha,
            delta=self.delta,
            abstention_rate=abstention,
            calibration_size=n,
            hoeffding_slack=slack,
            feasible=True,
            reason=(
                f"With probability at least {1 - self.delta:.2f}, at most "
                f"{best_risk + slack:.3f} of rows are failures accepted automatically. "
                f"{abstention:.1%} of rows are routed to human review."
            ),
        )

    def predict(self, nonconformity: float) -> str:
        """Classify one row as ``"accept"`` or ``"review"``.

        :raises RuntimeError: If :meth:`calibrate` has not been called.
        """
        if self._lambda is None:
            raise RuntimeError("ConformalRiskController.calibrate must be called first.")
        return "accept" if nonconformity <= self._lambda else "review"

    def apply(self, nonconformity: Sequence[float]) -> List[str]:
        """Classify a batch of rows."""
        return [self.predict(s) for s in nonconformity]
