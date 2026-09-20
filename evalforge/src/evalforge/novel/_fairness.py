# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""Optimal-Transport Counterfactual Fairness Audit (OT-CFA).

Problem
-------
Fairness reporting for scoring systems usually stops at a disparity number: a
gap in mean score, or a ratio of selection rates. That tells a team a problem
exists without telling them how large the intervention would have to be, where
in the score range the unfairness sits, or which attribute is responsible. And
a disparity measured across *different people* confounds bias with genuine
differences in the population.

Method
------
Two measurements, deliberately kept separate.

1. **Paired counterfactual probe.** Take one item and rewrite only a protected
   attribute -- a name, a pronoun set, an affiliation -- holding everything else
   fixed. Because the pair differs in exactly one respect, any score change is
   attributable to that attribute. This yields the counterfactual gap and the
   **decision flip rate**: how often the swap alone moves an item across the
   accept threshold.

2. **Distributional repair.** Take the two cohorts' score distributions and
   compute the Wasserstein-1 distance between them, then construct the
   **transport map to their barycenter**. For one-dimensional distributions the
   barycenter has the closed form ``Q_bary(u) = (Q_A(u) + Q_B(u)) / 2``, and
   the map that carries cohort A onto it is ``T_A(x) = Q_bary(F_A(x))``. The
   mean displacement ``E|T(x) - x|`` is the **repair cost**: the average score
   adjustment required to equalise the distributions.

The repair cost is a fairness metric whose units are score points, so it is
directly interpretable. More usefully, the map that measures the disparity is
also the map that removes it: :meth:`TransportFairnessAudit.repair` applies it,
and because each ``T`` is monotone, applying it preserves the ranking *within*
each cohort while equalising the distributions *between* them.

Attribution is by ablation: rerun the audit with one attribute swapped at a
time and compare the resulting repair costs to apportion the total.

What is novel
-------------
Counterfactual fairness testing, Wasserstein disparity measures and
distribution alignment each exist independently. The contribution is binding
them into one procedure for generative and scoring systems where (a) the metric
reported is the cost of the optimal repair rather than an abstract distance,
(b) the repair map is returned as an executable, rank-preserving remediation
rather than only a diagnostic, and (c) the total is decomposed across protected
attributes by ablation, so a team learns which rewrite drives the disparity.
"""

import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .._common._linalg import clamp, mean, quantile, wasserstein_1

__all__ = ["TransportFairnessAuditor", "TransportFairnessAudit", "swap_attributes"]


def swap_attributes(text: str, substitutions: Mapping[str, str]) -> str:
    """Rewrite ``text``, applying whole-word substitutions case-sensitively.

    Longer keys are applied first so that multi-word substitutions are not
    broken up by a shorter overlapping key.

    :param text: The source text.
    :param substitutions: ``{original: replacement}``.
    """
    out = text
    for source in sorted(substitutions, key=len, reverse=True):
        out = re.sub(rf"\b{re.escape(source)}\b", substitutions[source], out)
    return out


class TransportFairnessAudit(dict):
    """Result of an audit.

    Keys: ``wasserstein_distance``, ``repair_cost``, ``flip_rate``,
    ``mean_counterfactual_gap``, ``max_counterfactual_gap``, ``attribution``,
    ``cohort_scores``, ``reason``.
    """

    def __init__(self, *args: Any, repair_maps: Optional[Mapping[str, Callable[[float], float]]] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._repair_maps = dict(repair_maps or {})

    def repair(self, cohort: str, score: float) -> float:
        """Apply the fitted transport map for ``cohort`` to one score.

        :raises KeyError: If no map was fitted for that cohort.
        """
        if cohort not in self._repair_maps:
            raise KeyError(f"No repair map for cohort '{cohort}'. Known: {sorted(self._repair_maps)}")
        return self._repair_maps[cohort](score)

    @property
    def cohorts(self) -> List[str]:
        """Names of the cohorts that have a repair map."""
        return sorted(self._repair_maps)


def _empirical_cdf(samples: Sequence[float]) -> Callable[[float], float]:
    """Return the empirical CDF of ``samples`` with midpoint ranks."""
    ordered = sorted(samples)
    n = len(ordered)

    def cdf(x: float) -> float:
        if n == 0:
            return 0.5
        below = sum(1 for v in ordered if v < x)
        equal = sum(1 for v in ordered if v == x)
        # Midpoint convention keeps the map well behaved on ties.
        return clamp((below + 0.5 * max(equal, 1)) / n, 0.0, 1.0)

    return cdf


def _barycentric_map(
    source: Sequence[float], other: Sequence[float]
) -> Callable[[float], float]:
    """Monotone map carrying ``source`` onto the barycentre of both cohorts."""
    cdf = _empirical_cdf(source)
    src = sorted(source)
    oth = sorted(other)

    def transport(x: float) -> float:
        u = cdf(x)
        return 0.5 * (quantile(src, u) + quantile(oth, u))

    return transport


class TransportFairnessAuditor:
    """Runs paired counterfactual and distributional fairness audits.

    :param threshold: Decision threshold used for the flip rate.
    """

    def __init__(self, *, threshold: float = 0.5) -> None:
        self._threshold = threshold

    def audit(
        self,
        items: Sequence[str],
        score_fn: Callable[[str], float],
        attribute_swaps: Mapping[str, Mapping[str, str]],
        *,
        cohort_names: Tuple[str, str] = ("baseline", "counterfactual"),
    ) -> TransportFairnessAudit:
        """Audit a scoring function for counterfactual attribute sensitivity.

        :param items: The texts to score, e.g. résumés or applications.
        :param score_fn: Maps a text to a score. Called once per item per cohort.
        :param attribute_swaps: ``{attribute_name: {original: replacement}}``.
            All attributes are applied together for the headline numbers, and
            each is applied alone for the attribution breakdown.
        :param cohort_names: Labels for the two cohorts.
        :raises ValueError: If ``items`` or ``attribute_swaps`` is empty.
        """
        if not items:
            raise ValueError("audit requires at least one item")
        if not attribute_swaps:
            raise ValueError("audit requires at least one attribute swap")

        combined: Dict[str, str] = {}
        for substitutions in attribute_swaps.values():
            combined.update(substitutions)

        baseline_scores = [float(score_fn(item)) for item in items]
        counterfactual_scores = [float(score_fn(swap_attributes(item, combined))) for item in items]

        gaps = [b - c for b, c in zip(baseline_scores, counterfactual_scores)]
        flips = sum(
            1
            for b, c in zip(baseline_scores, counterfactual_scores)
            if (b >= self._threshold) != (c >= self._threshold)
        )

        distance = wasserstein_1(baseline_scores, counterfactual_scores)
        to_baseline = _barycentric_map(baseline_scores, counterfactual_scores)
        to_counterfactual = _barycentric_map(counterfactual_scores, baseline_scores)
        repair_cost = 0.5 * (
            mean([abs(to_baseline(s) - s) for s in baseline_scores])
            + mean([abs(to_counterfactual(s) - s) for s in counterfactual_scores])
        )

        # Attribution: rerun with one attribute at a time.
        attribution: Dict[str, float] = {}
        for name, substitutions in attribute_swaps.items():
            single = [float(score_fn(swap_attributes(item, substitutions))) for item in items]
            attribution[name] = wasserstein_1(baseline_scores, single)
        total = sum(attribution.values())
        attribution_share = (
            {k: v / total for k, v in attribution.items()} if total > 1e-12 else {k: 0.0 for k in attribution}
        )

        return TransportFairnessAudit(
            {
                "wasserstein_distance": distance,
                "repair_cost": repair_cost,
                "flip_rate": flips / len(items),
                "mean_counterfactual_gap": mean(gaps),
                "mean_absolute_counterfactual_gap": mean([abs(g) for g in gaps]),
                "max_counterfactual_gap": max(gaps, key=abs) if gaps else 0.0,
                "attribution": attribution,
                "attribution_share": attribution_share,
                "cohort_scores": {
                    cohort_names[0]: baseline_scores,
                    cohort_names[1]: counterfactual_scores,
                },
                "items_audited": len(items),
                "threshold": self._threshold,
                "reason": self._explain(distance, repair_cost, flips / len(items), gaps, attribution_share),
            },
            repair_maps={cohort_names[0]: to_baseline, cohort_names[1]: to_counterfactual},
        )

    @staticmethod
    def _explain(
        distance: float,
        repair_cost: float,
        flip_rate: float,
        gaps: Sequence[float],
        shares: Mapping[str, float],
    ) -> str:
        """State the finding in units a reviewer can act on."""
        if distance < 1e-6 and flip_rate == 0.0:
            return "Scores are invariant to the protected-attribute swap; no disparity detected."
        direction = "favours the baseline cohort" if mean(gaps) > 0 else "favours the counterfactual cohort"
        leading = max(shares, key=lambda k: shares[k]) if shares else None
        driver = f" The largest single contributor is '{leading}' ({shares[leading]:.0%})." if leading else ""
        return (
            f"Scoring {direction}: Wasserstein distance {distance:.4f}, repair cost "
            f"{repair_cost:.4f} score points, and {flip_rate:.1%} of items change decision "
            f"on the attribute swap alone.{driver}"
        )
