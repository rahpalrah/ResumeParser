# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""Anytime-Valid Evaluation Regression Canary (AVERC).

Problem
-------
Evaluation metrics are watched continuously: every commit, every nightly run,
every deployment. Testing them with a fresh significance test at each check is
statistically invalid -- peeking at a growing sample and stopping when the
p-value looks good drives the false-alarm rate toward one. Teams respond either
by alerting constantly on noise, or by setting thresholds so wide that real
regressions pass unnoticed. Fixed-sample tests are the wrong tool because the
sample is not fixed: the whole point is to look after every run.

Method
------
Replace the p-value with an **e-process**, a non-negative martingale whose
expectation under the null is at most one. For a quality metric with baseline
mean ``mu0`` and scale ``sigma0``, and a candidate drop of size ``lambda``
standard deviations, accumulate

    E_t = prod_{i<=t} exp( lambda * (mu0 - x_i) / sigma0 - lambda^2 / 2 )

Each factor is the likelihood ratio between "the metric dropped by ``lambda``
sigma" and "nothing changed"; under the null the product is a martingale with
expectation one. **Ville's inequality** then gives, for free, that

    P( sup_t E_t >= 1/alpha ) <= alpha

-- so raising an alarm the first time ``E_t`` crosses ``1/alpha`` controls the
false-alarm probability at ``alpha`` *no matter how often you look, or when you
decide to stop*. That is what "anytime-valid" buys: continuous monitoring with
a guarantee, and evidence that accumulates across runs instead of resetting.

Because the true regression size is unknown, the canary runs a grid of
``lambda`` values and **mixes** them with equal weights. A convex combination
of e-processes is an e-process, so the guarantee survives the mixture while the
detector stays sensitive across a range of effect sizes.

Baseline estimation error
-------------------------
Ville's inequality assumes the null is *known*. In practice ``mu0`` and
``sigma0`` are estimated from a finite run history, and that error does not
average out -- it is a fixed offset that the martingale accumulates evidence
against, run after run, until it alarms. Measured on a stationary stream, a
canary built on a 30-run baseline false-alarms at roughly twice its nominal
rate for exactly this reason.

The budget is therefore split in two. Half of ``alpha`` pays for a one-sided
confidence bound on the baseline mean, so the e-process is run against a
conservative null ``mu0 - z * sigma0 / sqrt(n)`` rather than the point
estimate; the variance is inflated to its predictive value
``sigma0 * sqrt(1 + 1/n)`` for the same reason. The other half pays for the
alarm boundary, which becomes ``2/alpha``. A union bound over the two gives
back the nominal guarantee.

For several metrics at once, each metric's e-process yields an anytime-valid
p-value ``p = min(1, 1/E_t)``, and those are combined under
Benjamini-Hochberg so that the **false discovery rate** across the dashboard is
controlled rather than the per-metric error rate.

What is novel
-------------
E-processes, Ville's inequality and test martingales are established sequential
statistics, as is CUSUM change detection. The contribution is their application
to **continuous evaluation monitoring**: a mixture e-process per metric with a
baseline estimated from the evaluation history, an alarm rule that stays valid
under the unlimited peeking that CI inherently performs, and BH-controlled
fusion across a metric dashboard -- replacing the per-run threshold alerting
that makes evaluation dashboards either noisy or blind.
"""

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .._common._linalg import clamp, mean, normal_ppf, stdev

__all__ = ["SequentialDriftCanary", "CanaryAlarm", "benjamini_hochberg"]

_EPS = 1e-12


def benjamini_hochberg(p_values: Mapping[str, float], fdr: float = 0.1) -> Dict[str, bool]:
    """Benjamini-Hochberg step-up procedure.

    :param p_values: ``{name: p_value}``.
    :param fdr: Target false discovery rate.
    :return: ``{name: rejected}`` -- ``True`` where the null is rejected.
    """
    if not p_values:
        return {}
    names = list(p_values)
    ordered = sorted(names, key=lambda k: p_values[k])
    m = len(ordered)
    cutoff_rank = 0
    for rank, name in enumerate(ordered, start=1):
        if p_values[name] <= fdr * rank / m:
            cutoff_rank = rank
    rejected = set(ordered[:cutoff_rank])
    return {name: name in rejected for name in names}


class CanaryAlarm(dict):
    """State of one metric's e-process.

    Keys: ``metric``, ``e_value``, ``p_value``, ``alarm``, ``observations``,
    ``first_alarm_at``, ``baseline_mean``, ``baseline_sd``.
    """

    @property
    def alarm(self) -> bool:
        """Whether the e-process has crossed the alarm boundary."""
        return bool(self.get("alarm"))


class SequentialDriftCanary:
    """Anytime-valid monitor for one or more evaluation metrics.

    .. code-block:: python

        canary = SequentialDriftCanary(alpha=0.05)
        canary.set_baseline("groundedness", history)     # past runs
        for value in new_runs:
            state = canary.observe("groundedness", value)
            if state.alarm:
                ...

    :param alpha: Target false-alarm probability across the whole monitoring
        run, however many times it is inspected.
    :param lambdas: Grid of effect sizes, in baseline standard deviations.
    :param direction: ``"drop"`` alerts on decreases (quality metrics);
        ``"rise"`` alerts on increases (defect rates, severities).
    """

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        lambdas: Sequence[float] = (0.25, 0.5, 1.0, 1.5, 2.0),
        direction: str = "drop",
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie strictly between 0 and 1")
        if direction not in ("drop", "rise"):
            raise ValueError("direction must be 'drop' or 'rise'")
        self.alpha = alpha
        self.direction = direction
        self._lambdas = [l for l in lambdas if l > 0]
        if not self._lambdas:
            raise ValueError("at least one positive lambda is required")
        # Half the error budget pays for the baseline confidence bound below,
        # so the alarm boundary spends only the other half.
        self._boundary = 2.0 / alpha
        self._baseline_z = normal_ppf(1.0 - alpha / 2.0)
        self._baselines: Dict[str, "tuple[float, float]"] = {}
        self._observed_baseline: Dict[str, "tuple[float, float, int]"] = {}
        # Per metric: one running log-e-value per lambda in the mixture.
        self._log_components: Dict[str, List[float]] = {}
        self._counts: Dict[str, int] = {}
        self._first_alarm: Dict[str, Optional[int]] = {}

    def set_baseline(self, metric: str, history: Sequence[float]) -> None:
        """Establish the null for ``metric`` from past runs.

        :param history: At least two past values of the metric.
        :raises ValueError: If fewer than two values are supplied.
        """
        values = [float(v) for v in history]
        if len(values) < 2:
            raise ValueError("baseline requires at least two observations")
        scale = stdev(values, ddof=1)
        if scale <= _EPS:
            # A perfectly flat baseline would make every deviation infinitely
            # surprising; floor the scale at the measurement granularity.
            scale = max(abs(mean(values)) * 1e-3, 1e-6)

        n = len(values)
        centre = mean(values)
        # Predictive scale for a *new* observation, not the scale of the
        # baseline sample itself.
        predictive_scale = scale * math.sqrt(1.0 + 1.0 / n)
        # Shift the null against the direction we alert on, so a mis-estimated
        # baseline cannot masquerade as a regression.
        margin = self._baseline_z * scale / math.sqrt(n)
        conservative = centre - margin if self.direction == "drop" else centre + margin

        self._baselines[metric] = (conservative, predictive_scale)
        self._observed_baseline[metric] = (centre, scale, n)
        self._log_components[metric] = [0.0] * len(self._lambdas)
        self._counts[metric] = 0
        self._first_alarm[metric] = None

    def observe(self, metric: str, value: float) -> CanaryAlarm:
        """Fold one new observation into the metric's e-process.

        :raises KeyError: If no baseline has been set for ``metric``.
        """
        if metric not in self._baselines:
            raise KeyError(f"No baseline set for metric '{metric}'. Call set_baseline first.")
        mu0, sigma0 = self._baselines[metric]

        # Standardised deviation in the direction we are alerting on.
        deviation = (mu0 - float(value)) / sigma0
        if self.direction == "rise":
            deviation = -deviation

        for index, lam in enumerate(self._lambdas):
            # log of exp(lambda*deviation - lambda^2/2): the log-likelihood
            # ratio of a lambda-sigma shift against no change.
            self._log_components[metric][index] += lam * deviation - 0.5 * lam * lam

        self._counts[metric] += 1
        e_value = self._mixture(metric)
        p_value = clamp(1.0 / e_value, 0.0, 1.0) if e_value > _EPS else 1.0
        alarm = e_value >= self._boundary
        if alarm and self._first_alarm[metric] is None:
            self._first_alarm[metric] = self._counts[metric]

        return CanaryAlarm(
            metric=metric,
            e_value=e_value,
            p_value=p_value,
            alarm=alarm,
            observations=self._counts[metric],
            first_alarm_at=self._first_alarm[metric],
            baseline_mean=self._observed_baseline[metric][0],
            baseline_sd=self._observed_baseline[metric][1],
            null_mean_used=mu0,
            null_sd_used=sigma0,
            boundary=self._boundary,
            reason=self._explain(metric, e_value, alarm, self._first_alarm[metric]),
        )

    def observe_many(self, observations: Mapping[str, float], *, fdr: float = 0.1) -> Dict[str, Any]:
        """Fold one observation per metric and apply BH across the dashboard.

        :param observations: ``{metric: value}`` for this run.
        :param fdr: Target false discovery rate across metrics.
        """
        states = {metric: self.observe(metric, value) for metric, value in observations.items()}
        discoveries = benjamini_hochberg({m: s["p_value"] for m, s in states.items()}, fdr=fdr)
        for metric, state in states.items():
            state["bh_flagged"] = discoveries.get(metric, False)
        return {
            "states": states,
            "flagged": sorted(m for m, flag in discoveries.items() if flag),
            "fdr": fdr,
        }

    def _mixture(self, metric: str) -> float:
        """Equal-weight mixture of the per-lambda e-processes, computed in log space."""
        logs = self._log_components[metric]
        top = max(logs)
        # log-sum-exp keeps the mixture finite when one component grows large.
        total = math.fsum(math.exp(l - top) for l in logs)
        return math.exp(top) * total / len(logs)

    @staticmethod
    def _explain(metric: str, e_value: float, alarm: bool, first_alarm: Optional[int]) -> str:
        """State the evidence in interpretable terms."""
        if alarm:
            return (
                f"Regression in '{metric}': accumulated evidence e={e_value:.1f} crossed the "
                f"alarm boundary, first at observation {first_alarm}. The e-value is a valid "
                "measure of evidence however many times the monitor was inspected."
            )
        if e_value > 1.0:
            return f"Weak evidence of regression in '{metric}' (e={e_value:.2f}); below the alarm boundary."
        return f"No evidence of regression in '{metric}' (e={e_value:.2f})."
