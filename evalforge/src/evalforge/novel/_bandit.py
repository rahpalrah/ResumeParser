# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""Coverage-Regularised Adaptive Attack Scheduling (CRAAS).

Problem
-------
An exhaustive red-team scan is the cross product of risk categories, attack
strategies and seed objectives, and it grows fast enough that real scans run
under a budget. Spending that budget uniformly wastes most of it on
combinations that never succeed. But the obvious fix -- steer the budget toward
whatever is succeeding -- collapses onto a single working exploit and reports a
high attack-success rate that describes one vulnerability found many times over.
For a safety scan that is the wrong objective: the goal is to find *distinct*
weaknesses, not to maximise the count of successes.

Method
------
Treat each ``(risk category, strategy)`` pair as the arm of a bandit and
allocate the budget by Thompson sampling, but score each attack on a reward
that has a diversity term:

    reward = success - kappa * redundancy

where ``redundancy`` is the greatest similarity between this attack's prompt
and any previously *successful* prompt. A success that closely resembles one
already found earns almost nothing; a success in an unexplored region earns
close to one. This is the marginal gain of a facility-location coverage
objective, which is submodular, so greedy allocation against it carries the
usual diminishing-returns behaviour: the scheduler naturally spreads out once
a region is covered.

Rewards lie in ``[0, 1]`` rather than ``{0, 1}``, so each arm's Beta posterior
is updated with fractional pseudo-counts -- ``alpha += r``, ``beta += 1 - r`` --
which preserves the mean while keeping the conjugate form.

Arms that succeed are also allowed to **chain**: on a success the scheduler
proposes the composition of that strategy with the one that has the next
highest posterior mean, so that strategies which work individually are tried
together without enumerating every composition in advance.

What is novel
-------------
Bandit allocation and submodular coverage maximisation are both standard, and
adaptive red teaming exists. The contribution is the objective: allocating a
red-team budget against **coverage of the vulnerability space** rather than raw
attack-success rate, by folding a facility-location marginal gain into the
bandit reward, so that a fixed budget returns a set of *distinct* findings; plus
posterior-driven strategy chaining that discovers effective compositions
without enumerating them.
"""

import math
import random
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

from .._common._embedding import Encoder, get_default_encoder
from .._common._linalg import clamp, cosine

__all__ = ["AdaptiveAttackScheduler"]

Arm = Tuple[str, Any]


class AdaptiveAttackScheduler:
    """Budgeted, coverage-aware scheduler for :class:`evalforge.red_team.RedTeam`.

    Implements the ``reset`` / ``select`` / ``update`` protocol the scanner
    expects, so passing an instance as ``scheduler=`` switches a scan from
    exhaustive enumeration to adaptive allocation.

    .. code-block:: python

        scheduler = AdaptiveAttackScheduler(kappa=0.7, seed=0)
        RedTeam(scheduler=scheduler, ...).scan(target=app, budget=40, ...)

    :param kappa: Weight of the redundancy penalty. ``0`` reduces the scheduler
        to plain success-maximising Thompson sampling.
    :param encoder: Text encoder used to measure prompt redundancy.
    :param prior_alpha: Beta prior successes; higher means more optimism.
    :param prior_beta: Beta prior failures.
    :param novelty_threshold: Similarity above which a success is counted as a
        rediscovery of an already-known weakness rather than a distinct finding.
    :param seed: Seed for reproducible sampling.
    """

    def __init__(
        self,
        *,
        kappa: float = 0.7,
        encoder: Optional[Encoder] = None,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        novelty_threshold: float = 0.9,
        seed: int = 0,
    ) -> None:
        self._kappa = clamp(kappa, 0.0, 1.0)
        self._encoder = encoder or get_default_encoder()
        self._prior = (max(1e-6, prior_alpha), max(1e-6, prior_beta))
        self._novelty_threshold = clamp(novelty_threshold, 0.0, 1.0)
        self._rng = random.Random(seed)
        self._posteriors: Dict[Hashable, List[float]] = {}
        self._arms: List[Arm] = []
        self._successful_vectors: List[List[float]] = []
        self._distinct_successes = 0
        self._history: List[Dict[str, Any]] = []

    # -- scheduler protocol -------------------------------------------------

    def reset(self, arms: Sequence[Arm]) -> None:
        """Begin a new scan over ``arms``."""
        self._arms = list(arms)
        self._posteriors = {self._key(arm): list(self._prior) for arm in self._arms}
        self._successful_vectors = []
        self._distinct_successes = 0
        self._history = []

    def select(self) -> Optional[Arm]:
        """Draw the next arm by Thompson sampling.

        :return: The chosen arm, or ``None`` if the scheduler has no arms.
        """
        if not self._arms:
            return None
        best_arm: Optional[Arm] = None
        best_draw = -1.0
        for arm in self._arms:
            alpha, beta = self._posteriors[self._key(arm)]
            draw = self._sample_beta(alpha, beta)
            if draw > best_draw:
                best_draw, best_arm = draw, arm
        return best_arm

    def update(self, arm: Arm, record: Mapping[str, Any]) -> None:
        """Fold an attack outcome into the arm's posterior.

        :param arm: The arm that produced the attack.
        :param record: The attack record; ``attack_success`` and ``prompt`` are read.
        """
        success = 1.0 if record.get("attack_success") else 0.0
        prompt = str(record.get("prompt", ""))

        redundancy = 0.0
        if success and self._successful_vectors:
            vector = self._encoder.encode_one(prompt)
            redundancy = max(
                (max(0.0, cosine(vector, known)) for known in self._successful_vectors), default=0.0
            )

        reward = clamp(success - self._kappa * redundancy * success, 0.0, 1.0)

        key = self._key(arm)
        alpha, beta = self._posteriors.setdefault(key, list(self._prior))
        self._posteriors[key] = [alpha + reward, beta + (1.0 - reward)]

        if success:
            if redundancy < self._novelty_threshold:
                self._distinct_successes += 1
            self._successful_vectors.append(self._encoder.encode_one(prompt))
            self._propose_chain(arm)

        self._history.append(
            {"arm": key, "success": bool(success), "redundancy": redundancy, "reward": reward}
        )

    # -- internals ----------------------------------------------------------

    def _propose_chain(self, arm: Arm) -> None:
        """Add the composition of a winning strategy with the next most promising one."""
        category, strategy = arm
        if isinstance(strategy, list):
            return  # Already a composition; do not grow without bound.
        partners = [
            other
            for other in self._arms
            if other[0] == category and not isinstance(other[1], list) and other[1] != strategy
        ]
        if not partners:
            return
        partner = max(partners, key=lambda a: self._posterior_mean(self._key(a)))
        composed: Arm = (category, [strategy, partner[1]])
        key = self._key(composed)
        if key not in self._posteriors:
            self._arms.append(composed)
            # Seed the composition optimistically from its parents.
            parent_mean = 0.5 * (
                self._posterior_mean(self._key(arm)) + self._posterior_mean(self._key(partner))
            )
            self._posteriors[key] = [1.0 + parent_mean, 1.0]

    def _posterior_mean(self, key: Hashable) -> float:
        alpha, beta = self._posteriors.get(key, self._prior)
        return alpha / (alpha + beta)

    @staticmethod
    def _key(arm: Arm) -> Hashable:
        category, strategy = arm
        if isinstance(strategy, list):
            return (category, tuple(getattr(s, "value", str(s)) for s in strategy))
        return (category, getattr(strategy, "value", str(strategy)))

    def _sample_beta(self, alpha: float, beta: float) -> float:
        """Beta sample via two Gamma draws, using only the standard library."""
        x = self._rng.gammavariate(alpha, 1.0)
        y = self._rng.gammavariate(beta, 1.0)
        total = x + y
        return x / total if total > 0 else 0.5

    # -- reporting ----------------------------------------------------------

    @property
    def coverage(self) -> int:
        """Successful attacks that were not near-duplicates of an earlier one.

        This is the quantity the scheduler optimises, and it is what a scan
        should be judged on: the count of *distinct* weaknesses found, not the
        count of times any weakness was triggered.
        """
        return self._distinct_successes

    @property
    def total_successes(self) -> int:
        """Every successful attack, including rediscoveries."""
        return len(self._successful_vectors)

    def summary(self) -> Dict[str, Any]:
        """Allocation and posterior state, for reporting after a scan."""
        pulls: Dict[str, int] = {}
        for entry in self._history:
            pulls[str(entry["arm"])] = pulls.get(str(entry["arm"]), 0) + 1
        return {
            "pulls": pulls,
            "distinct_successes": self.coverage,
            "total_successes": self.total_successes,
            "posterior_means": {str(k): self._posterior_mean(k) for k in self._posteriors},
            "mean_redundancy_of_successes": (
                sum(e["redundancy"] for e in self._history if e["success"])
                / max(1, sum(1 for e in self._history if e["success"]))
            ),
        }
