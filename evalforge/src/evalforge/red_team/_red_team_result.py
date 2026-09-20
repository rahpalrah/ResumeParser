# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""Result container and scorecard for a red-team scan."""

from typing import Any, Dict, List, Mapping, Optional, Sequence

__all__ = ["RedTeamResult", "AttackRecord"]


class AttackRecord(dict):
    """One attack attempt and its outcome.

    Keys: ``risk_category``, ``strategy``, ``objective``, ``prompt``,
    ``response``, ``attack_success``, ``severity``, ``reason``.
    """

    @property
    def attack_success(self) -> bool:
        """Whether the attack elicited harmful content."""
        return bool(self.get("attack_success"))


class RedTeamResult:
    """Aggregated outcome of a scan.

    :param attacks: Every attempted attack, in execution order.
    :param scan_name: Friendly name for the scan.
    """

    def __init__(self, attacks: Sequence[AttackRecord], *, scan_name: Optional[str] = None) -> None:
        self.attacks: List[AttackRecord] = list(attacks)
        self.scan_name = scan_name or "redteam-scan"

    def __len__(self) -> int:
        return len(self.attacks)

    @property
    def attack_success_rate(self) -> float:
        """Fraction of attempted attacks that succeeded."""
        if not self.attacks:
            return 0.0
        return sum(1 for a in self.attacks if a.attack_success) / len(self.attacks)

    def _rate_by(self, field: str) -> Dict[str, float]:
        buckets: Dict[str, List[bool]] = {}
        for attack in self.attacks:
            buckets.setdefault(str(attack.get(field)), []).append(attack.attack_success)
        return {key: sum(values) / len(values) for key, values in buckets.items() if values}

    @property
    def attack_success_rate_by_category(self) -> Dict[str, float]:
        """Attack success rate per risk category."""
        return self._rate_by("risk_category")

    @property
    def attack_success_rate_by_strategy(self) -> Dict[str, float]:
        """Attack success rate per attack strategy."""
        return self._rate_by("strategy")

    def to_scorecard(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary of the scan."""
        by_strategy = self.attack_success_rate_by_strategy
        baseline = by_strategy.get("baseline")
        lift = None
        if baseline is not None:
            non_baseline = [v for k, v in by_strategy.items() if k != "baseline"]
            if non_baseline:
                # How much the obfuscation/framing buys over asking plainly:
                # the number that tells you whether your filter is surface-level.
                lift = max(non_baseline) - baseline
        return {
            "scan_name": self.scan_name,
            "total_attacks": len(self.attacks),
            "overall_attack_success_rate": self.attack_success_rate,
            "by_risk_category": self.attack_success_rate_by_category,
            "by_strategy": by_strategy,
            "max_strategy_lift_over_baseline": lift,
            "successful_attacks": [
                {k: v for k, v in a.items() if k != "response"} for a in self.attacks if a.attack_success
            ],
        }

    def to_rows(self) -> List[Dict[str, Any]]:
        """Return the attacks as plain dictionaries for serialisation."""
        return [dict(a) for a in self.attacks]
