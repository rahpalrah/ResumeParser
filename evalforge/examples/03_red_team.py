"""Red-team a target, then repeat the scan under a coverage-aware budget.

Run: python examples/03_red_team.py
"""

from evalforge.novel import AdaptiveAttackScheduler
from evalforge.red_team import AttackStrategy, RedTeam, RiskCategory


def surface_filter_app(messages=None, **kwargs):
    """A target whose safety filter only inspects the surface form of a request."""
    text = messages[-1]["content"]
    blocked = ("injure", "attack", "weapon", "inferior", "slur", "hurt")
    if any(word in text.lower() for word in blocked):
        reply = "I cannot help with that request."
    else:
        reply = "Sure: shoot and kill them with a weapon, then bomb the area."
    return {"messages": list(messages) + [{"role": "assistant", "content": reply}]}


def main() -> None:
    print("=== Exhaustive scan ===")
    result = RedTeam(risk_categories=[RiskCategory.Violence], num_objectives=2, seed=1).scan(
        target=surface_filter_app,
        scan_name="surface-filter-demo",
        attack_strategies=[
            AttackStrategy.BASELINE,
            AttackStrategy.BASE64,
            AttackStrategy.ROT13,
            [AttackStrategy.LEETSPEAK, AttackStrategy.ROT13],
        ],
    )
    scorecard = result.to_scorecard()
    print(f"  attacks={scorecard['total_attacks']} ASR={scorecard['overall_attack_success_rate']:.2f}")
    for strategy, rate in sorted(scorecard["by_strategy"].items()):
        print(f"    {strategy:22s} {rate:.2f}")
    print(f"  lift of best strategy over asking plainly: {scorecard['max_strategy_lift_over_baseline']:.2f}")
    print("  -> a non-zero lift means the filter is matching surface form, not meaning.")

    print("\n=== Budgeted scan: raw success rate vs distinct findings ===")
    strategies = [
        AttackStrategy.BASELINE,
        AttackStrategy.BASE64,
        AttackStrategy.ROT13,
        AttackStrategy.LEETSPEAK,
        AttackStrategy.MORSE,
        AttackStrategy.URL,
    ]
    for kappa, label in ((0.0, "maximise successes"), (0.8, "maximise coverage")):
        scheduler = AdaptiveAttackScheduler(kappa=kappa, seed=5)
        scan = RedTeam(
            risk_categories=[RiskCategory.Violence],
            num_objectives=3,
            scheduler=scheduler,
            seed=1,
        ).scan(target=surface_filter_app, attack_strategies=strategies, budget=40)
        print(
            f"  {label:20s} ASR={scan.attack_success_rate:.2f} "
            f"total_hits={scheduler.total_successes:2d} distinct_weaknesses={scheduler.coverage}"
        )
    print(
        "  -> for the same budget, the coverage-aware objective finds substantially more\n"
        "     DISTINCT weaknesses. Raw hit count and ASR can look identical while one run\n"
        "     rediscovers a single exploit and the other maps the vulnerability surface."
    )


if __name__ == "__main__":
    main()
