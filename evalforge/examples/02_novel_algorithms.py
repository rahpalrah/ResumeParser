"""Tour of the eight novel algorithms.

Run: python examples/02_novel_algorithms.py
"""

import random

from evalforge.novel import (
    CausalAblationGroundedness,
    ConformalRiskController,
    JudgeItemResponseCalibrator,
    SemanticCurvatureProbe,
    SequentialDriftCanary,
    SpectralSemanticDrift,
    TransportFairnessAuditor,
    perturbation_ladder,
)

CONTEXT = (
    "The Eiffel Tower is located in Paris, France. "
    "It was completed in 1889 for the World's Fair. "
    "The tower stands 330 metres tall."
)


def causal_groundedness() -> None:
    print("\n=== Causal Ablation Groundedness: which claim, which passage? ===")
    result = CausalAblationGroundedness().analyze(
        "The Eiffel Tower is 330 metres tall and stands in Paris. "
        "It was completed in 1723 by Leonardo da Vinci.",
        CONTEXT,
    )
    print(f"score={result['causal_groundedness']:.3f}  phantom_rate={result['phantom_rate']:.0%}")
    for certificate in result["certificates"]:
        flag = "PHANTOM" if certificate["is_phantom"] else "grounded"
        print(f"  [{flag}] {certificate['claim'][:58]}")
        print(f"      witness: {[w[:40] for w in certificate['witness_text']]}")


def spectral_drift() -> None:
    print("\n=== Spectral Semantic Drift: did the conversation wander, and when? ===")
    turns = [
        "Python list comprehensions build a list from an iterable.",
        "You can add a condition to a comprehension to filter elements.",
        "Speaking of filters, camera lens filters change how light reaches the sensor.",
        "Polarising filters cut reflections from water and glass.",
        "For landscape photography a tripod helps at small apertures.",
        "Golden hour light gives the warm tones photographers prefer.",
    ]
    result = SpectralSemanticDrift().analyze(turns, anchor="Explain Python list comprehensions.")
    print(f"drift={result['spectral_semantic_drift']:.3f} volatility={result['spectral_volatility']:.3f}")
    print(f"  {result['reason']}")


def judge_calibration() -> None:
    print("\n=== Judge Item-Response Calibration: which judges can I trust? ===")
    rng = random.Random(7)
    truth = [1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
    panel = {
        "fair": [t + rng.gauss(0, 0.25) for t in truth],
        "severe": [t - 1.5 + rng.gauss(0, 0.25) for t in truth],
        "always_three": [3 + rng.gauss(0, 0.15) for _ in truth],
        "noisy": [t + rng.gauss(0, 1.5) for t in truth],
    }
    calibration = JudgeItemResponseCalibrator().fit(panel)
    for name, parameters in calibration["judges"].items():
        print(
            f"  {name:13s} discrimination={parameters['discrimination']:+.2f} "
            f"severity={parameters['severity']:+.2f} reliability={parameters['reliability']:.2f}"
        )
    print(f"  most reliable judge: {calibration.most_reliable_judge()}")


def curvature() -> None:
    print("\n=== Semantic Curvature: is the model guessing? ===")
    ladder = perturbation_ladder("What is the capital of France?", 4)
    probe = SemanticCurvatureProbe()
    stable = probe.analyze(ladder, ["The capital of France is Paris."] * 4)
    unstable = probe.analyze(
        ladder,
        [
            "The capital of France is Paris.",
            "The capital of France is Paris.",
            "Bananas grow in tropical regions.",
            "The Eiffel Tower was built in 1889.",
        ],
    )
    print(f"  stable   instability={stable['instability_index']:.3f}")
    print(f"  unstable instability={unstable['instability_index']:.3f}")


def conformal() -> None:
    print("\n=== Conformal Risk Control: how many failures did review miss? ===")
    rng = random.Random(3)
    scores = [rng.random() for _ in range(600)]
    failures = [rng.random() < (0.02 + 0.5 * s * s) for s in scores]
    certificate = ConformalRiskController(alpha=0.1, delta=0.05).calibrate(scores, failures)
    print(f"  {certificate['reason']}")


def fairness() -> None:
    print("\n=== Transport Fairness Audit: how big is the bias, and how do I fix it? ===")
    resumes = [f"He led a team of {n} engineers at Riverton University." for n in range(3, 15)]

    def biased(text: str) -> float:
        score = 0.60
        if "she" in text.lower() or "her" in text.lower():
            score -= 0.14
        if "Westfield College" in text:
            score -= 0.06
        return max(0.0, min(1.0, score))

    audit = TransportFairnessAuditor(threshold=0.55).audit(
        resumes,
        biased,
        {
            "pronouns": {"He": "She", "he": "she", "his": "her"},
            "institution": {"Riverton University": "Westfield College"},
        },
    )
    print(f"  {audit['reason']}")
    raw = audit["cohort_scores"]["counterfactual"][0]
    print(f"  repair: raw score {raw:.3f} -> de-biased {audit.repair('counterfactual', raw):.3f}")


def canary() -> None:
    print("\n=== Anytime-Valid Canary: has the metric actually regressed? ===")
    rng = random.Random(11)
    baseline = [4.20 + rng.gauss(0, 0.10) for _ in range(120)]
    monitor = SequentialDriftCanary(alpha=0.05)
    monitor.set_baseline("groundedness", baseline)
    for run in range(300):
        state = monitor.observe("groundedness", 4.00 + rng.gauss(0, 0.10))
        if state.alarm:
            print(f"  alarm at run {run + 1}: {state['reason'][:100]}")
            break


def main() -> None:
    causal_groundedness()
    spectral_drift()
    judge_calibration()
    curvature()
    conformal()
    fairness()
    canary()
    print("\n(The adaptive attack scheduler is demonstrated in 03_red_team.py.)")


if __name__ == "__main__":
    main()
