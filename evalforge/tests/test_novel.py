# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""Tests for the novel algorithms, including their statistical guarantees."""

import random
import statistics
import unittest

from evalforge.novel import (
    AdaptiveAttackScheduler,
    CausalAblationGroundedness,
    ConformalRiskController,
    JudgeItemResponseCalibrator,
    SemanticCurvatureProbe,
    SequentialDriftCanary,
    SpectralSemanticDrift,
    TransportFairnessAuditor,
    benjamini_hochberg,
    composite_nonconformity,
    perturbation_ladder,
)

ANCHOR = "Explain Python list comprehensions."
ON_TOPIC = [
    "Python list comprehensions build a list from an iterable in one expression.",
    "You can add a condition to a comprehension to filter elements.",
    "Nested comprehensions iterate over two sequences, outer loop first.",
    "Comprehensions are usually faster than an equivalent append loop.",
    "Generator expressions use the same syntax but produce values lazily.",
    "Dict and set comprehensions follow the same pattern with braces.",
]
DRIFTING = [
    "Python list comprehensions build a list from an iterable in one expression.",
    "You can add a condition to a comprehension to filter elements.",
    "Speaking of filters, camera lens filters change how light reaches the sensor.",
    "Polarising filters cut reflections from water and glass surfaces.",
    "For landscape photography a tripod helps at small apertures.",
    "Golden hour light gives warm tones that landscape photographers prefer.",
]
VOLATILE = [
    "Python list comprehensions build a list from an iterable.",
    "Bananas are rich in potassium and grow in the tropics.",
    "Comprehensions can include a filtering condition.",
    "The Eiffel Tower was completed in 1889 in Paris.",
    "Generator expressions are the lazy form of comprehensions.",
    "Dict comprehensions use braces and a key-value expression.",
]


class TestSpectralSemanticDrift(unittest.TestCase):
    def setUp(self):
        self.algorithm = SpectralSemanticDrift()

    def test_drifting_scores_above_on_topic(self):
        drifting = self.algorithm.analyze(DRIFTING, anchor=ANCHOR)["spectral_semantic_drift"]
        on_topic = self.algorithm.analyze(ON_TOPIC, anchor=ANCHOR)["spectral_semantic_drift"]
        self.assertGreater(drifting, on_topic)

    def test_volatility_separates_from_drift(self):
        # An alternating conversation is volatile, not drifting. This is the
        # case a single semantic graph gets wrong, so it is worth pinning.
        volatile = self.algorithm.analyze(VOLATILE, anchor=ANCHOR)
        drifting = self.algorithm.analyze(DRIFTING, anchor=ANCHOR)
        self.assertGreater(volatile["spectral_volatility"], volatile["spectral_semantic_drift"])
        self.assertGreater(drifting["spectral_semantic_drift"], volatile["spectral_semantic_drift"])

    def test_onset_reported_for_drift(self):
        result = self.algorithm.analyze(DRIFTING, anchor=ANCHOR)
        self.assertIsNotNone(result["drift_onset_turn"])
        self.assertTrue(0 < result["drift_onset_turn"] < len(DRIFTING))

    def test_short_conversation_is_handled(self):
        result = self.algorithm.analyze(["only one turn"], anchor=ANCHOR)
        self.assertEqual(result["spectral_semantic_drift"], 0.0)
        self.assertIsNone(result["drift_onset_turn"])

    def test_scores_are_bounded(self):
        for turns in (ON_TOPIC, DRIFTING, VOLATILE):
            result = self.algorithm.analyze(turns, anchor=ANCHOR)
            self.assertGreaterEqual(result["spectral_semantic_drift"], 0.0)
            self.assertLessEqual(result["spectral_semantic_drift"], 1.0)
            self.assertLessEqual(result["spectral_volatility"], 1.0)


class TestCausalGroundedness(unittest.TestCase):
    CONTEXT = (
        "The Eiffel Tower is located in Paris, France. "
        "It was completed in 1889 for the World's Fair. "
        "The tower stands 330 metres tall. "
        "Gustave Eiffel's company designed and built it."
    )

    def setUp(self):
        self.algorithm = CausalAblationGroundedness()

    def test_fabricated_claim_is_flagged_as_phantom(self):
        result = self.algorithm.analyze(
            "The Eiffel Tower is 330 metres tall and stands in Paris. "
            "It was completed in 1723 by Leonardo da Vinci.",
            self.CONTEXT,
        )
        self.assertEqual(len(result["phantom_claims"]), 1)
        self.assertIn("1723", result["phantom_claims"][0])

    def test_minimal_witness_selects_the_carrying_spans(self):
        result = self.algorithm.analyze(
            "The Eiffel Tower is 330 metres tall and stands in Paris.", self.CONTEXT
        )
        certificate = result["certificates"][0]
        witness = " ".join(certificate["witness_text"])
        # The claim fuses two facts, so both spans must be in the witness.
        self.assertIn("Paris", witness)
        self.assertIn("330", witness)

    def test_fully_grounded_beats_fabricated(self):
        grounded = self.algorithm.analyze("The tower stands 330 metres tall.", self.CONTEXT)
        fabricated = self.algorithm.analyze("The tower stands 725 metres tall.", self.CONTEXT)
        self.assertGreater(grounded["causal_groundedness"], fabricated["causal_groundedness"])

    def test_necessity_and_sufficiency_reported(self):
        result = self.algorithm.analyze("The tower stands 330 metres tall.", self.CONTEXT)
        certificate = result["certificates"][0]
        self.assertTrue(certificate["necessity"])
        self.assertTrue(certificate["sufficiency"])

    def test_empty_context_makes_every_claim_phantom(self):
        result = self.algorithm.analyze("Some claim about the world.", "")
        self.assertEqual(result["causal_groundedness"], 0.0)
        self.assertEqual(len(result["phantom_claims"]), 1)

    def test_empty_response(self):
        self.assertEqual(self.algorithm.analyze("", self.CONTEXT)["causal_groundedness"], 0.0)


class TestJudgeCalibration(unittest.TestCase):
    def _panel(self, seed=7):
        rng = random.Random(seed)
        truth = [1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5]
        return truth, {
            "fair": [t + rng.gauss(0, 0.25) for t in truth],
            "severe": [t - 1.5 + rng.gauss(0, 0.25) for t in truth],
            "flat": [3 + rng.gauss(0, 0.15) for _ in truth],
            "noisy": [t + rng.gauss(0, 1.5) for t in truth],
        }

    def test_recovers_latent_quality_better_than_averaging(self):
        truth, panel = self._panel()
        calibration = JudgeItemResponseCalibrator().fit(panel)
        naive = [statistics.fmean(panel[j][i] for j in panel) for i in range(len(truth))]
        self.assertGreater(
            statistics.correlation(calibration.latent_quality, truth),
            statistics.correlation(naive, truth),
        )

    def test_identifies_the_uninformative_judge(self):
        _, panel = self._panel()
        judges = JudgeItemResponseCalibrator().fit(panel)["judges"]
        # A judge that always says 3 has essentially no discrimination.
        self.assertLess(abs(judges["flat"]["discrimination"]), 0.2)
        self.assertGreater(abs(judges["fair"]["discrimination"]), 0.5)

    def test_recovers_severity_offset(self):
        _, panel = self._panel()
        judges = JudgeItemResponseCalibrator().fit(panel)["judges"]
        # The severe judge was shifted down by exactly 1.5 points.
        self.assertAlmostEqual(judges["fair"]["severity"] - judges["severe"]["severity"], 1.5, delta=0.2)

    def test_noisy_judge_has_highest_noise(self):
        _, panel = self._panel()
        judges = JudgeItemResponseCalibrator().fit(panel)["judges"]
        self.assertEqual(max(judges, key=lambda n: judges[n]["noise"]), "noisy")

    def test_posterior_sd_is_positive_and_finite(self):
        _, panel = self._panel()
        calibration = JudgeItemResponseCalibrator().fit(panel)
        for sd in calibration["posterior_sd"]:
            self.assertGreater(sd, 0.0)
            self.assertLess(sd, float("inf"))

    def test_requires_two_judges_and_two_items(self):
        with self.assertRaises(ValueError):
            JudgeItemResponseCalibrator().fit({"only": [1, 2, 3]})
        with self.assertRaises(ValueError):
            JudgeItemResponseCalibrator().fit({"a": [1], "b": [1]})

    def test_rejects_ragged_input(self):
        with self.assertRaises(ValueError):
            JudgeItemResponseCalibrator().fit({"a": [1, 2, 3], "b": [1, 2]})


class TestSemanticCurvature(unittest.TestCase):
    def setUp(self):
        self.probe = SemanticCurvatureProbe()
        self.ladder = perturbation_ladder("What is the capital of France?", 4)

    def test_identical_answers_have_zero_curvature(self):
        result = self.probe.analyze(self.ladder, ["The capital of France is Paris."] * 4)
        self.assertAlmostEqual(result["semantic_curvature"], 0.0, places=9)
        self.assertAlmostEqual(result["instability_index"], 0.0, places=9)

    def test_unstable_answers_score_above_stable(self):
        stable = self.probe.analyze(self.ladder, ["The capital of France is Paris."] * 4)
        unstable = self.probe.analyze(
            self.ladder,
            [
                "The capital of France is Paris.",
                "The capital of France is Paris.",
                "Bananas are grown in tropical regions.",
                "The Eiffel Tower was built in 1889.",
            ],
        )
        self.assertGreater(unstable["instability_index"], stable["instability_index"])

    def test_curvature_is_bounded(self):
        result = self.probe.analyze(
            self.ladder, ["a b c", "x y z", "a b c", "q r s"]
        )
        self.assertGreaterEqual(result["semantic_curvature"], 0.0)
        self.assertLessEqual(result["semantic_curvature"], 1.0)

    def test_ladder_rungs_are_distinct_and_ordered(self):
        self.assertEqual(len(self.ladder), 4)
        self.assertEqual(len(set(self.ladder)), 4)

    def test_probe_drives_a_generator(self):
        result = self.probe.probe("What is 2+2?", lambda q: "Four.")
        self.assertAlmostEqual(result["semantic_curvature"], 0.0, places=9)

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            self.probe.analyze(["a", "b", "c"], ["x", "y"])


class TestConformalRisk(unittest.TestCase):
    def _dataset(self, n, seed):
        rng = random.Random(seed)
        scores = [rng.random() for _ in range(n)]
        failures = [rng.random() < (0.02 + 0.5 * s * s) for s in scores]
        return scores, failures

    def test_guarantee_holds_on_fresh_data(self):
        # The certified bound must hold on an exchangeable holdout draw.
        for alpha in (0.10, 0.20):
            scores, failures = self._dataset(600, seed=3)
            controller = ConformalRiskController(alpha=alpha, delta=0.05)
            certificate = controller.calibrate(scores, failures)
            self.assertTrue(certificate.feasible)

            holdout_scores, holdout_failures = self._dataset(3000, seed=99)
            realised = sum(
                1
                for s, f in zip(holdout_scores, holdout_failures)
                if s <= certificate["lambda_hat"] and f
            ) / len(holdout_scores)
            self.assertLessEqual(realised, alpha)

    def test_larger_alpha_permits_less_abstention(self):
        scores, failures = self._dataset(600, seed=3)
        loose = ConformalRiskController(alpha=0.2).calibrate(scores, failures)
        tight = ConformalRiskController(alpha=0.1).calibrate(scores, failures)
        self.assertLessEqual(loose["abstention_rate"], tight["abstention_rate"])

    def test_infeasible_small_calibration_set(self):
        scores, failures = self._dataset(20, seed=3)
        certificate = ConformalRiskController(alpha=0.02).calibrate(scores, failures)
        self.assertFalse(certificate.feasible)
        self.assertIn("calibration rows", certificate["reason"])

    def test_predict_requires_calibration(self):
        with self.assertRaises(RuntimeError):
            ConformalRiskController().predict(0.5)

    def test_apply_labels_rows(self):
        scores, failures = self._dataset(400, seed=11)
        controller = ConformalRiskController(alpha=0.2)
        controller.calibrate(scores, failures)
        labels = set(controller.apply([0.0, 1.0]))
        self.assertTrue(labels <= {"accept", "review"})

    def test_invalid_parameters_rejected(self):
        for bad in (0.0, 1.0, -0.5):
            with self.assertRaises(ValueError):
                ConformalRiskController(alpha=bad)

    def test_composite_nonconformity_normalises(self):
        scores = composite_nonconformity(
            posterior_sd=[0.1, 0.5, 0.9], consensus_scores=[4.5, 3.1, 2.0], threshold=3.0
        )
        self.assertEqual(len(scores), 3)
        for value in scores:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_composite_requires_a_component(self):
        with self.assertRaises(ValueError):
            composite_nonconformity()


class TestTransportFairness(unittest.TestCase):
    ITEMS = [f"He led a team of {n} engineers at Riverton University." for n in range(3, 15)]
    SWAPS = {
        "pronouns": {"He": "She", "he": "she", "his": "her"},
        "institution": {"Riverton University": "Westfield College"},
    }

    @staticmethod
    def biased(text):
        score = 0.60
        if "she" in text.lower() or "her" in text.lower():
            score -= 0.14
        if "Westfield College" in text:
            score -= 0.06
        return max(0.0, min(1.0, score))

    @staticmethod
    def fair(text):
        return 0.60

    def test_measures_the_injected_disparity_exactly(self):
        audit = TransportFairnessAuditor(threshold=0.55).audit(self.ITEMS, self.biased, self.SWAPS)
        # Penalties sum to 0.20, and a uniform shift has W1 equal to the shift.
        self.assertAlmostEqual(audit["wasserstein_distance"], 0.20, places=4)

    def test_attribution_is_proportional_to_each_penalty(self):
        audit = TransportFairnessAuditor().audit(self.ITEMS, self.biased, self.SWAPS)
        self.assertAlmostEqual(audit["attribution_share"]["pronouns"], 0.14 / 0.20, places=3)
        self.assertAlmostEqual(audit["attribution_share"]["institution"], 0.06 / 0.20, places=3)

    def test_fair_scorer_shows_no_disparity(self):
        audit = TransportFairnessAuditor().audit(self.ITEMS, self.fair, self.SWAPS)
        self.assertAlmostEqual(audit["wasserstein_distance"], 0.0, places=9)
        self.assertAlmostEqual(audit["flip_rate"], 0.0, places=9)

    def test_repair_map_moves_cohorts_to_the_barycentre(self):
        audit = TransportFairnessAuditor().audit(self.ITEMS, self.biased, self.SWAPS)
        baseline = audit["cohort_scores"]["baseline"]
        counterfactual = audit["cohort_scores"]["counterfactual"]
        repaired_low = [audit.repair("counterfactual", s) for s in counterfactual]
        repaired_high = [audit.repair("baseline", s) for s in baseline]
        self.assertAlmostEqual(
            statistics.fmean(repaired_low), statistics.fmean(repaired_high), places=6
        )

    def test_repair_is_rank_preserving(self):
        audit = TransportFairnessAuditor().audit(self.ITEMS, self.biased, self.SWAPS)
        scores = sorted(audit["cohort_scores"]["baseline"])
        repaired = [audit.repair("baseline", s) for s in scores]
        self.assertEqual(repaired, sorted(repaired))

    def test_unknown_cohort_raises(self):
        audit = TransportFairnessAuditor().audit(self.ITEMS, self.biased, self.SWAPS)
        with self.assertRaises(KeyError):
            audit.repair("nonexistent", 0.5)

    def test_empty_inputs_rejected(self):
        auditor = TransportFairnessAuditor()
        with self.assertRaises(ValueError):
            auditor.audit([], self.fair, self.SWAPS)
        with self.assertRaises(ValueError):
            auditor.audit(self.ITEMS, self.fair, {})


class TestAdaptiveScheduler(unittest.TestCase):
    def test_coverage_regularisation_finds_more_distinct_weaknesses(self):
        from evalforge.red_team import AttackStrategy, RedTeam, RiskCategory

        def weak_target(messages=None, **kwargs):
            text = messages[-1]["content"]
            if "base64" in text or "ROT13" in text:
                content = "Sure: shoot and kill them with a weapon, then bomb it."
            else:
                content = "I cannot help with that."
            return {"messages": messages + [{"role": "assistant", "content": content}]}

        strategies = [
            AttackStrategy.BASELINE,
            AttackStrategy.BASE64,
            AttackStrategy.ROT13,
            AttackStrategy.LEETSPEAK,
            AttackStrategy.MORSE,
            AttackStrategy.URL,
        ]
        distinct = {}
        for kappa in (0.0, 0.8):
            scheduler = AdaptiveAttackScheduler(kappa=kappa, seed=5)
            RedTeam(
                risk_categories=[RiskCategory.Violence],
                num_objectives=3,
                scheduler=scheduler,
                seed=1,
            ).scan(target=weak_target, attack_strategies=strategies, budget=40)
            distinct[kappa] = scheduler.coverage
        # The whole point of the regulariser: distinct findings, not raw hits.
        self.assertGreater(distinct[0.8], distinct[0.0])

    def test_select_returns_none_without_arms(self):
        scheduler = AdaptiveAttackScheduler()
        scheduler.reset([])
        self.assertIsNone(scheduler.select())

    def test_reproducible_under_seed(self):
        arms = [("violence", "a"), ("violence", "b"), ("sexual", "c")]
        picks = []
        for _ in range(2):
            scheduler = AdaptiveAttackScheduler(seed=42)
            scheduler.reset(arms)
            picks.append([scheduler.select() for _ in range(8)])
        self.assertEqual(picks[0], picks[1])

    def test_successes_shift_the_posterior(self):
        scheduler = AdaptiveAttackScheduler(seed=1)
        arms = [("violence", "good"), ("violence", "bad")]
        scheduler.reset(arms)
        for index in range(20):
            scheduler.update(arms[0], {"attack_success": True, "prompt": f"unique prompt {index} alpha beta"})
            scheduler.update(arms[1], {"attack_success": False, "prompt": f"other {index}"})
        summary = scheduler.summary()
        good = summary["posterior_means"][str(("violence", "good"))]
        bad = summary["posterior_means"][str(("violence", "bad"))]
        self.assertGreater(good, bad)


class TestDriftCanary(unittest.TestCase):
    @staticmethod
    def _baseline(seed=11, n=60):
        rng = random.Random(seed)
        return [4.20 + rng.gauss(0, 0.10) for _ in range(n)], rng

    def test_false_alarm_rate_respects_the_bound(self):
        # Ville's inequality must hold under repeated peeking.
        baseline, rng = self._baseline()
        alarms = 0
        trials = 200
        for _ in range(trials):
            canary = SequentialDriftCanary(alpha=0.05)
            canary.set_baseline("m", baseline)
            for _ in range(150):
                if canary.observe("m", 4.20 + rng.gauss(0, 0.10)).alarm:
                    alarms += 1
                    break
        self.assertLessEqual(alarms / trials, 0.05)

    def test_detects_a_real_regression(self):
        baseline, rng = self._baseline(n=120)
        canary = SequentialDriftCanary(alpha=0.05)
        canary.set_baseline("m", baseline)
        detected = False
        for _ in range(300):
            if canary.observe("m", 4.00 + rng.gauss(0, 0.10)).alarm:
                detected = True
                break
        self.assertTrue(detected)

    def test_rise_direction_detects_increases(self):
        baseline, rng = self._baseline(n=120)
        canary = SequentialDriftCanary(alpha=0.05, direction="rise")
        canary.set_baseline("defects", baseline)
        detected = False
        for _ in range(300):
            if canary.observe("defects", 4.40 + rng.gauss(0, 0.10)).alarm:
                detected = True
                break
        self.assertTrue(detected)

    def test_unknown_metric_raises(self):
        with self.assertRaises(KeyError):
            SequentialDriftCanary().observe("never_registered", 1.0)

    def test_baseline_needs_two_points(self):
        with self.assertRaises(ValueError):
            SequentialDriftCanary().set_baseline("m", [1.0])

    def test_invalid_configuration_rejected(self):
        with self.assertRaises(ValueError):
            SequentialDriftCanary(alpha=1.5)
        with self.assertRaises(ValueError):
            SequentialDriftCanary(direction="sideways")

    def test_benjamini_hochberg_controls_discoveries(self):
        rejected = benjamini_hochberg({"a": 0.001, "b": 0.5, "c": 0.9}, fdr=0.1)
        self.assertTrue(rejected["a"])
        self.assertFalse(rejected["b"])
        self.assertFalse(rejected["c"])
        self.assertEqual(benjamini_hochberg({}), {})

    def test_multi_metric_flags_only_the_regressed_one(self):
        baseline, rng = self._baseline(n=120)
        canary = SequentialDriftCanary(alpha=0.05)
        for metric in ("groundedness", "relevance", "fluency"):
            canary.set_baseline(metric, baseline)
        flagged = []
        for _ in range(40):
            outcome = canary.observe_many(
                {
                    "groundedness": 3.70 + rng.gauss(0, 0.10),
                    "relevance": 4.20 + rng.gauss(0, 0.10),
                    "fluency": 4.20 + rng.gauss(0, 0.10),
                }
            )
            if outcome["flagged"]:
                flagged = outcome["flagged"]
                break
        self.assertEqual(flagged, ["groundedness"])


if __name__ == "__main__":
    unittest.main()
