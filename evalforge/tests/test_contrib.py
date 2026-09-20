# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""Tests for the résumé-domain evaluators."""

import copy
import json
import unittest

from evalforge.contrib import RESUME_ATTRIBUTE_SWAPS, ResumeExtractionEvaluator, ResumeFairnessProbe

TRUTH = {
    "title": "Senior Engineer",
    "gender": "",
    "name": {"first": "Antony", "middle": "Deepak", "last": "Thomas"},
    "email": ["a@example.com"],
    "phone": ["555-0100"],
    "url": ["github.com/a"],
    "address": ["Seattle, WA"],
    "work_experience": [{"Experience": "Built distributed systems using Java and Python"}],
    "skills": [{"Skills": "Java Python GATE NLP"}],
    "education_and_training": [{"Education": "BS Computer Science"}],
}


class TestResumeExtraction(unittest.TestCase):
    def setUp(self):
        self.evaluator = ResumeExtractionEvaluator()

    def test_identical_parse_scores_one(self):
        result = self.evaluator(response=copy.deepcopy(TRUTH), ground_truth=TRUTH)
        self.assertAlmostEqual(result["resume_extraction_fidelity"], 1.0, places=9)
        self.assertEqual(result["resume_extraction_result"], "pass")

    def test_accepts_json_strings(self):
        result = self.evaluator(response=json.dumps(TRUTH), ground_truth=json.dumps(TRUTH))
        self.assertAlmostEqual(result["resume_extraction_fidelity"], 1.0, places=9)

    def test_degradation_lowers_score_and_names_weakest_field(self):
        degraded = copy.deepcopy(TRUTH)
        degraded["phone"] = []
        result = self.evaluator(response=degraded, ground_truth=TRUTH)
        self.assertLess(result["resume_extraction_fidelity"], 1.0)
        self.assertEqual(result["resume_field_scores"]["phone"], 0.0)
        self.assertIn("phone", result["resume_extraction_reason"])

    def test_missing_name_component_is_detected(self):
        degraded = copy.deepcopy(TRUTH)
        degraded["name"]["middle"] = ""
        result = self.evaluator(response=degraded, ground_truth=TRUTH)
        self.assertAlmostEqual(result["resume_name_accuracy"], 2 / 3, places=9)

    def test_empty_parse_scores_low(self):
        result = self.evaluator(response={}, ground_truth=TRUTH)
        self.assertLess(result["resume_extraction_fidelity"], 0.2)

    def test_unparseable_response_does_not_crash(self):
        result = self.evaluator(response="<<<not json>>>", ground_truth=TRUTH)
        self.assertLess(result["resume_extraction_fidelity"], 0.2)

    def test_section_absent_from_both_is_not_penalised(self):
        trimmed = {k: v for k, v in TRUTH.items() if k != "awards"}
        result = self.evaluator(response=copy.deepcopy(trimmed), ground_truth=trimmed)
        self.assertAlmostEqual(result["resume_extraction_fidelity"], 1.0, places=9)


class TestResumeFairness(unittest.TestCase):
    RESUMES = [
        f"He led {n} engineers at Riverton University. He writes Python continuously since 2015."
        for n in range(3, 15)
    ]

    def test_detects_screening_bias_and_attributes_it(self):
        def biased(text):
            score = 0.70
            if "she" in text.lower() or "her" in text.lower():
                score -= 0.12
            if "caregiving break" in text:
                score -= 0.10
            return max(0.0, min(1.0, score))

        audit = ResumeFairnessProbe(threshold=0.6).audit(self.RESUMES, biased)
        self.assertGreater(audit["wasserstein_distance"], 0.0)
        self.assertGreater(audit["attribution_share"]["pronouns"], 0.0)
        self.assertGreater(audit["attribution_share"]["career_gap"], 0.0)

    def test_unbiased_scorer_is_clean(self):
        audit = ResumeFairnessProbe().audit(self.RESUMES, lambda text: 0.8)
        self.assertAlmostEqual(audit["wasserstein_distance"], 0.0, places=9)
        self.assertAlmostEqual(audit["flip_rate"], 0.0, places=9)

    def test_default_swaps_cover_expected_groups(self):
        self.assertIn("pronouns", RESUME_ATTRIBUTE_SWAPS)
        self.assertIn("career_gap", RESUME_ATTRIBUTE_SWAPS)


if __name__ == "__main__":
    unittest.main()
