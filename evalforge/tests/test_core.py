# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""Tests for the core primitives: math, text, encoding and configuration."""

import math
import unittest

from evalforge._common._embedding import HashingEncoder
from evalforge._common._heuristics import evidence_recall, harm_severity, score_metric
from evalforge._common._linalg import (
    cosine,
    jacobi_eigh,
    normal_ppf,
    quantile,
    solve_linear,
    wasserstein_1,
)
from evalforge._common._text import lcs_length, split_claims, split_sentences, tokenize


class TestLinearAlgebra(unittest.TestCase):
    def test_jacobi_matches_known_spectrum(self):
        # Tridiagonal [2,1;1,2,1;1,2] has eigenvalues 2-sqrt2, 2, 2+sqrt2.
        matrix = [[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]]
        values, vectors = jacobi_eigh(matrix)
        expected = [2 - math.sqrt(2), 2.0, 2 + math.sqrt(2)]
        for got, want in zip(values, expected):
            self.assertAlmostEqual(got, want, places=8)
        # Eigenvectors must satisfy A v = lambda v.
        for value, vector in zip(values, vectors):
            product = [sum(matrix[i][j] * vector[j] for j in range(3)) for i in range(3)]
            for i in range(3):
                self.assertAlmostEqual(product[i], value * vector[i], places=8)

    def test_jacobi_handles_identity_and_empty(self):
        values, _ = jacobi_eigh([[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(values, [1.0, 1.0])
        self.assertEqual(jacobi_eigh([]), ([], []))

    def test_solve_linear_exact(self):
        solution = solve_linear([[3.0, 1.0], [1.0, 2.0]], [9.0, 8.0])
        self.assertAlmostEqual(solution[0], 2.0, places=9)
        self.assertAlmostEqual(solution[1], 3.0, places=9)

    def test_wasserstein_properties(self):
        self.assertAlmostEqual(wasserstein_1([1, 2, 3], [1, 2, 3]), 0.0, places=9)
        # A pure shift of size c has W1 exactly c.
        self.assertAlmostEqual(wasserstein_1([1, 2, 3], [4, 5, 6]), 3.0, places=6)
        # Symmetry.
        self.assertAlmostEqual(
            wasserstein_1([1, 5, 9], [2, 2, 8]), wasserstein_1([2, 2, 8], [1, 5, 9]), places=9
        )

    def test_quantile_interpolates(self):
        self.assertAlmostEqual(quantile([0, 10], 0.5), 5.0, places=9)
        self.assertAlmostEqual(quantile([1, 2, 3, 4], 0.0), 1.0, places=9)
        self.assertAlmostEqual(quantile([1, 2, 3, 4], 1.0), 4.0, places=9)

    def test_normal_ppf_known_values(self):
        self.assertAlmostEqual(normal_ppf(0.5), 0.0, places=8)
        self.assertAlmostEqual(normal_ppf(0.975), 1.959964, places=5)
        self.assertAlmostEqual(normal_ppf(0.025), -1.959964, places=5)

    def test_cosine_bounds(self):
        self.assertAlmostEqual(cosine([1, 0], [1, 0]), 1.0, places=9)
        self.assertAlmostEqual(cosine([1, 0], [-1, 0]), -1.0, places=9)
        self.assertEqual(cosine([0, 0], [1, 1]), 0.0)


class TestText(unittest.TestCase):
    def test_sentence_split_guards_abbreviations(self):
        text = "Dr. Smith went to Washington. He met J. Doe at 3.5 p.m. It rained."
        self.assertEqual(len(split_sentences(text)), 3)

    def test_claim_split_separates_independent_assertions(self):
        claims = split_claims("Paris is the capital; it has two million residents.")
        self.assertEqual(len(claims), 2)

    def test_lcs(self):
        self.assertEqual(lcs_length("a b c d".split(), "a c d e".split()), 3)
        self.assertEqual(lcs_length([], ["a"]), 0)

    def test_tokenize_is_unicode_aware(self):
        self.assertEqual(tokenize("Café — naïve"), ["café", "naïve"])


class TestEncoder(unittest.TestCase):
    def test_deterministic_across_instances(self):
        a, b = HashingEncoder(), HashingEncoder()
        self.assertEqual(a.encode_one("hello world"), b.encode_one("hello world"))

    def test_unit_norm(self):
        vector = HashingEncoder().encode_one("some representative text")
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in vector)), 1.0, places=9)

    def test_related_text_scores_higher_than_unrelated(self):
        encoder = HashingEncoder()
        vectors = encoder.encode(
            ["the capital of France is Paris", "Paris is the French capital", "bananas are yellow"]
        )
        self.assertGreater(cosine(vectors[0], vectors[1]), cosine(vectors[0], vectors[2]))

    def test_empty_text_gives_zero_vector(self):
        self.assertEqual(set(HashingEncoder().encode_one("")), {0.0})

    def test_rejects_tiny_dimension(self):
        with self.assertRaises(ValueError):
            HashingEncoder(dimension=2)


class TestHeuristics(unittest.TestCase):
    CONTEXT = "The Eiffel Tower is in Paris, France. It was completed in 1889 and is 330 metres tall."

    def test_groundedness_ordering(self):
        grounded = score_metric("groundedness", {"response": "It is 330 metres tall and in Paris.", "context": self.CONTEXT})[0]
        contradicted = score_metric("groundedness", {"response": "It is in Berlin and was built in 1723.", "context": self.CONTEXT})[0]
        unrelated = score_metric("groundedness", {"response": "Bananas are rich in potassium.", "context": self.CONTEXT})[0]
        self.assertGreater(grounded, contradicted)
        self.assertGreater(contradicted, unrelated)

    def test_evidence_recall_penalises_wrong_specifics(self):
        # Swapping a numeral and a proper noun must cost more than a stopword.
        right = evidence_recall(self.CONTEXT, "The tower is 330 metres tall in Paris")
        wrong = evidence_recall(self.CONTEXT, "The tower is 725 metres tall in Berlin")
        self.assertGreater(right, wrong + 0.2)

    def test_refusal_damps_harm_severity(self):
        raw = harm_severity("I will kill him with a gun", "violence")
        refusal = harm_severity("I cannot help you kill anyone; please seek help", "violence")
        self.assertGreater(raw, refusal)
        self.assertEqual(harm_severity("Let us discuss gardening", "violence"), 0)

    def test_unknown_metric_raises(self):
        with self.assertRaises(KeyError):
            score_metric("not_a_metric", {"response": "x"})


if __name__ == "__main__":
    unittest.main()
