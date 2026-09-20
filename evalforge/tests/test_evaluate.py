# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""Tests for the batch evaluation engine."""

import json
import logging
import os
import tempfile
import unittest

from evalforge import EvaluationException, F1ScoreEvaluator, GroundednessEvaluator, evaluate
from evalforge._evaluate._utils import aggregate_metrics, apply_column_mapping, load_data

ROWS = [
    {"query": "How tall?", "context": "The tower is 330 metres tall.", "answer": "It is 330 metres tall.", "truth": "330 metres tall"},
    {"query": "Where?", "context": "The tower is in Paris.", "answer": "It is in Berlin.", "truth": "in Paris"},
]


class TestDataLoading(unittest.TestCase):
    def test_round_trip_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "data.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(json.dumps(r) for r in ROWS))
            self.assertEqual(load_data(path), ROWS)

    def test_reads_csv_and_json(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = os.path.join(directory, "d.csv")
            with open(csv_path, "w", encoding="utf-8") as handle:
                handle.write("a,b\n1,2\n")
            self.assertEqual(load_data(csv_path), [{"a": "1", "b": "2"}])

            json_path = os.path.join(directory, "d.json")
            with open(json_path, "w", encoding="utf-8") as handle:
                json.dump(ROWS, handle)
            self.assertEqual(load_data(json_path), ROWS)

    def test_accepts_in_memory_rows(self):
        self.assertEqual(load_data(ROWS), ROWS)

    def test_missing_file_raises(self):
        with self.assertRaises(EvaluationException):
            load_data("/nonexistent/path/data.jsonl")

    def test_malformed_jsonl_names_the_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "bad.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write('{"ok": 1}\nnot json\n')
            with self.assertRaises(EvaluationException) as context:
                load_data(path)
            self.assertIn("line 2", str(context.exception))


class TestColumnMapping(unittest.TestCase):
    def test_data_and_target_references(self):
        mapping = {"response": "${target.out}", "context": "${data.context}", "literal": "fixed"}
        resolved = apply_column_mapping(mapping, {"context": "C"}, {"out": "O"})
        self.assertEqual(resolved, {"response": "O", "context": "C", "literal": "fixed"})

    def test_run_outputs_alias(self):
        resolved = apply_column_mapping({"r": "${run.outputs.x}"}, {}, {"x": 1})
        self.assertEqual(resolved, {"r": 1})

    def test_unresolved_reference_is_omitted(self):
        self.assertEqual(apply_column_mapping({"a": "${data.absent}"}, {}, {}), {})


class TestAggregation(unittest.TestCase):
    def test_means_pass_rates_and_defect_rates(self):
        rows = [
            {"outputs.e.score": 2.0, "outputs.e.score_result": "pass", "outputs.e.flag_label": True},
            {"outputs.e.score": 4.0, "outputs.e.score_result": "fail", "outputs.e.flag_label": False},
        ]
        metrics = aggregate_metrics(rows, ["e"])
        self.assertAlmostEqual(metrics["e.score"], 3.0, places=9)
        self.assertAlmostEqual(metrics["e.score_result.pass_rate"], 0.5, places=9)
        self.assertAlmostEqual(metrics["e.flag_label.defect_rate"], 0.5, places=9)


F1_MAPPING = {"f1": {"column_mapping": {"response": "${data.answer}", "ground_truth": "${data.truth}"}}}


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        # Two tests below deliberately trigger per-row evaluator errors; the
        # engine logs them by design, which would otherwise clutter the report.
        logging.getLogger("evalforge._evaluate._evaluate").setLevel(logging.CRITICAL)

    def tearDown(self):
        logging.getLogger("evalforge._evaluate._evaluate").setLevel(logging.NOTSET)

    def test_end_to_end_with_mapping_and_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "out.json")
            result = evaluate(
                data=ROWS,
                evaluators={"grounded": GroundednessEvaluator(), "f1": F1ScoreEvaluator()},
                evaluator_config={
                    "default": {"column_mapping": {"response": "${data.answer}"}},
                    "f1": {"column_mapping": {"response": "${data.answer}", "ground_truth": "${data.truth}"}},
                },
                output_path=output,
                evaluation_name="unit-test",
            )
            self.assertEqual(len(result["rows"]), 2)
            self.assertIn("grounded.groundedness", result["metrics"])
            self.assertIn("f1.f1_score", result["metrics"])
            self.assertEqual(result["run_info"]["name"], "unit-test")
            self.assertTrue(os.path.isfile(output))
            with open(output, encoding="utf-8") as handle:
                self.assertIn("metrics", json.load(handle))

    def test_input_columns_are_prefixed(self):
        result = evaluate(
            data=ROWS,
            evaluators={"f1": F1ScoreEvaluator()},
            evaluator_config=F1_MAPPING,
        )
        self.assertIn("inputs.query", result["rows"][0])
        self.assertIn("outputs.f1.f1_score", result["rows"][0])

    def test_target_outputs_are_addressable(self):
        def target(**row):
            return {"response": row["context"]}

        result = evaluate(
            data=ROWS,
            evaluators={"g": GroundednessEvaluator()},
            target=target,
            evaluator_config={"g": {"column_mapping": {"response": "${target.response}", "context": "${data.context}"}}},
        )
        # Echoing the context back is perfectly grounded by construction.
        self.assertGreaterEqual(result["metrics"]["g.groundedness"], 4.0)
        self.assertIn("outputs.response", result["rows"][0])

    def test_jsonl_output_writes_rows_only(self):
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "rows.jsonl")
            evaluate(
                data=ROWS,
                evaluators={"f1": F1ScoreEvaluator()},
                evaluator_config=F1_MAPPING,
                output_path=output,
            )
            with open(output, encoding="utf-8") as handle:
                lines = [line for line in handle if line.strip()]
            self.assertEqual(len(lines), 2)

    def test_evaluator_error_is_recorded_not_raised(self):
        class Exploding:
            _singleton_inputs = ["response"]

            def __call__(self, **kwargs):
                raise RuntimeError("boom")

        result = evaluate(data=ROWS, evaluators={"bad": Exploding()})
        self.assertIn("outputs.bad.error", result["rows"][0])

    def test_fail_on_evaluator_errors_raises(self):
        class Exploding:
            _singleton_inputs = ["response"]

            def __call__(self, **kwargs):
                raise RuntimeError("boom")

        with self.assertRaises(EvaluationException):
            evaluate(data=ROWS, evaluators={"bad": Exploding()}, fail_on_evaluator_errors=True)

    def test_empty_inputs_rejected(self):
        with self.assertRaises(EvaluationException):
            evaluate(data=[], evaluators={"f1": F1ScoreEvaluator()})
        with self.assertRaises(EvaluationException):
            evaluate(data=ROWS, evaluators={})

    def test_studio_url_only_with_project(self):
        without = evaluate(data=ROWS, evaluators={"f1": F1ScoreEvaluator()}, evaluator_config=F1_MAPPING)
        self.assertIsNone(without["studio_url"])
        with_project = evaluate(
            data=ROWS,
            evaluators={"f1": F1ScoreEvaluator()},
            evaluator_config=F1_MAPPING,
            azure_ai_project={"subscription_id": "s", "resource_group_name": "r", "project_name": "p"},
        )
        self.assertIn("ai.azure.com", with_project["studio_url"])


if __name__ == "__main__":
    unittest.main()
