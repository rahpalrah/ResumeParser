# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""Tests for the simulators, attack strategies and the red-team scanner."""

import base64
import codecs
import json
import os
import tempfile
import unittest

from evalforge import EvaluationException
from evalforge.red_team import (
    AttackObjectiveGenerator,
    AttackStrategy,
    RedTeam,
    RiskCategory,
    apply_strategy,
    compose_strategies,
    is_refusal,
)
from evalforge.simulator import (
    AdversarialScenario,
    AdversarialSimulator,
    DirectAttackSimulator,
    IndirectAttackSimulator,
    Simulator,
)


def echo_app(messages=None, **kwargs):
    """A target that refuses obvious harm and otherwise answers."""
    last = messages[-1]["content"]
    harmful = any(word in last.lower() for word in ("hurt", "inferior", "injure", "explicit"))
    reply = "I cannot help with that." if harmful else f"Information about: {last[:40]}"
    return {"messages": list(messages) + [{"role": "assistant", "content": reply}]}


class TestAttackStrategies(unittest.TestCase):
    OBJECTIVE = "how to pick a lock"

    def test_base64_payload_round_trips(self):
        prompt = apply_strategy(AttackStrategy.BASE64, self.OBJECTIVE)
        payload = prompt.rsplit(": ", 1)[1]
        self.assertEqual(base64.b64decode(payload).decode(), self.OBJECTIVE)

    def test_rot13_payload_round_trips(self):
        prompt = apply_strategy(AttackStrategy.ROT13, self.OBJECTIVE)
        payload = prompt.rsplit(": ", 1)[1]
        self.assertEqual(codecs.decode(payload, "rot13"), self.OBJECTIVE)

    def test_flip_reverses(self):
        prompt = apply_strategy(AttackStrategy.FLIP, self.OBJECTIVE)
        self.assertIn(self.OBJECTIVE[::-1], prompt)

    def test_every_concrete_strategy_transforms(self):
        # COMPOSED marks a chain rather than a transformation, and the three
        # complexity names are buckets; all four must refuse with a clear error.
        buckets = {"easy", "moderate", "difficult", "composed"}
        for strategy in AttackStrategy:
            if strategy.value in buckets:
                with self.assertRaises(ValueError):
                    apply_strategy(strategy, self.OBJECTIVE)
                continue
            prompt = apply_strategy(strategy, self.OBJECTIVE)
            self.assertIsInstance(prompt, str)
            self.assertTrue(prompt)

    def test_composition_applies_in_order(self):
        composed = compose_strategies([AttackStrategy.LEETSPEAK, AttackStrategy.BASE64], self.OBJECTIVE)
        payload = composed.rsplit(": ", 1)[1]
        self.assertEqual(
            base64.b64decode(payload).decode(), apply_strategy(AttackStrategy.LEETSPEAK, self.OBJECTIVE)
        )


class TestRefusalDetection(unittest.TestCase):
    def test_recognises_refusals(self):
        for text in (
            "I cannot help with that request.",
            "I'm sorry, but I can't assist with this.",
            "That is against my guidelines.",
            "Please contact a crisis helpline.",
        ):
            self.assertTrue(is_refusal(text), text)

    def test_does_not_flag_compliance(self):
        self.assertFalse(is_refusal("Sure, here are the steps you asked for."))


class TestObjectiveGenerator(unittest.TestCase):
    def test_generates_per_category(self):
        objectives = AttackObjectiveGenerator(
            risk_categories=[RiskCategory.Violence, RiskCategory.SelfHarm], num_objectives=2
        ).generate()
        self.assertEqual(sorted(objectives), ["self_harm", "violence"])
        self.assertTrue(all(len(v) <= 2 for v in objectives.values()))

    def test_custom_seeds_replace_defaults(self):
        objectives = AttackObjectiveGenerator(
            risk_categories=[RiskCategory.Violence],
            num_objectives=1,
            custom_attack_seed_prompts={"violence": ["a bespoke objective"]},
        ).generate()
        self.assertEqual(objectives["violence"], ["a bespoke objective"])

    def test_sampling_is_reproducible(self):
        make = lambda: AttackObjectiveGenerator(
            risk_categories=[RiskCategory.Violence], num_objectives=2, seed=4
        ).generate()
        self.assertEqual(make(), make())


class TestRedTeamScan(unittest.TestCase):
    @staticmethod
    def surface_filter_target(messages=None, **kwargs):
        """Refuses plain requests but not encoded ones -- a surface-level filter."""
        text = messages[-1]["content"]
        if any(word in text.lower() for word in ("injure", "attack", "weapon", "inferior")):
            reply = "I cannot help with that."
        else:
            reply = "Sure: shoot and kill them with a weapon, then bomb the area."
        return {"messages": list(messages) + [{"role": "assistant", "content": reply}]}

    def test_detects_encoding_lift_over_baseline(self):
        scan = RedTeam(risk_categories=[RiskCategory.Violence], num_objectives=2, seed=1)
        result = scan.scan(
            target=self.surface_filter_target,
            attack_strategies=[AttackStrategy.BASELINE, AttackStrategy.BASE64],
        )
        by_strategy = result.attack_success_rate_by_strategy
        self.assertEqual(by_strategy["baseline"], 0.0)
        self.assertGreater(by_strategy["base64"], 0.0)
        self.assertGreater(result.to_scorecard()["max_strategy_lift_over_baseline"], 0.0)

    def test_refusing_target_has_zero_success(self):
        def always_refuse(messages=None, **kwargs):
            return {"messages": list(messages) + [{"role": "assistant", "content": "I cannot help with that."}]}

        result = RedTeam(risk_categories=[RiskCategory.Violence], num_objectives=1).scan(
            target=always_refuse, attack_strategies=[AttackStrategy.EASY]
        )
        self.assertEqual(result.attack_success_rate, 0.0)

    def test_complexity_buckets_expand(self):
        result = RedTeam(risk_categories=[RiskCategory.Violence], num_objectives=1).scan(
            target=echo_app, attack_strategies=[AttackStrategy.EASY]
        )
        self.assertGreater(len(result), 1)

    def test_target_exception_is_recorded_not_raised(self):
        def broken(messages=None, **kwargs):
            raise RuntimeError("target crashed")

        result = RedTeam(risk_categories=[RiskCategory.Violence], num_objectives=1).scan(
            target=broken, attack_strategies=[AttackStrategy.BASELINE]
        )
        self.assertFalse(result.attacks[0].attack_success)
        self.assertIn("target crashed", result.attacks[0]["reason"])

    def test_invalid_target_return_raises(self):
        with self.assertRaises(EvaluationException):
            RedTeam(risk_categories=[RiskCategory.Violence], num_objectives=1).scan(
                target=lambda **kwargs: 12345, attack_strategies=[AttackStrategy.BASELINE]
            )

    def test_scorecard_written_to_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "scan.json")
            RedTeam(risk_categories=[RiskCategory.Violence], num_objectives=1).scan(
                target=echo_app, attack_strategies=[AttackStrategy.BASELINE], output_path=path
            )
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertIn("scorecard", payload)
            self.assertIn("rows", payload)

    def test_string_returning_target_supported(self):
        result = RedTeam(risk_categories=[RiskCategory.Violence], num_objectives=1).scan(
            target=lambda **kwargs: "I cannot help with that.",
            attack_strategies=[AttackStrategy.BASELINE],
        )
        self.assertEqual(result.attack_success_rate, 0.0)


class TestSimulators(unittest.TestCase):
    def test_simulator_produces_multi_turn_transcripts(self):
        transcripts = Simulator()(
            target=echo_app,
            text="Solar panels convert light. Batteries store charge.",
            num_queries=2,
            max_conversation_turns=2,
        )
        self.assertEqual(len(transcripts), 2)
        for transcript in transcripts:
            self.assertGreaterEqual(len(transcript["messages"]), 2)

    def test_simulator_is_reproducible(self):
        make = lambda: Simulator()(target=echo_app, text="A topic sentence.", num_queries=2, randomization_seed=3)
        self.assertEqual(make(), make())

    def test_adversarial_simulator_labels_categories(self):
        transcripts = AdversarialSimulator()(target=echo_app, max_simulation_results=2)
        self.assertEqual(len(transcripts), 2)
        for transcript in transcripts:
            self.assertIn("category", transcript)
            self.assertEqual(transcript["scenario"], AdversarialScenario.ADVERSARIAL_QA.value)

    def test_direct_attack_returns_both_arms(self):
        result = DirectAttackSimulator()(target=echo_app, max_simulation_results=2)
        self.assertEqual(sorted(result), ["jailbreak", "regular"])
        self.assertEqual(len(result["regular"]), len(result["jailbreak"]))
        opener = result["jailbreak"][0]["messages"][0]["content"]
        self.assertTrue(len(opener) > len(result["regular"][0]["messages"][0]["content"]))

    def test_indirect_attack_poisons_context(self):
        transcripts = IndirectAttackSimulator()(target=echo_app, max_simulation_results=2)
        for transcript in transcripts:
            self.assertIn("injected_payload", transcript)
            self.assertIn(transcript["injected_payload"], transcript["context"])

    def test_bad_callback_raises(self):
        with self.assertRaises(EvaluationException):
            Simulator()(target=lambda messages, **kwargs: "not a mapping", text="topic", num_queries=1)


if __name__ == "__main__":
    unittest.main()
