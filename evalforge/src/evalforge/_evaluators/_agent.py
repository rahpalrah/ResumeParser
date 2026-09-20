# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""Agent-trajectory evaluators."""

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .._common._text import tokenize
from ._common._base import EvaluatorBase

__all__ = ["ToolCallAccuracyEvaluator"]


def _normalise_tool_calls(raw: Any) -> List[Dict[str, Any]]:
    """Accept the several shapes a tool call arrives in and flatten them."""
    if raw in (None, "", []):
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return []
    if isinstance(raw, Mapping):
        raw = [raw]
    calls: List[Dict[str, Any]] = []
    for item in raw or []:
        if not isinstance(item, Mapping):
            continue
        # OpenAI shape: {"type": "function", "function": {"name", "arguments"}}
        function = item.get("function") if isinstance(item.get("function"), Mapping) else item
        name = function.get("name") or item.get("name")
        arguments = function.get("arguments", item.get("arguments", {}))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = {"_raw": arguments}
        if name:
            calls.append({"name": str(name), "arguments": dict(arguments or {})})
    return calls


class ToolCallAccuracyEvaluator(EvaluatorBase):
    """Scores whether an agent selected the right tools with the right arguments.

    Three components are combined:

    * **selection** -- did the agent call the tools it should have, and no others?
    * **parameters** -- do the arguments match the expected arguments?
    * **admissibility** -- was every called tool actually declared in
      ``tool_definitions``? Calling an undeclared tool is always a defect.

    .. code-block:: python

        ToolCallAccuracyEvaluator()(
            query="Weather in Paris?",
            tool_calls=[{"name": "get_weather", "arguments": {"city": "Paris"}}],
            tool_definitions=[{"name": "get_weather", "parameters": {"city": "string"}}],
        )

    :param threshold: Minimum passing accuracy. Default 0.8.
    """

    _singleton_inputs = ["query", "tool_calls", "tool_definitions", "ground_truth"]
    id = "evalforge.evaluators.tool_call_accuracy"

    def __init__(self, model_config: Optional[Mapping[str, Any]] = None, *, threshold: float = 0.8, **kwargs: Any) -> None:
        super().__init__(threshold=threshold, higher_is_better=True, **kwargs)
        self._model_config = dict(model_config) if model_config else None

    def _derive_singleton_inputs(self) -> List[str]:
        return ["tool_calls"]

    @staticmethod
    def _argument_match(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> float:
        """Fraction of expected arguments present with an equivalent value."""
        if not expected:
            return 1.0
        hits = 0.0
        for key, want in expected.items():
            if key not in actual:
                continue
            got = actual[key]
            if got == want:
                hits += 1.0
            elif isinstance(got, str) and isinstance(want, str):
                # Tolerate case/whitespace differences in free-text arguments.
                if tokenize(got) == tokenize(want):
                    hits += 1.0
                elif tokenize(want) and set(tokenize(want)) <= set(tokenize(got)):
                    hits += 0.5
        return hits / len(expected)

    async def _do_eval(self, eval_input: Dict[str, Any]) -> Dict[str, Any]:
        actual = _normalise_tool_calls(eval_input.get("tool_calls"))
        expected = _normalise_tool_calls(eval_input.get("ground_truth"))
        definitions = {
            str(d.get("name"))
            for d in (eval_input.get("tool_definitions") or [])
            if isinstance(d, Mapping) and d.get("name")
        }

        details: Dict[str, Any] = {"calls": len(actual), "expected_calls": len(expected)}

        if not actual:
            score = 0.0
            reason = "The agent made no tool calls."
            details["undeclared_tools"] = []
        else:
            undeclared = sorted({c["name"] for c in actual if definitions and c["name"] not in definitions})
            details["undeclared_tools"] = undeclared
            admissibility = 0.0 if undeclared else 1.0

            if expected:
                expected_names = [c["name"] for c in expected]
                actual_names = [c["name"] for c in actual]
                matched = 0.0
                parameter_scores: List[float] = []
                remaining = list(expected)
                for call in actual:
                    for candidate in remaining:
                        if candidate["name"] == call["name"]:
                            matched += 1.0
                            parameter_scores.append(self._argument_match(call["arguments"], candidate["arguments"]))
                            remaining.remove(candidate)
                            break
                precision = matched / len(actual_names)
                recall = matched / len(expected_names)
                selection = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
                parameters = sum(parameter_scores) / len(parameter_scores) if parameter_scores else 0.0
                details.update({"selection": selection, "parameters": parameters, "admissibility": admissibility})
                score = 0.5 * selection + 0.3 * parameters + 0.2 * admissibility
                reason = (
                    f"Selection F1 {selection:.2f} over {len(expected_names)} expected call(s); "
                    f"argument match {parameters:.2f}; "
                    + ("all tools declared." if admissibility else f"undeclared tool(s): {undeclared}.")
                )
            else:
                # No ground truth: admissibility is the only objective signal.
                details.update({"selection": None, "parameters": None, "admissibility": admissibility})
                score = admissibility
                reason = (
                    "No ground-truth tool calls supplied; scored on tool admissibility only. "
                    + ("All called tools were declared." if admissibility else f"Undeclared tool(s): {undeclared}.")
                )

        return {
            "tool_call_accuracy": score,
            "tool_call_accuracy_reason": reason,
            "tool_call_accuracy_result": self._passed(score),
            "tool_call_accuracy_threshold": self._threshold,
            "tool_call_accuracy_details": details,
        }
