# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""Evaluator base classes.

An evaluator is a callable object. Calling it with keyword arguments returns a
flat ``dict`` of metric columns. Conversation-aware evaluators additionally
accept a ``conversation`` argument and aggregate per-turn results.

The class hierarchy mirrors the reference SDK:

``EvaluatorBase``            input normalisation, conversation fan-out, aggregation
  ``PromptyEvaluatorBase``   LLM-judged quality metrics on a Likert scale
  ``RaiServiceEvaluatorBase``harm/safety metrics scored by a safety service
  ``MultiEvaluatorBase``     composites that fan out to several sub-evaluators
"""

import json
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from ..._common._async_utils import run_allowing_running_loop
from ..._common._chat import ChatClient, client_from_configuration
from ..._constants import EVALUATION_PASS_FAIL_MAPPING
from ..._exceptions import ErrorBlame, ErrorCategory, ErrorTarget, EvaluationException

__all__ = [
    "EvaluatorBase",
    "PromptyEvaluatorBase",
    "RaiServiceEvaluatorBase",
    "MultiEvaluatorBase",
    "AggregationType",
    "DerivedEvaluatorName",
]

#: Keys that are never treated as a single-turn scalar input.
NOT_SINGLETON_INPUTS = ["conversation", "messages", "tool_definitions", "tool_calls"]

DerivedEvaluatorName = str


class AggregationType(str, Enum):
    """How per-turn conversation scores are reduced to a single score."""

    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    SUM = "sum"
    CUSTOM = "custom"


def _reduce(values: Sequence[float], how: AggregationType) -> float:
    numeric = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not numeric:
        return float("nan")
    if how is AggregationType.MAX:
        return max(numeric)
    if how is AggregationType.MIN:
        return min(numeric)
    if how is AggregationType.SUM:
        return sum(numeric)
    return sum(numeric) / len(numeric)


class EvaluatorBase(ABC):
    """Base class for every evaluator.

    :param not_singleton_inputs: Argument names that must not be interpreted as
        per-turn scalars when a conversation is expanded.
    :param eval_last_turn: Score only the final turn of a conversation.
    :param conversation_aggregation_type: Reduction applied across turns.
    :param threshold: Score at or beyond which the row is reported as ``pass``.
    :param higher_is_better: Direction of the metric's quality axis.
    """

    def __init__(
        self,
        *,
        not_singleton_inputs: Optional[List[str]] = None,
        eval_last_turn: bool = False,
        conversation_aggregation_type: AggregationType = AggregationType.MEAN,
        threshold: Optional[float] = None,
        higher_is_better: bool = True,
    ) -> None:
        self._not_singleton_inputs = list(not_singleton_inputs or NOT_SINGLETON_INPUTS)
        self._eval_last_turn = eval_last_turn
        self._conversation_aggregation_type = conversation_aggregation_type
        self._threshold = threshold
        self._higher_is_better = higher_is_better

    # -- naming -------------------------------------------------------------

    @property
    def id(self) -> str:
        """Stable identifier used in result column names and telemetry."""
        return f"evalforge.evaluators.{type(self).__name__}"

    # -- input handling -----------------------------------------------------

    def _derive_singleton_inputs(self) -> List[str]:
        """Names of the per-turn inputs this evaluator consumes."""
        return [
            name
            for name in getattr(self, "_singleton_inputs", [])
            if name not in self._not_singleton_inputs
        ]

    def _derive_conversation_converter(self, conversation: Mapping[str, Any]) -> List[Dict[str, Any]]:
        """Expand a conversation into one evaluation input per assistant turn.

        Each assistant message becomes a row whose ``query`` is the preceding
        user message and whose ``context`` is drawn from either the assistant
        turn's own context or the retrieved context attached to the user turn.
        """
        messages = list(conversation.get("messages") or [])
        global_context = conversation.get("context")
        rows: List[Dict[str, Any]] = []
        pending_query: Optional[str] = None
        pending_context: Optional[Any] = None

        for message in messages:
            role = message.get("role")
            content = message.get("content")
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            if role == "user":
                pending_query = text
                pending_context = message.get("context")
            elif role == "assistant":
                row: Dict[str, Any] = {"response": text}
                if pending_query is not None:
                    row["query"] = pending_query
                context = message.get("context") or pending_context or global_context
                if context is not None:
                    row["context"] = context if isinstance(context, str) else json.dumps(
                        context, ensure_ascii=False
                    )
                rows.append(row)

        if self._eval_last_turn and rows:
            return rows[-1:]
        return rows

    def _convert_kwargs_to_eval_input(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """Normalise call arguments into a list of evaluation inputs."""
        conversation = kwargs.get("conversation")
        singletons = {k: v for k, v in kwargs.items() if v is not None and k != "conversation"}

        if conversation is not None:
            if singletons:
                raise EvaluationException(
                    "Mixing 'conversation' with per-turn inputs "
                    f"({sorted(singletons)}) is not supported; pass one or the other.",
                    target=ErrorTarget.CONVERSATION,
                    category=ErrorCategory.INVALID_VALUE,
                    blame=ErrorBlame.USER_ERROR,
                )
            rows = self._derive_conversation_converter(conversation)
            if not rows:
                raise EvaluationException(
                    "The conversation contains no assistant turn to evaluate.",
                    target=ErrorTarget.CONVERSATION,
                    category=ErrorCategory.MISSING_FIELD,
                    blame=ErrorBlame.USER_ERROR,
                )
            return rows

        required = self._derive_singleton_inputs()
        # Only an absent key is a configuration error. An empty string is data:
        # a model that returned nothing must score zero, not fail the row, or a
        # batch run breaks on exactly the cases worth measuring.
        missing = [name for name in required if kwargs.get(name) is None]
        if missing:
            raise EvaluationException(
                f"{type(self).__name__} requires {sorted(required)}; missing {sorted(missing)}.",
                target=ErrorTarget.UNKNOWN,
                category=ErrorCategory.MISSING_FIELD,
                blame=ErrorBlame.USER_ERROR,
            )
        return [singletons]

    # -- aggregation --------------------------------------------------------

    def _aggregate_results(self, per_turn: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Reduce per-turn result dictionaries to a single result dictionary."""
        if len(per_turn) == 1:
            return dict(per_turn[0])

        aggregated: Dict[str, Any] = {}
        numeric_keys = [
            key
            for key in per_turn[0]
            if all(isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool) for row in per_turn)
        ]
        for key in numeric_keys:
            if key.endswith("_threshold"):
                aggregated[key] = per_turn[0][key]
                continue
            aggregated[key] = _reduce([row[key] for row in per_turn], self._conversation_aggregation_type)

        for key in per_turn[0]:
            if key.endswith("_result"):
                values = [row.get(key) for row in per_turn]
                aggregated[key] = EVALUATION_PASS_FAIL_MAPPING[all(v == "pass" for v in values)]
            elif key.endswith("_reason"):
                aggregated[key] = " | ".join(str(row.get(key, "")) for row in per_turn)

        aggregated["evaluation_per_turn"] = {
            key: [row.get(key) for row in per_turn] for key in per_turn[0]
        }
        return aggregated

    def _passed(self, score: float) -> str:
        """Apply the evaluator's threshold to ``score``."""
        if self._threshold is None:
            return EVALUATION_PASS_FAIL_MAPPING[True]
        ok = score >= self._threshold if self._higher_is_better else score <= self._threshold
        return EVALUATION_PASS_FAIL_MAPPING[bool(ok)]

    # -- execution ----------------------------------------------------------

    @abstractmethod
    async def _do_eval(self, eval_input: Dict[str, Any]) -> Dict[str, Any]:
        """Score a single normalised evaluation input."""

    async def _real_call(self, **kwargs: Any) -> Dict[str, Any]:
        rows = self._convert_kwargs_to_eval_input(**kwargs)
        results = [await self._do_eval(row) for row in rows]
        return self._aggregate_results(results)

    def __call__(self, **kwargs: Any) -> Dict[str, Any]:
        """Evaluate and return a flat dictionary of metric columns."""
        return run_allowing_running_loop(lambda: self._real_call(**kwargs))


class PromptyEvaluatorBase(EvaluatorBase):
    """Base class for LLM-judged quality metrics on an integer Likert scale.

    :param result_key: Metric name; determines every emitted column name.
    :param model_config: Judge model configuration. ``None`` selects the
        deterministic offline judge.
    :param threshold: Pass/fail cut-off applied to the score.
    """

    _LEGACY_PREFIX = "gpt_"

    def __init__(
        self,
        *,
        result_key: str,
        model_config: Optional[Mapping[str, Any]] = None,
        threshold: float = 3.0,
        higher_is_better: bool = True,
        scale: "tuple[int, int]" = (1, 5),
        client: Optional[ChatClient] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, higher_is_better=higher_is_better, **kwargs)
        self._result_key = result_key
        self._model_config = dict(model_config) if model_config else None
        self._scale = scale
        self._client = client or client_from_configuration(model_config)

    @property
    def result_key(self) -> str:
        """Metric name emitted by this evaluator."""
        return self._result_key

    def _build_messages(self, eval_input: Mapping[str, Any]) -> List[Dict[str, str]]:
        """Render the judge prompt for ``eval_input``."""
        from ._prompty import render_judge_prompt

        return render_judge_prompt(self._result_key, eval_input, scale=self._scale)

    @staticmethod
    def _parse_judgement(raw: str) -> Dict[str, Any]:
        """Extract ``{"score", "reason"}`` from a judge response.

        Tolerates fenced code blocks and prose padding, because hosted judges
        do not always honour a strict JSON instruction.
        """
        text = (raw or "").strip()
        fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.S)
        if fenced:
            text = fenced.group(1).strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, TypeError):
            pass
        brace = re.search(r"\{.*\}", text, re.S)
        if brace:
            try:
                parsed = json.loads(brace.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except (ValueError, TypeError):
                pass
        number = re.search(r"(\d+(?:\.\d+)?)", text)
        if number:
            return {"score": float(number.group(1)), "reason": text}
        raise EvaluationException(
            "Could not parse a score out of the judge response.",
            internal_message="judge response was not JSON and contained no numeric score",
            target=ErrorTarget.UNKNOWN,
            category=ErrorCategory.FAILED_EXECUTION,
            blame=ErrorBlame.SYSTEM_ERROR,
        )

    async def _do_eval(self, eval_input: Dict[str, Any]) -> Dict[str, Any]:
        messages = self._build_messages(eval_input)
        raw = self._client.chat(
            messages,
            hint={"metric": self._result_key, "fields": eval_input, "scale": self._scale},
            temperature=0.0,
            max_tokens=800,
        )
        judgement = self._parse_judgement(raw)
        low, high = self._scale
        try:
            score = float(judgement.get("score", low))
        except (TypeError, ValueError):
            score = float(low)
        score = min(max(score, float(low)), float(high))
        reason = str(judgement.get("reason", ""))

        key = self._result_key
        return {
            key: score,
            f"{self._LEGACY_PREFIX}{key}": score,
            f"{key}_reason": reason,
            f"{key}_result": self._passed(score),
            f"{key}_threshold": self._threshold,
        }


class RaiServiceEvaluatorBase(EvaluatorBase):
    """Base class for harm/safety metrics reported on a 0-7 severity scale.

    Severity is mapped to a ``very low``/``low``/``medium``/``high`` label, and
    the pass/fail column is inverted relative to quality metrics: *lower* is
    better.
    """

    SEVERITY_LABELS = ("Very low", "Low", "Medium", "High")

    def __init__(
        self,
        *,
        eval_metric: str,
        azure_ai_project: Optional[Mapping[str, Any]] = None,
        credential: Optional[Any] = None,
        threshold: float = 3.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, higher_is_better=False, **kwargs)
        self._eval_metric = eval_metric
        self._azure_ai_project = dict(azure_ai_project) if azure_ai_project else None
        self._credential = credential

    @classmethod
    def _severity_label(cls, severity: float) -> str:
        """Bucket a 0-7 severity into its human readable band."""
        index = min(int(severity) // 2, len(cls.SEVERITY_LABELS) - 1)
        return cls.SEVERITY_LABELS[index]

    async def _do_eval(self, eval_input: Dict[str, Any]) -> Dict[str, Any]:
        from ..._common._heuristics import harm_severity

        payload = " ".join(
            str(eval_input.get(key, "")) for key in ("query", "response", "context") if eval_input.get(key)
        )
        severity = float(harm_severity(payload, self._eval_metric))
        key = self._eval_metric
        return {
            key: self._severity_label(severity),
            f"{key}_score": severity,
            f"{key}_reason": (
                f"Severity {int(severity)}/7 assigned for category '{key}' by the offline "
                f"safety scorer."
            ),
            f"{key}_result": self._passed(severity),
            f"{key}_threshold": self._threshold,
        }


class MultiEvaluatorBase(EvaluatorBase):
    """Composite evaluator that merges the output of several sub-evaluators."""

    def __init__(self, evaluators: Sequence[EvaluatorBase], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._evaluators = list(evaluators)

    async def _do_eval(self, eval_input: Dict[str, Any]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        for evaluator in self._evaluators:
            accepted = set(evaluator._derive_singleton_inputs())
            arguments = {k: v for k, v in eval_input.items() if k in accepted}
            try:
                merged.update(await evaluator._real_call(**arguments))
            except EvaluationException as exc:
                # A composite should report which member failed rather than
                # discarding the results of the members that succeeded.
                name = type(evaluator).__name__
                merged[f"{name}_error"] = str(exc)
        return merged

    def _convert_kwargs_to_eval_input(self, **kwargs: Any) -> List[Dict[str, Any]]:
        conversation = kwargs.get("conversation")
        if conversation is not None:
            return self._derive_conversation_converter(conversation)
        return [{k: v for k, v in kwargs.items() if v is not None}]
