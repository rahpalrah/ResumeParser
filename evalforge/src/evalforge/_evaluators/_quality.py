# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""AI-assisted quality evaluators.

Each class scores one property of a model output on a 1-5 Likert scale using an
LLM judge, falling back to the deterministic offline scorer when no credentials
are configured. All of them accept either per-turn arguments or a
``conversation``.
"""

from typing import Any, List, Mapping, Optional

from ._common._base import AggregationType, PromptyEvaluatorBase

__all__ = [
    "GroundednessEvaluator",
    "GroundednessProEvaluator",
    "RelevanceEvaluator",
    "CoherenceEvaluator",
    "FluencyEvaluator",
    "SimilarityEvaluator",
    "RetrievalEvaluator",
    "IntentResolutionEvaluator",
    "TaskAdherenceEvaluator",
    "ResponseCompletenessEvaluator",
]


class GroundednessEvaluator(PromptyEvaluatorBase):
    """Measures how well a response is supported by its context.

    A response is grounded when every claim it makes follows from the supplied
    context. Contradictions and unsupported specifics both lower the score.

    .. code-block:: python

        evaluator = GroundednessEvaluator(model_config)
        evaluator(
            query="How tall is the tower?",
            context="The tower is 330 metres tall.",
            response="It is 330 metres tall.",
        )
        # {'groundedness': 5.0, 'groundedness_result': 'pass', ...}

    :param model_config: Judge model configuration; ``None`` uses the offline judge.
    :param threshold: Minimum passing score. Default 3.
    """

    _singleton_inputs = ["response", "context", "query"]
    id = "evalforge.evaluators.groundedness"

    def __init__(self, model_config: Optional[Mapping[str, Any]] = None, *, threshold: float = 3, **kwargs: Any) -> None:
        super().__init__(result_key="groundedness", model_config=model_config, threshold=threshold, **kwargs)

    def _derive_singleton_inputs(self) -> List[str]:
        # 'query' is optional context for the judge, not a requirement.
        return ["response", "context"]


class GroundednessProEvaluator(PromptyEvaluatorBase):
    """Service-backed groundedness with per-claim detail.

    Emits the same columns as :class:`GroundednessEvaluator` plus a boolean
    ``groundedness_pro_label`` suitable for defect-rate aggregation.

    :param azure_ai_project: Project the safety service call is billed to.
    :param credential: Token credential for that project.
    :param threshold: Minimum passing score. Default 3.
    """

    _singleton_inputs = ["response", "context", "query"]
    id = "evalforge.evaluators.groundedness_pro"

    def __init__(
        self,
        azure_ai_project: Optional[Mapping[str, Any]] = None,
        credential: Optional[Any] = None,
        *,
        threshold: float = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(result_key="groundedness", threshold=threshold, **kwargs)
        self._azure_ai_project = dict(azure_ai_project) if azure_ai_project else None
        self._credential = credential

    def _derive_singleton_inputs(self) -> List[str]:
        return ["response", "context"]

    async def _do_eval(self, eval_input: dict) -> dict:
        result = await super()._do_eval(eval_input)
        score = float(result["groundedness"])
        passed = result["groundedness_result"] == "pass"
        return {
            "groundedness_pro_label": passed,
            "groundedness_pro_reason": result["groundedness_reason"],
            "groundedness_pro_score": score,
            **result,
        }


class RelevanceEvaluator(PromptyEvaluatorBase):
    """Measures how well a response addresses the query.

    :param model_config: Judge model configuration.
    :param threshold: Minimum passing score. Default 3.
    """

    _singleton_inputs = ["query", "response"]
    id = "evalforge.evaluators.relevance"

    def __init__(self, model_config: Optional[Mapping[str, Any]] = None, *, threshold: float = 3, **kwargs: Any) -> None:
        super().__init__(result_key="relevance", model_config=model_config, threshold=threshold, **kwargs)


class CoherenceEvaluator(PromptyEvaluatorBase):
    """Measures the logical flow and organisation of a response.

    :param model_config: Judge model configuration.
    :param threshold: Minimum passing score. Default 3.
    """

    _singleton_inputs = ["query", "response"]
    id = "evalforge.evaluators.coherence"

    def __init__(self, model_config: Optional[Mapping[str, Any]] = None, *, threshold: float = 3, **kwargs: Any) -> None:
        super().__init__(result_key="coherence", model_config=model_config, threshold=threshold, **kwargs)


class FluencyEvaluator(PromptyEvaluatorBase):
    """Measures the linguistic quality of a response, independent of accuracy.

    :param model_config: Judge model configuration.
    :param threshold: Minimum passing score. Default 3.
    """

    _singleton_inputs = ["response"]
    id = "evalforge.evaluators.fluency"

    def __init__(self, model_config: Optional[Mapping[str, Any]] = None, *, threshold: float = 3, **kwargs: Any) -> None:
        super().__init__(result_key="fluency", model_config=model_config, threshold=threshold, **kwargs)


class SimilarityEvaluator(PromptyEvaluatorBase):
    """Measures semantic equivalence between a response and a ground truth.

    :param model_config: Judge model configuration.
    :param threshold: Minimum passing score. Default 3.
    """

    _singleton_inputs = ["query", "response", "ground_truth"]
    id = "evalforge.evaluators.similarity"

    def __init__(self, model_config: Optional[Mapping[str, Any]] = None, *, threshold: float = 3, **kwargs: Any) -> None:
        super().__init__(result_key="similarity", model_config=model_config, threshold=threshold, **kwargs)

    def _derive_singleton_inputs(self) -> List[str]:
        return ["response", "ground_truth"]


class RetrievalEvaluator(PromptyEvaluatorBase):
    """Measures how well retrieved context serves the query.

    Scores the *retrieval*, not the generated answer: whether the relevant
    chunks are present and ranked ahead of irrelevant ones.

    :param model_config: Judge model configuration.
    :param threshold: Minimum passing score. Default 3.
    """

    _singleton_inputs = ["query", "context"]
    id = "evalforge.evaluators.retrieval"

    def __init__(self, model_config: Optional[Mapping[str, Any]] = None, *, threshold: float = 3, **kwargs: Any) -> None:
        super().__init__(result_key="retrieval", model_config=model_config, threshold=threshold, **kwargs)


class IntentResolutionEvaluator(PromptyEvaluatorBase):
    """Measures whether an agent correctly identified and resolved user intent.

    :param model_config: Judge model configuration.
    :param threshold: Minimum passing score. Default 3.
    """

    _singleton_inputs = ["query", "response"]
    id = "evalforge.evaluators.intent_resolution"

    def __init__(self, model_config: Optional[Mapping[str, Any]] = None, *, threshold: float = 3, **kwargs: Any) -> None:
        super().__init__(result_key="intent_resolution", model_config=model_config, threshold=threshold, **kwargs)


class TaskAdherenceEvaluator(PromptyEvaluatorBase):
    """Measures adherence to the instructions and constraints of a task.

    :param model_config: Judge model configuration.
    :param threshold: Minimum passing score. Default 3.
    """

    _singleton_inputs = ["query", "response", "instructions"]
    id = "evalforge.evaluators.task_adherence"

    def __init__(self, model_config: Optional[Mapping[str, Any]] = None, *, threshold: float = 3, **kwargs: Any) -> None:
        super().__init__(result_key="task_adherence", model_config=model_config, threshold=threshold, **kwargs)

    def _derive_singleton_inputs(self) -> List[str]:
        return ["query", "response"]


class ResponseCompletenessEvaluator(PromptyEvaluatorBase):
    """Measures whether a response omits information required by ground truth.

    :param model_config: Judge model configuration.
    :param threshold: Minimum passing score. Default 3.
    """

    _singleton_inputs = ["response", "ground_truth"]
    id = "evalforge.evaluators.response_completeness"

    def __init__(self, model_config: Optional[Mapping[str, Any]] = None, *, threshold: float = 3, **kwargs: Any) -> None:
        super().__init__(
            result_key="response_completeness",
            model_config=model_config,
            threshold=threshold,
            conversation_aggregation_type=AggregationType.MEAN,
            **kwargs,
        )
