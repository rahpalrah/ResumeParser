# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""Composite evaluators that bundle a standard set of metrics."""

from typing import Any, List, Mapping, Optional

from ._common._base import MultiEvaluatorBase
from ._nlp import F1ScoreEvaluator
from ._quality import (
    CoherenceEvaluator,
    FluencyEvaluator,
    GroundednessEvaluator,
    RelevanceEvaluator,
    SimilarityEvaluator,
)

__all__ = ["QAEvaluator"]


class QAEvaluator(MultiEvaluatorBase):
    """The standard question-answering bundle.

    Runs groundedness, relevance, coherence, fluency, similarity and token F1
    in one call and merges their columns.

    .. code-block:: python

        QAEvaluator(model_config)(
            query="How tall is the tower?",
            context="The tower is 330 metres tall.",
            response="330 metres.",
            ground_truth="330 metres.",
        )

    :param model_config: Judge model configuration shared by the AI-assisted members.
    :param threshold: Pass threshold applied to every AI-assisted member.
    """

    _singleton_inputs = ["query", "context", "response", "ground_truth"]
    id = "evalforge.evaluators.qa"

    def __init__(self, model_config: Optional[Mapping[str, Any]] = None, *, threshold: float = 3, **kwargs: Any) -> None:
        evaluators = [
            GroundednessEvaluator(model_config, threshold=threshold),
            RelevanceEvaluator(model_config, threshold=threshold),
            CoherenceEvaluator(model_config, threshold=threshold),
            FluencyEvaluator(model_config, threshold=threshold),
            SimilarityEvaluator(model_config, threshold=threshold),
            F1ScoreEvaluator(),
        ]
        super().__init__(evaluators, **kwargs)

    def _derive_singleton_inputs(self) -> List[str]:
        return ["query", "response"]
