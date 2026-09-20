# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""EvalForge -- a generative-AI evaluation SDK.

EvalForge is an API-compatible reimplementation of the Azure AI Evaluation SDK
surface, written from the published API contract with no third-party
dependencies, plus :mod:`evalforge.novel`: eight evaluation algorithms with no
counterpart in the reference SDK.

Quick start
-----------
.. code-block:: python

    from evalforge import evaluate, GroundednessEvaluator, F1ScoreEvaluator

    result = evaluate(
        data="rows.jsonl",
        evaluators={
            "groundedness": GroundednessEvaluator(model_config),
            "f1": F1ScoreEvaluator(),
        },
    )
    print(result["metrics"])

Every AI-assisted evaluator accepts a model configuration. Omit it and the
evaluator falls back to a deterministic in-process scorer, so the whole surface
is runnable and testable without credentials or network access.

Compatibility
-------------
Public names, call signatures and result-column names match the reference SDK.
Code written against ``azure.ai.evaluation`` runs here by changing the import;
see ``docs/COMPATIBILITY.md`` for the mapping and for the documented
differences.
"""

__version__ = "1.0.0"

from ._constants import EvaluationMetrics
from ._evaluate import evaluate
from ._evaluators._agent import ToolCallAccuracyEvaluator
from ._evaluators._common import (
    AggregationType,
    EvaluatorBase,
    MultiEvaluatorBase,
    PromptyEvaluatorBase,
    RaiServiceEvaluatorBase,
)
from ._evaluators._composite import QAEvaluator
from ._evaluators._content_safety import (
    CodeVulnerabilityEvaluator,
    ContentSafetyEvaluator,
    HateUnfairnessEvaluator,
    IndirectAttackEvaluator,
    ProtectedMaterialEvaluator,
    SelfHarmEvaluator,
    SexualEvaluator,
    UngroundedAttributesEvaluator,
    ViolenceEvaluator,
)
from ._evaluators._nlp import (
    BleuScoreEvaluator,
    F1ScoreEvaluator,
    GleuScoreEvaluator,
    MeteorScoreEvaluator,
    RougeScoreEvaluator,
    RougeType,
)
from ._evaluators._quality import (
    CoherenceEvaluator,
    FluencyEvaluator,
    GroundednessEvaluator,
    GroundednessProEvaluator,
    IntentResolutionEvaluator,
    RelevanceEvaluator,
    ResponseCompletenessEvaluator,
    RetrievalEvaluator,
    SimilarityEvaluator,
    TaskAdherenceEvaluator,
)
from ._exceptions import ErrorBlame, ErrorCategory, ErrorTarget, EvaluationException
from ._model_configurations import (
    AzureAIProject,
    AzureOpenAIModelConfiguration,
    Conversation,
    EvaluationResult,
    EvaluatorConfig,
    LocalModelConfiguration,
    Message,
    OpenAIModelConfiguration,
)

__all__ = [
    "__version__",
    # Entry point
    "evaluate",
    # Quality evaluators
    "GroundednessEvaluator",
    "GroundednessProEvaluator",
    "RelevanceEvaluator",
    "CoherenceEvaluator",
    "FluencyEvaluator",
    "SimilarityEvaluator",
    "RetrievalEvaluator",
    "QAEvaluator",
    # Agent evaluators
    "IntentResolutionEvaluator",
    "TaskAdherenceEvaluator",
    "ToolCallAccuracyEvaluator",
    "ResponseCompletenessEvaluator",
    # NLP evaluators
    "F1ScoreEvaluator",
    "BleuScoreEvaluator",
    "GleuScoreEvaluator",
    "RougeScoreEvaluator",
    "RougeType",
    "MeteorScoreEvaluator",
    # Safety evaluators
    "ViolenceEvaluator",
    "SexualEvaluator",
    "SelfHarmEvaluator",
    "HateUnfairnessEvaluator",
    "ContentSafetyEvaluator",
    "ProtectedMaterialEvaluator",
    "IndirectAttackEvaluator",
    "CodeVulnerabilityEvaluator",
    "UngroundedAttributesEvaluator",
    # Configuration types
    "AzureOpenAIModelConfiguration",
    "OpenAIModelConfiguration",
    "LocalModelConfiguration",
    "AzureAIProject",
    "EvaluatorConfig",
    "EvaluationResult",
    "Conversation",
    "Message",
    "EvaluationMetrics",
    # Errors
    "EvaluationException",
    "ErrorBlame",
    "ErrorCategory",
    "ErrorTarget",
    # Extension points
    "EvaluatorBase",
    "PromptyEvaluatorBase",
    "RaiServiceEvaluatorBase",
    "MultiEvaluatorBase",
    "AggregationType",
]
