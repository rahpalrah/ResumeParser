# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""Error taxonomy.

Mirrors the ``azure.ai.evaluation`` exception surface so that user code that
catches ``EvaluationException`` and inspects ``category``/``blame``/``target``
behaves identically against either implementation.
"""

from enum import Enum
from typing import Optional

__all__ = [
    "ErrorBlame",
    "ErrorCategory",
    "ErrorTarget",
    "EvaluationException",
    "MissingRequiredFieldError",
]


class ErrorBlame(str, Enum):
    """Whether a failure is attributable to the caller or to the service."""

    USER_ERROR = "UserError"
    SYSTEM_ERROR = "SystemError"
    UNKNOWN = "Unknown"


class ErrorCategory(str, Enum):
    """Coarse classification used for aggregated error reporting."""

    INVALID_VALUE = "INVALID VALUE"
    UNKNOWN_FIELD = "UNKNOWN FIELD"
    MISSING_FIELD = "MISSING REQUIRED FIELD"
    FILE_OR_FOLDER_NOT_FOUND = "FILE OR FOLDER NOT FOUND"
    RESOURCE_NOT_FOUND = "RESOURCE NOT FOUND"
    FAILED_EXECUTION = "FAILED EXECUTION"
    SERVICE_UNAVAILABLE = "SERVICE UNAVAILABLE"
    UPLOAD_ERROR = "UPLOAD ERROR"
    NOT_APPLICABLE = "NOT APPLICABLE"
    UNKNOWN = "UNKNOWN"


class ErrorTarget(str, Enum):
    """Component in which a failure occurred."""

    EVAL_RUN = "EvalRun"
    EVALUATE = "Evaluate"
    CODE_CLIENT = "CodeClient"
    RAI_CLIENT = "RAIClient"
    GROUNDEDNESS_EVALUATOR = "GroundednessEvaluator"
    RELEVANCE_EVALUATOR = "RelevanceEvaluator"
    COHERENCE_EVALUATOR = "CoherenceEvaluator"
    FLUENCY_EVALUATOR = "FluencyEvaluator"
    SIMILARITY_EVALUATOR = "SimilarityEvaluator"
    RETRIEVAL_EVALUATOR = "RetrievalEvaluator"
    F1_EVALUATOR = "F1Evaluator"
    ROUGE_EVALUATOR = "RougeScoreEvaluator"
    CONTENT_SAFETY_EVALUATOR = "ContentSafetyEvaluator"
    PROTECTED_MATERIAL_EVALUATOR = "ProtectedMaterialEvaluator"
    INDIRECT_ATTACK_EVALUATOR = "IndirectAttackEvaluator"
    TOOL_CALL_ACCURACY_EVALUATOR = "ToolCallAccuracyEvaluator"
    ADVERSARIAL_SIMULATOR = "AdversarialSimulator"
    DIRECT_ATTACK_SIMULATOR = "DirectAttackSimulator"
    INDIRECT_ATTACK_SIMULATOR = "IndirectAttackSimulator"
    RED_TEAM = "RedTeam"
    CONVERSATION = "Conversation"
    MODELS = "Models"
    NOVEL_ALGORITHM = "NovelAlgorithm"
    UNKNOWN = "Unknown"


class EvaluationException(Exception):
    """Base exception raised by every EvalForge component.

    :param message: Human readable description of the failure.
    :param internal_message: Optional message safe for telemetry (no user data).
    :param target: Component that raised the error.
    :param category: Coarse error classification.
    :param blame: Whether the user or the system is responsible.
    """

    def __init__(
        self,
        message: str,
        internal_message: Optional[str] = None,
        *,
        target: ErrorTarget = ErrorTarget.UNKNOWN,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        blame: ErrorBlame = ErrorBlame.UNKNOWN,
        **kwargs: object,
    ) -> None:
        self.category = category
        self.target = target
        self.blame = blame
        self.internal_message = internal_message
        self.details = dict(kwargs)
        super().__init__(message)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"{type(self).__name__}(message={str(self)!r}, target={self.target.value!r}, "
            f"category={self.category.value!r}, blame={self.blame.value!r})"
        )


class MissingRequiredFieldError(EvaluationException):
    """Raised when a required evaluator input is absent."""

    def __init__(self, message: str, **kwargs: object) -> None:
        kwargs.setdefault("category", ErrorCategory.MISSING_FIELD)
        kwargs.setdefault("blame", ErrorBlame.USER_ERROR)
        super().__init__(message, **kwargs)  # type: ignore[arg-type]
