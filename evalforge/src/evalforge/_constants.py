# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""Shared constants and metric name registries."""

from enum import Enum

__all__ = ["EvaluationMetrics", "DefaultOpenEncoding", "Prefixes", "EVALUATION_PASS_FAIL_MAPPING"]


class EvaluationMetrics(str, Enum):
    """Canonical metric identifiers understood by the evaluation service."""

    GROUNDEDNESS = "groundedness"
    RELEVANCE = "relevance"
    COHERENCE = "coherence"
    FLUENCY = "fluency"
    SIMILARITY = "similarity"
    RETRIEVAL = "retrieval"
    F1_SCORE = "f1_score"
    GLEU_SCORE = "gleu_score"
    BLEU_SCORE = "bleu_score"
    ROUGE_SCORE = "rouge_score"
    METEOR_SCORE = "meteor_score"
    VIOLENCE = "violence"
    SEXUAL = "sexual"
    SELF_HARM = "self_harm"
    HATE_UNFAIRNESS = "hate_unfairness"
    HATE_FAIRNESS = "hate_fairness"
    PROTECTED_MATERIAL = "protected_material"
    XPIA = "xpia"
    CODE_VULNERABILITY = "code_vulnerability"
    UNGROUNDED_ATTRIBUTES = "ungrounded_attributes"
    INTENT_RESOLUTION = "intent_resolution"
    TASK_ADHERENCE = "task_adherence"
    TOOL_CALL_ACCURACY = "tool_call_accuracy"
    RESPONSE_COMPLETENESS = "response_completeness"
    # EvalForge novel metrics (no counterpart in the reference SDK).
    SPECTRAL_SEMANTIC_DRIFT = "spectral_semantic_drift"
    CAUSAL_GROUNDEDNESS = "causal_groundedness"
    JUDGE_LATENT_QUALITY = "judge_latent_quality"
    SEMANTIC_CURVATURE = "semantic_curvature"
    CONFORMAL_RISK = "conformal_risk"
    TRANSPORT_FAIRNESS = "transport_fairness"


class DefaultOpenEncoding:
    """Encodings used whenever EvalForge touches the filesystem."""

    READ = "utf-8-sig"
    WRITE = "utf-8"


class Prefixes:
    """Column-name prefixes used by :func:`evalforge.evaluate`."""

    INPUTS = "inputs."
    OUTPUTS = "outputs."
    TSG_OUTPUTS = "__outputs."
    RUN_OUTPUTS = "${run.outputs."
    DATA = "${data."
    TARGET = "${target."


#: Threshold semantics shared by every "<metric>_result" column.
EVALUATION_PASS_FAIL_MAPPING = {True: "pass", False: "fail"}

#: Default number of parallel workers used by the batch engine.
DEFAULT_MAX_WORKERS = 4

#: Likert range used by every AI-assisted quality evaluator.
QUALITY_SCORE_MIN = 1
QUALITY_SCORE_MAX = 5

#: Harm severity range used by every content-safety evaluator.
HARM_SEVERITY_MIN = 0
HARM_SEVERITY_MAX = 7
