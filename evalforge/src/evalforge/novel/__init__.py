# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""Novel evaluation algorithms.

Eight algorithms that have no counterpart in the reference SDK. Each targets a
specific failure of conventional evaluation, and each module's docstring states
the problem, the method, and precisely what is claimed to be new.

===========================================  =============================================
Algorithm                                    Answers
===========================================  =============================================
:class:`SpectralSemanticDrift`               Did this conversation slowly go off-goal, and when?
:class:`CausalAblationGroundedness`          Which claim is unsupported, and by which passage?
:class:`JudgeItemResponseCalibrator`         Which of my judges can I trust, and how much?
:class:`SemanticCurvatureProbe`              Is the model guessing, or answering?
:class:`ConformalRiskController`             How many failures did automated review miss?
:class:`TransportFairnessAuditor`            How large is the bias, and how do I remove it?
:class:`AdaptiveAttackScheduler`             Where should a limited red-team budget go?
:class:`SequentialDriftCanary`               Has the metric regressed, given I check constantly?
===========================================  =============================================
"""

from ._bandit import AdaptiveAttackScheduler
from ._canary import CanaryAlarm, SequentialDriftCanary, benjamini_hochberg
from ._causal import (
    CausalAblationGroundedness,
    CausalGroundednessEvaluator,
    ProvenanceCertificate,
)
from ._conformal import ConformalRiskController, RiskCertificate, composite_nonconformity
from ._curvature import SemanticCurvatureEvaluator, SemanticCurvatureProbe
from ._drift import SpectralDriftEvaluator, SpectralSemanticDrift
from ._fairness import TransportFairnessAudit, TransportFairnessAuditor, swap_attributes
from ._irt import JudgeCalibration, JudgeItemResponseCalibrator
from ._paraphrase import paraphrase, perturbation_ladder

__all__ = [
    # Spectral Semantic Drift
    "SpectralSemanticDrift",
    "SpectralDriftEvaluator",
    # Causal Ablation Groundedness
    "CausalAblationGroundedness",
    "CausalGroundednessEvaluator",
    "ProvenanceCertificate",
    # Judge Item-Response Calibration
    "JudgeItemResponseCalibrator",
    "JudgeCalibration",
    # Semantic Curvature Probe
    "SemanticCurvatureProbe",
    "SemanticCurvatureEvaluator",
    "perturbation_ladder",
    "paraphrase",
    # Conformal Risk-Calibrated Evaluation
    "ConformalRiskController",
    "RiskCertificate",
    "composite_nonconformity",
    # Optimal-Transport Fairness Audit
    "TransportFairnessAuditor",
    "TransportFairnessAudit",
    "swap_attributes",
    # Coverage-Regularised Adaptive Attack Scheduling
    "AdaptiveAttackScheduler",
    # Anytime-Valid Evaluation Regression Canary
    "SequentialDriftCanary",
    "CanaryAlarm",
    "benjamini_hochberg",
]
