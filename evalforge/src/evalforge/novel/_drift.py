# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""Spectral Semantic Drift (SSD).

Problem
-------
Long conversations fail gradually. Each individual turn looks acceptable to a
per-turn evaluator, yet by turn fifteen the assistant is answering a different
question than the one it was asked. Per-turn scoring cannot see this, and a
simple first-vs-last similarity cannot distinguish *drift* (a sustained regime
change) from *volatility* (one noisy turn) or from legitimate topic development.

Method
------
Treat the conversation as a graph signal.

1. Embed each assistant turn, giving ``v_1..v_T``, and embed the conversation's
   anchor (its opening goal or system instruction), giving ``a``.
2. Define the *anchor-alignment signal* ``s_i = cos(v_i, a)`` -- how on-goal
   each turn is. Drift and volatility are both properties of this signal; they
   differ only in *how* it varies.
3. Decompose ``s`` on a **temporal graph** whose edges depend only on turn
   distance, ``W^t_ij = exp(-|i-j| / tau)``. Its normalised Laplacian has a
   fixed, signal-independent eigenbasis: low eigenvectors are smooth ramps,
   high eigenvectors alternate turn to turn. Energy that ``s`` puts in the low
   band (excluding the DC mode) is a *sustained* change -- drift. Energy in the
   high band is turn-to-turn thrash -- volatility.

   The graph used for this decomposition must not depend on semantics. Building
   edges from ``cos(v_i, v_j)`` seems natural but is self-defeating: it lowers
   exactly the edges the signal jumps across, so an alternating conversation
   becomes *smooth* with respect to its own graph and its volatility vanishes.
4. Separately build a **semantic graph** ``W^s_ij = max(0, cos(v_i, v_j)) *
   exp(-|i-j| / tau)``, whose Fiedler vector partitions the conversation into
   two coherent blocks. Its sign change localises *when* the regime changed,
   and its eigenvalue (the algebraic connectivity) says how cleanly the
   conversation splits in two at all.
5. Drift is reported only when alignment actually decays, and both quantities
   are gated on the absolute power of ``s`` so that a flat, on-goal
   conversation cannot report a large ratio of two near-zero energies.

What is novel
-------------
Per-turn quality scoring, embedding-variance self-consistency, and first-vs-last
similarity are all established. The contribution here is the **dual-graph
decomposition**: a semantics-independent temporal graph that separates drift
from volatility by spectral band, paired with a semantic graph whose Fiedler
vector localises the onset turn and quantifies how cleanly the conversation
splits. Using one graph for both -- the obvious construction -- provably
suppresses the volatility signal, which is why the two are kept distinct.
"""

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .._common._embedding import Encoder, get_default_encoder
from .._common._linalg import clamp, cosine, jacobi_eigh, mean, normalized_laplacian
from .._evaluators._common._base import EvaluatorBase

__all__ = ["SpectralSemanticDrift", "SpectralDriftEvaluator"]


class SpectralSemanticDrift:
    """Computes the spectral drift decomposition of a conversation.

    :param encoder: Text encoder. Defaults to the process encoder.
    :param tau: Temporal kernel width, in turns. Smaller values make the metric
        more sensitive to abrupt changes.
    :param band_width: Width of the Fiedler band as a fraction of the spectrum.
    """

    def __init__(
        self,
        *,
        encoder: Optional[Encoder] = None,
        tau: float = 3.0,
        band_width: float = 0.25,
    ) -> None:
        self._encoder = encoder or get_default_encoder()
        self._tau = max(1e-6, tau)
        self._band_width = clamp(band_width, 0.05, 1.0)

    def _temporal_affinity(self, n: int) -> List[List[float]]:
        """Turn-distance kernel only. Deliberately independent of content.

        The eigenbasis of this graph is what gives "low frequency" its meaning
        (a smooth ramp across turns) and "high frequency" its meaning
        (turn-to-turn alternation), regardless of what was said.
        """
        w = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                value = math.exp(-abs(i - j) / self._tau)
                w[i][j] = value
                w[j][i] = value
        return w

    def _semantic_affinity(self, vectors: Sequence[Sequence[float]]) -> List[List[float]]:
        """Semantic similarity modulated by the temporal kernel."""
        n = len(vectors)
        w = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                value = max(0.0, cosine(vectors[i], vectors[j])) * math.exp(-abs(i - j) / self._tau)
                w[i][j] = value
                w[j][i] = value
        return w

    def analyze(
        self,
        turns: Sequence[str],
        *,
        anchor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Decompose a turn sequence into drift, volatility and onset.

        :param turns: Assistant turns, in order.
        :param anchor: The conversation's goal. Defaults to the first turn,
            which measures drift away from where the conversation started.
        :return: Metric dictionary; see the module docstring for the meaning of
            each quantity.
        """
        turns = [t for t in turns if t and str(t).strip()]
        n = len(turns)
        if n < 3:
            return {
                "spectral_semantic_drift": 0.0,
                "spectral_volatility": 0.0,
                "drift_onset_turn": None,
                "anchor_alignment": [],
                "fiedler_value": 0.0,
                "turns_analyzed": n,
                "reason": "At least three turns are required for a spectral decomposition.",
            }

        vectors = self._encoder.encode(list(turns))
        anchor_vector = self._encoder.encode_one(anchor) if anchor else vectors[0]

        # Anchor-alignment signal, mean-centred so the DC component carries the
        # average alignment and the remaining energy carries its variation.
        alignment = [cosine(v, anchor_vector) for v in vectors]
        mu = mean(alignment)
        signal = [a - mu for a in alignment]

        # Band decomposition on the semantics-independent temporal graph.
        temporal_eigenvalues, temporal_eigenvectors = jacobi_eigh(
            normalized_laplacian(self._temporal_affinity(n))
        )
        coefficients = [
            sum(s * u for s, u in zip(signal, vector)) for vector in temporal_eigenvectors
        ]
        # Index 0 is the DC/trivial mode; drift lives in the variation.
        energies = [c * c for c in coefficients[1:]]
        total_energy = math.fsum(energies)

        # Onset localisation on the semantic graph.
        semantic_eigenvalues, semantic_eigenvectors = jacobi_eigh(
            normalized_laplacian(self._semantic_affinity(vectors))
        )
        fiedler_value = semantic_eigenvalues[1] if len(semantic_eigenvalues) > 1 else 0.0

        # Both fractions are ratios, so they are meaningless when the alignment
        # signal is essentially flat: a perfectly on-topic conversation has
        # near-zero variation energy and would otherwise report whatever numeric
        # noise the ratio produces. Gate on absolute signal power.
        alignment_rms = math.sqrt(total_energy / n) if n else 0.0
        activity = clamp(alignment_rms / 0.10, 0.0, 1.0)

        if total_energy <= 1e-12:
            drift_fraction = 0.0
            volatility_fraction = 0.0
        else:
            # Weight each non-DC mode by how "smooth" it is on the temporal
            # graph: weight 1 at the smoothest non-trivial mode, decaying with
            # eigenvalue. Drift is the smooth part, volatility the rest.
            low = temporal_eigenvalues[1]
            high = temporal_eigenvalues[-1]
            spread = max(high - low, 1e-9)
            width = max(self._band_width * spread, 1e-9)
            band_weights = [
                math.exp(-((temporal_eigenvalues[k + 1] - low) ** 2) / (2 * width * width))
                for k in range(len(energies))
            ]
            drift_fraction = math.fsum(e * w for e, w in zip(energies, band_weights)) / total_energy
            volatility_fraction = math.fsum(
                e * (1.0 - w) for e, w in zip(energies, band_weights)
            ) / total_energy

        # Direction and magnitude: drift only counts when alignment *decays*.
        half = n // 2
        early = mean(alignment[:half]) if half else alignment[0]
        late = mean(alignment[half:])
        decay = early - late
        magnitude = clamp(decay / 0.30, 0.0, 1.0) if decay > 0 else 0.0

        drift_index = clamp(drift_fraction * magnitude * activity, 0.0, 1.0)
        volatility_index = clamp(volatility_fraction * activity, 0.0, 1.0)

        onset = (
            self._onset(semantic_eigenvectors[1])
            if len(semantic_eigenvectors) > 1 and drift_index > 0.05
            else None
        )

        return {
            "spectral_semantic_drift": drift_index,
            "spectral_volatility": volatility_index,
            "drift_onset_turn": onset,
            "anchor_alignment": [round(a, 6) for a in alignment],
            "anchor_alignment_early": early,
            "anchor_alignment_late": late,
            "fiedler_value": fiedler_value,
            "drift_band_energy_fraction": drift_fraction,
            "turns_analyzed": n,
            "alignment_rms": alignment_rms,
            "reason": self._explain(drift_index, volatility_index, onset, decay),
        }

    @staticmethod
    def _onset(fiedler_vector: Sequence[float]) -> Optional[int]:
        """Turn index at which the Fiedler vector changes sign."""
        for i in range(1, len(fiedler_vector)):
            if fiedler_vector[i - 1] == 0.0:
                continue
            if (fiedler_vector[i - 1] > 0) != (fiedler_vector[i] > 0):
                return i
        return None

    @staticmethod
    def _explain(drift: float, volatility: float, onset: Optional[int], decay: float) -> str:
        """Human-readable summary of the decomposition."""
        if drift < 0.1 and volatility < 0.4:
            return "The conversation stayed aligned with its anchor; no sustained drift detected."
        if drift < 0.1:
            return (
                f"No sustained drift, but turn-to-turn volatility is high ({volatility:.2f}): "
                "alignment fluctuates without a regime change."
            )
        where = f" beginning around turn {onset}" if onset is not None else ""
        return (
            f"Sustained drift detected (index {drift:.2f}){where}: anchor alignment fell by "
            f"{decay:.3f} between the first and second half of the conversation."
        )


class SpectralDriftEvaluator(EvaluatorBase):
    """Evaluator wrapper around :class:`SpectralSemanticDrift`.

    Accepts a ``conversation`` and reports drift for the whole exchange, so it
    slots into :func:`evalforge.evaluate` beside the per-turn evaluators.

    .. code-block:: python

        SpectralDriftEvaluator()(conversation={"messages": [...]})

    :param threshold: Maximum passing drift index. Lower is better.
    :param anchor_from: ``"first_user"`` anchors on the opening user request;
        ``"first_turn"`` anchors on the first assistant turn.
    """

    id = "evalforge.evaluators.spectral_semantic_drift"
    _singleton_inputs = ["conversation"]

    def __init__(
        self,
        *,
        threshold: float = 0.35,
        anchor_from: str = "first_user",
        encoder: Optional[Encoder] = None,
        tau: float = 3.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, higher_is_better=False, **kwargs)
        self._algorithm = SpectralSemanticDrift(encoder=encoder, tau=tau)
        self._anchor_from = anchor_from

    def _derive_singleton_inputs(self) -> List[str]:
        return []

    def _convert_kwargs_to_eval_input(self, **kwargs: Any) -> List[Dict[str, Any]]:
        conversation = kwargs.get("conversation")
        if conversation is None:
            from .._exceptions import ErrorBlame, ErrorCategory, ErrorTarget, EvaluationException

            raise EvaluationException(
                "SpectralDriftEvaluator requires a 'conversation'.",
                target=ErrorTarget.CONVERSATION,
                category=ErrorCategory.MISSING_FIELD,
                blame=ErrorBlame.USER_ERROR,
            )
        return [{"conversation": conversation}]

    async def _do_eval(self, eval_input: Dict[str, Any]) -> Dict[str, Any]:
        conversation: Mapping[str, Any] = eval_input["conversation"]
        messages = list(conversation.get("messages") or [])
        turns = [
            m.get("content")
            for m in messages
            if m.get("role") == "assistant" and isinstance(m.get("content"), str)
        ]
        anchor = None
        if self._anchor_from == "first_user":
            anchor = next(
                (m.get("content") for m in messages if m.get("role") == "user" and isinstance(m.get("content"), str)),
                None,
            )
        result = self._algorithm.analyze(turns, anchor=anchor)
        score = result["spectral_semantic_drift"]
        return {
            "spectral_semantic_drift": score,
            "spectral_semantic_drift_reason": result["reason"],
            "spectral_semantic_drift_result": self._passed(score),
            "spectral_semantic_drift_threshold": self._threshold,
            "spectral_volatility": result["spectral_volatility"],
            "drift_onset_turn": result["drift_onset_turn"],
            "fiedler_value": result["fiedler_value"],
        }
