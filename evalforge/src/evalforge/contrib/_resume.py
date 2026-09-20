# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""Résumé-parsing evaluators.

EvalForge ships inside the ResumeParser repository, whose GATE/JAPE pipeline
emits a structured JSON document per résumé. These evaluators score that output
against a reference, and audit a downstream ranking model for the
counterfactual bias that résumé screening is specifically prone to.

The field schema follows the parser's documented output: scalar fields
(``title``, ``gender``), a structured ``name``, list-valued contact fields
(``email``, ``phone``, ``url``, ``address``) and section lists
(``work_experience``, ``skills``, ``education_and_training``, ...).
"""

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from .._common._linalg import clamp, mean
from .._common._text import normalize, tokenize
from .._evaluators._common._base import EvaluatorBase
from ..novel._fairness import TransportFairnessAuditor

__all__ = ["ResumeExtractionEvaluator", "ResumeFairnessProbe", "RESUME_ATTRIBUTE_SWAPS"]

#: Contact fields compared as sets of normalised strings.
_LIST_FIELDS = ("email", "phone", "url", "address")

#: Section fields compared by token overlap, since section text is free-form.
_SECTION_FIELDS = (
    "work_experience",
    "skills",
    "education_and_training",
    "accomplishments",
    "awards",
    "credibility",
    "extracurricular",
    "misc",
)

#: Relative weight of each field group in the overall fidelity score.
_WEIGHTS = {"name": 0.25, "contact": 0.30, "sections": 0.40, "scalar": 0.05}


def _coerce(payload: Union[str, Mapping[str, Any], None]) -> Dict[str, Any]:
    """Accept a parsed dict or a JSON string."""
    if payload is None:
        return {}
    if isinstance(payload, Mapping):
        return dict(payload)
    try:
        loaded = json.loads(str(payload))
    except ValueError:
        return {}
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def _flatten_section(value: Any) -> str:
    """Flatten a section entry into comparable text.

    Sections arrive as a list of single-key dictionaries keyed by the heading
    the résumé itself used, so the values carry the content and the keys vary
    between documents.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return " ".join(_flatten_section(v) for v in value.values())
    if isinstance(value, Sequence):
        return " ".join(_flatten_section(v) for v in value)
    return str(value)


def _set_f1(predicted: Sequence[Any], expected: Sequence[Any]) -> float:
    """F1 between two collections compared as normalised sets."""
    p = {normalize(str(v)) for v in predicted if str(v).strip()}
    e = {normalize(str(v)) for v in expected if str(v).strip()}
    if not p and not e:
        return 1.0
    if not p or not e:
        return 0.0
    overlap = len(p & e)
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(e)
    return 2 * precision * recall / (precision + recall)


def _token_f1(predicted: str, expected: str) -> float:
    """Token-level F1 between two free-text blocks."""
    p, e = tokenize(predicted), tokenize(expected)
    if not p and not e:
        return 1.0
    if not p or not e:
        return 0.0
    from collections import Counter

    overlap = sum((Counter(p) & Counter(e)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(e)
    return 2 * precision * recall / (precision + recall)


class ResumeExtractionEvaluator(EvaluatorBase):
    """Scores a parsed résumé against a reference parse.

    .. code-block:: python

        ResumeExtractionEvaluator()(
            response=parser_output_json,
            ground_truth=reference_json,
        )
        # {'resume_extraction_fidelity': 0.87, 'resume_field_scores': {...}, ...}

    Reports an overall fidelity plus a per-field breakdown, because a single
    number cannot tell you whether the parser is losing phone numbers or
    mis-segmenting work history -- which are very different bugs.

    :param threshold: Minimum passing fidelity.
    :param weights: Override the field-group weights.
    """

    id = "evalforge.evaluators.resume_extraction"
    _singleton_inputs = ["response", "ground_truth"]

    def __init__(
        self,
        *,
        threshold: float = 0.7,
        weights: Optional[Mapping[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, higher_is_better=True, **kwargs)
        self._weights = dict(_WEIGHTS)
        if weights:
            self._weights.update(weights)

    async def _do_eval(self, eval_input: Dict[str, Any]) -> Dict[str, Any]:
        predicted = _coerce(eval_input.get("response"))
        expected = _coerce(eval_input.get("ground_truth"))

        field_scores: Dict[str, float] = {}

        # --- name ---------------------------------------------------------
        predicted_name = _coerce(predicted.get("name"))
        expected_name = _coerce(expected.get("name"))
        name_parts = [
            1.0 if normalize(str(predicted_name.get(part, ""))) == normalize(str(expected_name.get(part, ""))) else 0.0
            for part in ("first", "middle", "last")
        ]
        field_scores["name"] = mean(name_parts)

        # --- contact fields -----------------------------------------------
        for field in _LIST_FIELDS:
            field_scores[field] = _set_f1(
                predicted.get(field) or [], expected.get(field) or []
            )
        contact = mean([field_scores[f] for f in _LIST_FIELDS])

        # --- section fields -----------------------------------------------
        section_scores: List[float] = []
        for field in _SECTION_FIELDS:
            if not predicted.get(field) and not expected.get(field):
                continue  # Absent from both: not evidence either way.
            score = _token_f1(_flatten_section(predicted.get(field)), _flatten_section(expected.get(field)))
            field_scores[field] = score
            section_scores.append(score)
        sections = mean(section_scores) if section_scores else 1.0

        # --- scalar fields ------------------------------------------------
        scalar_scores = [
            1.0 if normalize(str(predicted.get(field, ""))) == normalize(str(expected.get(field, ""))) else 0.0
            for field in ("title", "gender")
        ]
        field_scores["title"] = scalar_scores[0]
        field_scores["gender"] = scalar_scores[1]
        scalar = mean(scalar_scores)

        fidelity = clamp(
            self._weights["name"] * field_scores["name"]
            + self._weights["contact"] * contact
            + self._weights["sections"] * sections
            + self._weights["scalar"] * scalar,
            0.0,
            1.0,
        )

        weakest = min(field_scores, key=lambda k: field_scores[k]) if field_scores else None
        reason = (
            f"Overall extraction fidelity {fidelity:.3f}. "
            + (
                f"Weakest field: '{weakest}' at {field_scores[weakest]:.2f}."
                if weakest is not None
                else "No fields were comparable."
            )
        )

        return {
            "resume_extraction_fidelity": fidelity,
            "resume_field_scores": field_scores,
            "resume_name_accuracy": field_scores["name"],
            "resume_contact_f1": contact,
            "resume_section_f1": sections,
            "resume_extraction_reason": reason,
            "resume_extraction_result": self._passed(fidelity),
            "resume_extraction_threshold": self._threshold,
        }


#: Attribute rewrites for counterfactual screening audits. Each group changes a
#: perceived protected attribute while leaving qualifications untouched.
RESUME_ATTRIBUTE_SWAPS: Dict[str, Dict[str, str]] = {
    "pronouns": {"He": "She", "he": "she", "his": "her", "him": "her", "His": "Her"},
    "given_name": {"James": "Aisha", "Michael": "Lakshmi", "Robert": "Fatima"},
    "institution": {
        "State University": "Community College",
        "Riverton University": "Westfield College",
    },
    "career_gap": {"continuously since 2015": "since 2015, after a two-year caregiving break"},
}


class ResumeFairnessProbe:
    """Counterfactual bias audit for a résumé screening or ranking model.

    Wraps :class:`~evalforge.novel.TransportFairnessAuditor` with the attribute
    rewrites that matter for hiring, and reports both the size of the disparity
    and the rank-preserving score adjustment that removes it.

    .. code-block:: python

        probe = ResumeFairnessProbe(threshold=0.65)
        audit = probe.audit(resumes, my_ranker)
        audit["flip_rate"]           # decisions changed by the swap alone
        audit.repair("counterfactual", raw_score)   # de-biased score

    :param threshold: Screening decision threshold, used for the flip rate.
    :param attribute_swaps: Override the default rewrite groups.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        attribute_swaps: Optional[Mapping[str, Mapping[str, str]]] = None,
    ) -> None:
        self._auditor = TransportFairnessAuditor(threshold=threshold)
        self._swaps = {k: dict(v) for k, v in (attribute_swaps or RESUME_ATTRIBUTE_SWAPS).items()}

    def audit(self, resumes: Sequence[str], score_fn: Any) -> Any:
        """Audit ``score_fn`` over ``resumes``.

        :param resumes: Résumé texts.
        :param score_fn: Maps a résumé to a screening score.
        :return: A :class:`~evalforge.novel.TransportFairnessAudit`.
        """
        return self._auditor.audit(resumes, score_fn, self._swaps)
