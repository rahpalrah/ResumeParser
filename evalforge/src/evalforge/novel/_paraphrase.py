# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""Deterministic, meaning-preserving perturbation operators.

Used to build the graded *perturbation ladders* that the curvature probe
differentiates along. Operators are deliberately surface-level: they change how
a question is phrased without changing what is being asked, so any change in
the answer is attributable to the model's instability rather than to the
question having changed.
"""

import re
from typing import Callable, List, Sequence

__all__ = ["PARAPHRASE_OPERATORS", "perturbation_ladder", "paraphrase"]

#: Small, safe synonym table. Only substitutions that preserve question meaning.
_SYNONYMS = {
    "explain": "describe",
    "describe": "explain",
    "how": "in what way",
    "tell me": "let me know",
    "what is": "what exactly is",
    "why": "for what reason",
    "list": "enumerate",
    "find": "locate",
    "big": "large",
    "begin": "start",
    "buy": "purchase",
}


def _politeness(text: str) -> str:
    """Wrap the request in a polite frame."""
    stripped = text.strip()
    if stripped.lower().startswith(("could you", "would you", "please")):
        return stripped
    return f"Could you please tell me: {stripped[0].lower() + stripped[1:] if stripped else stripped}"


def _indirect(text: str) -> str:
    """Restate the request indirectly."""
    stripped = text.strip().rstrip("?")
    return f"I would like to know {stripped[0].lower() + stripped[1:] if stripped else stripped}."


def _synonym(text: str) -> str:
    """Substitute a single safe synonym, longest phrase first."""
    lowered = text.lower()
    for source in sorted(_SYNONYMS, key=len, reverse=True):
        if source in lowered:
            index = lowered.index(source)
            return text[:index] + _SYNONYMS[source] + text[index + len(source) :]
    return text


def _filler(text: str) -> str:
    """Insert a semantically inert filler word."""
    if " " not in text.strip():
        return text
    parts = text.split(" ", 1)
    return f"{parts[0]} actually {parts[1]}"


def _trailing_context(text: str) -> str:
    """Append an inert clarifier."""
    return text.rstrip() + " I am asking for my own understanding."


#: Ordered operator bank. Order matters: a ladder applies a prefix of this list.
PARAPHRASE_OPERATORS: Sequence[Callable[[str], str]] = (
    _politeness,
    _synonym,
    _filler,
    _indirect,
    _trailing_context,
)


def paraphrase(text: str, level: int) -> str:
    """Apply the first ``level`` operators cumulatively."""
    out = text
    for operator in PARAPHRASE_OPERATORS[: max(0, level)]:
        out = operator(out)
    return out


def perturbation_ladder(text: str, rungs: int = 3) -> List[str]:
    """Build a graded ladder of increasingly perturbed phrasings.

    Rung ``k`` applies ``k`` operators, so successive rungs are separated by one
    operator each. That even spacing is what makes the second difference across
    three consecutive rungs a meaningful discrete curvature estimate.

    :param text: The original query.
    :param rungs: Number of rungs, including rung 0 (the original).
    :return: ``rungs`` phrasings, from unmodified to most perturbed.
    """
    return [paraphrase(text, level) for level in range(max(2, rungs))]
