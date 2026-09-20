# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""Judge prompt templates and a minimal ``.prompty`` loader.

Prompts live as data, not code, so they can be audited and versioned. Built-in
templates are registered per metric; custom ones can be loaded from a
``.prompty`` file (YAML front matter followed by a role-delimited body), which
is parsed here with a deliberately small YAML subset so no parser dependency is
introduced.
"""

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..._constants import DefaultOpenEncoding
from ..._exceptions import ErrorBlame, ErrorCategory, ErrorTarget, EvaluationException

__all__ = ["Prompty", "render_judge_prompt", "JUDGE_TEMPLATES"]

_ENVELOPE = (
    "Respond with a single JSON object and nothing else, in the form "
    '{{"score": <integer {low} to {high}>, "reason": "<one or two sentences>"}}.'
)

#: ``metric -> (criteria, ordered input field names)``
JUDGE_TEMPLATES: Dict[str, Tuple[str, Sequence[str]]] = {
    "groundedness": (
        "Judge whether every claim in RESPONSE is supported by CONTEXT. A claim that "
        "contradicts CONTEXT, or that adds specifics CONTEXT does not state, is not "
        "grounded. Ignore whether the response is helpful or well written. "
        "{low} means no claim is supported; {high} means every claim is fully supported.",
        ("query", "context", "response"),
    ),
    "relevance": (
        "Judge how well RESPONSE addresses QUERY. {low} means it does not address the "
        "query at all; {high} means it fully and directly addresses it. Do not reward "
        "extra information that was not asked for.",
        ("query", "response"),
    ),
    "coherence": (
        "Judge the logical flow and organisation of RESPONSE. {low} means disjointed or "
        "self-contradictory; {high} means ideas build in a clear, connected order. "
        "Repetition is not coherence.",
        ("query", "response"),
    ),
    "fluency": (
        "Judge the linguistic quality of RESPONSE on its own terms: grammar, word "
        "choice and sentence structure. {low} means barely readable; {high} means "
        "natural, polished prose. Ignore factual accuracy.",
        ("response",),
    ),
    "similarity": (
        "Judge how closely RESPONSE matches GROUND_TRUTH in meaning. {low} means "
        "unrelated or contradictory; {high} means equivalent in substance. Differences "
        "in wording alone should not reduce the score.",
        ("query", "response", "ground_truth"),
    ),
    "retrieval": (
        "Judge the quality of the retrieved CONTEXT for answering QUERY: are the most "
        "relevant chunks present, and are they ranked ahead of irrelevant ones? {low} "
        "means the context is useless for the query; {high} means it is complete and "
        "well ordered. Judge the context, not the response.",
        ("query", "context"),
    ),
    "intent_resolution": (
        "Judge whether RESPONSE correctly identifies and resolves the user's underlying "
        "intent in QUERY. {low} means the intent was misread or ignored; {high} means it "
        "was correctly understood and fully resolved.",
        ("query", "response"),
    ),
    "task_adherence": (
        "Judge whether RESPONSE follows the instructions and constraints stated in the "
        "task. {low} means the constraints were ignored; {high} means every stated "
        "constraint was respected.",
        ("query", "response", "instructions"),
    ),
    "response_completeness": (
        "Judge whether RESPONSE contains all the information required by GROUND_TRUTH. "
        "{low} means essential information is missing; {high} means nothing required is "
        "missing. Extra correct information does not reduce the score.",
        ("response", "ground_truth"),
    ),
}


def render_judge_prompt(
    metric: str, fields: Mapping[str, Any], *, scale: Tuple[int, int] = (1, 5)
) -> List[Dict[str, str]]:
    """Render the chat messages that ask a judge to score ``metric``.

    :param metric: Metric name registered in :data:`JUDGE_TEMPLATES`.
    :param fields: Evaluation input; only the fields the template declares are
        included in the rendered prompt.
    :param scale: Inclusive ``(low, high)`` integer scoring range.
    """
    low, high = scale
    if metric not in JUDGE_TEMPLATES:
        raise EvaluationException(
            f"No judge template registered for metric '{metric}'.",
            target=ErrorTarget.UNKNOWN,
            category=ErrorCategory.INVALID_VALUE,
            blame=ErrorBlame.SYSTEM_ERROR,
        )
    criteria, field_names = JUDGE_TEMPLATES[metric]
    system = (
        "You are an impartial evaluator. You score one property of a model output "
        "against a fixed rubric. You never rewrite the output and you never explain "
        "how to improve it."
    )
    body = [criteria.format(low=low, high=high), ""]
    for name in field_names:
        value = fields.get(name)
        if value in (None, ""):
            continue
        body.append(f"{name.upper()}:\n{value}\n")
    body.append(_ENVELOPE.format(low=low, high=high))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(body)},
    ]


class Prompty:
    """A parsed ``.prompty`` template.

    The file format is YAML front matter delimited by ``---`` followed by a body
    whose turns are introduced by ``system:``, ``user:`` or ``assistant:`` on
    their own line. Placeholders use ``{{name}}``.
    """

    _ROLE_RE = re.compile(r"^(system|user|assistant)\s*:\s*$", re.I | re.M)

    def __init__(self, metadata: Dict[str, Any], body: str) -> None:
        self.metadata = metadata
        self.body = body

    @property
    def name(self) -> str:
        """Template name from the front matter, if present."""
        return str(self.metadata.get("name", ""))

    @property
    def parameters(self) -> Dict[str, Any]:
        """Model parameters declared under ``model.parameters``."""
        model = self.metadata.get("model")
        if isinstance(model, dict) and isinstance(model.get("parameters"), dict):
            return dict(model["parameters"])
        return {}

    @classmethod
    def load(cls, path: str) -> "Prompty":
        """Read and parse a ``.prompty`` file."""
        try:
            with open(path, "r", encoding=DefaultOpenEncoding.READ) as handle:
                raw = handle.read()
        except OSError as exc:
            raise EvaluationException(
                f"Could not read prompty file '{path}': {exc}",
                target=ErrorTarget.UNKNOWN,
                category=ErrorCategory.FILE_OR_FOLDER_NOT_FOUND,
                blame=ErrorBlame.USER_ERROR,
            ) from exc
        return cls.loads(raw)

    @classmethod
    def loads(cls, raw: str) -> "Prompty":
        """Parse prompty content from a string."""
        metadata: Dict[str, Any] = {}
        body = raw
        if raw.lstrip().startswith("---"):
            stripped = raw.lstrip()
            end = stripped.find("\n---", 3)
            if end != -1:
                metadata = _parse_simple_yaml(stripped[3:end])
                body = stripped[end + 4 :].lstrip("-\n")
        return cls(metadata, body)

    def render(self, **values: Any) -> List[Dict[str, str]]:
        """Substitute ``{{placeholders}}`` and split the body into chat turns."""
        text = self.body
        for key, value in values.items():
            text = text.replace("{{" + key + "}}", "" if value is None else str(value))
        text = re.sub(r"\{\{\s*\w+\s*\}\}", "", text)

        matches = list(self._ROLE_RE.finditer(text))
        if not matches:
            return [{"role": "user", "content": text.strip()}]
        messages: List[Dict[str, str]] = []
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            if content:
                messages.append({"role": match.group(1).lower(), "content": content})
        return messages


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Parse the indentation-based ``key: value`` subset used by front matter.

    Supports nested mappings, scalar values, and inline ``[a, b]`` lists, which
    covers every field a prompty header actually uses. Anything more elaborate
    should be expressed in code.
    """
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, root)]

    for line in text.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if ":" not in line:
            continue
        key, _, value = line.strip().partition(":")
        key, value = key.strip(), value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            stack = [(-1, root)]
        parent = stack[-1][1]

        if not value:
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _coerce_scalar(value)
    return root


def _coerce_scalar(value: str) -> Any:
    """Convert a YAML scalar string to a Python value."""
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [_coerce_scalar(part) for part in inner.split(",")] if inner else []
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none", "~"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text
