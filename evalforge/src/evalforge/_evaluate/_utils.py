# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""I/O and column-plumbing helpers for the batch evaluation engine."""

import csv
import json
import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .._constants import EVALUATION_PASS_FAIL_MAPPING, DefaultOpenEncoding
from .._exceptions import ErrorBlame, ErrorCategory, ErrorTarget, EvaluationException

__all__ = [
    "load_data",
    "write_jsonl",
    "write_results",
    "resolve_column_mapping",
    "apply_column_mapping",
    "aggregate_metrics",
]

_REFERENCE_RE = re.compile(r"^\$\{(data|target|run\.outputs)\.([^}]+)\}$")


def load_data(data: Any) -> List[Dict[str, Any]]:
    """Load evaluation rows from a path (``.jsonl``/``.json``/``.csv``) or an iterable."""
    if data is None:
        raise EvaluationException(
            "'data' is required.",
            target=ErrorTarget.EVALUATE,
            category=ErrorCategory.MISSING_FIELD,
            blame=ErrorBlame.USER_ERROR,
        )
    if isinstance(data, (list, tuple)):
        return [dict(row) for row in data]
    if not isinstance(data, (str, os.PathLike)):
        if isinstance(data, Iterable):
            return [dict(row) for row in data]
        raise EvaluationException(
            f"Unsupported 'data' type: {type(data).__name__}.",
            target=ErrorTarget.EVALUATE,
            category=ErrorCategory.INVALID_VALUE,
            blame=ErrorBlame.USER_ERROR,
        )

    path = os.fspath(data)
    if not os.path.isfile(path):
        raise EvaluationException(
            f"Data file not found: {path}",
            target=ErrorTarget.EVALUATE,
            category=ErrorCategory.FILE_OR_FOLDER_NOT_FOUND,
            blame=ErrorBlame.USER_ERROR,
        )

    suffix = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding=DefaultOpenEncoding.READ) as handle:
        if suffix == ".csv":
            return [dict(row) for row in csv.DictReader(handle)]
        if suffix == ".json":
            payload = json.load(handle)
            return [dict(row) for row in (payload if isinstance(payload, list) else [payload])]
        rows: List[Dict[str, Any]] = []
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(dict(json.loads(line)))
            except ValueError as exc:
                raise EvaluationException(
                    f"Malformed JSON on line {number} of {path}: {exc}",
                    target=ErrorTarget.EVALUATE,
                    category=ErrorCategory.INVALID_VALUE,
                    blame=ErrorBlame.USER_ERROR,
                ) from exc
        return rows


def write_jsonl(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write ``rows`` as JSON Lines."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding=DefaultOpenEncoding.WRITE) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_results(path: str, result: Mapping[str, Any]) -> None:
    """Persist an evaluation result.

    A ``.jsonl`` path receives one line per row; any other path receives the
    whole result object as JSON.
    """
    if str(path).lower().endswith(".jsonl"):
        write_jsonl(path, result.get("rows", []))
        return
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding=DefaultOpenEncoding.WRITE) as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, default=str)


def resolve_column_mapping(
    evaluator: Any, mapping: Optional[Mapping[str, str]], default_mapping: Optional[Mapping[str, str]]
) -> Dict[str, str]:
    """Merge the default and per-evaluator column mappings.

    Parameters the evaluator declares but nobody maps default to the identically
    named data column, which is what makes the common case configuration free.
    """
    resolved: Dict[str, str] = dict(default_mapping or {})
    resolved.update(dict(mapping or {}))
    for name in getattr(evaluator, "_singleton_inputs", []):
        resolved.setdefault(name, "${data.%s}" % name)
    resolved.setdefault("conversation", "${data.conversation}")
    return resolved


def apply_column_mapping(
    mapping: Mapping[str, str], row: Mapping[str, Any], target_row: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """Resolve a column mapping against one data row.

    Values of the form ``${data.x}`` read the input row, ``${target.x}`` and
    ``${run.outputs.x}`` read the target's output. Anything else is a literal.
    Unresolved references are omitted so the evaluator's own required-field
    validation produces the error message, not a ``None`` sneaking through.
    """
    resolved: Dict[str, Any] = {}
    for parameter, reference in mapping.items():
        if not isinstance(reference, str):
            resolved[parameter] = reference
            continue
        match = _REFERENCE_RE.match(reference.strip())
        if not match:
            resolved[parameter] = reference
            continue
        source, key = match.group(1), match.group(2)
        if source == "data":
            if key in row:
                resolved[parameter] = row[key]
        else:
            if target_row and key in target_row:
                resolved[parameter] = target_row[key]
            elif key in row:
                resolved[parameter] = row[key]
    return resolved


def aggregate_metrics(rows: Sequence[Mapping[str, Any]], evaluator_names: Sequence[str]) -> Dict[str, Any]:
    """Reduce per-row evaluator outputs to run-level metrics.

    * numeric columns -> mean
    * ``*_result`` columns -> ``<name>.pass_rate``
    * boolean ``*_label`` columns -> ``<name>.defect_rate``
    """
    metrics: Dict[str, Any] = {}
    numeric: Dict[str, List[float]] = {}
    passes: Dict[str, List[bool]] = {}
    labels: Dict[str, List[bool]] = {}

    for row in rows:
        for column, value in row.items():
            if not column.startswith("outputs."):
                continue
            name = column[len("outputs.") :]
            if column.endswith("_result"):
                if value in EVALUATION_PASS_FAIL_MAPPING.values():
                    passes.setdefault(name, []).append(value == "pass")
            elif column.endswith("_label") and isinstance(value, bool):
                labels.setdefault(name, []).append(value)
            elif isinstance(value, bool):
                labels.setdefault(name, []).append(value)
            elif isinstance(value, (int, float)):
                numeric.setdefault(name, []).append(float(value))

    for name, values in numeric.items():
        if values:
            metrics[name] = sum(values) / len(values)
    for name, values in passes.items():
        if values:
            metrics[f"{name}.pass_rate"] = sum(values) / len(values)
    for name, values in labels.items():
        if values:
            metrics[f"{name}.defect_rate"] = sum(values) / len(values)
    return metrics
