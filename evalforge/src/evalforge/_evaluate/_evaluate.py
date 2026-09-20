# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""The batch evaluation entry point."""

import concurrent.futures
import logging
import os
import time
import uuid
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .._constants import DEFAULT_MAX_WORKERS, Prefixes
from .._exceptions import ErrorBlame, ErrorCategory, ErrorTarget, EvaluationException
from ._utils import (
    aggregate_metrics,
    apply_column_mapping,
    load_data,
    resolve_column_mapping,
    write_results,
)

__all__ = ["evaluate"]

LOGGER = logging.getLogger(__name__)


def _run_target(
    target: Callable[..., Any], rows: Sequence[Mapping[str, Any]], max_workers: int
) -> List[Dict[str, Any]]:
    """Invoke ``target`` once per row, preserving row order."""

    def call(row: Mapping[str, Any]) -> Dict[str, Any]:
        try:
            result = target(**dict(row))
        except Exception as exc:  # noqa: BLE001 - surfaced per row, never fatal
            LOGGER.warning("Target callable failed for a row: %s", exc)
            return {"__error__": str(exc)}
        if isinstance(result, Mapping):
            return dict(result)
        return {"response": result}

    if max_workers <= 1 or len(rows) <= 1:
        return [call(row) for row in rows]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(call, rows))


def _flatten(prefix: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Prefix every key of ``payload``."""
    return {f"{prefix}{key}": value for key, value in payload.items()}


def evaluate(
    *,
    data: Any,
    evaluators: Mapping[str, Callable[..., Mapping[str, Any]]],
    evaluation_name: Optional[str] = None,
    target: Optional[Callable[..., Any]] = None,
    evaluator_config: Optional[Mapping[str, Mapping[str, Any]]] = None,
    azure_ai_project: Optional[Mapping[str, Any]] = None,
    output_path: Optional[str] = None,
    fail_on_evaluator_errors: bool = False,
    max_workers: int = DEFAULT_MAX_WORKERS,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Evaluate a dataset with one or more evaluators.

    .. code-block:: python

        result = evaluate(
            data="data.jsonl",
            evaluators={"groundedness": GroundednessEvaluator(), "f1": F1ScoreEvaluator()},
            evaluator_config={
                "groundedness": {"column_mapping": {"response": "${data.answer}"}}
            },
            output_path="./results.json",
        )
        result["metrics"]["groundedness.groundedness"]

    :param data: Path to a ``.jsonl``/``.json``/``.csv`` file, or a list of dicts.
    :param evaluators: Mapping of run-local name to evaluator callable. The name
        prefixes every column that evaluator produces.
    :param evaluation_name: Friendly name recorded on the run.
    :param target: Optional callable invoked per row to produce the output under
        evaluation. Its returned keys are addressable as ``${target.<key>}``.
    :param evaluator_config: Per-evaluator ``{"column_mapping": {...}}`` options.
    :param azure_ai_project: Project to associate results with, for the studio URL.
    :param output_path: Where to persist results. ``.jsonl`` writes rows only.
    :param fail_on_evaluator_errors: Raise instead of recording per-row errors.
    :param max_workers: Row-level parallelism.
    :return: ``{"metrics": ..., "rows": ..., "studio_url": ...}``
    :raises EvaluationException: On invalid arguments, or on the first evaluator
        error when ``fail_on_evaluator_errors`` is set.
    """
    if not evaluators:
        raise EvaluationException(
            "At least one evaluator is required.",
            target=ErrorTarget.EVALUATE,
            category=ErrorCategory.MISSING_FIELD,
            blame=ErrorBlame.USER_ERROR,
        )

    started = time.time()
    input_rows = load_data(data)
    if not input_rows:
        raise EvaluationException(
            "The evaluation dataset is empty.",
            target=ErrorTarget.EVALUATE,
            category=ErrorCategory.INVALID_VALUE,
            blame=ErrorBlame.USER_ERROR,
        )

    target_rows: List[Dict[str, Any]] = (
        _run_target(target, input_rows, max_workers) if target is not None else [{} for _ in input_rows]
    )

    default_mapping = dict((evaluator_config or {}).get("default", {}).get("column_mapping", {}))
    mappings = {
        name: resolve_column_mapping(
            evaluator,
            (evaluator_config or {}).get(name, {}).get("column_mapping"),
            default_mapping,
        )
        for name, evaluator in evaluators.items()
    }

    def evaluate_row(index: int) -> Dict[str, Any]:
        row = input_rows[index]
        target_row = target_rows[index]
        record: Dict[str, Any] = _flatten(Prefixes.INPUTS, row)
        if target_row:
            record.update(_flatten(Prefixes.OUTPUTS, target_row))

        for name, evaluator in evaluators.items():
            arguments = apply_column_mapping(mappings[name], row, target_row)
            arguments = {k: v for k, v in arguments.items() if v is not None}
            try:
                output = evaluator(**arguments)
            except Exception as exc:  # noqa: BLE001 - recorded per row by design
                if fail_on_evaluator_errors:
                    raise EvaluationException(
                        f"Evaluator '{name}' failed on row {index}: {exc}",
                        target=ErrorTarget.EVALUATE,
                        category=ErrorCategory.FAILED_EXECUTION,
                        blame=ErrorBlame.UNKNOWN,
                    ) from exc
                LOGGER.warning("Evaluator '%s' failed on row %d: %s", name, index, exc)
                record[f"{Prefixes.OUTPUTS}{name}.error"] = str(exc)
                continue
            if isinstance(output, Mapping):
                record.update(_flatten(f"{Prefixes.OUTPUTS}{name}.", output))
            else:
                record[f"{Prefixes.OUTPUTS}{name}.score"] = output
        return record

    indices = range(len(input_rows))
    if max_workers <= 1 or len(input_rows) <= 1:
        rows = [evaluate_row(i) for i in indices]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            rows = list(pool.map(evaluate_row, indices))

    metrics = aggregate_metrics(rows, list(evaluators))
    run_id = str(uuid.uuid4())
    studio_url = None
    if azure_ai_project:
        studio_url = (
            "https://ai.azure.com/build/evaluation/"
            f"{run_id}?wsid=/subscriptions/{azure_ai_project.get('subscription_id')}"
            f"/resourceGroups/{azure_ai_project.get('resource_group_name')}"
            f"/providers/Microsoft.MachineLearningServices/workspaces/"
            f"{azure_ai_project.get('project_name')}"
        )

    result: Dict[str, Any] = {
        "metrics": metrics,
        "rows": rows,
        "studio_url": studio_url,
        "run_info": {
            "id": run_id,
            "name": evaluation_name or f"evalforge-{run_id[:8]}",
            "row_count": len(rows),
            "evaluators": sorted(evaluators),
            "duration_seconds": round(time.time() - started, 4),
        },
    }

    if output_path:
        write_results(os.fspath(output_path), result)
    return result
