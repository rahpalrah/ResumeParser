# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""Judge model clients.

Three backends share one interface:

* :class:`AzureOpenAIChatClient` -- Azure OpenAI chat completions over HTTPS.
* :class:`OpenAIChatClient` -- OpenAI chat completions over HTTPS.
* :class:`LocalHeuristicChatClient` -- deterministic in-process scorer.

Only the standard library is used for transport (``urllib``), so no SDK client
package is required. Clients are selected from a model configuration by
:func:`client_from_configuration`; a missing or ``local`` configuration yields
the deterministic client, which is what makes the whole evaluator surface
runnable offline.
"""

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .._exceptions import ErrorBlame, ErrorCategory, ErrorTarget, EvaluationException
from ._embedding import Encoder
from ._heuristics import likert, score_metric

__all__ = [
    "ChatClient",
    "LocalHeuristicChatClient",
    "AzureOpenAIChatClient",
    "OpenAIChatClient",
    "client_from_configuration",
]

DEFAULT_API_VERSION = "2024-10-21"
DEFAULT_TIMEOUT = 60.0
DEFAULT_RETRIES = 3


class ChatClient:
    """Interface implemented by every judge backend."""

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        hint: Optional[Mapping[str, Any]] = None,
        **parameters: Any,
    ) -> str:
        """Return the assistant message content for ``messages``.

        :param messages: OpenAI-style ``{"role", "content"}`` dictionaries.
        :param hint: Structured description of the judging task
            (``{"metric": ..., "fields": {...}, "scale": (lo, hi)}``). Remote
            backends ignore it; the offline backend scores from it directly.
        """
        raise NotImplementedError


class LocalHeuristicChatClient(ChatClient):
    """Deterministic judge that never leaves the process.

    Produces the same JSON envelope a remote judge is prompted to emit, so the
    calling evaluator cannot tell the backends apart.
    """

    def __init__(self, *, encoder: Optional[Encoder] = None, seed: int = 0) -> None:
        self.encoder = encoder
        self.seed = seed

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        hint: Optional[Mapping[str, Any]] = None,
        **parameters: Any,
    ) -> str:
        if not hint or "metric" not in hint:
            raise EvaluationException(
                "The offline judge requires a structured hint naming the metric. "
                "Configure a real model to evaluate free-form prompts.",
                target=ErrorTarget.MODELS,
                category=ErrorCategory.MISSING_FIELD,
                blame=ErrorBlame.USER_ERROR,
            )
        metric = str(hint["metric"])
        fields = {k: ("" if v is None else str(v)) for k, v in dict(hint.get("fields", {})).items()}
        low, high = hint.get("scale", (1, 5))
        unit, rationale = score_metric(metric, fields, encoder=self.encoder)
        if hint.get("raw_scale"):
            score: float = round(unit * high, 4)
        else:
            score = likert(unit, low=int(low), high=int(high))
        return json.dumps({"score": score, "reason": rationale, "unit_score": round(unit, 6)})


class _HttpChatClient(ChatClient):
    """Shared HTTP plumbing for the hosted judge backends."""

    def __init__(self, *, timeout: float = DEFAULT_TIMEOUT, retries: int = DEFAULT_RETRIES) -> None:
        self.timeout = timeout
        self.retries = retries

    def _endpoint(self) -> str:
        raise NotImplementedError

    def _headers(self) -> Dict[str, str]:
        raise NotImplementedError

    def _payload(self, messages: Sequence[Mapping[str, Any]], parameters: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        hint: Optional[Mapping[str, Any]] = None,
        **parameters: Any,
    ) -> str:
        body = json.dumps(self._payload(messages, dict(parameters))).encode("utf-8")
        request = urllib.request.Request(
            self._endpoint(), data=body, headers=self._headers(), method="POST"
        )
        last_error: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    parsed = json.loads(response.read().decode("utf-8"))
                return str(parsed["choices"][0]["message"]["content"])
            except (urllib.error.URLError, KeyError, IndexError, ValueError) as exc:
                last_error = exc
                status = getattr(exc, "code", None)
                # 4xx other than throttling will not succeed on retry.
                if status is not None and 400 <= status < 500 and status != 429:
                    break
                if attempt < self.retries - 1:
                    time.sleep(2.0**attempt)
        raise EvaluationException(
            f"Judge model request failed after {self.retries} attempt(s): {last_error}",
            target=ErrorTarget.MODELS,
            category=ErrorCategory.SERVICE_UNAVAILABLE,
            blame=ErrorBlame.SYSTEM_ERROR,
        )


class AzureOpenAIChatClient(_HttpChatClient):
    """Chat completions against an Azure OpenAI deployment."""

    def __init__(self, configuration: Mapping[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.endpoint = str(configuration.get("azure_endpoint", "")).rstrip("/")
        self.deployment = str(configuration.get("azure_deployment", ""))
        self.api_key = configuration.get("api_key")
        self.api_version = configuration.get("api_version") or DEFAULT_API_VERSION
        if not self.endpoint or not self.deployment:
            raise EvaluationException(
                "AzureOpenAIModelConfiguration requires 'azure_endpoint' and 'azure_deployment'.",
                target=ErrorTarget.MODELS,
                category=ErrorCategory.MISSING_FIELD,
                blame=ErrorBlame.USER_ERROR,
            )

    def _endpoint(self) -> str:
        return (
            f"{self.endpoint}/openai/deployments/{self.deployment}"
            f"/chat/completions?api-version={self.api_version}"
        )

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["api-key"] = str(self.api_key)
        return headers

    def _payload(self, messages: Sequence[Mapping[str, Any]], parameters: Dict[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"messages": [dict(m) for m in messages]}
        payload.update(parameters)
        return payload


class OpenAIChatClient(_HttpChatClient):
    """Chat completions against the OpenAI API."""

    def __init__(self, configuration: Mapping[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = str(configuration.get("model", ""))
        self.api_key = configuration.get("api_key")
        self.base_url = str(configuration.get("base_url") or "https://api.openai.com/v1").rstrip("/")
        self.organization = configuration.get("organization")
        if not self.model:
            raise EvaluationException(
                "OpenAIModelConfiguration requires 'model'.",
                target=ErrorTarget.MODELS,
                category=ErrorCategory.MISSING_FIELD,
                blame=ErrorBlame.USER_ERROR,
            )

    def _endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.organization:
            headers["OpenAI-Organization"] = str(self.organization)
        return headers

    def _payload(self, messages: Sequence[Mapping[str, Any]], parameters: Dict[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": self.model, "messages": [dict(m) for m in messages]}
        payload.update(parameters)
        return payload


def client_from_configuration(
    configuration: Optional[Mapping[str, Any]], *, encoder: Optional[Encoder] = None
) -> ChatClient:
    """Build the judge client described by ``configuration``.

    ``None``, an empty mapping, or ``type="local"`` selects the deterministic
    offline judge. A configuration that names a hosted model but carries no
    credential also falls back to the offline judge rather than failing a whole
    evaluation run at call time.
    """
    if not configuration:
        return LocalHeuristicChatClient(encoder=encoder)

    config = dict(configuration)
    declared = config.get("type")
    if declared == "local":
        return LocalHeuristicChatClient(encoder=encoder, seed=int(config.get("seed", 0)))

    if declared == "azure_openai" or ("azure_endpoint" in config or "azure_deployment" in config):
        if not config.get("api_key"):
            return LocalHeuristicChatClient(encoder=encoder)
        return AzureOpenAIChatClient(config)

    if declared == "openai" or "model" in config:
        if not config.get("api_key"):
            return LocalHeuristicChatClient(encoder=encoder)
        return OpenAIChatClient(config)

    raise EvaluationException(
        f"Unrecognised model configuration: {sorted(config)}",
        target=ErrorTarget.MODELS,
        category=ErrorCategory.INVALID_VALUE,
        blame=ErrorBlame.USER_ERROR,
    )
