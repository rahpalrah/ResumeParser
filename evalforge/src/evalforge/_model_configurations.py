# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""Typed model/project configuration dictionaries.

These are ``TypedDict`` definitions rather than classes, matching the reference
SDK: a configuration is an ordinary ``dict`` so it can be loaded straight from
JSON/YAML or from environment variables.
"""

import sys
from typing import Any, Dict, List, Optional, Union

if sys.version_info >= (3, 8):
    from typing import Literal, TypedDict
else:  # pragma: no cover - Python 3.7 fallback
    from typing_extensions import Literal, TypedDict

__all__ = [
    "AzureOpenAIModelConfiguration",
    "OpenAIModelConfiguration",
    "LocalModelConfiguration",
    "AzureAIProject",
    "EvaluatorConfig",
    "EvaluationResult",
    "Conversation",
    "Message",
]


class AzureOpenAIModelConfiguration(TypedDict, total=False):
    """Connection details for an Azure OpenAI deployment used as an LLM judge."""

    type: Literal["azure_openai"]
    azure_endpoint: str
    azure_deployment: str
    api_key: Optional[str]
    api_version: Optional[str]


class OpenAIModelConfiguration(TypedDict, total=False):
    """Connection details for an OpenAI model used as an LLM judge."""

    type: Literal["openai"]
    api_key: Optional[str]
    model: str
    base_url: Optional[str]
    organization: Optional[str]


class LocalModelConfiguration(TypedDict, total=False):
    """EvalForge extension: run judges against a deterministic in-process scorer.

    This has no counterpart in the reference SDK. It exists so that the whole
    evaluation surface, including AI-assisted evaluators, is executable and
    testable with no network access and no credentials.
    """

    type: Literal["local"]
    model: str
    seed: int


ModelConfiguration = Union[
    AzureOpenAIModelConfiguration, OpenAIModelConfiguration, LocalModelConfiguration
]


class AzureAIProject(TypedDict, total=False):
    """Identifies the workspace that evaluation results are logged to."""

    subscription_id: str
    resource_group_name: str
    project_name: str


class EvaluatorConfig(TypedDict, total=False):
    """Per-evaluator options accepted by :func:`evalforge.evaluate`.

    ``column_mapping`` maps an evaluator parameter name to a data reference such
    as ``"${data.question}"``, ``"${target.response}"`` or a literal string.
    """

    column_mapping: Dict[str, str]


class Message(TypedDict, total=False):
    """A single conversation turn."""

    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, List[Dict[str, Any]]]
    context: Optional[Any]
    tool_calls: Optional[List[Dict[str, Any]]]


class Conversation(TypedDict, total=False):
    """A multi-turn exchange accepted by conversation-aware evaluators."""

    messages: List[Message]
    context: Optional[Dict[str, Any]]


class EvaluationResult(TypedDict, total=False):
    """Return value of :func:`evalforge.evaluate`."""

    metrics: Dict[str, Any]
    studio_url: Optional[str]
    rows: List[Dict[str, Any]]
