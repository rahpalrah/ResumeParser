# Copyright (c) Contributors to the EvalForge project.
# Licensed under the Apache License, Version 2.0.
"""Scenario and language enumerations for the simulators."""

from enum import Enum

__all__ = ["AdversarialScenario", "SupportedLanguages", "AdversarialScenarioJailbreak"]


class AdversarialScenario(str, Enum):
    """Task shapes an adversarial simulator can generate conversations for."""

    ADVERSARIAL_QA = "adv_qa"
    ADVERSARIAL_CONVERSATION = "adv_conversation"
    ADVERSARIAL_SUMMARIZATION = "adv_summarization"
    ADVERSARIAL_SEARCH = "adv_search"
    ADVERSARIAL_REWRITE = "adv_rewrite"
    ADVERSARIAL_CONTENT_GEN_UNGROUNDED = "adv_content_gen_ungrounded"
    ADVERSARIAL_CONTENT_GEN_GROUNDED = "adv_content_gen_grounded"
    ADVERSARIAL_CONTENT_PROTECTED_MATERIAL = "adv_content_protected_material"
    ADVERSARIAL_CODE_VULNERABILITY = "adv_code_vulnerability"


class AdversarialScenarioJailbreak(str, Enum):
    """Scenarios that carry an explicit jailbreak payload."""

    ADVERSARIAL_CONVERSATION_JAILBREAK = "adv_conversation_jailbreak"
    ADVERSARIAL_QA_JAILBREAK = "adv_qa_jailbreak"
    ADVERSARIAL_INDIRECT_JAILBREAK = "adv_indirect_jailbreak"


class SupportedLanguages(str, Enum):
    """Languages the simulators can emit prompts in."""

    English = "en"
    Spanish = "es"
    French = "fr"
    German = "de"
    Italian = "it"
    Portuguese = "pt"
    Japanese = "ja"
    Korean = "ko"
    SimplifiedChinese = "zh-cn"
