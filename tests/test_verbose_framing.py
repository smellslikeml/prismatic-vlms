"""
test_verbose_framing.py

Tests for the verbose-framing "pad the prompt" recipe (arXiv:2609.20139) and its wiring
into the existing prompt builders. These import the real, pre-existing prompter modules
to prove the transform integrates with `wrap_human` uniformly across families.
"""

import pytest

# Existing (non-new) prompt-builder modules -- the integration surface.
from prismatic.models.backbones.llm.prompting import (
    LLaMa2ChatPromptBuilder,
    MistralInstructPromptBuilder,
    PhiPromptBuilder,
    PurePromptBuilder,
    VicunaV15ChatPromptBuilder,
)
from prismatic.models.backbones.llm.prompting.verbose_framing import (
    DEFAULT_VERBOSE_PREAMBLE,
    apply_verbose_framing,
    frame_verbose,
)

# Families that do not require a `model_family`-keyed system prompt to construct.
BASIC_BUILDERS = [PurePromptBuilder, MistralInstructPromptBuilder, PhiPromptBuilder]
# Families that look up a default system prompt via `model_family="prismatic"`.
SYSTEM_BUILDERS = [LLaMa2ChatPromptBuilder, VicunaV15ChatPromptBuilder]


def test_frame_verbose_pads_question():
    assert frame_verbose("Is there a cat?") == "Please look carefully and answer: Is there a cat?"


def test_frame_verbose_is_idempotent():
    once = frame_verbose("Is there a cat?")
    assert frame_verbose(once) == once


def test_frame_verbose_noop_on_empty():
    assert frame_verbose("") == ""
    assert frame_verbose("   ") == "   "


def test_frame_verbose_custom_preamble():
    assert frame_verbose("what colour?", preamble="Think step by step:") == "Think step by step: what colour?"


@pytest.mark.parametrize("builder_cls", BASIC_BUILDERS)
def test_apply_verbose_framing_reaches_prompt(builder_cls):
    """Framing composed into `wrap_human` shows up in the rendered prompt, not the raw turn."""
    plain = builder_cls(model_family="prismatic")
    plain.add_turn(role="human", message="Is there a cat?")
    assert DEFAULT_VERBOSE_PREAMBLE not in plain.get_prompt()

    framed = apply_verbose_framing(builder_cls(model_family="prismatic"))
    framed.add_turn(role="human", message="Is there a cat?")
    rendered = framed.get_prompt()
    assert DEFAULT_VERBOSE_PREAMBLE in rendered
    assert "Is there a cat?" in rendered


@pytest.mark.parametrize("builder_cls", SYSTEM_BUILDERS)
def test_apply_verbose_framing_preserves_system_prompt(builder_cls):
    """Framing only touches the question; the family's system prompt is untouched."""
    framed = apply_verbose_framing(builder_cls(model_family="prismatic"))
    framed.add_turn(role="human", message="Is there a cat?")
    rendered = framed.get_prompt()
    assert DEFAULT_VERBOSE_PREAMBLE in rendered
    assert framed.system_prompt.strip() in rendered


def test_apply_verbose_framing_only_affects_human_turns():
    framed = apply_verbose_framing(PurePromptBuilder(model_family="prismatic"))
    framed.add_turn(role="human", message="Is there a cat?")
    framed.add_turn(role="gpt", message="Yes, there is a cat.")
    rendered = framed.get_prompt()
    # Preamble appears exactly once (only the human turn was framed).
    assert rendered.count(DEFAULT_VERBOSE_PREAMBLE) == 1
