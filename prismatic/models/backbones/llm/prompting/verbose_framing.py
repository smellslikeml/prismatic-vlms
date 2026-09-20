"""
verbose_framing.py

Prompt-level "verbose framing" for improving VLM robustness under image corruption.

Adapted from:
    Cross-Modal Attention Acts as a Frequency Filter: Why Verbose Prompts Improve
    Robustness in Vision-Language Models (arXiv:2609.20139).

The paper shows that question wording modulates question-conditioned cross-modal
attention, which acts as a spectral filter over image patches. Padding a question with
a short, content-free verbose preamble (e.g. "Is there a cat?" -> "Please look
carefully and answer: is there a cat?") broadens the frequency support of that filter,
reducing answer drift under image corruption (the paper reports 70-81% drift-variance
reduction on 8B models, plus accuracy gains from the practical "pad the prompt" recipe).

This module implements that practical recipe as a parameter-free str->str transform.
It deliberately does *not* reproduce the paper's spectral analysis, drift-variance
measurement, or corruption-benchmark suite -- those belong in a downstream evaluation.
The transform composes into any `PromptBuilder`'s `wrap_human` step, so it applies
uniformly across every prompt family without touching the forward/generate contract.
"""

from typing import Callable

# Content-free preamble that pads the question. Mirrors the paper's canonical example
# ("Please look carefully and answer: ...") -- verbose, but semantically neutral so it
# does not change the question being asked.
DEFAULT_VERBOSE_PREAMBLE = "Please look carefully and answer:"


def frame_verbose(message: str, preamble: str = DEFAULT_VERBOSE_PREAMBLE) -> str:
    """Prepend a verbose, content-free preamble to a human question.

    The transform is idempotent (a message already starting with ``preamble`` is
    returned unchanged) and a no-op on empty / whitespace-only messages, so it is safe
    to apply blindly to every human turn.
    """
    stripped = message.strip()
    if not stripped:
        return message

    preamble = preamble.strip()
    if stripped.lower().startswith(preamble.lower()):
        return message

    return f"{preamble} {stripped}"


def compose_verbose_framing(
    wrap_human: Callable[[str], str], preamble: str = DEFAULT_VERBOSE_PREAMBLE
) -> Callable[[str], str]:
    """Return a new ``wrap_human`` that verbose-frames the question before wrapping it."""

    def _wrapped(msg: str) -> str:
        return wrap_human(frame_verbose(msg, preamble))

    return _wrapped


def apply_verbose_framing(prompt_builder, preamble: str = DEFAULT_VERBOSE_PREAMBLE):
    """Enable verbose framing on an existing ``PromptBuilder`` in place.

    Composes :func:`frame_verbose` into the builder's ``wrap_human`` step. Because every
    family prompter exposes ``wrap_human`` as an instance attribute, this works uniformly
    across all families and only affects the human/question turns.
    """
    prompt_builder.wrap_human = compose_verbose_framing(prompt_builder.wrap_human, preamble)
    return prompt_builder
