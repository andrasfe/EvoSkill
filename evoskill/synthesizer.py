"""Pluggable skill synthesis — bring your own LLM."""

from __future__ import annotations

from typing import Awaitable, Callable, Union

from .config import get_api_key, get_model
from .skill import Skill
from .store import SkillStore

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a concise skill extractor. Given an agent role, the input it received, \
what it produced, and the feedback or failure, produce ONE new skill \
(1-3 sentences) that would help the agent do better next time. \
Output ONLY the skill text — no preamble, no bullet points, no quotes."""

_USER_TEMPLATE = """\
Role: {role}

Input prompt:
{input_prompt}

Agent output:
{agent_output}

Feedback / failure:
{feedback}

Existing skills:
{existing_skills}

Produce a brief, actionable skill (1-3 sentences)."""


# ---------------------------------------------------------------------------
# Default OpenAI adapter
# ---------------------------------------------------------------------------


def default_openai_llm(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
) -> str:
    """Default LLM implementation using the OpenAI SDK.

    This is used when the caller does not supply their own *llm* callable.
    """
    from openai import OpenAI

    client = OpenAI(api_key=get_api_key())
    response = client.chat.completions.create(
        model=model or get_model(),
        messages=messages,  # type: ignore[arg-type]
        temperature=0.3,
        max_tokens=256,
    )
    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

LLMCallable = Callable[[list[dict[str, str]]], str]
"""Type alias: ``(messages) -> str``."""

AsyncLLMCallable = Callable[[list[dict[str, str]]], Awaitable[str]]
"""Type alias: ``async (messages) -> str``."""


def synthesize_skill(
    role: str,
    input_prompt: str,
    failure: str,
    store: SkillStore,
    *,
    llm: LLMCallable | None = None,
    tags: list[str] | None = None,
) -> Skill:
    """Synthesize a skill from an input + failure.

    Backward-compatible entry point: sets *agent_output* to ``"(not captured)"``
    and forwards to :func:`synthesize_skill_with_context`.
    """
    return synthesize_skill_with_context(
        role=role,
        input_prompt=input_prompt,
        agent_output="(not captured)",
        feedback=failure,
        store=store,
        llm=llm,
        tags=tags,
    )


def synthesize_skill_with_context(
    role: str,
    input_prompt: str,
    agent_output: str,
    feedback: str,
    store: SkillStore,
    *,
    llm: LLMCallable | None = None,
    tags: list[str] | None = None,
) -> Skill:
    """Synthesize a skill using the full context.

    Parameters
    ----------
    role:
        Agent role this skill belongs to.
    input_prompt:
        The original input / prompt the agent received.
    agent_output:
        What the agent actually produced.
    feedback:
        The failure traceback **or** structured reviewer feedback.
    store:
        The :class:`SkillStore` to persist the result into.
    llm:
        Optional LLM callable ``(messages) -> str``.  Falls back to the
        built-in OpenAI adapter when ``None``.
    tags:
        Optional tags to attach to the new skill.
    """
    existing = store.get_skills(role)
    existing_text = (
        "\n".join(f"- {s.content}" for s in existing) if existing else "(none)"
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _USER_TEMPLATE.format(
                role=role,
                input_prompt=input_prompt,
                agent_output=agent_output,
                feedback=feedback,
                existing_skills=existing_text,
            ),
        },
    ]

    llm_fn = llm or default_openai_llm
    content = llm_fn(messages).strip()

    skill = Skill(
        role=role, content=content, source="learned", tags=tags or [],
    )
    store.add_skill(skill)
    return skill


async def asynthesize_skill(
    role: str,
    input_prompt: str,
    failure: str,
    store: SkillStore,
    *,
    llm: AsyncLLMCallable | None = None,
    tags: list[str] | None = None,
) -> Skill:
    """Async version of :func:`synthesize_skill`."""
    return await asynthesize_skill_with_context(
        role=role,
        input_prompt=input_prompt,
        agent_output="(not captured)",
        feedback=failure,
        store=store,
        llm=llm,
        tags=tags,
    )


async def asynthesize_skill_with_context(
    role: str,
    input_prompt: str,
    agent_output: str,
    feedback: str,
    store: SkillStore,
    *,
    llm: AsyncLLMCallable | None = None,
    tags: list[str] | None = None,
) -> Skill:
    """Async version of :func:`synthesize_skill_with_context`.

    Parameters
    ----------
    llm:
        Async LLM callable ``async (messages) -> str``.  Falls back to the
        built-in OpenAI adapter (wrapped in a sync-to-async shim) when ``None``.
    """
    existing = store.get_skills(role)
    existing_text = (
        "\n".join(f"- {s.content}" for s in existing) if existing else "(none)"
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _USER_TEMPLATE.format(
                role=role,
                input_prompt=input_prompt,
                agent_output=agent_output,
                feedback=feedback,
                existing_skills=existing_text,
            ),
        },
    ]

    if llm is not None:
        content = (await llm(messages)).strip()
    else:
        content = default_openai_llm(messages).strip()

    skill = Skill(
        role=role, content=content, source="learned", tags=tags or [],
    )
    store.add_skill(skill)
    return skill
