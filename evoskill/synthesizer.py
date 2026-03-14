"""Pluggable skill synthesis -- bring your own LLM."""

from __future__ import annotations

import re
from typing import Awaitable, Callable

from .config import get_api_key, get_model
from .skill import Skill

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "You are a concise skill extractor. Given an agent role, the input it received, "
    "what it produced, and the feedback or failure, produce ONE new skill "
    "(1-3 sentences) that would help the agent do better next time. "
    "Output ONLY the skill text -- no preamble, no bullet points, no quotes."
)

DEFAULT_USER_TEMPLATE = (
    "Role: {role}\n\n"
    "Input prompt:\n{input_prompt}\n\n"
    "Agent output:\n{agent_output}\n\n"
    "Feedback / failure:\n{feedback}\n\n"
    "Existing skills:\n{existing_skills}\n\n"
    "Produce a brief, actionable skill (1-3 sentences)."
)

_BATCH_SYSTEM_PROMPT = (
    "You are a concise skill extractor. You will receive multiple feedback items "
    "for an agent role. For each item, produce ONE brief, actionable skill "
    "(1-3 sentences). Output a numbered list -- one skill per item -- and nothing else."
)

_BATCH_USER_TEMPLATE = (
    "Role: {role}\n\n"
    "{items_text}\n\n"
    "Existing skills:\n{existing_skills}\n\n"
    "Produce one brief, actionable skill per item above. Output a numbered list."
)


# ---------------------------------------------------------------------------
# Default OpenAI adapter
# ---------------------------------------------------------------------------


def default_openai_llm(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
) -> str:
    """Default LLM implementation using the OpenAI SDK."""
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
# Public API -- types
# ---------------------------------------------------------------------------

LLMCallable = Callable[[list[dict[str, str]]], str]
"""Type alias: ``(messages) -> str``."""

AsyncLLMCallable = Callable[[list[dict[str, str]]], Awaitable[str]]
"""Type alias: ``async (messages) -> str``."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_numbered_list(text: str) -> list[str]:
    """Parse '1. ...' style numbered list from LLM output."""
    lines = text.strip().splitlines()
    result: list[str] = []
    for line in lines:
        line = line.strip()
        m = re.match(r"^\d+[.\)]\s*(.+)$", line)
        if m:
            result.append(m.group(1).strip())
    return result


def _format_batch_items(items: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for i, item in enumerate(items, 1):
        parts.append(
            f"--- Item {i} ---\n"
            f"Input: {item['input_prompt']}\n"
            f"Agent output: {item['agent_output']}\n"
            f"Feedback: {item['reviewer_feedback']}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Sync synthesis
# ---------------------------------------------------------------------------


def synthesize_skill(
    role: str,
    input_prompt: str,
    failure: str,
    store: "SkillStore",
    *,
    llm: LLMCallable | None = None,
    tags: list[str] | None = None,
    system_prompt: str | None = None,
    user_template: str | None = None,
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
        system_prompt=system_prompt,
        user_template=user_template,
    )


def synthesize_skill_with_context(
    role: str,
    input_prompt: str,
    agent_output: str,
    feedback: str,
    store: "SkillStore",
    *,
    llm: LLMCallable | None = None,
    tags: list[str] | None = None,
    system_prompt: str | None = None,
    user_template: str | None = None,
) -> Skill:
    """Synthesize a skill using the full context.

    Parameters
    ----------
    system_prompt:
        Override the default system prompt for synthesis.
    user_template:
        Override the default user template.  Must contain ``{role}``,
        ``{input_prompt}``, ``{agent_output}``, ``{feedback}``, and
        ``{existing_skills}`` placeholders.
    """
    existing = store.get_skills(role)
    existing_text = (
        "\n".join(f"- {s.content}" for s in existing) if existing else "(none)"
    )

    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    usr_tmpl = user_template or DEFAULT_USER_TEMPLATE

    messages: list[dict[str, str]] = [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": usr_tmpl.format(
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


# ---------------------------------------------------------------------------
# Async synthesis
# ---------------------------------------------------------------------------


async def asynthesize_skill(
    role: str,
    input_prompt: str,
    failure: str,
    store: "SkillStore",
    *,
    llm: AsyncLLMCallable | None = None,
    tags: list[str] | None = None,
    system_prompt: str | None = None,
    user_template: str | None = None,
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
        system_prompt=system_prompt,
        user_template=user_template,
    )


async def asynthesize_skill_with_context(
    role: str,
    input_prompt: str,
    agent_output: str,
    feedback: str,
    store: "SkillStore",
    *,
    llm: AsyncLLMCallable | None = None,
    tags: list[str] | None = None,
    system_prompt: str | None = None,
    user_template: str | None = None,
) -> Skill:
    """Async version of :func:`synthesize_skill_with_context`."""
    existing = store.get_skills(role)
    existing_text = (
        "\n".join(f"- {s.content}" for s in existing) if existing else "(none)"
    )

    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    usr_tmpl = user_template or DEFAULT_USER_TEMPLATE

    messages: list[dict[str, str]] = [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": usr_tmpl.format(
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


# ---------------------------------------------------------------------------
# Batch synthesis
# ---------------------------------------------------------------------------


def synthesize_skill_batch(
    role: str,
    items: list[dict[str, str]],
    store: "SkillStore",
    *,
    llm: LLMCallable | None = None,
    tags: list[str] | None = None,
    system_prompt: str | None = None,
) -> list[Skill]:
    """Synthesize multiple skills in one LLM call from a batch of feedback items."""
    existing = store.get_skills(role)
    existing_text = (
        "\n".join(f"- {s.content}" for s in existing) if existing else "(none)"
    )

    sys_prompt = system_prompt or _BATCH_SYSTEM_PROMPT
    items_text = _format_batch_items(items)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": _BATCH_USER_TEMPLATE.format(
                role=role,
                items_text=items_text,
                existing_skills=existing_text,
            ),
        },
    ]

    llm_fn = llm or default_openai_llm
    raw = llm_fn(messages).strip()
    contents = _parse_numbered_list(raw)

    skills: list[Skill] = []
    for text in contents:
        skill = Skill(role=role, content=text, source="learned", tags=tags or [])
        store.add_skill(skill)
        skills.append(skill)
    return skills


async def asynthesize_skill_batch(
    role: str,
    items: list[dict[str, str]],
    store: "SkillStore",
    *,
    llm: AsyncLLMCallable | None = None,
    tags: list[str] | None = None,
    system_prompt: str | None = None,
) -> list[Skill]:
    """Async version of :func:`synthesize_skill_batch`."""
    existing = store.get_skills(role)
    existing_text = (
        "\n".join(f"- {s.content}" for s in existing) if existing else "(none)"
    )

    sys_prompt = system_prompt or _BATCH_SYSTEM_PROMPT
    items_text = _format_batch_items(items)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": _BATCH_USER_TEMPLATE.format(
                role=role,
                items_text=items_text,
                existing_skills=existing_text,
            ),
        },
    ]

    if llm is not None:
        raw = (await llm(messages)).strip()
    else:
        raw = default_openai_llm(messages).strip()

    contents = _parse_numbered_list(raw)

    skills: list[Skill] = []
    for text in contents:
        skill = Skill(role=role, content=text, source="learned", tags=tags or [])
        store.add_skill(skill)
        skills.append(skill)
    return skills
