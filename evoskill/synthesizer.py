"""Pluggable skill synthesis -- bring your own LLM."""

from __future__ import annotations

import math
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

EmbeddingCallable = Callable[[str], list[float]]
"""Type alias: ``(text) -> list[float]``."""

AsyncEmbeddingCallable = Callable[[str], Awaitable[list[float]]]
"""Type alias: ``async (text) -> list[float]``."""


# ---------------------------------------------------------------------------
# Default OpenAI embedding adapter
# ---------------------------------------------------------------------------


def default_openai_embedding(text: str) -> list[float]:
    """Default embedding implementation using the OpenAI SDK."""
    from openai import OpenAI

    client = OpenAI(api_key=get_api_key())
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _find_duplicate(
    feedback_text: str,
    existing_skills: list[Skill],
    embed_fn: EmbeddingCallable,
    threshold: float,
) -> Skill | None:
    """Return the best-matching existing skill if above *threshold*, else ``None``.

    Also computes and caches embeddings on skills that don't have one yet.
    """
    feedback_embedding = embed_fn(feedback_text)
    best_skill: Skill | None = None
    best_sim = -1.0
    for skill in existing_skills:
        if skill.embedding is None:
            skill.embedding = embed_fn(skill.content)
        sim = _cosine_similarity(feedback_embedding, skill.embedding)
        if sim > best_sim:
            best_sim = sim
            best_skill = skill
    if best_sim >= threshold and best_skill is not None:
        return best_skill
    return None


async def _afind_duplicate(
    feedback_text: str,
    existing_skills: list[Skill],
    embed_fn: AsyncEmbeddingCallable | EmbeddingCallable,
    threshold: float,
) -> Skill | None:
    """Async version of :func:`_find_duplicate`."""
    import asyncio

    result = embed_fn(feedback_text)
    if asyncio.iscoroutine(result) or asyncio.isfuture(result):
        feedback_embedding = await result
    else:
        feedback_embedding = result  # type: ignore[assignment]

    best_skill: Skill | None = None
    best_sim = -1.0
    for skill in existing_skills:
        if skill.embedding is None:
            emb_result = embed_fn(skill.content)
            if asyncio.iscoroutine(emb_result) or asyncio.isfuture(emb_result):
                skill.embedding = await emb_result
            else:
                skill.embedding = emb_result  # type: ignore[assignment]
        sim = _cosine_similarity(feedback_embedding, skill.embedding)
        if sim > best_sim:
            best_sim = sim
            best_skill = skill
    if best_sim >= threshold and best_skill is not None:
        return best_skill
    return None


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
    deduplicate: bool = True,
    similarity_threshold: float = 0.85,
    embed: EmbeddingCallable | None = None,
) -> Skill | None:
    """Synthesize a skill from an input + failure.

    Backward-compatible entry point: sets *agent_output* to ``"(not captured)"``
    and forwards to :func:`synthesize_skill_with_context`.

    Returns ``None`` when *deduplicate* is ``True`` and an existing skill
    already covers the feedback.
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
        deduplicate=deduplicate,
        similarity_threshold=similarity_threshold,
        embed=embed,
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
    deduplicate: bool = True,
    similarity_threshold: float = 0.85,
    embed: EmbeddingCallable | None = None,
) -> Skill | None:
    """Synthesize a skill using the full context.

    Parameters
    ----------
    system_prompt:
        Override the default system prompt for synthesis.
    user_template:
        Override the default user template.  Must contain ``{role}``,
        ``{input_prompt}``, ``{agent_output}``, ``{feedback}``, and
        ``{existing_skills}`` placeholders.
    deduplicate:
        When ``True`` (default), embed the feedback and compare against
        existing skills.  If a match exceeds *similarity_threshold*, skip
        synthesis and return ``None``.
    similarity_threshold:
        Cosine-similarity cutoff (default 0.85).
    embed:
        Embedding callable ``(text) -> list[float]``.  Falls back to
        :func:`default_openai_embedding` when ``None``.

    Returns ``None`` when deduplication matches an existing skill.
    """
    existing = store.get_skills(role, tags=tags)
    existing_text = (
        "\n".join(f"- {s.content}" for s in existing) if existing else "(none)"
    )

    # -- deduplication check --------------------------------------------------
    embed_fn = embed  # None means "skip embedding" — caller must supply a callable
    if deduplicate and existing and embed_fn is not None:
        matched = _find_duplicate(
            feedback, existing, embed_fn, similarity_threshold,
        )
        if matched is not None:
            # Persist any newly-computed embeddings (without overwriting unrelated skills)
            store._update_embeddings(role, existing)
            return None
        # Persist computed embeddings even when no match
        store._update_embeddings(role, existing)

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

    # Embed the new skill so future dedup checks have it cached
    new_embedding: list[float] | None = None
    if deduplicate and embed_fn is not None:
        new_embedding = embed_fn(content)

    skill = Skill(
        role=role, content=content, source="learned", tags=tags or [],
        embedding=new_embedding,
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
    deduplicate: bool = True,
    similarity_threshold: float = 0.85,
    embed: AsyncEmbeddingCallable | EmbeddingCallable | None = None,
) -> Skill | None:
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
        deduplicate=deduplicate,
        similarity_threshold=similarity_threshold,
        embed=embed,
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
    deduplicate: bool = True,
    similarity_threshold: float = 0.85,
    embed: AsyncEmbeddingCallable | EmbeddingCallable | None = None,
) -> Skill | None:
    """Async version of :func:`synthesize_skill_with_context`."""
    existing = store.get_skills(role, tags=tags)
    existing_text = (
        "\n".join(f"- {s.content}" for s in existing) if existing else "(none)"
    )

    # -- deduplication check --------------------------------------------------
    embed_fn = embed  # None means "skip embedding"
    if deduplicate and existing and embed_fn is not None:
        matched = await _afind_duplicate(
            feedback, existing, embed_fn, similarity_threshold,
        )
        if matched is not None:
            store._update_embeddings(role, existing)
            return None
        store._update_embeddings(role, existing)

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

    # Embed the new skill
    new_embedding: list[float] | None = None
    if deduplicate and embed_fn is not None:
        import asyncio

        emb_result = embed_fn(content)
        if asyncio.iscoroutine(emb_result) or asyncio.isfuture(emb_result):
            new_embedding = await emb_result
        else:
            new_embedding = emb_result  # type: ignore[assignment]

    skill = Skill(
        role=role, content=content, source="learned", tags=tags or [],
        embedding=new_embedding,
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
    user_template: str | None = None,
    deduplicate: bool = True,
    similarity_threshold: float = 0.85,
    embed: EmbeddingCallable | None = None,
) -> list[Skill]:
    """Synthesize multiple skills in one LLM call from a batch of feedback items.

    When *deduplicate* is ``True``, each synthesized skill is checked against
    existing skills before being stored.  Skills that match an existing skill
    above *similarity_threshold* are silently dropped from the returned list.
    """
    existing = store.get_skills(role, tags=tags)
    existing_text = (
        "\n".join(f"- {s.content}" for s in existing) if existing else "(none)"
    )

    sys_prompt = system_prompt or _BATCH_SYSTEM_PROMPT
    items_text = _format_batch_items(items)
    usr_template = user_template or _BATCH_USER_TEMPLATE

    messages: list[dict[str, str]] = [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": usr_template.format(
                role=role,
                items_text=items_text,
                existing_skills=existing_text,
            ),
        },
    ]

    llm_fn = llm or default_openai_llm
    raw = llm_fn(messages).strip()
    contents = _parse_numbered_list(raw)

    embed_fn = embed  # None means "skip embedding"

    skills: list[Skill] = []
    for text in contents:
        if deduplicate and embed_fn is not None and existing:
            matched = _find_duplicate(text, existing, embed_fn, similarity_threshold)
            if matched is not None:
                continue

        new_embedding: list[float] | None = None
        if deduplicate and embed_fn is not None:
            new_embedding = embed_fn(text)

        skill = Skill(
            role=role, content=text, source="learned", tags=tags or [],
            embedding=new_embedding,
        )
        store.add_skill(skill)
        skills.append(skill)
        # Add to existing so subsequent items in the batch can dedup against it
        existing.append(skill)

    # Persist any newly-computed embeddings on pre-existing skills
    if deduplicate and embed_fn is not None and existing:
        store._update_embeddings(role, existing)

    return skills


async def asynthesize_skill_batch(
    role: str,
    items: list[dict[str, str]],
    store: "SkillStore",
    *,
    llm: AsyncLLMCallable | None = None,
    tags: list[str] | None = None,
    system_prompt: str | None = None,
    user_template: str | None = None,
    deduplicate: bool = True,
    similarity_threshold: float = 0.85,
    embed: AsyncEmbeddingCallable | EmbeddingCallable | None = None,
) -> list[Skill]:
    """Async version of :func:`synthesize_skill_batch`."""
    existing = store.get_skills(role, tags=tags)
    existing_text = (
        "\n".join(f"- {s.content}" for s in existing) if existing else "(none)"
    )

    sys_prompt = system_prompt or _BATCH_SYSTEM_PROMPT
    items_text = _format_batch_items(items)
    usr_template = user_template or _BATCH_USER_TEMPLATE

    messages: list[dict[str, str]] = [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": usr_template.format(
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

    embed_fn = embed  # None means "skip embedding"

    skills: list[Skill] = []
    for text in contents:
        if deduplicate and embed_fn is not None and existing:
            matched = await _afind_duplicate(
                text, existing, embed_fn, similarity_threshold,
            )
            if matched is not None:
                continue

        new_embedding: list[float] | None = None
        if deduplicate and embed_fn is not None:
            import asyncio

            emb_result = embed_fn(text)
            if asyncio.iscoroutine(emb_result) or asyncio.isfuture(emb_result):
                new_embedding = await emb_result
            else:
                new_embedding = emb_result  # type: ignore[assignment]

        skill = Skill(
            role=role, content=text, source="learned", tags=tags or [],
            embedding=new_embedding,
        )
        store.add_skill(skill)
        skills.append(skill)
        existing.append(skill)

    if deduplicate and embed_fn is not None and existing:
        store._update_embeddings(role, existing)

    return skills
