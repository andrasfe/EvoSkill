"""SkillStore — primary API for skill management."""

from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .backend import FileBackend, StorageBackend
from .config import get_storage_path
from .skill import Skill

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path


@dataclass
class _RoleBuffer:
    """Internal buffer configuration for a single role."""

    items: list[dict[str, str]] = field(default_factory=list)
    llm: Any = None
    tags: list[str] | None = None
    system_prompt: str | None = None
    user_template: str | None = None
    batch_size: int = 10
    deduplicate: bool = True
    similarity_threshold: float = 0.85
    embed: Any = None


_INJECTION_HEADER = "[EvoSkill] Learned skills for this role:"


class SkillStore:
    """Primary API for storing, retrieving, and managing skills.

    Accepts an optional *backend* that implements the
    :class:`~evoskill.backend.StorageBackend` protocol.  When ``None``,
    a :class:`~evoskill.backend.FileBackend` is used with the path from
    ``EVOSKILL_STORAGE_PATH`` (or *storage_path*).
    """

    def __init__(
        self,
        storage_path: Path | None = None,
        *,
        backend: StorageBackend | None = None,
    ) -> None:
        if backend is not None:
            self._backend = backend
        else:
            path = storage_path or get_storage_path()
            self._backend = FileBackend(path)
        self._buffers: dict[str, _RoleBuffer] = {}
        self._buffer_lock = threading.Lock()

    # -- read / query --------------------------------------------------------

    def get_skills(
        self,
        role: str,
        *,
        tags: list[str] | None = None,
        enabled_only: bool = True,
    ) -> list[Skill]:
        """Return skills for *role*, optionally filtered by *tags*.

        If *tags* is given, only skills whose ``tags`` field contains **all**
        of the requested tags are returned.  Disabled skills are excluded by
        default (pass ``enabled_only=False`` to include them).
        """
        with self._backend.lock(role):
            skills = self._backend.read(role)
        if enabled_only:
            skills = [s for s in skills if s.enabled]
        if tags:
            tag_set = set(tags)
            skills = [s for s in skills if tag_set.issubset(s.tags)]
        return skills

    def get_skills_text(
        self,
        role: str,
        *,
        tags: list[str] | None = None,
        max_skills: int | None = None,
        query: str | None = None,
        embed: Callable[[str], list[float]] | None = None,
        relevance_threshold: float = 0.1,
        recency_half_life: float | None = None,
        max_tokens: int | None = None,
        compact: bool = False,
        llm: Callable[[list[dict[str, str]]], str] | None = None,
    ) -> str:
        """Return the formatted skill block ready to paste into a prompt.

        Returns an empty string if there are no matching skills.

        Parameters
        ----------
        query:
            When provided (together with *embed*), skills are ranked by
            semantic relevance to the query and only those above
            *relevance_threshold* are included.
        embed:
            Embedding callable ``(text) -> list[float]`` used for
            semantic retrieval.  Required when *query* is set.
        relevance_threshold:
            Minimum cosine similarity for a skill to be included when
            *query* is provided.  Default ``0.1``.
        recency_half_life:
            Half-life in days for recency decay.  When set, newer skills
            receive a higher ranking boost via an exponential decay
            multiplier.  A skill whose age equals *recency_half_life*
            retains 50 % of the recency bonus.  When ``None`` (default),
            recency is not factored into ranking.  Without semantic
            ranking, skills are always returned most-recent-first.
        max_tokens:
            Approximate token budget for the returned block.  Skills are
            added in rank order until the budget is exhausted.  Uses a
            simple ``len(text.split()) * 1.3`` heuristic.
        compact:
            When ``True``, the selected skills are compressed into a
            single concise paragraph via *llm* before returning.
            Requires *llm* to be set; silently falls back to the normal
            bullet list when *llm* is ``None``.
        llm:
            LLM callable used for *compact* mode.
        """
        skills = self.get_skills(role, tags=tags)
        if not skills:
            return ""

        # -- semantic ranking ----------------------------------------------------
        ranked = False
        if query and embed is not None:
            all_skills = list(skills)  # snapshot before filtering
            skills = _rank_by_relevance(
                skills,
                query,
                embed,
                relevance_threshold,
                recency_half_life=recency_half_life,
            )
            # Persist embeddings for ALL skills (including those below
            # threshold) to avoid redundant embed calls on future invocations.
            self._update_embeddings(role, all_skills)
            ranked = True
            if not skills:
                return ""

        # -- max_skills cap (applied after relevance ranking) --------------------
        if max_skills is not None:
            skills = skills[:max_skills] if ranked else skills[-max_skills:]

        if not skills:
            return ""

        # -- token-budget filtering ----------------------------------------------
        if max_tokens is not None:
            skills = _fit_token_budget(skills, max_tokens)
            if not skills:
                return ""

        # -- compact mode --------------------------------------------------------
        if compact and llm is not None:
            return _compact_skills(skills, role, llm)

        lines = [_INJECTION_HEADER]
        for s in skills:
            lines.append(f"- {s.content}")
        return "\n".join(lines) + "\n\n"

    def list_roles(self) -> list[str]:
        """Return roles that have at least one stored skill."""
        return self._backend.list_roles()

    # -- write ---------------------------------------------------------------

    def add_skill(self, skill: Skill) -> None:
        """Persist a single skill."""
        with self._backend.lock(skill.role):
            skills = self._backend.read(skill.role)
            skills.append(skill)
            self._backend.write(skill.role, skills)

    def _save_skills(self, role: str, skills: list[Skill]) -> None:
        """Persist *skills* for *role*, overwriting existing data.

        Used internally to flush back skills whose embeddings were
        computed during a deduplication check.

        .. warning:: This overwrites **all** skills for *role*.  Prefer
           :meth:`_update_embeddings` when you only need to persist
           newly-computed embeddings from a filtered subset.
        """
        with self._backend.lock(role):
            self._backend.write(role, skills)

    def _update_embeddings(
        self, role: str, skills_with_embeddings: list[Skill]
    ) -> None:
        """Merge cached embeddings back into the full (unfiltered) skill list.

        Reads **all** skills for *role* from the backend, updates the
        ``embedding`` field on any skill whose ``content`` matches one
        of the provided *skills_with_embeddings*, and writes the full
        list back.  This avoids the data-loss problem of overwriting
        all skills with a tag/enabled-filtered subset.
        """
        embedding_map: dict[str, list[float]] = {
            s.content: s.embedding
            for s in skills_with_embeddings
            if s.embedding is not None
        }
        if not embedding_map:
            return
        with self._backend.lock(role):
            all_skills = self._backend.read(role)
            changed = False
            for s in all_skills:
                if s.embedding is None and s.content in embedding_map:
                    s.embedding = embedding_map[s.content]
                    changed = True
            if changed:
                self._backend.write(role, all_skills)

    def add_manual_skill(
        self,
        role: str,
        content: str,
        *,
        tags: list[str] | None = None,
    ) -> None:
        """Add a manual skill for *role*."""
        self.add_skill(
            Skill(role=role, content=content, source="manual", tags=tags or [])
        )

    # -- lifecycle -----------------------------------------------------------

    def remove_skill(self, role: str, content: str) -> bool:
        """Remove the first skill matching *content* for *role*.

        Returns ``True`` if a skill was removed.
        """
        with self._backend.lock(role):
            skills = self._backend.read(role)
            for i, s in enumerate(skills):
                if s.content == content:
                    skills.pop(i)
                    self._backend.write(role, skills)
                    return True
        return False

    def disable_skill(self, role: str, content: str) -> bool:
        """Disable (soft-delete) the first skill matching *content*.

        Returns ``True`` if a skill was disabled.
        """
        with self._backend.lock(role):
            skills = self._backend.read(role)
            for s in skills:
                if s.content == content and s.enabled:
                    s.enabled = False
                    self._backend.write(role, skills)
                    return True
        return False

    def enable_skill(self, role: str, content: str) -> bool:
        """Re-enable a previously disabled skill.

        Returns ``True`` if a skill was enabled.
        """
        with self._backend.lock(role):
            skills = self._backend.read(role)
            for s in skills:
                if s.content == content and not s.enabled:
                    s.enabled = True
                    self._backend.write(role, skills)
                    return True
        return False

    # -- effectiveness tracking ----------------------------------------------

    def mark_hit(self, role: str, content: str) -> bool:
        """Record that a skill demonstrably prevented a repeat failure.

        Returns ``True`` if the skill was found and updated.
        """
        with self._backend.lock(role):
            skills = self._backend.read(role)
            for s in skills:
                if s.content == content:
                    s.hit_count += 1
                    self._backend.write(role, skills)
                    return True
        return False

    def mark_miss(self, role: str, content: str) -> bool:
        """Record that a skill did NOT help — the failure repeated.

        Returns ``True`` if the skill was found and updated.
        """
        with self._backend.lock(role):
            skills = self._backend.read(role)
            for s in skills:
                if s.content == content:
                    s.miss_count += 1
                    self._backend.write(role, skills)
                    return True
        return False

    # -- consolidation -------------------------------------------------------

    def consolidate(
        self,
        role: str,
        llm: Callable[[list[dict[str, str]]], str],
        *,
        max_skills: int | None = None,
        drop_zero_hit: bool = False,
    ) -> list[Skill]:
        """Deduplicate / merge skills for *role* using the provided *llm*.

        If *drop_zero_hit* is ``True``, skills with ``hit_count == 0`` **and**
        at least one ``miss_count`` are removed before consolidation.
        """
        with self._backend.lock(role):
            skills = self._backend.read(role)

        if drop_zero_hit:
            skills = [s for s in skills if not (s.hit_count == 0 and s.miss_count > 0)]

        if len(skills) < 2:
            with self._backend.lock(role):
                self._backend.write(role, skills)
            return skills

        skill_lines: list[str] = []
        for s in skills:
            rate_info = ""
            total = s.hit_count + s.miss_count
            if total > 0:
                rate_info = f" [hit_rate={s.hit_rate:.0%}, hits={s.hit_count}, misses={s.miss_count}]"
            skill_lines.append(f"- {s.content}{rate_info}")
        skill_list = "\n".join(skill_lines)

        limit_note = f" Keep at most {max_skills} skills." if max_skills else ""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a skill consolidator. Given a list of skills, "
                    "merge duplicates, remove contradictions, and return a "
                    "clean numbered list of concise skills (1-3 sentences each). "
                    "Prefer skills with higher hit rates. "
                    "Output ONLY the numbered list, nothing else." + limit_note
                ),
            },
            {
                "role": "user",
                "content": f"Skills for role '{role}':\n{skill_list}",
            },
        ]

        raw = llm(messages).strip()
        new_contents = _parse_numbered_list(raw)
        if not new_contents:
            return skills

        manual_contents = {s.content for s in skills if s.source == "manual"}
        new_skills: list[Skill] = []
        for text in new_contents:
            source = "manual" if text in manual_contents else "learned"
            new_skills.append(Skill(role=role, content=text, source=source))

        with self._backend.lock(role):
            self._backend.write(role, new_skills)
        return new_skills

    # -- feedback learning ---------------------------------------------------

    def learn_from_feedback(
        self,
        role: str,
        llm: Callable[[list[dict[str, str]]], str],
        *,
        input_prompt: str,
        agent_output: str,
        reviewer_feedback: str,
        tags: list[str] | None = None,
        system_prompt: str | None = None,
        user_template: str | None = None,
        deduplicate: bool = True,
        similarity_threshold: float = 0.85,
        embed: Callable[[str], list[float]] | None = None,
    ) -> Skill | None:
        """Synthesize a skill from another agent's structured feedback.

        Returns ``None`` when *deduplicate* is ``True`` and an existing skill
        already covers the feedback.
        """
        from .synthesizer import synthesize_skill_with_context

        return synthesize_skill_with_context(
            role=role,
            input_prompt=input_prompt,
            agent_output=agent_output,
            feedback=reviewer_feedback,
            store=self,
            llm=llm,
            tags=tags,
            system_prompt=system_prompt,
            user_template=user_template,
            deduplicate=deduplicate,
            similarity_threshold=similarity_threshold,
            embed=embed,
        )

    async def alearn_from_feedback(
        self,
        role: str,
        llm: Callable[[list[dict[str, str]]], Awaitable[str]],
        *,
        input_prompt: str,
        agent_output: str,
        reviewer_feedback: str,
        tags: list[str] | None = None,
        system_prompt: str | None = None,
        user_template: str | None = None,
        deduplicate: bool = True,
        similarity_threshold: float = 0.85,
        embed: Callable | None = None,
    ) -> Skill | None:
        """Async version of :meth:`learn_from_feedback`."""
        from .synthesizer import asynthesize_skill_with_context

        return await asynthesize_skill_with_context(
            role=role,
            input_prompt=input_prompt,
            agent_output=agent_output,
            feedback=reviewer_feedback,
            store=self,
            llm=llm,
            tags=tags,
            system_prompt=system_prompt,
            user_template=user_template,
            deduplicate=deduplicate,
            similarity_threshold=similarity_threshold,
            embed=embed,
        )

    # -- batch feedback ------------------------------------------------------

    def learn_from_feedback_batch(
        self,
        role: str,
        llm: Callable[[list[dict[str, str]]], str],
        *,
        items: list[dict[str, str]],
        tags: list[str] | None = None,
        system_prompt: str | None = None,
        deduplicate: bool = True,
        similarity_threshold: float = 0.85,
        embed: Callable[[str], list[float]] | None = None,
    ) -> list[Skill]:
        """Synthesize multiple skills from a batch of feedback items in one LLM call.

        *items* is a list of dicts, each with keys:
        ``input_prompt``, ``agent_output``, ``reviewer_feedback``.

        When *deduplicate* is ``True``, synthesized skills that are too similar
        to existing skills are silently dropped.
        """
        from .synthesizer import synthesize_skill_batch

        return synthesize_skill_batch(
            role=role,
            items=items,
            store=self,
            llm=llm,
            tags=tags,
            system_prompt=system_prompt,
            deduplicate=deduplicate,
            similarity_threshold=similarity_threshold,
            embed=embed,
        )

    async def alearn_from_feedback_batch(
        self,
        role: str,
        llm: Callable[[list[dict[str, str]]], Awaitable[str]],
        *,
        items: list[dict[str, str]],
        tags: list[str] | None = None,
        system_prompt: str | None = None,
        deduplicate: bool = True,
        similarity_threshold: float = 0.85,
        embed: Callable | None = None,
    ) -> list[Skill]:
        """Async version of :meth:`learn_from_feedback_batch`."""
        from .synthesizer import asynthesize_skill_batch

        return await asynthesize_skill_batch(
            role=role,
            items=items,
            store=self,
            llm=llm,
            tags=tags,
            system_prompt=system_prompt,
            deduplicate=deduplicate,
            similarity_threshold=similarity_threshold,
            embed=embed,
        )

    # -- buffering -----------------------------------------------------------

    def _buffer_item(
        self,
        role: str,
        item: dict[str, str],
        *,
        llm: Any = None,
        tags: list[str] | None = None,
        system_prompt: str | None = None,
        user_template: str | None = None,
        batch_size: int = 10,
        deduplicate: bool = True,
        similarity_threshold: float = 0.85,
        embed: Any = None,
    ) -> list[Skill]:
        """Add a feedback item to the internal buffer and auto-flush when full.

        Returns any skills created by an auto-flush, or an empty list.
        """
        with self._buffer_lock:
            if role not in self._buffers:
                self._buffers[role] = _RoleBuffer(
                    llm=llm,
                    tags=tags,
                    system_prompt=system_prompt,
                    user_template=user_template,
                    batch_size=batch_size,
                    deduplicate=deduplicate,
                    similarity_threshold=similarity_threshold,
                    embed=embed,
                )
            buf = self._buffers[role]
            buf.items.append(item)
            if len(buf.items) >= buf.batch_size:
                items_to_flush = buf.items[:]
                buf.items.clear()
            else:
                return []

        from .synthesizer import synthesize_skill_batch

        return synthesize_skill_batch(
            role=role,
            items=items_to_flush,
            store=self,
            llm=buf.llm,
            tags=buf.tags,
            system_prompt=buf.system_prompt,
            user_template=buf.user_template,
            deduplicate=buf.deduplicate,
            similarity_threshold=buf.similarity_threshold,
            embed=buf.embed,
        )

    async def _abuffer_item(
        self,
        role: str,
        item: dict[str, str],
        *,
        llm: Any = None,
        tags: list[str] | None = None,
        system_prompt: str | None = None,
        user_template: str | None = None,
        batch_size: int = 10,
        deduplicate: bool = True,
        similarity_threshold: float = 0.85,
        embed: Any = None,
    ) -> list[Skill]:
        """Async version of :meth:`_buffer_item`."""
        with self._buffer_lock:
            if role not in self._buffers:
                self._buffers[role] = _RoleBuffer(
                    llm=llm,
                    tags=tags,
                    system_prompt=system_prompt,
                    user_template=user_template,
                    batch_size=batch_size,
                    deduplicate=deduplicate,
                    similarity_threshold=similarity_threshold,
                    embed=embed,
                )
            buf = self._buffers[role]
            buf.items.append(item)
            if len(buf.items) >= buf.batch_size:
                items_to_flush = buf.items[:]
                buf.items.clear()
            else:
                return []

        from .synthesizer import asynthesize_skill_batch

        return await asynthesize_skill_batch(
            role=role,
            items=items_to_flush,
            store=self,
            llm=buf.llm,
            tags=buf.tags,
            system_prompt=buf.system_prompt,
            user_template=buf.user_template,
            deduplicate=buf.deduplicate,
            similarity_threshold=buf.similarity_threshold,
            embed=buf.embed,
        )

    def flush(self, role: str | None = None) -> list[Skill]:
        """Drain buffered items and synthesize skills via batch synthesis.

        When *role* is ``None``, all buffered roles are flushed.
        Returns all newly created skills.
        """
        from .synthesizer import synthesize_skill_batch

        with self._buffer_lock:
            if role is not None:
                roles = [role] if role in self._buffers else []
            else:
                roles = list(self._buffers.keys())

            to_flush: list[tuple[str, list[dict[str, str]], _RoleBuffer]] = []
            for r in roles:
                buf = self._buffers[r]
                if buf.items:
                    to_flush.append((r, buf.items[:], buf))
                    buf.items.clear()

        all_skills: list[Skill] = []
        for r, items, buf in to_flush:
            skills = synthesize_skill_batch(
                role=r,
                items=items,
                store=self,
                llm=buf.llm,
                tags=buf.tags,
                system_prompt=buf.system_prompt,
                user_template=buf.user_template,
                deduplicate=buf.deduplicate,
                similarity_threshold=buf.similarity_threshold,
                embed=buf.embed,
            )
            all_skills.extend(skills)
        return all_skills

    async def aflush(self, role: str | None = None) -> list[Skill]:
        """Async version of :meth:`flush`."""
        from .synthesizer import asynthesize_skill_batch

        with self._buffer_lock:
            if role is not None:
                roles = [role] if role in self._buffers else []
            else:
                roles = list(self._buffers.keys())

            to_flush: list[tuple[str, list[dict[str, str]], _RoleBuffer]] = []
            for r in roles:
                buf = self._buffers[r]
                if buf.items:
                    to_flush.append((r, buf.items[:], buf))
                    buf.items.clear()

        all_skills: list[Skill] = []
        for r, items, buf in to_flush:
            skills = await asynthesize_skill_batch(
                role=r,
                items=items,
                store=self,
                llm=buf.llm,
                tags=buf.tags,
                system_prompt=buf.system_prompt,
                user_template=buf.user_template,
                deduplicate=buf.deduplicate,
                similarity_threshold=buf.similarity_threshold,
                embed=buf.embed,
            )
            all_skills.extend(skills)
        return all_skills

    @property
    def pending_buffer_count(self) -> int:
        """Total number of items waiting in all buffers."""
        with self._buffer_lock:
            return sum(len(buf.items) for buf in self._buffers.values())

    # -- export / import -----------------------------------------------------

    def export_skills(self, role: str) -> list[dict]:
        """Export all skills for *role* as a list of plain dicts."""
        with self._backend.lock(role):
            skills = self._backend.read(role)
        return [s.to_dict() for s in skills]

    def import_skills(self, role: str, data: list[dict]) -> list[Skill]:
        """Import skills from a list of dicts (as produced by :meth:`export_skills`).

        Appends to existing skills for *role*.  Returns the imported skills.
        """
        imported = [Skill.from_dict(d) for d in data]
        with self._backend.lock(role):
            existing = self._backend.read(role)
            existing.extend(imported)
            self._backend.write(role, existing)
        return imported

    # -- legacy compatibility ------------------------------------------------

    @property
    def _path(self) -> Path:
        if isinstance(self._backend, FileBackend):
            return self._backend._path
        raise AttributeError("_path is only available with FileBackend")

    @_path.setter
    def _path(self, value: Path) -> None:
        if isinstance(self._backend, FileBackend):
            self._backend._path = value
            self._backend._path.mkdir(parents=True, exist_ok=True)
        else:
            raise AttributeError("_path is only available with FileBackend")


def _parse_numbered_list(text: str) -> list[str]:
    """Parse '1. ...' style numbered list from LLM output."""
    lines = text.strip().splitlines()
    result: list[str] = []
    for line in lines:
        line = line.strip()
        m = re.match(r"^\d+[\.\)]\s*(.+)$", line)
        if m:
            result.append(m.group(1).strip())
    return result


# ---------------------------------------------------------------------------
# Smart injection helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _recency_weight(
    skill: Skill,
    half_life_days: float | None,
    *,
    now: datetime | None = None,
) -> float:
    """Exponential recency multiplier in ``(0, 1]``.

    Returns ``1.0`` when *half_life_days* is ``None`` (recency disabled).
    A skill whose age equals *half_life_days* scores ``0.5``.
    """
    if half_life_days is None or half_life_days <= 0:
        return 1.0
    if now is None:
        now = datetime.now(UTC)
    age_days = max((now - skill.created_at).total_seconds() / 86_400, 0.0)
    return math.pow(0.5, age_days / half_life_days)


def _rank_by_relevance(
    skills: list[Skill],
    query: str,
    embed: Callable[[str], list[float]],
    threshold: float,
    *,
    recency_half_life: float | None = None,
) -> list[Skill]:
    """Return *skills* ranked by relevance to *query*, filtered by *threshold*.

    Ranking score::

        similarity x (0.5 + 0.5 x hit_rate) x recency_weight

    Skills with no hit/miss data get a neutral effectiveness weight of 0.5.
    ``recency_weight`` is an exponential decay controlled by
    *recency_half_life* (days).  When ``None``, recency is neutral (1.0).
    """
    query_embedding = embed(query)
    now = datetime.now(UTC)
    scored: list[tuple[float, Skill]] = []
    for skill in skills:
        if skill.embedding is None:
            skill.embedding = embed(skill.content)
        sim = _cosine_similarity(query_embedding, skill.embedding)
        effectiveness = 0.5 + 0.5 * skill.hit_rate
        recency = _recency_weight(skill, recency_half_life, now=now)
        score = sim * effectiveness * recency
        if sim >= threshold:
            scored.append((score, skill))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [skill for _, skill in scored]


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ``ceil(word_count x 1.3)``."""
    return math.ceil(len(text.split()) * 1.3)


def _fit_token_budget(skills: list[Skill], max_tokens: int) -> list[Skill]:
    """Return a prefix of *skills* that fits within *max_tokens*.

    Accounts for the injection header and bullet formatting.
    """
    header_tokens = _estimate_tokens(_INJECTION_HEADER)
    budget = max_tokens - header_tokens
    if budget <= 0:
        return []
    result: list[Skill] = []
    used = 0
    for skill in skills:
        line_tokens = _estimate_tokens(f"- {skill.content}")
        if used + line_tokens > budget:
            break
        result.append(skill)
        used += line_tokens
    return result


_COMPACT_SYSTEM = (
    "You are a concise summarizer. Given a list of agent skills, compress "
    "them into a single short paragraph (2-4 sentences) that captures the "
    "key guidance. Output ONLY the paragraph — no preamble, no bullet points."
)


def _compact_skills(
    skills: list[Skill],
    role: str,
    llm: Callable[[list[dict[str, str]]], str],
) -> str:
    """Compress *skills* into a compact paragraph via *llm*."""
    bullet_list = "\n".join(f"- {s.content}" for s in skills)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _COMPACT_SYSTEM},
        {
            "role": "user",
            "content": f"Role: {role}\n\nSkills:\n{bullet_list}",
        },
    ]
    compressed = llm(messages).strip()
    return f"[EvoSkill] Guidance for this role:\n{compressed}\n\n"
