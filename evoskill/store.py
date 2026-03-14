"""SkillStore — persist and load skills to disk (process-safe)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from filelock import FileLock

from .config import get_storage_path
from .skill import Skill


class SkillStore:
    """Process-safe, file-backed store for skills.

    Uses ``filelock`` for cross-process safety so multiple workers can
    read/write the same storage directory without races.
    """

    def __init__(self, storage_path: Path | None = None) -> None:
        self._path = storage_path or get_storage_path()
        self._path.mkdir(parents=True, exist_ok=True)

    # -- public API ----------------------------------------------------------

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
        with self._lock_for(role):
            skills = self._read_role(role)
        if enabled_only:
            skills = [s for s in skills if s.enabled]
        if tags:
            tag_set = set(tags)
            skills = [s for s in skills if tag_set.issubset(s.tags)]
        return skills

    def add_skill(self, skill: Skill) -> None:
        """Persist a single skill."""
        with self._lock_for(skill.role):
            skills = self._read_role(skill.role)
            skills.append(skill)
            self._write_role(skill.role, skills)

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

    def remove_skill(self, role: str, content: str) -> bool:
        """Remove the first skill matching *content* for *role*.

        Returns ``True`` if a skill was removed.
        """
        with self._lock_for(role):
            skills = self._read_role(role)
            for i, s in enumerate(skills):
                if s.content == content:
                    skills.pop(i)
                    self._write_role(role, skills)
                    return True
        return False

    def disable_skill(self, role: str, content: str) -> bool:
        """Disable (soft-delete) the first skill matching *content*.

        Returns ``True`` if a skill was disabled.
        """
        with self._lock_for(role):
            skills = self._read_role(role)
            for s in skills:
                if s.content == content and s.enabled:
                    s.enabled = False
                    self._write_role(role, skills)
                    return True
        return False

    def enable_skill(self, role: str, content: str) -> bool:
        """Re-enable a previously disabled skill.

        Returns ``True`` if a skill was enabled.
        """
        with self._lock_for(role):
            skills = self._read_role(role)
            for s in skills:
                if s.content == content and not s.enabled:
                    s.enabled = True
                    self._write_role(role, skills)
                    return True
        return False

    def list_roles(self) -> list[str]:
        """Return roles that have at least one stored skill."""
        return [p.stem for p in sorted(self._path.glob("*.json"))]

    def consolidate(
        self,
        role: str,
        llm: Callable[[list[dict[str, str]]], str],
        *,
        max_skills: int | None = None,
    ) -> list[Skill]:
        """Deduplicate / merge skills for *role* using the provided *llm*.

        *llm* is a callable that accepts a list of ``{"role": ..., "content": ...}``
        message dicts (chat-style) and returns a plain string response.

        If *max_skills* is given the LLM is asked to keep at most that many
        skills.  The consolidated list replaces whatever was on disk.
        """
        with self._lock_for(role):
            skills = self._read_role(role)
        if len(skills) < 2:
            return skills

        skill_list = "\n".join(f"- {s.content}" for s in skills)
        limit_note = (
            f" Keep at most {max_skills} skills." if max_skills else ""
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a skill consolidator. Given a list of skills, "
                    "merge duplicates, remove contradictions, and return a "
                    "clean numbered list of concise skills (1-3 sentences each). "
                    "Output ONLY the numbered list, nothing else."
                    + limit_note
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
            return skills  # LLM returned nothing useful, keep originals

        # Preserve manual source tag for skills that survived consolidation
        manual_contents = {s.content for s in skills if s.source == "manual"}
        new_skills: list[Skill] = []
        for text in new_contents:
            source = "manual" if text in manual_contents else "learned"
            new_skills.append(Skill(role=role, content=text, source=source))

        with self._lock_for(role):
            self._write_role(role, new_skills)
        return new_skills

    def learn_from_feedback(
        self,
        role: str,
        llm: Callable[[list[dict[str, str]]], str],
        *,
        input_prompt: str,
        agent_output: str,
        reviewer_feedback: str,
        tags: list[str] | None = None,
    ) -> Skill:
        """Synthesize a skill from another agent's structured feedback.

        *llm* is a callable that accepts chat-style messages and returns a
        plain string.
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
        )

    # -- internals -----------------------------------------------------------

    def _lock_for(self, role: str) -> FileLock:
        """Return a cross-process file lock for the given role."""
        return FileLock(str(self._path / f".{role}.lock"))

    def _role_file(self, role: str) -> Path:
        return self._path / f"{role}.json"

    def _read_role(self, role: str) -> list[Skill]:
        path = self._role_file(role)
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return [Skill.from_dict(d) for d in data]

    def _write_role(self, role: str, skills: list[Skill]) -> None:
        path = self._role_file(role)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps([s.to_dict() for s in skills], indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)  # atomic on POSIX


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
