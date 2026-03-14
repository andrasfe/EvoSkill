"""SkillStore — persist and load skills to disk."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from .config import get_storage_path
from .skill import Skill


class SkillStore:
    """Thread-safe, file-backed store for skills."""

    def __init__(self, storage_path: Path | None = None) -> None:
        self._path = storage_path or get_storage_path()
        self._path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # -- public API ----------------------------------------------------------

    def get_skills(self, role: str) -> list[Skill]:
        """Return all skills for *role*."""
        with self._lock:
            return self._read_role(role)

    def add_skill(self, skill: Skill) -> None:
        """Persist a single skill."""
        with self._lock:
            skills = self._read_role(skill.role)
            skills.append(skill)
            self._write_role(skill.role, skills)

    def add_manual_skill(self, role: str, content: str) -> None:
        """Convenience method: add a manual skill for *role*."""
        self.add_skill(Skill(role=role, content=content, source="manual"))

    def list_roles(self) -> list[str]:
        """Return roles that have at least one stored skill."""
        with self._lock:
            return [
                p.stem for p in sorted(self._path.glob("*.json"))
            ]

    # -- internals -----------------------------------------------------------

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
