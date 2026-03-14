"""Storage backend protocol — pluggable persistence layer."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

from filelock import FileLock

from .skill import Skill


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol that any storage backend must satisfy.

    The default implementation is :class:`FileBackend` (file-based JSON with
    ``filelock``).  Consumers can provide a SQLite, Redis, or any other
    backend by implementing this protocol and passing it to
    :class:`~evoskill.store.SkillStore`.
    """

    def read(self, role: str) -> list[Skill]:
        """Return all skills stored for *role*."""
        ...

    def write(self, role: str, skills: list[Skill]) -> None:
        """Overwrite the skill list for *role*."""
        ...

    def lock(self, role: str) -> Iterator[None]:
        """Context manager that acquires an exclusive lock for *role*.

        Usage::

            with backend.lock(role):
                skills = backend.read(role)
                ...
                backend.write(role, skills)
        """
        ...

    def list_roles(self) -> list[str]:
        """Return all roles that have stored skills."""
        ...


# ---------------------------------------------------------------------------
# Default file-based backend
# ---------------------------------------------------------------------------


class FileBackend:
    """JSON-file-per-role backend with ``filelock`` for process safety."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.mkdir(parents=True, exist_ok=True)

    def read(self, role: str) -> list[Skill]:
        fpath = self._role_file(role)
        if not fpath.exists():
            return []
        data = json.loads(fpath.read_text(encoding="utf-8"))
        return [Skill.from_dict(d) for d in data]

    def write(self, role: str, skills: list[Skill]) -> None:
        fpath = self._role_file(role)
        tmp = fpath.with_suffix(".tmp")
        tmp.write_text(
            json.dumps([s.to_dict() for s in skills], indent=2),
            encoding="utf-8",
        )
        tmp.replace(fpath)  # atomic on POSIX

    @contextmanager
    def lock(self, role: str) -> Iterator[None]:
        fl = FileLock(str(self._path / f".{role}.lock"))
        with fl:
            yield

    def list_roles(self) -> list[str]:
        return [p.stem for p in sorted(self._path.glob("*.json"))]

    # -- helpers -------------------------------------------------------------

    def _role_file(self, role: str) -> Path:
        return self._path / f"{role}.json"
