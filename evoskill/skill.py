"""Skill data model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Skill:
    """A single learned or manual skill."""

    role: str
    content: str
    source: str  # 'manual' | 'learned'
    tags: list[str] = field(default_factory=list)
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "source": self.source,
            "tags": self.tags,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Skill:
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        else:
            created_at = datetime.now(timezone.utc)
        return cls(
            role=data["role"],
            content=data["content"],
            source=data["source"],
            tags=data.get("tags", []),
            enabled=data.get("enabled", True),
            created_at=created_at,
        )
