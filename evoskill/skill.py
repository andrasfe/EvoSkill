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
    hit_count: int = 0
    miss_count: int = 0
    embedding: list[float] | None = field(default=None, repr=False)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def hit_rate(self) -> float:
        """Fraction of total marks that were hits.  Returns 0.0 if unmarked."""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total else 0.0

    def to_dict(self) -> dict:
        d: dict = {
            "role": self.role,
            "content": self.content,
            "source": self.source,
            "tags": self.tags,
            "enabled": self.enabled,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "created_at": self.created_at.isoformat(),
        }
        if self.embedding is not None:
            d["embedding"] = self.embedding
        return d

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
            hit_count=data.get("hit_count", 0),
            miss_count=data.get("miss_count", 0),
            embedding=data.get("embedding"),
            created_at=created_at,
        )
