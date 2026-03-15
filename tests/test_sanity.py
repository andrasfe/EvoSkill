"""Basic sanity checks that the public API is importable and coherent."""

from evoskill import Skill, SkillStore, __all__


def test_public_api_is_non_empty():
    """The package exposes a non-empty public API."""
    assert len(__all__) > 0


def test_skillstore_list_roles_returns_list():
    """SkillStore.list_roles() returns a list (possibly empty on a fresh store)."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        store = SkillStore(storage_path=Path(tmp))
        roles = store.list_roles()
        assert isinstance(roles, list)


def test_skill_dataclass_fields():
    """Skill instances expose the expected core fields."""
    skill = Skill(role="test", content="do something", source="manual")
    assert skill.role == "test"
    assert skill.content == "do something"
    assert skill.source == "manual"
    assert skill.enabled is True
    assert skill.hit_count == 0
    assert skill.miss_count == 0
