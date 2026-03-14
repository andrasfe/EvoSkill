"""Tests for SkillStore."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evoskill.skill import Skill
from evoskill.store import SkillStore


@pytest.fixture()
def store(tmp_path: Path) -> SkillStore:
    return SkillStore(storage_path=tmp_path)


class TestGetAndAddSkills:
    def test_empty_store_returns_no_skills(self, store: SkillStore) -> None:
        assert store.get_skills("analyst") == []

    def test_add_and_get(self, store: SkillStore) -> None:
        skill = Skill(role="analyst", content="validate input", source="manual")
        store.add_skill(skill)
        loaded = store.get_skills("analyst")
        assert len(loaded) == 1
        assert loaded[0].content == "validate input"
        assert loaded[0].source == "manual"

    def test_role_filtering(self, store: SkillStore) -> None:
        store.add_skill(Skill(role="analyst", content="skill a", source="manual"))
        store.add_skill(Skill(role="coder", content="skill b", source="learned"))
        assert len(store.get_skills("analyst")) == 1
        assert len(store.get_skills("coder")) == 1
        assert store.get_skills("analyst")[0].content == "skill a"

    def test_add_manual_skill(self, store: SkillStore) -> None:
        store.add_manual_skill("analyst", "always use UTC")
        skills = store.get_skills("analyst")
        assert len(skills) == 1
        assert skills[0].source == "manual"
        assert skills[0].content == "always use UTC"


class TestPersistence:
    def test_skills_survive_new_store_instance(self, tmp_path: Path) -> None:
        s1 = SkillStore(storage_path=tmp_path)
        s1.add_manual_skill("dev", "write tests")

        s2 = SkillStore(storage_path=tmp_path)
        skills = s2.get_skills("dev")
        assert len(skills) == 1
        assert skills[0].content == "write tests"

    def test_json_file_created(self, store: SkillStore, tmp_path: Path) -> None:
        store.add_manual_skill("analyst", "check nulls")
        path = tmp_path / "analyst.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data) == 1
        assert data[0]["content"] == "check nulls"


class TestListRoles:
    def test_empty(self, store: SkillStore) -> None:
        assert store.list_roles() == []

    def test_multiple_roles(self, store: SkillStore) -> None:
        store.add_manual_skill("analyst", "s1")
        store.add_manual_skill("coder", "s2")
        roles = store.list_roles()
        assert set(roles) == {"analyst", "coder"}
