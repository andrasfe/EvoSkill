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


class TestTags:
    def test_add_skill_with_tags(self, store: SkillStore) -> None:
        store.add_manual_skill("analyst", "use pandas", tags=["python", "data"])
        skills = store.get_skills("analyst")
        assert len(skills) == 1
        assert skills[0].tags == ["python", "data"]

    def test_filter_by_tags(self, store: SkillStore) -> None:
        store.add_manual_skill("analyst", "use pandas", tags=["python", "data"])
        store.add_manual_skill("analyst", "use SQL", tags=["sql", "data"])
        store.add_manual_skill("analyst", "be concise", tags=[])

        assert len(store.get_skills("analyst", tags=["data"])) == 2
        assert len(store.get_skills("analyst", tags=["python"])) == 1
        assert len(store.get_skills("analyst", tags=["python", "data"])) == 1
        assert len(store.get_skills("analyst")) == 3

    def test_filter_by_tags_requires_all(self, store: SkillStore) -> None:
        store.add_manual_skill("analyst", "skill a", tags=["x"])
        assert store.get_skills("analyst", tags=["x", "y"]) == []


class TestDisableEnable:
    def test_disable_skill(self, store: SkillStore) -> None:
        store.add_manual_skill("dev", "write tests")
        assert store.disable_skill("dev", "write tests") is True
        assert store.get_skills("dev") == []
        assert len(store.get_skills("dev", enabled_only=False)) == 1

    def test_enable_skill(self, store: SkillStore) -> None:
        store.add_manual_skill("dev", "write tests")
        store.disable_skill("dev", "write tests")
        assert store.enable_skill("dev", "write tests") is True
        assert len(store.get_skills("dev")) == 1

    def test_disable_nonexistent_returns_false(self, store: SkillStore) -> None:
        assert store.disable_skill("dev", "nope") is False


class TestRemoveSkill:
    def test_remove_existing(self, store: SkillStore) -> None:
        store.add_manual_skill("dev", "s1")
        store.add_manual_skill("dev", "s2")
        assert store.remove_skill("dev", "s1") is True
        assert len(store.get_skills("dev")) == 1
        assert store.get_skills("dev")[0].content == "s2"

    def test_remove_nonexistent(self, store: SkillStore) -> None:
        assert store.remove_skill("dev", "nope") is False


class TestConsolidate:
    def test_consolidate_merges_skills(self, store: SkillStore) -> None:
        store.add_manual_skill("dev", "always write tests")
        store.add_skill(Skill(role="dev", content="make sure to write unit tests", source="learned"))
        store.add_skill(Skill(role="dev", content="handle errors gracefully", source="learned"))

        def fake_llm(messages: list[dict[str, str]]) -> str:
            return "1. Always write unit tests.\n2. Handle errors gracefully."

        result = store.consolidate("dev", fake_llm)
        assert len(result) == 2
        assert result[0].content == "Always write unit tests."
        assert result[1].content == "Handle errors gracefully."

    def test_consolidate_with_max_skills(self, store: SkillStore) -> None:
        for i in range(5):
            store.add_manual_skill("dev", f"skill {i}")

        def fake_llm(messages: list[dict[str, str]]) -> str:
            # Check that the system prompt mentions the limit
            assert "3" in messages[0]["content"]
            return "1. combined skill a\n2. combined skill b\n3. combined skill c"

        result = store.consolidate("dev", fake_llm, max_skills=3)
        assert len(result) == 3

    def test_consolidate_single_skill_is_noop(self, store: SkillStore) -> None:
        store.add_manual_skill("dev", "only one")

        def should_not_be_called(messages: list[dict[str, str]]) -> str:
            raise AssertionError("LLM should not be called")

        result = store.consolidate("dev", should_not_be_called)
        assert len(result) == 1


class TestLearnFromFeedback:
    def test_learn_from_feedback(self, store: SkillStore) -> None:
        def fake_llm(messages: list[dict[str, str]]) -> str:
            user_msg = messages[1]["content"]
            assert "my input" in user_msg
            assert "agent said this" in user_msg
            assert "reviewer says fix it" in user_msg
            return "Always double-check output before returning."

        skill = store.learn_from_feedback(
            role="writer",
            llm=fake_llm,
            input_prompt="my input",
            agent_output="agent said this",
            reviewer_feedback="reviewer says fix it",
        )
        assert skill.content == "Always double-check output before returning."
        assert skill.source == "learned"
        assert skill.role == "writer"
        assert len(store.get_skills("writer")) == 1

    def test_learn_from_feedback_with_tags(self, store: SkillStore) -> None:
        def fake_llm(messages: list[dict[str, str]]) -> str:
            return "Tag-aware skill."

        skill = store.learn_from_feedback(
            role="writer",
            llm=fake_llm,
            input_prompt="input",
            agent_output="output",
            reviewer_feedback="feedback",
            tags=["review", "writing"],
        )
        assert skill.tags == ["review", "writing"]
