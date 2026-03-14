"""Tests for SkillStore."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evoskill.backend import FileBackend, StorageBackend
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


class TestAlearnFromFeedback:
    @pytest.mark.asyncio
    async def test_async_learn_from_feedback(self, store: SkillStore) -> None:
        async def fake_async_llm(messages: list[dict[str, str]]) -> str:
            user_msg = messages[1]["content"]
            assert "async input" in user_msg
            assert "async output" in user_msg
            assert "async feedback" in user_msg
            return "Async feedback skill."

        skill = await store.alearn_from_feedback(
            role="writer",
            llm=fake_async_llm,
            input_prompt="async input",
            agent_output="async output",
            reviewer_feedback="async feedback",
        )
        assert skill.content == "Async feedback skill."
        assert skill.source == "learned"
        assert len(store.get_skills("writer")) == 1

    @pytest.mark.asyncio
    async def test_async_learn_from_feedback_with_tags(self, store: SkillStore) -> None:
        async def fake_async_llm(messages: list[dict[str, str]]) -> str:
            return "Async tagged skill."

        skill = await store.alearn_from_feedback(
            role="writer",
            llm=fake_async_llm,
            input_prompt="input",
            agent_output="output",
            reviewer_feedback="feedback",
            tags=["async", "review"],
        )
        assert skill.tags == ["async", "review"]


# ---------------------------------------------------------------------------
# 1. get_skills_text (SkillStore as primary API)
# ---------------------------------------------------------------------------


class TestGetSkillsText:
    def test_empty_returns_empty_string(self, store: SkillStore) -> None:
        assert store.get_skills_text("analyst") == ""

    def test_returns_formatted_block(self, store: SkillStore) -> None:
        store.add_manual_skill("analyst", "always validate")
        store.add_manual_skill("analyst", "check nulls")
        text = store.get_skills_text("analyst")
        assert "[EvoSkill]" in text
        assert "- always validate" in text
        assert "- check nulls" in text
        # Should end with double newline for easy concatenation
        assert text.endswith("\n\n")

    def test_max_skills_limits_output(self, store: SkillStore) -> None:
        for i in range(5):
            store.add_manual_skill("dev", f"skill {i}")
        text = store.get_skills_text("dev", max_skills=2)
        assert "skill 3" in text
        assert "skill 4" in text
        assert "skill 0" not in text

    def test_tags_filter(self, store: SkillStore) -> None:
        store.add_manual_skill("dev", "python tip", tags=["python"])
        store.add_manual_skill("dev", "sql tip", tags=["sql"])
        text = store.get_skills_text("dev", tags=["python"])
        assert "python tip" in text
        assert "sql tip" not in text


# ---------------------------------------------------------------------------
# 3. Skill effectiveness tracking
# ---------------------------------------------------------------------------


class TestEffectivenessTracking:
    def test_mark_hit(self, store: SkillStore) -> None:
        store.add_manual_skill("dev", "write tests")
        assert store.mark_hit("dev", "write tests") is True
        skills = store.get_skills("dev")
        assert skills[0].hit_count == 1
        assert skills[0].miss_count == 0

    def test_mark_miss(self, store: SkillStore) -> None:
        store.add_manual_skill("dev", "write tests")
        assert store.mark_miss("dev", "write tests") is True
        skills = store.get_skills("dev")
        assert skills[0].miss_count == 1

    def test_mark_hit_nonexistent(self, store: SkillStore) -> None:
        assert store.mark_hit("dev", "nope") is False

    def test_mark_miss_nonexistent(self, store: SkillStore) -> None:
        assert store.mark_miss("dev", "nope") is False

    def test_hit_rate_property(self) -> None:
        s = Skill(role="dev", content="x", source="learned", hit_count=3, miss_count=1)
        assert s.hit_rate == 0.75

    def test_hit_rate_zero_when_unmarked(self) -> None:
        s = Skill(role="dev", content="x", source="learned")
        assert s.hit_rate == 0.0

    def test_consolidate_drop_zero_hit(self, store: SkillStore) -> None:
        s1 = Skill(role="dev", content="good skill", source="learned", hit_count=5, miss_count=1)
        s2 = Skill(role="dev", content="bad skill", source="learned", hit_count=0, miss_count=3)
        s3 = Skill(role="dev", content="new skill", source="learned", hit_count=0, miss_count=0)
        store.add_skill(s1)
        store.add_skill(s2)
        store.add_skill(s3)

        def fake_llm(messages: list[dict[str, str]]) -> str:
            # After dropping zero-hit s2, only s1 and s3 remain
            return "1. good skill\n2. new skill"

        result = store.consolidate("dev", fake_llm, drop_zero_hit=True)
        contents = [s.content for s in result]
        assert "bad skill" not in contents
        assert "good skill" in contents
        assert "new skill" in contents


# ---------------------------------------------------------------------------
# 4. Batch feedback ingestion
# ---------------------------------------------------------------------------


class TestBatchFeedback:
    def test_learn_from_feedback_batch(self, store: SkillStore) -> None:
        def fake_llm(messages: list[dict[str, str]]) -> str:
            # Should see all items in one call
            user_msg = messages[1]["content"]
            assert "Item 1" in user_msg
            assert "Item 2" in user_msg
            return "1. Skill from item one.\n2. Skill from item two."

        items = [
            {"input_prompt": "inp1", "agent_output": "out1", "reviewer_feedback": "fb1"},
            {"input_prompt": "inp2", "agent_output": "out2", "reviewer_feedback": "fb2"},
        ]
        skills = store.learn_from_feedback_batch(
            role="writer", llm=fake_llm, items=items,
        )
        assert len(skills) == 2
        assert skills[0].content == "Skill from item one."
        assert skills[1].content == "Skill from item two."
        assert len(store.get_skills("writer")) == 2

    @pytest.mark.asyncio
    async def test_alearn_from_feedback_batch(self, store: SkillStore) -> None:
        async def fake_async_llm(messages: list[dict[str, str]]) -> str:
            return "1. Async skill one.\n2. Async skill two."

        items = [
            {"input_prompt": "a", "agent_output": "b", "reviewer_feedback": "c"},
            {"input_prompt": "d", "agent_output": "e", "reviewer_feedback": "f"},
        ]
        skills = await store.alearn_from_feedback_batch(
            role="writer", llm=fake_async_llm, items=items,
        )
        assert len(skills) == 2
        assert len(store.get_skills("writer")) == 2


# ---------------------------------------------------------------------------
# 5. Storage backend protocol
# ---------------------------------------------------------------------------


class TestStorageBackend:
    def test_file_backend_satisfies_protocol(self, tmp_path: Path) -> None:
        backend = FileBackend(tmp_path / "data")
        assert isinstance(backend, StorageBackend)

    def test_custom_backend(self, tmp_path: Path) -> None:
        """A minimal in-memory backend that satisfies the protocol."""
        from contextlib import contextmanager

        class MemoryBackend:
            def __init__(self):
                self._data: dict[str, list[Skill]] = {}

            def read(self, role: str) -> list[Skill]:
                return list(self._data.get(role, []))

            def write(self, role: str, skills: list[Skill]) -> None:
                self._data[role] = list(skills)

            @contextmanager
            def lock(self, role: str):
                yield

            def list_roles(self) -> list[str]:
                return list(self._data.keys())

        backend = MemoryBackend()
        store = SkillStore(backend=backend)
        store.add_manual_skill("dev", "test skill")
        skills = store.get_skills("dev")
        assert len(skills) == 1
        assert skills[0].content == "test skill"
        assert store.list_roles() == ["dev"]


# ---------------------------------------------------------------------------
# 6. Skill export/import
# ---------------------------------------------------------------------------


class TestExportImport:
    def test_export_skills(self, store: SkillStore) -> None:
        store.add_manual_skill("dev", "write tests")
        store.add_manual_skill("dev", "handle errors")
        exported = store.export_skills("dev")
        assert len(exported) == 2
        assert isinstance(exported[0], dict)
        assert exported[0]["content"] == "write tests"
        assert exported[1]["content"] == "handle errors"

    def test_import_skills(self, store: SkillStore) -> None:
        data = [
            {"role": "dev", "content": "imported skill", "source": "learned"},
        ]
        imported = store.import_skills("dev", data)
        assert len(imported) == 1
        assert imported[0].content == "imported skill"
        assert len(store.get_skills("dev")) == 1

    def test_round_trip(self, store: SkillStore) -> None:
        store.add_manual_skill("dev", "skill A")
        store.add_manual_skill("dev", "skill B")
        exported = store.export_skills("dev")

        store2 = SkillStore(storage_path=store._path / "other")
        store2.import_skills("dev", exported)
        skills = store2.get_skills("dev")
        assert len(skills) == 2
        assert {s.content for s in skills} == {"skill A", "skill B"}

    def test_import_appends_to_existing(self, store: SkillStore) -> None:
        store.add_manual_skill("dev", "existing")
        data = [{"role": "dev", "content": "new one", "source": "learned"}]
        store.import_skills("dev", data)
        assert len(store.get_skills("dev")) == 2
