"""Tests for skill synthesis (LLM calls are mocked)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evoskill.store import SkillStore
from evoskill.synthesizer import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_TEMPLATE,
    asynthesize_skill,
    asynthesize_skill_batch,
    asynthesize_skill_with_context,
    synthesize_skill,
    synthesize_skill_batch,
    synthesize_skill_with_context,
)


@pytest.fixture()
def store(tmp_path: Path) -> SkillStore:
    return SkillStore(storage_path=tmp_path)


def _make_mock_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class TestSynthesizeSkillDefault:
    """Tests using the default OpenAI adapter (mocked)."""

    @patch("evoskill.synthesizer.get_api_key", return_value="sk-fake")
    @patch("evoskill.synthesizer.default_openai_llm", return_value="Always validate JSON before parsing.")
    def test_calls_default_llm_and_stores_skill(
        self,
        mock_llm: MagicMock,
        mock_key: MagicMock,
        store: SkillStore,
    ) -> None:
        skill = synthesize_skill(
            role="analyst",
            input_prompt="parse this data",
            failure="JSONDecodeError: invalid json",
            store=store,
        )

        assert skill.content == "Always validate JSON before parsing."
        assert skill.source == "learned"
        assert skill.role == "analyst"

        persisted = store.get_skills("analyst")
        assert len(persisted) == 1


class TestSynthesizeSkillCustomLLM:
    """Tests using a user-supplied LLM callable."""

    def test_custom_llm_is_called(self, store: SkillStore) -> None:
        def my_llm(messages: list[dict[str, str]]) -> str:
            return "Custom skill from my LLM."

        skill = synthesize_skill(
            role="analyst",
            input_prompt="parse this",
            failure="error",
            store=store,
            llm=my_llm,
        )
        assert skill.content == "Custom skill from my LLM."
        assert len(store.get_skills("analyst")) == 1

    def test_custom_llm_receives_correct_messages(self, store: SkillStore) -> None:
        store.add_manual_skill("analyst", "existing skill one")
        captured_messages: list = []

        def my_llm(messages: list[dict[str, str]]) -> str:
            captured_messages.extend(messages)
            return "New skill"

        synthesize_skill(
            role="analyst",
            input_prompt="do something",
            failure="some error",
            store=store,
            llm=my_llm,
        )

        assert len(captured_messages) == 2
        user_msg = captured_messages[1]["content"]
        assert "existing skill one" in user_msg
        assert "do something" in user_msg
        assert "some error" in user_msg

    def test_no_existing_skills_says_none(self, store: SkillStore) -> None:
        captured: list = []

        def my_llm(messages: list[dict[str, str]]) -> str:
            captured.extend(messages)
            return "s"

        synthesize_skill(
            role="analyst",
            input_prompt="do something",
            failure="error",
            store=store,
            llm=my_llm,
        )

        user_msg = captured[1]["content"]
        assert "(none)" in user_msg

    def test_tags_attached_to_skill(self, store: SkillStore) -> None:
        def my_llm(messages: list[dict[str, str]]) -> str:
            return "Tagged skill"

        skill = synthesize_skill(
            role="analyst",
            input_prompt="x",
            failure="y",
            store=store,
            llm=my_llm,
            tags=["python", "data"],
        )
        assert skill.tags == ["python", "data"]


class TestSynthesizeSkillWithContext:
    """Tests for the richer synthesis entry point."""

    def test_includes_agent_output_and_feedback(self, store: SkillStore) -> None:
        captured: list = []

        def my_llm(messages: list[dict[str, str]]) -> str:
            captured.extend(messages)
            return "Contextual skill"

        skill = synthesize_skill_with_context(
            role="writer",
            input_prompt="write a poem",
            agent_output="roses are red",
            feedback="too cliché, be more creative",
            store=store,
            llm=my_llm,
        )

        user_msg = captured[1]["content"]
        assert "write a poem" in user_msg
        assert "roses are red" in user_msg
        assert "too cliché" in user_msg
        assert skill.content == "Contextual skill"
        assert skill.role == "writer"


class TestAsynthesizeSkill:
    """Tests for the async synthesis entry points."""

    @pytest.mark.asyncio
    async def test_async_custom_llm_is_called(self, store: SkillStore) -> None:
        async def my_async_llm(messages: list[dict[str, str]]) -> str:
            return "Async custom skill."

        skill = await asynthesize_skill(
            role="analyst",
            input_prompt="parse this",
            failure="error",
            store=store,
            llm=my_async_llm,
        )
        assert skill.content == "Async custom skill."
        assert skill.source == "learned"
        assert len(store.get_skills("analyst")) == 1

    @pytest.mark.asyncio
    async def test_async_with_context(self, store: SkillStore) -> None:
        captured: list = []

        async def my_async_llm(messages: list[dict[str, str]]) -> str:
            captured.extend(messages)
            return "Async contextual skill"

        skill = await asynthesize_skill_with_context(
            role="writer",
            input_prompt="write an essay",
            agent_output="draft text",
            feedback="needs more detail",
            store=store,
            llm=my_async_llm,
        )

        user_msg = captured[1]["content"]
        assert "write an essay" in user_msg
        assert "draft text" in user_msg
        assert "needs more detail" in user_msg
        assert skill.content == "Async contextual skill"

    @pytest.mark.asyncio
    async def test_async_tags_attached(self, store: SkillStore) -> None:
        async def my_async_llm(messages: list[dict[str, str]]) -> str:
            return "Tagged async skill"

        skill = await asynthesize_skill(
            role="analyst",
            input_prompt="x",
            failure="y",
            store=store,
            llm=my_async_llm,
            tags=["async", "test"],
        )
        assert skill.tags == ["async", "test"]

    @pytest.mark.asyncio
    async def test_async_fallback_to_sync_default(
        self, store: SkillStore,
    ) -> None:
        """When no llm is provided, falls back to the sync default_openai_llm."""
        with patch(
            "evoskill.synthesizer.default_openai_llm",
            return_value="Fallback skill",
        ):
            skill = await asynthesize_skill(
                role="analyst",
                input_prompt="x",
                failure="y",
                store=store,
            )
        assert skill.content == "Fallback skill"


# ---------------------------------------------------------------------------
# Pluggable synthesis prompts
# ---------------------------------------------------------------------------


class TestPluggablePrompts:
    def test_custom_system_prompt(self, store: SkillStore) -> None:
        captured: list = []

        def my_llm(messages: list[dict[str, str]]) -> str:
            captured.extend(messages)
            return "Custom prompt skill"

        custom_sys = "You are a documentation expert."
        skill = synthesize_skill(
            role="writer",
            input_prompt="write docs",
            failure="too verbose",
            store=store,
            llm=my_llm,
            system_prompt=custom_sys,
        )
        assert captured[0]["content"] == custom_sys
        assert skill.content == "Custom prompt skill"

    def test_custom_user_template(self, store: SkillStore) -> None:
        captured: list = []

        def my_llm(messages: list[dict[str, str]]) -> str:
            captured.extend(messages)
            return "Template skill"

        custom_tmpl = "ROLE={role} FAIL={feedback} EXISTING={existing_skills} INPUT={input_prompt} OUTPUT={agent_output}"
        synthesize_skill(
            role="dev",
            input_prompt="code",
            failure="bug",
            store=store,
            llm=my_llm,
            user_template=custom_tmpl,
        )
        user_msg = captured[1]["content"]
        assert "ROLE=dev" in user_msg
        assert "FAIL=bug" in user_msg

    def test_default_prompts_used_when_none(self, store: SkillStore) -> None:
        captured: list = []

        def my_llm(messages: list[dict[str, str]]) -> str:
            captured.extend(messages)
            return "Default skill"

        synthesize_skill(
            role="dev",
            input_prompt="x",
            failure="y",
            store=store,
            llm=my_llm,
        )
        assert captured[0]["content"] == DEFAULT_SYSTEM_PROMPT

    def test_with_context_custom_prompts(self, store: SkillStore) -> None:
        captured: list = []

        def my_llm(messages: list[dict[str, str]]) -> str:
            captured.extend(messages)
            return "Context skill"

        custom_sys = "Custom system for context."
        synthesize_skill_with_context(
            role="dev",
            input_prompt="inp",
            agent_output="out",
            feedback="fb",
            store=store,
            llm=my_llm,
            system_prompt=custom_sys,
        )
        assert captured[0]["content"] == custom_sys

    @pytest.mark.asyncio
    async def test_async_custom_system_prompt(self, store: SkillStore) -> None:
        captured: list = []

        async def my_llm(messages: list[dict[str, str]]) -> str:
            captured.extend(messages)
            return "Async custom prompt skill"

        custom_sys = "You are an async expert."
        await asynthesize_skill(
            role="dev",
            input_prompt="x",
            failure="y",
            store=store,
            llm=my_llm,
            system_prompt=custom_sys,
        )
        assert captured[0]["content"] == custom_sys


# ---------------------------------------------------------------------------
# Batch synthesis
# ---------------------------------------------------------------------------


class TestBatchSynthesis:
    def test_synthesize_skill_batch(self, store: SkillStore) -> None:
        def fake_llm(messages: list[dict[str, str]]) -> str:
            user_msg = messages[1]["content"]
            assert "Item 1" in user_msg
            assert "Item 2" in user_msg
            return "1. First batch skill.\n2. Second batch skill."

        items = [
            {"input_prompt": "a", "agent_output": "b", "reviewer_feedback": "c"},
            {"input_prompt": "d", "agent_output": "e", "reviewer_feedback": "f"},
        ]
        skills = synthesize_skill_batch(
            role="dev", items=items, store=store, llm=fake_llm,
        )
        assert len(skills) == 2
        assert skills[0].content == "First batch skill."
        assert skills[1].content == "Second batch skill."

    def test_batch_with_custom_system_prompt(self, store: SkillStore) -> None:
        captured: list = []

        def fake_llm(messages: list[dict[str, str]]) -> str:
            captured.extend(messages)
            return "1. Skill."

        items = [{"input_prompt": "a", "agent_output": "b", "reviewer_feedback": "c"}]
        synthesize_skill_batch(
            role="dev", items=items, store=store, llm=fake_llm,
            system_prompt="Custom batch prompt.",
        )
        assert captured[0]["content"] == "Custom batch prompt."

    @pytest.mark.asyncio
    async def test_asynthesize_skill_batch(self, store: SkillStore) -> None:
        async def fake_llm(messages: list[dict[str, str]]) -> str:
            return "1. Async batch skill one.\n2. Async batch skill two."

        items = [
            {"input_prompt": "a", "agent_output": "b", "reviewer_feedback": "c"},
            {"input_prompt": "d", "agent_output": "e", "reviewer_feedback": "f"},
        ]
        skills = await asynthesize_skill_batch(
            role="dev", items=items, store=store, llm=fake_llm,
        )
        assert len(skills) == 2
        assert skills[0].content == "Async batch skill one."

    def test_batch_with_tags(self, store: SkillStore) -> None:
        def fake_llm(messages: list[dict[str, str]]) -> str:
            return "1. Tagged skill."

        items = [{"input_prompt": "a", "agent_output": "b", "reviewer_feedback": "c"}]
        skills = synthesize_skill_batch(
            role="dev", items=items, store=store, llm=fake_llm,
            tags=["batch", "test"],
        )
        assert skills[0].tags == ["batch", "test"]
