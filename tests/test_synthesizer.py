"""Tests for skill synthesis (LLM calls are mocked)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evoskill.store import SkillStore
from evoskill.synthesizer import synthesize_skill, synthesize_skill_with_context


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
