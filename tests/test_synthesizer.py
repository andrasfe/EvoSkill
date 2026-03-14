"""Tests for skill synthesis (OpenAI calls are mocked)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evoskill.store import SkillStore
from evoskill.synthesizer import synthesize_skill


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


class TestSynthesizeSkill:
    @patch("evoskill.synthesizer.get_api_key", return_value="sk-fake")
    @patch("evoskill.synthesizer.OpenAI")
    def test_calls_openai_and_stores_skill(
        self,
        mock_openai_cls: MagicMock,
        mock_key: MagicMock,
        store: SkillStore,
    ) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response(
            "Always validate JSON before parsing."
        )

        skill = synthesize_skill(
            role="analyst",
            input_prompt="parse this data",
            failure="JSONDecodeError: invalid json",
            store=store,
            model="gpt-4o-mini",
        )

        assert skill.content == "Always validate JSON before parsing."
        assert skill.source == "learned"
        assert skill.role == "analyst"

        # Verify the skill was persisted
        persisted = store.get_skills("analyst")
        assert len(persisted) == 1

    @patch("evoskill.synthesizer.get_api_key", return_value="sk-fake")
    @patch("evoskill.synthesizer.OpenAI")
    def test_prompt_includes_existing_skills(
        self,
        mock_openai_cls: MagicMock,
        mock_key: MagicMock,
        store: SkillStore,
    ) -> None:
        store.add_manual_skill("analyst", "existing skill one")

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response(
            "New skill"
        )

        synthesize_skill(
            role="analyst",
            input_prompt="do something",
            failure="some error",
            store=store,
        )

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_msg = messages[1]["content"]
        assert "existing skill one" in user_msg

    @patch("evoskill.synthesizer.get_api_key", return_value="sk-fake")
    @patch("evoskill.synthesizer.OpenAI")
    def test_no_existing_skills_says_none(
        self,
        mock_openai_cls: MagicMock,
        mock_key: MagicMock,
        store: SkillStore,
    ) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response("s")

        synthesize_skill(
            role="analyst",
            input_prompt="do something",
            failure="error",
            store=store,
        )

        call_args = mock_client.chat.completions.create.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        assert "(none)" in user_msg
