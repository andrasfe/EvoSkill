"""Tests for the @evoskill decorator."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evoskill.decorator import evoskill, _INJECTION_HEADER
from evoskill.store import SkillStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _store_path(tmp_path: Path) -> Path:
    """Return a unique sub-directory inside tmp_path for skill storage."""
    p = tmp_path / "skills"
    p.mkdir(exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Sync function tests
# ---------------------------------------------------------------------------


class TestSyncWrapping:
    def test_basic_call(self, tmp_path: Path) -> None:
        @evoskill(role="tester")
        def agent(prompt: str) -> str:
            return f"got: {prompt}"

        agent._evoskill_store._path = _store_path(tmp_path)

        result = agent("hello")
        assert "hello" in result

    def test_skill_prepending(self, tmp_path: Path) -> None:
        @evoskill(role="analyst")
        def agent(prompt: str) -> str:
            return prompt  # echo back the augmented prompt

        agent._evoskill_store._path = _store_path(tmp_path)
        agent._evoskill_store.add_manual_skill("analyst", "always validate")

        result = agent("do analysis")
        assert _INJECTION_HEADER in result
        assert "always validate" in result
        assert "do analysis" in result

    @patch("evoskill.decorator.synthesize_skill")
    def test_exception_triggers_learning(
        self, mock_synth: MagicMock, tmp_path: Path
    ) -> None:
        @evoskill(role="analyst")
        def agent(prompt: str) -> str:
            raise ValueError("bad data")

        agent._evoskill_store._path = _store_path(tmp_path)

        with pytest.raises(ValueError, match="bad data"):
            agent("analyze this")

        mock_synth.assert_called_once()
        call_args = mock_synth.call_args
        assert call_args[0][0] == "analyst"  # role

    @patch("evoskill.decorator.synthesize_skill")
    def test_learn_when_callback(
        self, mock_synth: MagicMock, tmp_path: Path
    ) -> None:
        def should_learn(inp: str, out: str) -> bool:
            return "bad" in out

        @evoskill(role="analyst", learn_when=should_learn)
        def agent(prompt: str) -> str:
            return "bad result"

        agent._evoskill_store._path = _store_path(tmp_path)

        agent("test")
        mock_synth.assert_called_once()

    @patch("evoskill.decorator.synthesize_skill")
    def test_no_learning_on_success(
        self, mock_synth: MagicMock, tmp_path: Path
    ) -> None:
        @evoskill(role="analyst")
        def agent(prompt: str) -> str:
            return "good result"

        agent._evoskill_store._path = _store_path(tmp_path)

        agent("test")
        mock_synth.assert_not_called()

    def test_manual_skills_via_decorator(self, tmp_path: Path) -> None:
        @evoskill(role="dev", skills=["use type hints", "write docstrings"])
        def agent(prompt: str) -> str:
            return prompt

        agent._evoskill_store._path = _store_path(tmp_path)

        result = agent("code review")
        assert "use type hints" in result
        assert "write docstrings" in result


# ---------------------------------------------------------------------------
# Async function tests
# ---------------------------------------------------------------------------


class TestAsyncWrapping:
    def test_async_basic_call(self, tmp_path: Path) -> None:
        @evoskill(role="tester")
        async def agent(prompt: str) -> str:
            return f"got: {prompt}"

        agent._evoskill_store._path = _store_path(tmp_path)

        result = asyncio.get_event_loop().run_until_complete(agent("hello"))
        assert "hello" in result

    def test_async_skill_prepending(self, tmp_path: Path) -> None:
        @evoskill(role="analyst")
        async def agent(prompt: str) -> str:
            return prompt

        agent._evoskill_store._path = _store_path(tmp_path)
        agent._evoskill_store.add_manual_skill("analyst", "check nulls")

        result = asyncio.get_event_loop().run_until_complete(agent("query"))
        assert "check nulls" in result

    @patch("evoskill.decorator.asynthesize_skill")
    def test_async_exception_triggers_learning(
        self, mock_asynth: MagicMock, tmp_path: Path
    ) -> None:
        # Make the mock return a coroutine
        async def _noop(*a, **kw):
            return MagicMock()

        mock_asynth.side_effect = _noop

        @evoskill(role="dev")
        async def agent(prompt: str) -> str:
            raise RuntimeError("oops")

        agent._evoskill_store._path = _store_path(tmp_path)

        with pytest.raises(RuntimeError, match="oops"):
            asyncio.get_event_loop().run_until_complete(agent("fix this"))

        mock_asynth.assert_called_once()

    @patch("evoskill.decorator.asynthesize_skill")
    def test_async_learn_when_uses_async_synthesis(
        self, mock_asynth: MagicMock, tmp_path: Path
    ) -> None:
        async def _noop(*a, **kw):
            return MagicMock()

        mock_asynth.side_effect = _noop

        def should_learn(inp: str, out: str) -> bool:
            return "bad" in out

        @evoskill(role="dev", learn_when=should_learn)
        async def agent(prompt: str) -> str:
            return "bad output"

        agent._evoskill_store._path = _store_path(tmp_path)

        asyncio.get_event_loop().run_until_complete(agent("test"))
        mock_asynth.assert_called_once()

    def test_async_with_async_llm(self, tmp_path: Path) -> None:
        """Async decorator passes async LLM through to synthesis."""

        async def my_async_llm(messages: list[dict[str, str]]) -> str:
            return "async skill"

        @evoskill(role="dev", llm=my_async_llm)
        async def agent(prompt: str) -> str:
            return prompt

        agent._evoskill_store._path = _store_path(tmp_path)

        result = asyncio.get_event_loop().run_until_complete(agent("hello"))
        assert "hello" in result


# ---------------------------------------------------------------------------
# Method / class tests
# ---------------------------------------------------------------------------


class TestMethodWrapping:
    def test_role_inferred_from_class_name(self, tmp_path: Path) -> None:
        class DataAnalyst:
            @evoskill()
            def run(self, prompt: str) -> str:
                return prompt

        obj = DataAnalyst()
        obj.run._evoskill_store._path = _store_path(tmp_path)
        obj.run._evoskill_store.add_manual_skill("DataAnalyst", "be precise")

        result = obj.run("analyze")
        assert "be precise" in result

    def test_explicit_role_overrides_class(self, tmp_path: Path) -> None:
        class DataAnalyst:
            @evoskill(role="custom_role")
            def run(self, prompt: str) -> str:
                return prompt

        obj = DataAnalyst()
        obj.run._evoskill_store._path = _store_path(tmp_path)
        obj.run._evoskill_store.add_manual_skill("custom_role", "custom skill")

        result = obj.run("test")
        assert "custom skill" in result


# ---------------------------------------------------------------------------
# Custom inject_skills callback
# ---------------------------------------------------------------------------


class TestInjectSkills:
    def test_custom_inject_skills(self, tmp_path: Path) -> None:
        """User provides a custom inject_skills to control where skills go."""

        def my_inject(args: tuple, kwargs: dict, skills_text: str) -> tuple[tuple, dict]:
            # Inject skills into a kwarg called 'system_prompt' instead
            new_kwargs = {**kwargs, "system_prompt": skills_text + kwargs.get("system_prompt", "")}
            return args, new_kwargs

        @evoskill(role="custom", inject_skills=my_inject)
        def agent(task: dict, system_prompt: str = "") -> str:
            return f"sys={system_prompt} task={task}"

        agent._evoskill_store._path = _store_path(tmp_path)
        agent._evoskill_store.add_manual_skill("custom", "be careful")

        result = agent({"type": "analyze"}, system_prompt="base prompt")
        assert "be careful" in result
        assert "base prompt" in result

    def test_structured_input_not_modified_without_inject(self, tmp_path: Path) -> None:
        """When input is not a string and no inject_skills, args pass through unchanged."""

        @evoskill(role="structured")
        def agent(task: dict) -> str:
            return str(task)

        agent._evoskill_store._path = _store_path(tmp_path)
        agent._evoskill_store.add_manual_skill("structured", "some skill")

        # The dict arg is not a string, so default inject is a no-op
        result = agent({"key": "value"})
        assert "key" in result


# ---------------------------------------------------------------------------
# BYO LLM
# ---------------------------------------------------------------------------


class TestBYOLLM:
    @patch("evoskill.decorator.synthesize_skill")
    def test_sync_llm_passed_to_synthesizer(
        self, mock_synth: MagicMock, tmp_path: Path
    ) -> None:
        def my_llm(messages: list[dict[str, str]]) -> str:
            return "custom skill"

        @evoskill(role="dev", llm=my_llm)
        def agent(prompt: str) -> str:
            raise ValueError("fail")

        agent._evoskill_store._path = _store_path(tmp_path)

        with pytest.raises(ValueError):
            agent("test")

        mock_synth.assert_called_once()
        assert mock_synth.call_args.kwargs["llm"] is my_llm

    @patch("evoskill.decorator.asynthesize_skill")
    def test_async_llm_passed_to_async_synthesizer(
        self, mock_asynth: MagicMock, tmp_path: Path
    ) -> None:
        async def _noop(*a, **kw):
            return MagicMock()

        mock_asynth.side_effect = _noop

        async def my_async_llm(messages: list[dict[str, str]]) -> str:
            return "async custom skill"

        @evoskill(role="dev", llm=my_async_llm)
        async def agent(prompt: str) -> str:
            raise ValueError("fail")

        agent._evoskill_store._path = _store_path(tmp_path)

        with pytest.raises(ValueError):
            asyncio.get_event_loop().run_until_complete(agent("test"))

        mock_asynth.assert_called_once()
        assert mock_asynth.call_args.kwargs["llm"] is my_async_llm


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------


class TestDecoratorTags:
    def test_tags_used_for_skill_retrieval(self, tmp_path: Path) -> None:
        @evoskill(role="analyst", tags=["python"])
        def agent(prompt: str) -> str:
            return prompt

        agent._evoskill_store._path = _store_path(tmp_path)
        agent._evoskill_store.add_manual_skill("analyst", "tagged skill", tags=["python"])
        agent._evoskill_store.add_manual_skill("analyst", "untagged skill", tags=["sql"])

        result = agent("do analysis")
        assert "tagged skill" in result
        # The sql-tagged skill should NOT appear since decorator filters by tags=["python"]
        assert "untagged skill" not in result


# ---------------------------------------------------------------------------
# max_skills
# ---------------------------------------------------------------------------


class TestMaxSkills:
    def test_max_skills_limits_injection(self, tmp_path: Path) -> None:
        @evoskill(role="dev", max_skills=2)
        def agent(prompt: str) -> str:
            return prompt

        agent._evoskill_store._path = _store_path(tmp_path)
        for i in range(5):
            agent._evoskill_store.add_manual_skill("dev", f"skill {i}")

        result = agent("test")
        # Only the last 2 skills should be injected
        assert "skill 3" in result
        assert "skill 4" in result
        assert "skill 0" not in result
