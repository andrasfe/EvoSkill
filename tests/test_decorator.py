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


def _patch_store(func, tmp_path: Path) -> SkillStore:
    """Replace the decorator's internal store with one backed by tmp_path."""
    store = SkillStore(storage_path=_store_path(tmp_path))
    func._evoskill_store = store  # type: ignore[attr-defined]
    # Also patch the store reference inside the closure
    # We'll use the store fixture directly via the decorator's attribute
    return store


# ---------------------------------------------------------------------------
# Sync function tests
# ---------------------------------------------------------------------------


class TestSyncWrapping:
    def test_basic_call(self, tmp_path: Path) -> None:
        @evoskill(role="tester")
        def agent(prompt: str) -> str:
            return f"got: {prompt}"

        # Replace store
        store = SkillStore(storage_path=_store_path(tmp_path))
        agent._evoskill_store.__dict__.update(store.__dict__)
        # Monkey-patch the store path
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

    @patch("evoskill.decorator.synthesize_skill")
    def test_async_exception_triggers_learning(
        self, mock_synth: MagicMock, tmp_path: Path
    ) -> None:
        @evoskill(role="dev")
        async def agent(prompt: str) -> str:
            raise RuntimeError("oops")

        agent._evoskill_store._path = _store_path(tmp_path)

        with pytest.raises(RuntimeError, match="oops"):
            asyncio.get_event_loop().run_until_complete(agent("fix this"))

        mock_synth.assert_called_once()


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
