"""Tests for the @evoskill decorator."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

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


# ---------------------------------------------------------------------------
# 1. inject_field — Pydantic model field injection
# ---------------------------------------------------------------------------


class AgentInput(BaseModel):
    task: str
    skill_context: str = ""


class TestInjectField:
    def test_injects_into_pydantic_field(self, tmp_path: Path) -> None:
        """inject_field sets the named field on the first Pydantic arg."""

        @evoskill(role="writer", inject_field="skill_context")
        def agent(input_data: AgentInput) -> str:
            return f"ctx={input_data.skill_context} task={input_data.task}"

        agent._evoskill_store._path = _store_path(tmp_path)
        agent._evoskill_store.add_manual_skill("writer", "use active voice")

        result = agent(AgentInput(task="write essay"))
        assert "use active voice" in result
        assert "write essay" in result

    def test_original_model_not_mutated(self, tmp_path: Path) -> None:
        """model_copy should produce a new object — original stays unchanged."""

        @evoskill(role="writer", inject_field="skill_context")
        def agent(input_data: AgentInput) -> str:
            return input_data.skill_context

        agent._evoskill_store._path = _store_path(tmp_path)
        agent._evoskill_store.add_manual_skill("writer", "be concise")

        original = AgentInput(task="draft")
        agent(original)
        assert original.skill_context == ""

    def test_no_skills_no_mutation(self, tmp_path: Path) -> None:
        """With no skills, the field stays at its default."""

        @evoskill(role="empty", inject_field="skill_context")
        def agent(input_data: AgentInput) -> str:
            return input_data.skill_context

        agent._evoskill_store._path = _store_path(tmp_path)
        result = agent(AgentInput(task="hello"))
        assert result == ""

    def test_inject_field_on_method(self, tmp_path: Path) -> None:
        """inject_field works correctly when the decorated function is a method."""

        class Writer:
            @evoskill(role="Writer", inject_field="skill_context")
            def run(self, input_data: AgentInput) -> str:
                return input_data.skill_context

        obj = Writer()
        obj.run._evoskill_store._path = _store_path(tmp_path)
        obj.run._evoskill_store.add_manual_skill("Writer", "cite sources")

        result = obj.run(AgentInput(task="report"))
        assert "cite sources" in result

    def test_inject_field_and_inject_skills_mutually_exclusive(self) -> None:
        """Passing both inject_field and inject_skills should raise."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            evoskill(
                role="x",
                inject_skills=lambda a, k, s: (a, k),
                inject_field="skill_context",
            )

    def test_inject_field_async(self, tmp_path: Path) -> None:
        """inject_field works with async functions."""

        @evoskill(role="writer", inject_field="skill_context")
        async def agent(input_data: AgentInput) -> str:
            return input_data.skill_context

        agent._evoskill_store._path = _store_path(tmp_path)
        agent._evoskill_store.add_manual_skill("writer", "short sentences")

        result = asyncio.get_event_loop().run_until_complete(
            agent(AgentInput(task="blog"))
        )
        assert "short sentences" in result


# ---------------------------------------------------------------------------
# 2. extract_input — pull prompt text from structured input
# ---------------------------------------------------------------------------


class TestExtractInput:
    @patch("evoskill.decorator.synthesize_skill")
    def test_extract_input_used_for_synthesis_context(
        self, mock_synth: MagicMock, tmp_path: Path
    ) -> None:
        """extract_input provides the prompt string for synthesis."""

        def my_extract(input_data: AgentInput) -> str:
            return input_data.task

        @evoskill(
            role="dev",
            extract_input=my_extract,
            inject_field="skill_context",
        )
        def agent(input_data: AgentInput) -> str:
            raise ValueError("oops")

        agent._evoskill_store._path = _store_path(tmp_path)

        with pytest.raises(ValueError):
            agent(AgentInput(task="my specific task"))

        mock_synth.assert_called_once()
        # The second positional arg to synthesize_skill is the input_prompt
        assert mock_synth.call_args[0][1] == "my specific task"

    @patch("evoskill.decorator.synthesize_skill")
    def test_default_prompt_extraction_for_non_string(
        self, mock_synth: MagicMock, tmp_path: Path
    ) -> None:
        """Without extract_input and non-string arg, prompt defaults to ''."""

        @evoskill(role="dev")
        def agent(data: dict) -> str:
            raise ValueError("fail")

        agent._evoskill_store._path = _store_path(tmp_path)

        with pytest.raises(ValueError):
            agent({"key": "val"})

        mock_synth.assert_called_once()
        assert mock_synth.call_args[0][1] == ""

    @patch("evoskill.decorator.synthesize_skill")
    def test_extract_input_receives_kwargs(
        self, mock_synth: MagicMock, tmp_path: Path
    ) -> None:
        """extract_input should receive both args and kwargs."""

        def my_extract(data: dict, mode: str = "") -> str:
            return f"{data['key']}-{mode}"

        @evoskill(role="dev", extract_input=my_extract)
        def agent(data: dict, mode: str = "") -> str:
            raise ValueError("fail")

        agent._evoskill_store._path = _store_path(tmp_path)

        with pytest.raises(ValueError):
            agent({"key": "val"}, mode="fast")

        mock_synth.assert_called_once()
        assert mock_synth.call_args[0][1] == "val-fast"


# ---------------------------------------------------------------------------
# 3. extract_output — pull content from structured output
# ---------------------------------------------------------------------------


class ReviewResult(BaseModel):
    score: int
    issues: str


class TestExtractOutput:
    @patch("evoskill.decorator.synthesize_skill")
    def test_extract_output_used_in_learn_when(
        self, mock_synth: MagicMock, tmp_path: Path
    ) -> None:
        """extract_output converts output before passing to synthesis.

        learn_when still receives the raw result object so it can inspect
        any field.  extract_output only affects the string sent to the
        synthesizer.
        """

        def my_extract(result: ReviewResult) -> str:
            return result.issues

        def should_learn(inp: str, out: ReviewResult) -> bool:
            return "error" in out.issues

        @evoskill(
            role="coder",
            learn_when=should_learn,
            extract_output=my_extract,
        )
        def agent(prompt: str) -> ReviewResult:
            return ReviewResult(score=3, issues="found error in line 5")

        agent._evoskill_store._path = _store_path(tmp_path)
        agent("check code")

        mock_synth.assert_called_once()
        # The failure text (3rd positional arg) should be the extracted issues
        assert "found error in line 5" in mock_synth.call_args[0][2]

    @patch("evoskill.decorator.synthesize_skill")
    def test_default_str_conversion_without_extract_output(
        self, mock_synth: MagicMock, tmp_path: Path
    ) -> None:
        """Without extract_output, str(output) is used."""

        def should_learn(_inp: str, _out: str) -> bool:
            return True

        @evoskill(role="dev", learn_when=should_learn)
        def agent(prompt: str) -> str:
            return "plain output"

        agent._evoskill_store._path = _store_path(tmp_path)
        agent("test")

        mock_synth.assert_called_once()
        assert "plain output" in mock_synth.call_args[0][2]

    @patch("evoskill.decorator.asynthesize_skill")
    def test_extract_output_async(
        self, mock_asynth: MagicMock, tmp_path: Path
    ) -> None:
        """extract_output works with async wrappers too."""

        async def _noop(*a, **kw):
            return MagicMock()

        mock_asynth.side_effect = _noop

        def my_extract(result: ReviewResult) -> str:
            return result.issues

        def should_learn(_inp: str, out: ReviewResult) -> bool:
            return "bug" in out.issues

        @evoskill(
            role="coder",
            learn_when=should_learn,
            extract_output=my_extract,
        )
        async def agent(prompt: str) -> ReviewResult:
            return ReviewResult(score=1, issues="bug in parser")

        agent._evoskill_store._path = _store_path(tmp_path)
        asyncio.get_event_loop().run_until_complete(agent("review"))

        mock_asynth.assert_called_once()
        assert "bug in parser" in mock_asynth.call_args[0][2]


# ---------------------------------------------------------------------------
# 4. teach_role — cross-agent teaching
# ---------------------------------------------------------------------------


class TestTeachRole:
    @patch("evoskill.decorator.synthesize_skill")
    def test_teach_role_on_exception(
        self, mock_synth: MagicMock, tmp_path: Path
    ) -> None:
        """Skills from exceptions are stored under teach_role, not the agent's own role."""

        @evoskill(role="reviewer", teach_role="writer")
        def agent(prompt: str) -> str:
            raise ValueError("bad")

        agent._evoskill_store._path = _store_path(tmp_path)

        with pytest.raises(ValueError):
            agent("review this")

        mock_synth.assert_called_once()
        # First positional arg is the role — should be 'writer', not 'reviewer'
        assert mock_synth.call_args[0][0] == "writer"

    @patch("evoskill.decorator.synthesize_skill")
    def test_teach_role_on_learn_when(
        self, mock_synth: MagicMock, tmp_path: Path
    ) -> None:
        """learn_when callback stores skills under teach_role."""

        def should_learn(_inp: str, _out: str) -> bool:
            return True

        @evoskill(role="reviewer", teach_role="writer", learn_when=should_learn)
        def agent(prompt: str) -> str:
            return "issues found"

        agent._evoskill_store._path = _store_path(tmp_path)
        agent("review")

        mock_synth.assert_called_once()
        assert mock_synth.call_args[0][0] == "writer"

    @patch("evoskill.decorator.asynthesize_skill")
    def test_teach_role_async(
        self, mock_asynth: MagicMock, tmp_path: Path
    ) -> None:
        """teach_role works with async wrappers."""

        async def _noop(*a, **kw):
            return MagicMock()

        mock_asynth.side_effect = _noop

        @evoskill(role="reviewer", teach_role="writer")
        async def agent(prompt: str) -> str:
            raise RuntimeError("fail")

        agent._evoskill_store._path = _store_path(tmp_path)

        with pytest.raises(RuntimeError):
            asyncio.get_event_loop().run_until_complete(agent("check"))

        mock_asynth.assert_called_once()
        assert mock_asynth.call_args[0][0] == "writer"

    @patch("evoskill.decorator.synthesize_skill")
    def test_no_teach_role_uses_own_role(
        self, mock_synth: MagicMock, tmp_path: Path
    ) -> None:
        """Without teach_role, skills are stored under the agent's own role."""

        @evoskill(role="analyst")
        def agent(prompt: str) -> str:
            raise ValueError("err")

        agent._evoskill_store._path = _store_path(tmp_path)

        with pytest.raises(ValueError):
            agent("test")

        mock_synth.assert_called_once()
        assert mock_synth.call_args[0][0] == "analyst"


# ---------------------------------------------------------------------------
# 5. Async wrapper uses async LLM callable
# ---------------------------------------------------------------------------


class TestAsyncNonBlocking:
    @patch("evoskill.decorator.asynthesize_skill")
    def test_async_wrapper_calls_asynthesize(
        self, mock_asynth: MagicMock, tmp_path: Path
    ) -> None:
        """Async wrapper must call asynthesize_skill, not synthesize_skill."""

        async def _noop(*a, **kw):
            return MagicMock()

        mock_asynth.side_effect = _noop

        async def my_async_llm(messages: list[dict[str, str]]) -> str:
            return "skill from async llm"

        @evoskill(role="dev", llm=my_async_llm)
        async def agent(prompt: str) -> str:
            raise ValueError("fail")

        agent._evoskill_store._path = _store_path(tmp_path)

        with pytest.raises(ValueError):
            asyncio.get_event_loop().run_until_complete(agent("test"))

        # Must use async synthesis, not sync
        mock_asynth.assert_called_once()
        assert mock_asynth.call_args.kwargs["llm"] is my_async_llm

    @patch("evoskill.decorator.asynthesize_skill")
    def test_async_learn_when_calls_asynthesize(
        self, mock_asynth: MagicMock, tmp_path: Path
    ) -> None:
        """Async learn_when path also uses asynthesize_skill."""

        async def _noop(*a, **kw):
            return MagicMock()

        mock_asynth.side_effect = _noop

        async def my_llm(msgs: list[dict[str, str]]) -> str:
            return "skill"

        @evoskill(
            role="dev",
            llm=my_llm,
            learn_when=lambda i, o: True,
        )
        async def agent(prompt: str) -> str:
            return "output"

        agent._evoskill_store._path = _store_path(tmp_path)
        asyncio.get_event_loop().run_until_complete(agent("test"))
        mock_asynth.assert_called_once()


# ---------------------------------------------------------------------------
# 6. is_method — explicit method declaration
# ---------------------------------------------------------------------------


class TestIsMethod:
    def test_is_method_true_skips_self(self, tmp_path: Path) -> None:
        """is_method=True treats the first arg as self even without 'self' param name."""

        class Agent:
            @evoskill(role="Agent", is_method=True)
            def run(this_is_not_self, prompt: str) -> str:
                return prompt

        obj = Agent()
        obj.run._evoskill_store._path = _store_path(tmp_path)
        obj.run._evoskill_store.add_manual_skill("Agent", "be helpful")

        result = obj.run("hello")
        assert "be helpful" in result
        assert "hello" in result

    def test_is_method_false_on_function_with_self_param(self, tmp_path: Path) -> None:
        """is_method=False prevents treating first arg as self."""

        @evoskill(role="agent", is_method=False)
        def process(self_data: str) -> str:
            return self_data

        process._evoskill_store._path = _store_path(tmp_path)
        process._evoskill_store.add_manual_skill("agent", "check types")

        result = process("my data")
        # Skills should be prepended to the first arg (self_data)
        assert "check types" in result
        assert "my data" in result

    def test_is_method_none_uses_heuristic(self, tmp_path: Path) -> None:
        """is_method=None (default) falls back to the self/cls heuristic."""

        class MyAgent:
            @evoskill()
            def run(self, prompt: str) -> str:
                return prompt

        obj = MyAgent()
        obj.run._evoskill_store._path = _store_path(tmp_path)
        obj.run._evoskill_store.add_manual_skill("MyAgent", "be precise")

        result = obj.run("test")
        assert "be precise" in result

    @patch("evoskill.decorator.synthesize_skill")
    def test_is_method_true_resolves_role_from_class(
        self, mock_synth: MagicMock, tmp_path: Path
    ) -> None:
        """is_method=True + no explicit role => role inferred from class name."""

        class Reviewer:
            @evoskill(is_method=True)
            def check(this, prompt: str) -> str:
                raise ValueError("err")

        obj = Reviewer()
        obj.check._evoskill_store._path = _store_path(tmp_path)

        with pytest.raises(ValueError):
            obj.check("test")

        mock_synth.assert_called_once()
        assert mock_synth.call_args[0][0] == "Reviewer"


# ---------------------------------------------------------------------------
# Combined features
# ---------------------------------------------------------------------------


class TestCombinedFeatures:
    @patch("evoskill.decorator.synthesize_skill")
    def test_inject_field_with_extract_input_and_teach_role(
        self, mock_synth: MagicMock, tmp_path: Path
    ) -> None:
        """All three features work together on a single decorator."""

        def my_extract(input_data: AgentInput) -> str:
            return input_data.task

        @evoskill(
            role="reviewer",
            teach_role="writer",
            inject_field="skill_context",
            extract_input=my_extract,
        )
        def agent(input_data: AgentInput) -> str:
            raise ValueError("bad writing")

        agent._evoskill_store._path = _store_path(tmp_path)

        with pytest.raises(ValueError):
            agent(AgentInput(task="review draft"))

        mock_synth.assert_called_once()
        # teach_role should direct the skill to 'writer'
        assert mock_synth.call_args[0][0] == "writer"
        # extract_input should provide the prompt
        assert mock_synth.call_args[0][1] == "review draft"

    @patch("evoskill.decorator.synthesize_skill")
    def test_extract_output_with_teach_role(
        self, mock_synth: MagicMock, tmp_path: Path
    ) -> None:
        """extract_output + teach_role + learn_when work together."""

        def my_extract_out(result: ReviewResult) -> str:
            return result.issues

        @evoskill(
            role="reviewer",
            teach_role="writer",
            extract_output=my_extract_out,
            learn_when=lambda i, o: True,
        )
        def agent(prompt: str) -> ReviewResult:
            return ReviewResult(score=2, issues="unclear thesis")

        agent._evoskill_store._path = _store_path(tmp_path)
        agent("check essay")

        mock_synth.assert_called_once()
        assert mock_synth.call_args[0][0] == "writer"
        assert "unclear thesis" in mock_synth.call_args[0][2]
