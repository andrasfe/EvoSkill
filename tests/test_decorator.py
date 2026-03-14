"""Tests for the @evoskill decorator."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from evoskill.decorator import _INJECTION_HEADER, evoskill
from evoskill.store import SkillStore

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _store_path(tmp_path: Path) -> Path:
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
            return prompt

        agent._evoskill_store._path = _store_path(tmp_path)
        agent._evoskill_store.add_manual_skill("analyst", "always validate")

        result = agent("do analysis")
        assert _INJECTION_HEADER in result
        assert "always validate" in result
        assert "do analysis" in result

    def test_exception_triggers_learning(self, tmp_path: Path) -> None:
        @evoskill(role="analyst")
        def agent(prompt: str) -> str:
            raise ValueError("bad data")

        agent._evoskill_store._path = _store_path(tmp_path)

        with pytest.raises(ValueError, match="bad data"):
            agent("analyze this")

        store = agent._evoskill_store
        assert store.pending_buffer_count == 1
        buf = store._buffers["analyst"]
        assert buf.items[0]["input_prompt"] == "analyze this"
        assert "bad data" in buf.items[0]["reviewer_feedback"]

    def test_learn_when_callback(self, tmp_path: Path) -> None:
        def should_learn(inp, out):
            return "bad" in out

        @evoskill(role="analyst", learn_when=should_learn)
        def agent(prompt: str) -> str:
            return "bad result"

        agent._evoskill_store._path = _store_path(tmp_path)
        agent("test")
        store = agent._evoskill_store
        assert store.pending_buffer_count == 1
        buf = store._buffers["analyst"]
        assert buf.items[0]["input_prompt"] == "test"

    def test_no_learning_on_success(self, tmp_path: Path) -> None:
        @evoskill(role="analyst")
        def agent(prompt: str) -> str:
            return "good result"

        agent._evoskill_store._path = _store_path(tmp_path)
        agent("test")
        assert agent._evoskill_store.pending_buffer_count == 0

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

    def test_async_exception_triggers_learning(self, tmp_path: Path) -> None:
        @evoskill(role="dev")
        async def agent(prompt: str) -> str:
            raise RuntimeError("oops")

        agent._evoskill_store._path = _store_path(tmp_path)

        with pytest.raises(RuntimeError, match="oops"):
            asyncio.get_event_loop().run_until_complete(agent("fix this"))

        store = agent._evoskill_store
        assert store.pending_buffer_count == 1
        buf = store._buffers["dev"]
        assert "oops" in buf.items[0]["reviewer_feedback"]

    def test_async_learn_when_uses_async_synthesis(self, tmp_path: Path) -> None:
        def should_learn(inp, out):
            return "bad" in out

        @evoskill(role="dev", learn_when=should_learn)
        async def agent(prompt: str) -> str:
            return "bad output"

        agent._evoskill_store._path = _store_path(tmp_path)
        asyncio.get_event_loop().run_until_complete(agent("test"))
        store = agent._evoskill_store
        assert store.pending_buffer_count == 1

    def test_async_with_async_llm(self, tmp_path: Path) -> None:
        async def my_async_llm(messages):
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
        def my_inject(args, kwargs, skills_text):
            new_kw = dict(kwargs)
            new_kw["system_prompt"] = skills_text + kwargs.get("system_prompt", "")
            return args, new_kw

        @evoskill(role="custom", inject_skills=my_inject)
        def agent(task: dict, system_prompt: str = "") -> str:
            return f"sys={system_prompt} task={task}"

        agent._evoskill_store._path = _store_path(tmp_path)
        agent._evoskill_store.add_manual_skill("custom", "be careful")
        result = agent({"type": "analyze"}, system_prompt="base prompt")
        assert "be careful" in result
        assert "base prompt" in result

    def test_structured_input_not_modified_without_inject(self, tmp_path: Path) -> None:
        @evoskill(role="structured")
        def agent(task: dict) -> str:
            return str(task)

        agent._evoskill_store._path = _store_path(tmp_path)
        agent._evoskill_store.add_manual_skill("structured", "some skill")
        result = agent({"key": "value"})
        assert "key" in result


# ---------------------------------------------------------------------------
# BYO LLM
# ---------------------------------------------------------------------------


class TestBYOLLM:
    def test_sync_llm_stored_in_buffer(self, tmp_path: Path) -> None:
        def my_llm(messages):
            return "custom skill"

        @evoskill(role="dev", llm=my_llm)
        def agent(prompt: str) -> str:
            raise ValueError("fail")

        agent._evoskill_store._path = _store_path(tmp_path)
        with pytest.raises(ValueError, match="fail"):
            agent("test")

        store = agent._evoskill_store
        assert store.pending_buffer_count == 1
        assert store._buffers["dev"].llm is my_llm

    def test_async_llm_stored_in_buffer(self, tmp_path: Path) -> None:
        async def my_async_llm(messages):
            return "async custom skill"

        @evoskill(role="dev", llm=my_async_llm)
        async def agent(prompt: str) -> str:
            raise ValueError("fail")

        agent._evoskill_store._path = _store_path(tmp_path)
        with pytest.raises(ValueError, match="fail"):
            asyncio.get_event_loop().run_until_complete(agent("test"))

        store = agent._evoskill_store
        assert store.pending_buffer_count == 1
        assert store._buffers["dev"].llm is my_async_llm


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
        assert "skill 3" in result
        assert "skill 4" in result
        assert "skill 0" not in result


# ---------------------------------------------------------------------------
# inject_field
# ---------------------------------------------------------------------------


class AgentInput(BaseModel):
    task: str
    skill_context: str = ""


class TestInjectField:
    def test_injects_into_pydantic_field(self, tmp_path: Path) -> None:
        @evoskill(role="writer", inject_field="skill_context")
        def agent(input_data: AgentInput) -> str:
            return f"ctx={input_data.skill_context} task={input_data.task}"

        agent._evoskill_store._path = _store_path(tmp_path)
        agent._evoskill_store.add_manual_skill("writer", "use active voice")
        result = agent(AgentInput(task="write essay"))
        assert "use active voice" in result
        assert "write essay" in result

    def test_original_model_not_mutated(self, tmp_path: Path) -> None:
        @evoskill(role="writer", inject_field="skill_context")
        def agent(input_data: AgentInput) -> str:
            return input_data.skill_context

        agent._evoskill_store._path = _store_path(tmp_path)
        agent._evoskill_store.add_manual_skill("writer", "be concise")
        original = AgentInput(task="draft")
        agent(original)
        assert original.skill_context == ""

    def test_no_skills_no_mutation(self, tmp_path: Path) -> None:
        @evoskill(role="empty", inject_field="skill_context")
        def agent(input_data: AgentInput) -> str:
            return input_data.skill_context

        agent._evoskill_store._path = _store_path(tmp_path)
        result = agent(AgentInput(task="hello"))
        assert result == ""

    def test_inject_field_on_method(self, tmp_path: Path) -> None:
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
        with pytest.raises(ValueError, match="mutually exclusive"):
            evoskill(
                role="x",
                inject_skills=lambda a, k, s: (a, k),
                inject_field="skill_context",
            )

    def test_inject_field_async(self, tmp_path: Path) -> None:
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
# extract_input
# ---------------------------------------------------------------------------


class TestExtractInput:
    def test_extract_input_used_for_synthesis_context(self, tmp_path: Path) -> None:
        def my_extract(input_data):
            return input_data.task

        @evoskill(role="dev", extract_input=my_extract, inject_field="skill_context")
        def agent(input_data: AgentInput) -> str:
            raise ValueError("oops")

        agent._evoskill_store._path = _store_path(tmp_path)
        with pytest.raises(ValueError, match="oops"):
            agent(AgentInput(task="my specific task"))

        store = agent._evoskill_store
        assert store.pending_buffer_count == 1
        assert store._buffers["dev"].items[0]["input_prompt"] == "my specific task"

    def test_default_prompt_extraction_for_non_string(self, tmp_path: Path) -> None:
        @evoskill(role="dev")
        def agent(data: dict) -> str:
            raise ValueError("fail")

        agent._evoskill_store._path = _store_path(tmp_path)
        with pytest.raises(ValueError, match="fail"):
            agent({"key": "val"})

        store = agent._evoskill_store
        assert store.pending_buffer_count == 1
        assert store._buffers["dev"].items[0]["input_prompt"] == ""

    def test_extract_input_receives_kwargs(self, tmp_path: Path) -> None:
        def my_extract(data, mode=""):
            return str(data["key"]) + "-" + mode

        @evoskill(role="dev", extract_input=my_extract)
        def agent(data: dict, mode: str = "") -> str:
            raise ValueError("fail")

        agent._evoskill_store._path = _store_path(tmp_path)
        with pytest.raises(ValueError, match="fail"):
            agent({"key": "val"}, mode="fast")

        store = agent._evoskill_store
        assert store.pending_buffer_count == 1
        assert store._buffers["dev"].items[0]["input_prompt"] == "val-fast"


# ---------------------------------------------------------------------------
# extract_output
# ---------------------------------------------------------------------------


class ReviewResult(BaseModel):
    score: int
    issues: str


class TestExtractOutput:
    def test_extract_output_used_in_learn_when(self, tmp_path: Path) -> None:
        def my_extract(result):
            return result.issues

        def should_learn(inp, out):
            return "error" in out.issues

        @evoskill(role="coder", learn_when=should_learn, extract_output=my_extract)
        def agent(prompt: str) -> ReviewResult:
            return ReviewResult(score=3, issues="found error in line 5")

        agent._evoskill_store._path = _store_path(tmp_path)
        agent("check code")
        store = agent._evoskill_store
        assert store.pending_buffer_count == 1
        assert "found error in line 5" in store._buffers["coder"].items[0]["reviewer_feedback"]

    def test_default_str_conversion_without_extract_output(self, tmp_path: Path) -> None:
        @evoskill(role="dev", learn_when=lambda i, o: True)
        def agent(prompt: str) -> str:
            return "plain output"

        agent._evoskill_store._path = _store_path(tmp_path)
        agent("test")
        store = agent._evoskill_store
        assert store.pending_buffer_count == 1
        assert "plain output" in store._buffers["dev"].items[0]["reviewer_feedback"]

    def test_extract_output_async(self, tmp_path: Path) -> None:
        def my_extract(result):
            return result.issues

        def should_learn(_inp, out):
            return "bug" in out.issues

        @evoskill(role="coder", learn_when=should_learn, extract_output=my_extract)
        async def agent(prompt: str) -> ReviewResult:
            return ReviewResult(score=1, issues="bug in parser")

        agent._evoskill_store._path = _store_path(tmp_path)
        asyncio.get_event_loop().run_until_complete(agent("review"))
        store = agent._evoskill_store
        assert store.pending_buffer_count == 1
        assert "bug in parser" in store._buffers["coder"].items[0]["reviewer_feedback"]


# ---------------------------------------------------------------------------
# teach_role
# ---------------------------------------------------------------------------


class TestTeachRole:
    def test_teach_role_on_exception(self, tmp_path: Path) -> None:
        @evoskill(role="reviewer", teach_role="writer")
        def agent(prompt: str) -> str:
            raise ValueError("bad")

        agent._evoskill_store._path = _store_path(tmp_path)
        with pytest.raises(ValueError, match="bad"):
            agent("review this")

        store = agent._evoskill_store
        assert store.pending_buffer_count == 1
        assert "writer" in store._buffers
        assert "reviewer" not in store._buffers

    def test_teach_role_on_learn_when(self, tmp_path: Path) -> None:
        @evoskill(role="reviewer", teach_role="writer", learn_when=lambda i, o: True)
        def agent(prompt: str) -> str:
            return "issues found"

        agent._evoskill_store._path = _store_path(tmp_path)
        agent("review")
        store = agent._evoskill_store
        assert store.pending_buffer_count == 1
        assert "writer" in store._buffers

    def test_teach_role_async(self, tmp_path: Path) -> None:
        @evoskill(role="reviewer", teach_role="writer")
        async def agent(prompt: str) -> str:
            raise RuntimeError("fail")

        agent._evoskill_store._path = _store_path(tmp_path)
        with pytest.raises(RuntimeError):
            asyncio.get_event_loop().run_until_complete(agent("check"))

        store = agent._evoskill_store
        assert store.pending_buffer_count == 1
        assert "writer" in store._buffers

    def test_no_teach_role_uses_own_role(self, tmp_path: Path) -> None:
        @evoskill(role="analyst")
        def agent(prompt: str) -> str:
            raise ValueError("err")

        agent._evoskill_store._path = _store_path(tmp_path)
        with pytest.raises(ValueError, match="err"):
            agent("test")

        store = agent._evoskill_store
        assert store.pending_buffer_count == 1
        assert "analyst" in store._buffers


# ---------------------------------------------------------------------------
# Async wrapper uses async buffer path
# ---------------------------------------------------------------------------


class TestAsyncNonBlocking:
    def test_async_wrapper_buffers_with_async_path(self, tmp_path: Path) -> None:
        async def my_async_llm(messages):
            return "skill from async llm"

        @evoskill(role="dev", llm=my_async_llm)
        async def agent(prompt: str) -> str:
            raise ValueError("fail")

        agent._evoskill_store._path = _store_path(tmp_path)
        with pytest.raises(ValueError, match="fail"):
            asyncio.get_event_loop().run_until_complete(agent("test"))

        store = agent._evoskill_store
        assert store.pending_buffer_count == 1
        assert store._buffers["dev"].llm is my_async_llm

    def test_async_learn_when_buffers(self, tmp_path: Path) -> None:
        async def my_llm(msgs):
            return "skill"

        @evoskill(role="dev", llm=my_llm, learn_when=lambda i, o: True)
        async def agent(prompt: str) -> str:
            return "output"

        agent._evoskill_store._path = _store_path(tmp_path)
        asyncio.get_event_loop().run_until_complete(agent("test"))
        assert agent._evoskill_store.pending_buffer_count == 1


# ---------------------------------------------------------------------------
# is_method
# ---------------------------------------------------------------------------


class TestIsMethod:
    def test_is_method_true_skips_self(self, tmp_path: Path) -> None:
        class Agent:
            @evoskill(role="Agent", is_method=True)
            def run(self, prompt: str) -> str:
                return prompt

        obj = Agent()
        obj.run._evoskill_store._path = _store_path(tmp_path)
        obj.run._evoskill_store.add_manual_skill("Agent", "be helpful")
        result = obj.run("hello")
        assert "be helpful" in result
        assert "hello" in result

    def test_is_method_false_on_function_with_self_param(self, tmp_path: Path) -> None:
        @evoskill(role="agent", is_method=False)
        def process(self_data: str) -> str:
            return self_data

        process._evoskill_store._path = _store_path(tmp_path)
        process._evoskill_store.add_manual_skill("agent", "check types")
        result = process("my data")
        assert "check types" in result
        assert "my data" in result

    def test_is_method_none_uses_heuristic(self, tmp_path: Path) -> None:
        class MyAgent:
            @evoskill()
            def run(self, prompt: str) -> str:
                return prompt

        obj = MyAgent()
        obj.run._evoskill_store._path = _store_path(tmp_path)
        obj.run._evoskill_store.add_manual_skill("MyAgent", "be precise")
        result = obj.run("test")
        assert "be precise" in result

    def test_is_method_true_resolves_role_from_class(self, tmp_path: Path) -> None:
        class Reviewer:
            @evoskill(is_method=True)
            def check(self, prompt: str) -> str:
                raise ValueError("err")

        obj = Reviewer()
        obj.check._evoskill_store._path = _store_path(tmp_path)
        with pytest.raises(ValueError, match="err"):
            obj.check("test")

        store = obj.check._evoskill_store
        assert store.pending_buffer_count == 1
        assert "Reviewer" in store._buffers


# ---------------------------------------------------------------------------
# Combined features
# ---------------------------------------------------------------------------


class TestCombinedFeatures:
    def test_inject_field_with_extract_input_and_teach_role(self, tmp_path: Path) -> None:
        def my_extract(input_data):
            return input_data.task

        @evoskill(
            role="reviewer", teach_role="writer",
            inject_field="skill_context", extract_input=my_extract,
        )
        def agent(input_data: AgentInput) -> str:
            raise ValueError("bad writing")

        agent._evoskill_store._path = _store_path(tmp_path)
        with pytest.raises(ValueError, match="bad writing"):
            agent(AgentInput(task="review draft"))

        store = agent._evoskill_store
        assert store.pending_buffer_count == 1
        assert "writer" in store._buffers
        assert store._buffers["writer"].items[0]["input_prompt"] == "review draft"

    def test_extract_output_with_teach_role(self, tmp_path: Path) -> None:
        def my_extract_out(result):
            return result.issues

        @evoskill(
            role="reviewer", teach_role="writer",
            extract_output=my_extract_out, learn_when=lambda i, o: True,
        )
        def agent(prompt: str) -> ReviewResult:
            return ReviewResult(score=2, issues="unclear thesis")

        agent._evoskill_store._path = _store_path(tmp_path)
        agent("check essay")
        store = agent._evoskill_store
        assert store.pending_buffer_count == 1
        assert "writer" in store._buffers
        assert "unclear thesis" in store._buffers["writer"].items[0]["reviewer_feedback"]


# ---------------------------------------------------------------------------
# batch_size and flush
# ---------------------------------------------------------------------------


class TestBatchSizeAndFlush:
    def test_default_batch_size_is_10(self, tmp_path: Path) -> None:
        @evoskill(role="dev")
        def agent(prompt: str) -> str:
            raise ValueError("fail")

        agent._evoskill_store._path = _store_path(tmp_path)
        with pytest.raises(ValueError, match="fail"):
            agent("test")

        buf = agent._evoskill_store._buffers["dev"]
        assert buf.batch_size == 10

    def test_custom_batch_size(self, tmp_path: Path) -> None:
        @evoskill(role="dev", batch_size=3)
        def agent(prompt: str) -> str:
            raise ValueError("fail")

        agent._evoskill_store._path = _store_path(tmp_path)
        for _ in range(2):
            with pytest.raises(ValueError, match="fail"):
                agent("test")

        assert agent._evoskill_store.pending_buffer_count == 2

    @patch("evoskill.synthesizer.synthesize_skill_batch")
    def test_auto_flush_at_batch_size(self, mock_batch, tmp_path: Path) -> None:
        mock_batch.return_value = []

        @evoskill(role="dev", batch_size=2)
        def agent(prompt: str) -> str:
            raise ValueError("fail")

        agent._evoskill_store._path = _store_path(tmp_path)

        with pytest.raises(ValueError, match="fail"):
            agent("first")
        mock_batch.assert_not_called()
        assert agent._evoskill_store.pending_buffer_count == 1

        with pytest.raises(ValueError, match="fail"):
            agent("second")
        mock_batch.assert_called_once()
        assert agent._evoskill_store.pending_buffer_count == 0

    @patch("evoskill.synthesizer.synthesize_skill_batch")
    def test_sync_flush_drains_buffer(self, mock_batch, tmp_path: Path) -> None:
        mock_batch.return_value = []

        @evoskill(role="dev", batch_size=10)
        def agent(prompt: str) -> str:
            raise ValueError("fail")

        agent._evoskill_store._path = _store_path(tmp_path)
        for _ in range(3):
            with pytest.raises(ValueError, match="fail"):
                agent("test")

        assert agent._evoskill_store.pending_buffer_count == 3
        agent.flush()
        mock_batch.assert_called_once()
        assert agent._evoskill_store.pending_buffer_count == 0

    @patch("evoskill.synthesizer.asynthesize_skill_batch")
    def test_async_flush_drains_buffer(self, mock_batch, tmp_path: Path) -> None:
        async def _return_empty(*a, **kw):
            return []

        mock_batch.side_effect = _return_empty

        @evoskill(role="dev", batch_size=10)
        async def agent(prompt: str) -> str:
            raise ValueError("fail")

        agent._evoskill_store._path = _store_path(tmp_path)
        for _ in range(3):
            with pytest.raises(ValueError, match="fail"):
                asyncio.get_event_loop().run_until_complete(agent("test"))

        assert agent._evoskill_store.pending_buffer_count == 3
        asyncio.get_event_loop().run_until_complete(agent.flush())
        mock_batch.assert_called_once()
        assert agent._evoskill_store.pending_buffer_count == 0

    @patch("evoskill.synthesizer.synthesize_skill_batch")
    def test_flush_on_store_directly(self, mock_batch, tmp_path: Path) -> None:
        mock_batch.return_value = []

        store = SkillStore(storage_path=_store_path(tmp_path))
        store._buffer_item(
            "dev",
            {"input_prompt": "p", "agent_output": "a", "reviewer_feedback": "f"},
            llm=lambda msgs: "1. skill",
            batch_size=10,
        )
        assert store.pending_buffer_count == 1
        store.flush()
        mock_batch.assert_called_once()
        assert store.pending_buffer_count == 0

    @patch("evoskill.synthesizer.synthesize_skill_batch")
    def test_flush_empty_buffer_is_noop(self, mock_batch, tmp_path: Path) -> None:
        store = SkillStore(storage_path=_store_path(tmp_path))
        store.flush()
        mock_batch.assert_not_called()

    @patch("evoskill.synthesizer.synthesize_skill_batch")
    def test_system_prompt_propagates_to_batch(self, mock_batch, tmp_path: Path) -> None:
        mock_batch.return_value = []

        @evoskill(role="dev", system_prompt="Be terse.", batch_size=2)
        def agent(prompt: str) -> str:
            raise ValueError("fail")

        agent._evoskill_store._path = _store_path(tmp_path)
        for _ in range(2):
            with pytest.raises(ValueError, match="fail"):
                agent("test")

        mock_batch.assert_called_once()
        assert mock_batch.call_args.kwargs.get("system_prompt") == "Be terse."

    @patch("evoskill.synthesizer.synthesize_skill_batch")
    def test_user_template_propagates_to_batch(self, mock_batch, tmp_path: Path) -> None:
        mock_batch.return_value = []

        @evoskill(role="dev", user_template="Custom: {role} {items_text} {existing_skills}", batch_size=2)
        def agent(prompt: str) -> str:
            raise ValueError("fail")

        agent._evoskill_store._path = _store_path(tmp_path)
        for _ in range(2):
            with pytest.raises(ValueError, match="fail"):
                agent("test")

        mock_batch.assert_called_once()
        assert mock_batch.call_args.kwargs.get("user_template") == "Custom: {role} {items_text} {existing_skills}"

    def test_wrapper_has_flush_attribute(self, tmp_path: Path) -> None:
        @evoskill(role="dev")
        def sync_agent(prompt: str) -> str:
            return prompt

        @evoskill(role="dev")
        async def async_agent(prompt: str) -> str:
            return prompt

        assert hasattr(sync_agent, "flush")
        assert hasattr(async_agent, "flush")
        assert callable(sync_agent.flush)
        assert callable(async_agent.flush)
