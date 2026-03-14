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


# ---------------------------------------------------------------------------
# Embedding-based deduplication
# ---------------------------------------------------------------------------


def _fake_embed(text: str) -> list[float]:
    """Deterministic fake embedding: hash-based unit vector."""
    import hashlib

    h = hashlib.md5(text.encode()).hexdigest()
    raw = [int(h[i : i + 2], 16) / 255.0 for i in range(0, 32, 2)]
    norm = sum(x * x for x in raw) ** 0.5
    return [x / norm for x in raw]


def _identical_embed(_text: str) -> list[float]:
    """Always returns the same vector — guarantees cosine sim = 1.0."""
    return [1.0, 0.0, 0.0, 0.0]


async def _async_fake_embed(text: str) -> list[float]:
    """Async version of _fake_embed."""
    return _fake_embed(text)


async def _async_identical_embed(_text: str) -> list[float]:
    return [1.0, 0.0, 0.0, 0.0]


class TestCosineHelpers:
    def test_cosine_identical_vectors(self) -> None:
        from evoskill.synthesizer import _cosine_similarity

        v = [1.0, 2.0, 3.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_cosine_orthogonal_vectors(self) -> None:
        from evoskill.synthesizer import _cosine_similarity

        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_cosine_zero_vector(self) -> None:
        from evoskill.synthesizer import _cosine_similarity

        assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


class TestFindDuplicate:
    def test_returns_match_above_threshold(self, store: SkillStore) -> None:
        from evoskill.synthesizer import _find_duplicate
        from evoskill.skill import Skill

        existing = [Skill(role="dev", content="test", source="learned")]
        # _identical_embed makes everything sim=1.0
        result = _find_duplicate("feedback", existing, _identical_embed, 0.85)
        assert result is not None
        assert result.content == "test"

    def test_returns_none_below_threshold(self, store: SkillStore) -> None:
        from evoskill.synthesizer import _find_duplicate
        from evoskill.skill import Skill

        existing = [Skill(role="dev", content="test", source="learned")]
        result = _find_duplicate("feedback", existing, _fake_embed, 0.9999)
        assert result is None

    def test_caches_embeddings_on_skills(self) -> None:
        from evoskill.synthesizer import _find_duplicate
        from evoskill.skill import Skill

        skill = Skill(role="dev", content="test skill", source="learned")
        assert skill.embedding is None
        _find_duplicate("query", [skill], _fake_embed, 0.99)
        assert skill.embedding is not None
        assert isinstance(skill.embedding, list)

    def test_reuses_cached_embedding(self) -> None:
        from evoskill.synthesizer import _find_duplicate
        from evoskill.skill import Skill

        cached = [0.5, 0.5, 0.5, 0.5]
        skill = Skill(
            role="dev", content="test", source="learned", embedding=cached,
        )
        call_count = 0

        def counting_embed(text: str) -> list[float]:
            nonlocal call_count
            call_count += 1
            return _fake_embed(text)

        _find_duplicate("query", [skill], counting_embed, 0.99)
        # Should only embed the query, NOT re-embed the skill
        assert call_count == 1


class TestDeduplicationSynthesizeSkill:
    def test_returns_none_when_duplicate_found(self, store: SkillStore) -> None:
        # Pre-populate with a skill
        from evoskill.skill import Skill

        store.add_skill(Skill(role="dev", content="existing skill", source="learned"))

        def should_not_call(messages: list[dict[str, str]]) -> str:
            raise AssertionError("LLM should not be called when dedup matches")

        result = synthesize_skill(
            role="dev",
            input_prompt="input",
            failure="feedback",
            store=store,
            llm=should_not_call,
            deduplicate=True,
            similarity_threshold=0.01,  # Very low — everything matches
            embed=_identical_embed,
        )
        assert result is None

    def test_synthesizes_when_no_duplicate(self, store: SkillStore) -> None:
        from evoskill.skill import Skill

        store.add_skill(Skill(role="dev", content="existing skill", source="learned"))

        def my_llm(messages: list[dict[str, str]]) -> str:
            return "Brand new skill"

        result = synthesize_skill(
            role="dev",
            input_prompt="input",
            failure="feedback",
            store=store,
            llm=my_llm,
            deduplicate=True,
            similarity_threshold=0.9999,  # Very high — nothing matches
            embed=_fake_embed,
        )
        assert result is not None
        assert result.content == "Brand new skill"
        assert result.embedding is not None  # New skill gets embedded

    def test_deduplicate_false_skips_check(self, store: SkillStore) -> None:
        from evoskill.skill import Skill

        store.add_skill(Skill(role="dev", content="existing", source="learned"))

        def my_llm(messages: list[dict[str, str]]) -> str:
            return "New skill"

        # Even with identical embeddings, deduplicate=False should proceed
        result = synthesize_skill(
            role="dev",
            input_prompt="input",
            failure="feedback",
            store=store,
            llm=my_llm,
            deduplicate=False,
            embed=_identical_embed,
        )
        assert result is not None
        assert result.content == "New skill"
        assert result.embedding is None  # No embedding when dedup disabled

    def test_no_existing_skills_proceeds(self, store: SkillStore) -> None:
        def my_llm(messages: list[dict[str, str]]) -> str:
            return "First skill"

        result = synthesize_skill(
            role="dev",
            input_prompt="input",
            failure="feedback",
            store=store,
            llm=my_llm,
            deduplicate=True,
            embed=_fake_embed,
        )
        assert result is not None
        assert result.content == "First skill"


class TestDeduplicationWithContext:
    def test_with_context_returns_none_on_match(self, store: SkillStore) -> None:
        from evoskill.skill import Skill

        store.add_skill(Skill(role="dev", content="existing", source="learned"))

        def should_not_call(messages: list[dict[str, str]]) -> str:
            raise AssertionError("LLM should not be called")

        result = synthesize_skill_with_context(
            role="dev",
            input_prompt="inp",
            agent_output="out",
            feedback="fb",
            store=store,
            llm=should_not_call,
            deduplicate=True,
            similarity_threshold=0.01,
            embed=_identical_embed,
        )
        assert result is None


class TestDeduplicationAsync:
    @pytest.mark.asyncio
    async def test_async_returns_none_on_match(self, store: SkillStore) -> None:
        from evoskill.skill import Skill

        store.add_skill(Skill(role="dev", content="existing", source="learned"))

        async def should_not_call(messages: list[dict[str, str]]) -> str:
            raise AssertionError("LLM should not be called")

        result = await asynthesize_skill(
            role="dev",
            input_prompt="inp",
            failure="fb",
            store=store,
            llm=should_not_call,
            deduplicate=True,
            similarity_threshold=0.01,
            embed=_async_identical_embed,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_async_synthesizes_when_no_match(self, store: SkillStore) -> None:
        from evoskill.skill import Skill

        store.add_skill(Skill(role="dev", content="existing", source="learned"))

        async def my_async_llm(messages: list[dict[str, str]]) -> str:
            return "Async new skill"

        result = await asynthesize_skill(
            role="dev",
            input_prompt="inp",
            failure="fb",
            store=store,
            llm=my_async_llm,
            deduplicate=True,
            similarity_threshold=0.9999,
            embed=_async_fake_embed,
        )
        assert result is not None
        assert result.content == "Async new skill"
        assert result.embedding is not None

    @pytest.mark.asyncio
    async def test_async_with_context_dedup(self, store: SkillStore) -> None:
        from evoskill.skill import Skill

        store.add_skill(Skill(role="dev", content="existing", source="learned"))

        async def should_not_call(messages: list[dict[str, str]]) -> str:
            raise AssertionError("LLM should not be called")

        result = await asynthesize_skill_with_context(
            role="dev",
            input_prompt="inp",
            agent_output="out",
            feedback="fb",
            store=store,
            llm=should_not_call,
            deduplicate=True,
            similarity_threshold=0.01,
            embed=_async_identical_embed,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_async_sync_embed_fallback(self, store: SkillStore) -> None:
        """Async functions should work with sync embed callables too."""
        from evoskill.skill import Skill

        store.add_skill(Skill(role="dev", content="existing", source="learned"))

        async def should_not_call(messages: list[dict[str, str]]) -> str:
            raise AssertionError("LLM should not be called")

        result = await asynthesize_skill(
            role="dev",
            input_prompt="inp",
            failure="fb",
            store=store,
            llm=should_not_call,
            deduplicate=True,
            similarity_threshold=0.01,
            embed=_identical_embed,  # sync embed in async context
        )
        assert result is None


class TestDeduplicationBatch:
    def test_batch_dedup_drops_matching_skills(self, store: SkillStore) -> None:
        from evoskill.skill import Skill

        store.add_skill(Skill(role="dev", content="existing", source="learned"))

        def fake_llm(messages: list[dict[str, str]]) -> str:
            return "1. Existing duplicate.\n2. Brand new insight."

        items = [
            {"input_prompt": "a", "agent_output": "b", "reviewer_feedback": "c"},
            {"input_prompt": "d", "agent_output": "e", "reviewer_feedback": "f"},
        ]
        skills = synthesize_skill_batch(
            role="dev", items=items, store=store, llm=fake_llm,
            deduplicate=True,
            similarity_threshold=0.01,  # Low threshold — everything matches existing
            embed=_identical_embed,
        )
        # Both synthesized skills match existing (identical embed), so both dropped
        assert len(skills) == 0

    def test_batch_dedup_keeps_unique_skills(self, store: SkillStore) -> None:
        def fake_llm(messages: list[dict[str, str]]) -> str:
            return "1. Unique skill one.\n2. Unique skill two."

        items = [
            {"input_prompt": "a", "agent_output": "b", "reviewer_feedback": "c"},
            {"input_prompt": "d", "agent_output": "e", "reviewer_feedback": "f"},
        ]
        # No existing skills, so nothing to dedup against initially
        skills = synthesize_skill_batch(
            role="dev", items=items, store=store, llm=fake_llm,
            deduplicate=True,
            similarity_threshold=0.9999,
            embed=_fake_embed,
        )
        assert len(skills) == 2

    def test_batch_dedup_disabled(self, store: SkillStore) -> None:
        from evoskill.skill import Skill

        store.add_skill(Skill(role="dev", content="existing", source="learned"))

        def fake_llm(messages: list[dict[str, str]]) -> str:
            return "1. Skill A.\n2. Skill B."

        items = [
            {"input_prompt": "a", "agent_output": "b", "reviewer_feedback": "c"},
            {"input_prompt": "d", "agent_output": "e", "reviewer_feedback": "f"},
        ]
        skills = synthesize_skill_batch(
            role="dev", items=items, store=store, llm=fake_llm,
            deduplicate=False,
        )
        assert len(skills) == 2

    @pytest.mark.asyncio
    async def test_async_batch_dedup(self, store: SkillStore) -> None:
        from evoskill.skill import Skill

        store.add_skill(Skill(role="dev", content="existing", source="learned"))

        async def fake_llm(messages: list[dict[str, str]]) -> str:
            return "1. Duplicate.\n2. Another duplicate."

        items = [
            {"input_prompt": "a", "agent_output": "b", "reviewer_feedback": "c"},
            {"input_prompt": "d", "agent_output": "e", "reviewer_feedback": "f"},
        ]
        skills = await asynthesize_skill_batch(
            role="dev", items=items, store=store, llm=fake_llm,
            deduplicate=True,
            similarity_threshold=0.01,
            embed=_async_identical_embed,
        )
        assert len(skills) == 0


class TestEmbeddingCaching:
    def test_embedding_persisted_on_new_skill(self, store: SkillStore) -> None:
        def my_llm(messages: list[dict[str, str]]) -> str:
            return "New synthesized skill"

        skill = synthesize_skill(
            role="dev",
            input_prompt="input",
            failure="feedback",
            store=store,
            llm=my_llm,
            deduplicate=True,
            embed=_fake_embed,
        )
        assert skill is not None
        assert skill.embedding is not None
        # Verify persisted
        stored = store.get_skills("dev")
        assert stored[0].embedding is not None

    def test_embedding_not_set_when_dedup_disabled(self, store: SkillStore) -> None:
        def my_llm(messages: list[dict[str, str]]) -> str:
            return "Skill without embedding"

        skill = synthesize_skill(
            role="dev",
            input_prompt="input",
            failure="feedback",
            store=store,
            llm=my_llm,
            deduplicate=False,
        )
        assert skill is not None
        assert skill.embedding is None

    def test_skill_embedding_roundtrip_json(self, store: SkillStore) -> None:
        from evoskill.skill import Skill

        embedding = [0.1, 0.2, 0.3]
        skill = Skill(
            role="dev", content="test", source="learned", embedding=embedding,
        )
        d = skill.to_dict()
        assert d["embedding"] == embedding
        restored = Skill.from_dict(d)
        assert restored.embedding == embedding

    def test_skill_no_embedding_in_dict(self) -> None:
        from evoskill.skill import Skill

        skill = Skill(role="dev", content="test", source="learned")
        d = skill.to_dict()
        assert "embedding" not in d


# ---------------------------------------------------------------------------
# Regression: _update_embeddings preserves unfiltered skills
# ---------------------------------------------------------------------------


class TestUpdateEmbeddingsNoDataLoss:
    """Verify that dedup embedding persistence does NOT overwrite skills
    with different tags, disabled skills, or skills from concurrent writes."""

    def test_dedup_preserves_skills_with_different_tags(self, store: SkillStore) -> None:
        """synthesize_skill_with_context with tags=["finance"] must not
        delete existing skills tagged ["marketing"]."""
        from evoskill.skill import Skill

        store.add_skill(Skill(role="writer", content="marketing tip", source="learned", tags=["marketing"]))
        store.add_skill(Skill(role="writer", content="finance tip", source="learned", tags=["finance"]))

        result = synthesize_skill_with_context(
            role="writer",
            input_prompt="write report",
            agent_output="draft",
            feedback="finance tip",  # will match via _identical_embed
            store=store,
            llm=lambda msgs: "should not be called",
            tags=["finance"],
            deduplicate=True,
            similarity_threshold=0.01,
            embed=_identical_embed,
        )
        assert result is None  # dedup matched

        # The marketing skill must still exist
        all_skills = store.get_skills("writer", tags=["marketing"])
        assert len(all_skills) == 1
        assert all_skills[0].content == "marketing tip"

        # Finance skill also still there
        finance_skills = store.get_skills("writer", tags=["finance"])
        assert len(finance_skills) == 1

    def test_dedup_preserves_disabled_skills(self, store: SkillStore) -> None:
        """Disabled skills must not be deleted when dedup persists embeddings."""
        from evoskill.skill import Skill

        store.add_skill(Skill(role="dev", content="active skill", source="learned"))
        store.add_skill(Skill(role="dev", content="disabled skill", source="learned"))
        store.disable_skill("dev", "disabled skill")

        result = synthesize_skill_with_context(
            role="dev",
            input_prompt="code",
            agent_output="output",
            feedback="active skill",
            store=store,
            llm=lambda msgs: "should not be called",
            deduplicate=True,
            similarity_threshold=0.01,
            embed=_identical_embed,
        )
        assert result is None

        # Disabled skill must still be in the store
        all_skills = store.get_skills("dev", enabled_only=False)
        disabled = [s for s in all_skills if s.content == "disabled skill"]
        assert len(disabled) == 1
        assert not disabled[0].enabled

    def test_async_dedup_preserves_other_tags(self, store: SkillStore) -> None:
        """Async variant must also not lose skills with different tags."""
        import asyncio
        from evoskill.skill import Skill

        store.add_skill(Skill(role="writer", content="seo tip", source="learned", tags=["seo"]))
        store.add_skill(Skill(role="writer", content="tone tip", source="learned", tags=["tone"]))

        async def run() -> Skill | None:
            return await asynthesize_skill_with_context(
                role="writer",
                input_prompt="write",
                agent_output="draft",
                feedback="tone tip",
                store=store,
                llm=lambda msgs: "should not be called",
                tags=["tone"],
                deduplicate=True,
                similarity_threshold=0.01,
                embed=_identical_embed,
            )

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result is None

        seo_skills = store.get_skills("writer", tags=["seo"])
        assert len(seo_skills) == 1
        assert seo_skills[0].content == "seo tip"

    def test_batch_dedup_preserves_other_tags(self, store: SkillStore) -> None:
        """synthesize_skill_batch must not delete skills with different tags
        when persisting embeddings."""
        from evoskill.skill import Skill

        store.add_skill(Skill(role="dev", content="python tip", source="learned", tags=["python"]))
        store.add_skill(Skill(role="dev", content="rust tip", source="learned", tags=["rust"]))

        def fake_llm(msgs: list[dict[str, str]]) -> str:
            return "1. New python skill."

        skills = synthesize_skill_batch(
            role="dev",
            items=[{"input_prompt": "a", "agent_output": "b", "reviewer_feedback": "c"}],
            store=store,
            llm=fake_llm,
            tags=["python"],
            deduplicate=True,
            embed=_fake_embed,
        )
        assert len(skills) == 1

        # Rust skills must still be there
        rust_skills = store.get_skills("dev", tags=["rust"])
        assert len(rust_skills) == 1
        assert rust_skills[0].content == "rust tip"

    def test_update_embeddings_merges_correctly(self, store: SkillStore) -> None:
        """_update_embeddings should only set embeddings on matching skills."""
        from evoskill.skill import Skill

        store.add_skill(Skill(role="dev", content="skill A", source="learned"))
        store.add_skill(Skill(role="dev", content="skill B", source="learned"))

        # Simulate: we computed an embedding for skill A only
        skills_with_emb = [
            Skill(role="dev", content="skill A", source="learned", embedding=[1.0, 0.0]),
        ]
        store._update_embeddings("dev", skills_with_emb)

        all_skills = store.get_skills("dev")
        a = [s for s in all_skills if s.content == "skill A"][0]
        b = [s for s in all_skills if s.content == "skill B"][0]
        assert a.embedding == [1.0, 0.0]
        assert b.embedding is None
