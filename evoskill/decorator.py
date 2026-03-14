"""@evoskill(...) decorator implementation."""

from __future__ import annotations

import asyncio
import functools
import inspect
import traceback
from typing import Any, Callable

from .skill import Skill
from .store import SkillStore
from .synthesizer import AsyncLLMCallable, LLMCallable, asynthesize_skill, synthesize_skill

_INJECTION_HEADER = "[EvoSkill] Learned skills for this role:"


def _build_skill_block(skills: list[Skill]) -> str:
    """Format skills for injection into the prompt."""
    if not skills:
        return ""
    lines = [_INJECTION_HEADER]
    for s in skills:
        lines.append(f"- {s.content}")
    return "\n".join(lines) + "\n\n"


def _default_inject(
    args: tuple,
    kwargs: dict,
    skills_text: str,
    is_method: bool,
) -> tuple[tuple, dict]:
    """Default injection: prepend *skills_text* to the first str positional arg."""
    if not skills_text:
        return args, kwargs
    prompt_idx = 1 if is_method else 0
    if len(args) <= prompt_idx:
        return args, kwargs
    original = args[prompt_idx]
    if not isinstance(original, str):
        return args, kwargs
    args_list = list(args)
    args_list[prompt_idx] = skills_text + original
    return tuple(args_list), kwargs


def _inject_into_field(
    args: tuple,
    kwargs: dict,
    skills_text: str,
    is_method: bool,
    field_name: str,
) -> tuple[tuple, dict]:
    """Inject *skills_text* into a named field on a Pydantic model.

    Finds the first positional arg that is a Pydantic ``BaseModel`` (skipping
    ``self`` when *is_method*), then creates a shallow copy with
    ``model_copy(update={field_name: skills_text})``.
    """
    if not skills_text:
        return args, kwargs
    start = 1 if is_method else 0
    args_list = list(args)
    for i in range(start, len(args_list)):
        obj = args_list[i]
        if _is_pydantic_model(obj):
            args_list[i] = obj.model_copy(update={field_name: skills_text})
            return tuple(args_list), kwargs
    return args, kwargs


def _is_pydantic_model(obj: Any) -> bool:
    """Return ``True`` if *obj* is a Pydantic BaseModel instance."""
    try:
        from pydantic import BaseModel
        return isinstance(obj, BaseModel)
    except ImportError:
        return False


def _infer_role(func: Callable) -> str:
    """Best-effort role: use the function name."""
    return func.__name__


def evoskill(
    role: str | None = None,
    learn_when: Callable[..., bool] | None = None,
    skills: list[str] | None = None,
    inject_skills: Callable[[tuple, dict, str], tuple[tuple, dict]] | None = None,
    inject_field: str | None = None,
    extract_input: Callable[..., str] | None = None,
    extract_output: Callable[..., str] | None = None,
    llm: LLMCallable | AsyncLLMCallable | None = None,
    tags: list[str] | None = None,
    max_skills: int | None = None,
    system_prompt: str | None = None,
    user_template: str | None = None,
    teach_role: str | None = None,
    is_method: bool | None = None,
    batch_size: int = 10,
) -> Callable:
    """Decorator that adds runtime skill learning to any function/method.

    Parameters
    ----------
    role:
        Agent role identifier.  Defaults to function/class name.
    learn_when:
        Callback ``(input, output) -> bool``.  When it returns ``True``
        skill synthesis is triggered even without an exception.
    skills:
        Manual skill strings to always include for this role.
    inject_skills:
        Custom callback ``(args, kwargs, skills_text) -> (args, kwargs)``
        that controls **where** skills get injected into the function's
        arguments.  When ``None`` the default behaviour prepends skills to
        the first positional ``str`` argument.
    inject_field:
        Name of a field on a Pydantic ``BaseModel`` argument.  When set,
        the decorator uses ``model_copy(update={inject_field: skills_text})``
        to inject skills into the named field.  Mutually exclusive with
        *inject_skills*.
    extract_input:
        Callback ``(*args, **kwargs) -> str`` that extracts the prompt
        text from structured input for synthesis context.  When ``None``,
        the first positional ``str`` argument is used.
    extract_output:
        Callback ``(output) -> str`` that extracts useful content from
        a structured output object.  When ``None``, ``str(output)`` is used.
    llm:
        LLM callable ``(messages) -> str`` used for skill synthesis.
        Falls back to the built-in OpenAI adapter when ``None``.
    tags:
        Default tags attached to skills learned by this decorator.
    max_skills:
        If set, only the most-recent *max_skills* skills are injected.
    system_prompt:
        Override the default system prompt used for skill synthesis.
    user_template:
        Override the default user template used for skill synthesis.
    teach_role:
        When set, skills learned from this agent's failures/feedback are
        stored under *teach_role* instead of the agent's own role.  This
        allows one agent's output to automatically teach another agent.
    is_method:
        Explicitly declare whether the decorated function is a method.
        When ``None`` (default), the decorator uses a heuristic (first
        param named ``self`` or ``cls``).  Set to ``True`` or ``False``
        to override.
    batch_size:
        Number of learning triggers to buffer before flushing via
        :func:`synthesize_skill_batch`.  Defaults to ``10``.  Call
        ``wrapper.flush()`` at end-of-run to drain remaining items.
    """

    if inject_skills is not None and inject_field is not None:
        raise ValueError(
            "inject_skills and inject_field are mutually exclusive — "
            "provide one or the other, not both."
        )

    store = SkillStore()

    def _ensure_manual_skills(resolved_role: str) -> None:
        if not skills:
            return
        existing = {
            s.content
            for s in store.get_skills(resolved_role, enabled_only=False)
            if s.source == "manual"
        }
        for text in skills:
            if text not in existing:
                store.add_manual_skill(resolved_role, text, tags=tags or [])

    def decorator(func: Callable) -> Callable:
        resolved_is_method = is_method if is_method is not None else _looks_like_method(func)

        def _prepare(args: tuple, kwargs: dict) -> tuple[str, tuple, dict, str]:
            """Resolve role, ensure manual skills, inject skills."""
            resolved_role = _resolve_role(func, role, resolved_is_method, args)
            _ensure_manual_skills(resolved_role)

            skills_text = store.get_skills_text(
                resolved_role, tags=tags, max_skills=max_skills,
            )

            # Determine original prompt for synthesis context
            if extract_input is not None:
                original_prompt = extract_input(*args, **kwargs)
            else:
                prompt_idx = 1 if resolved_is_method else 0
                original_prompt = ""
                if len(args) > prompt_idx and isinstance(args[prompt_idx], str):
                    original_prompt = args[prompt_idx]

            if inject_skills is not None:
                new_args, new_kwargs = inject_skills(args, kwargs, skills_text)
            elif inject_field is not None:
                new_args, new_kwargs = _inject_into_field(
                    args, kwargs, skills_text, resolved_is_method, inject_field,
                )
            else:
                new_args, new_kwargs = _default_inject(
                    args, kwargs, skills_text, resolved_is_method,
                )
            return resolved_role, new_args, new_kwargs, original_prompt

        def _learning_role(resolved_role: str) -> str:
            """Return the role that should receive the learned skill."""
            return teach_role if teach_role is not None else resolved_role

        def _extract_output_text(result: Any) -> str:
            """Convert *result* to a string for synthesis context."""
            if extract_output is not None:
                return extract_output(result)
            return str(result)

        def _buffer_kwargs() -> dict:
            """Common kwargs for ``_buffer_item`` / ``_abuffer_item``."""
            return dict(
                llm=llm, tags=tags, system_prompt=system_prompt,
                batch_size=batch_size,
            )

        def _after_success(
            resolved_role: str, original_prompt: str, result: Any,
        ) -> None:
            if learn_when is not None and learn_when(original_prompt, result):
                output_text = _extract_output_text(result)
                try:
                    store._buffer_item(
                        _learning_role(resolved_role),
                        {"input_prompt": original_prompt,
                         "agent_output": output_text,
                         "reviewer_feedback": output_text},
                        **_buffer_kwargs(),
                    )
                except Exception:
                    pass

        def _after_failure(
            resolved_role: str, original_prompt: str, exc: Exception,
        ) -> None:
            failure_text = "\n".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__),
            )
            try:
                store._buffer_item(
                    _learning_role(resolved_role),
                    {"input_prompt": original_prompt,
                     "agent_output": "(not captured)",
                     "reviewer_feedback": failure_text},
                    **_buffer_kwargs(),
                )
            except Exception:
                pass

        async def _aafter_success(
            resolved_role: str, original_prompt: str, result: Any,
        ) -> None:
            if learn_when is not None and learn_when(original_prompt, result):
                output_text = _extract_output_text(result)
                try:
                    await store._abuffer_item(
                        _learning_role(resolved_role),
                        {"input_prompt": original_prompt,
                         "agent_output": output_text,
                         "reviewer_feedback": output_text},
                        **_buffer_kwargs(),
                    )
                except Exception:
                    pass

        async def _aafter_failure(
            resolved_role: str, original_prompt: str, exc: Exception,
        ) -> None:
            failure_text = "\n".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__),
            )
            try:
                await store._abuffer_item(
                    _learning_role(resolved_role),
                    {"input_prompt": original_prompt,
                     "agent_output": "(not captured)",
                     "reviewer_feedback": failure_text},
                    **_buffer_kwargs(),
                )
            except Exception:
                pass

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            resolved_role, new_args, new_kwargs, original_prompt = _prepare(
                args, kwargs,
            )
            try:
                result = await func(*new_args, **new_kwargs)
            except Exception as exc:
                await _aafter_failure(resolved_role, original_prompt, exc)
                raise
            await _aafter_success(resolved_role, original_prompt, result)
            return result

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            resolved_role, new_args, new_kwargs, original_prompt = _prepare(
                args, kwargs,
            )
            try:
                result = func(*new_args, **new_kwargs)
            except Exception as exc:
                _after_failure(resolved_role, original_prompt, exc)
                raise
            _after_success(resolved_role, original_prompt, result)
            return result

        wrapper = async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        wrapper._evoskill_store = store  # type: ignore[attr-defined]
        if asyncio.iscoroutinefunction(func):
            wrapper.flush = store.aflush  # type: ignore[attr-defined]
        else:
            wrapper.flush = store.flush  # type: ignore[attr-defined]
        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _looks_like_method(func: Callable) -> bool:
    """Heuristic: first param named 'self' or 'cls'."""
    params = list(inspect.signature(func).parameters.keys())
    return bool(params) and params[0] in ("self", "cls")


def _resolve_role(
    func: Callable,
    explicit_role: str | None,
    is_method: bool,
    args: tuple,
) -> str:
    """Return the agent role for this call."""
    if explicit_role:
        return explicit_role
    if is_method and args:
        return type(args[0]).__name__
    return _infer_role(func)


def _learn_from_exception(
    role: str,
    prompt: str,
    exc: Exception,
    store: SkillStore,
    llm: LLMCallable | None,
    skill_tags: list[str] | None,
    system_prompt: str | None = None,
    user_template: str | None = None,
) -> None:
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    failure_text = "".join(tb)
    try:
        synthesize_skill(
            role, prompt, failure_text, store,
            llm=llm, tags=skill_tags,
            system_prompt=system_prompt, user_template=user_template,
        )
    except Exception:
        pass  # don't mask the original exception


def _learn_from_output(
    role: str,
    prompt: str,
    output_text: str,
    store: SkillStore,
    llm: LLMCallable | None,
    skill_tags: list[str] | None,
    system_prompt: str | None = None,
    user_template: str | None = None,
) -> None:
    try:
        synthesize_skill(
            role, prompt, output_text, store,
            llm=llm, tags=skill_tags,
            system_prompt=system_prompt, user_template=user_template,
        )
    except Exception:
        pass


async def _alearn_from_exception(
    role: str,
    prompt: str,
    exc: Exception,
    store: SkillStore,
    llm: AsyncLLMCallable | LLMCallable | None,
    skill_tags: list[str] | None,
    system_prompt: str | None = None,
    user_template: str | None = None,
) -> None:
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    failure_text = "".join(tb)
    try:
        await asynthesize_skill(
            role, prompt, failure_text, store,
            llm=llm, tags=skill_tags,
            system_prompt=system_prompt, user_template=user_template,
        )
    except Exception:
        pass


async def _alearn_from_output(
    role: str,
    prompt: str,
    output_text: str,
    store: SkillStore,
    llm: AsyncLLMCallable | LLMCallable | None,
    skill_tags: list[str] | None,
    system_prompt: str | None = None,
    user_template: str | None = None,
) -> None:
    try:
        await asynthesize_skill(
            role, prompt, output_text, store,
            llm=llm, tags=skill_tags,
            system_prompt=system_prompt, user_template=user_template,
        )
    except Exception:
        pass
