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


def _infer_role(func: Callable) -> str:
    """Best-effort role: use the function name."""
    return func.__name__


def evoskill(
    role: str | None = None,
    learn_when: Callable[..., bool] | None = None,
    skills: list[str] | None = None,
    inject_skills: Callable[[tuple, dict, str], tuple[tuple, dict]] | None = None,
    llm: LLMCallable | AsyncLLMCallable | None = None,
    tags: list[str] | None = None,
    max_skills: int | None = None,
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
    llm:
        LLM callable ``(messages) -> str`` used for skill synthesis.
        Falls back to the built-in OpenAI adapter when ``None``.
    tags:
        Default tags attached to skills learned by this decorator.
    max_skills:
        If set, only the most-recent *max_skills* skills are injected.
    """

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
        is_method = _looks_like_method(func)

        def _prepare(args: tuple, kwargs: dict) -> tuple[str, tuple, dict, str]:
            """Resolve role, ensure manual skills, inject skills."""
            resolved_role = _resolve_role(func, role, is_method, args)
            _ensure_manual_skills(resolved_role)

            all_skills = store.get_skills(resolved_role, tags=tags)
            if max_skills is not None:
                all_skills = all_skills[-max_skills:]
            skills_text = _build_skill_block(all_skills)

            # Determine original prompt for synthesis context
            prompt_idx = 1 if is_method else 0
            original_prompt = ""
            if len(args) > prompt_idx and isinstance(args[prompt_idx], str):
                original_prompt = args[prompt_idx]

            if inject_skills is not None:
                new_args, new_kwargs = inject_skills(args, kwargs, skills_text)
            else:
                new_args, new_kwargs = _default_inject(
                    args, kwargs, skills_text, is_method,
                )
            return resolved_role, new_args, new_kwargs, original_prompt

        def _after_success(
            resolved_role: str, original_prompt: str, result: Any,
        ) -> None:
            if learn_when is not None and learn_when(original_prompt, result):
                _learn_from_output(
                    resolved_role, original_prompt, result, store, llm, tags,
                )

        def _after_failure(
            resolved_role: str, original_prompt: str, exc: Exception,
        ) -> None:
            _learn_from_exception(
                resolved_role, original_prompt, exc, store, llm, tags,
            )

        async def _aafter_success(
            resolved_role: str, original_prompt: str, result: Any,
        ) -> None:
            if learn_when is not None and learn_when(original_prompt, result):
                await _alearn_from_output(
                    resolved_role, original_prompt, result, store, llm, tags,
                )

        async def _aafter_failure(
            resolved_role: str, original_prompt: str, exc: Exception,
        ) -> None:
            await _alearn_from_exception(
                resolved_role, original_prompt, exc, store, llm, tags,
            )

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
) -> None:
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    failure_text = "".join(tb)
    try:
        synthesize_skill(role, prompt, failure_text, store, llm=llm, tags=skill_tags)
    except Exception:
        pass  # don't mask the original exception


def _learn_from_output(
    role: str,
    prompt: str,
    output: Any,
    store: SkillStore,
    llm: LLMCallable | None,
    skill_tags: list[str] | None,
) -> None:
    try:
        synthesize_skill(role, prompt, str(output), store, llm=llm, tags=skill_tags)
    except Exception:
        pass


async def _alearn_from_exception(
    role: str,
    prompt: str,
    exc: Exception,
    store: SkillStore,
    llm: AsyncLLMCallable | LLMCallable | None,
    skill_tags: list[str] | None,
) -> None:
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    failure_text = "".join(tb)
    try:
        await asynthesize_skill(role, prompt, failure_text, store, llm=llm, tags=skill_tags)
    except Exception:
        pass


async def _alearn_from_output(
    role: str,
    prompt: str,
    output: Any,
    store: SkillStore,
    llm: AsyncLLMCallable | LLMCallable | None,
    skill_tags: list[str] | None,
) -> None:
    try:
        await asynthesize_skill(role, prompt, str(output), store, llm=llm, tags=skill_tags)
    except Exception:
        pass
