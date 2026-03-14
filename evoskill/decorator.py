"""@evoskill(...) decorator implementation."""

from __future__ import annotations

import asyncio
import functools
import inspect
import traceback
from typing import Any, Callable

from .skill import Skill
from .store import SkillStore
from .synthesizer import synthesize_skill

_INJECTION_HEADER = "[EvoSkill] Learned skills for this role:"


def _build_skill_block(skills: list[Skill]) -> str:
    """Format skills for injection into the prompt."""
    if not skills:
        return ""
    lines = [_INJECTION_HEADER]
    for s in skills:
        lines.append(f"- {s.content}")
    return "\n".join(lines) + "\n\n"


def _infer_role(func: Callable) -> str:
    """Best-effort role: use the function name."""
    return func.__name__


def evoskill(
    role: str | None = None,
    learn_when: Callable[[str, Any], bool] | None = None,
    skills: list[str] | None = None,
) -> Callable:
    """Decorator that adds runtime skill learning to any function/method."""

    store = SkillStore()

    # Persist any manual skills supplied via the decorator
    def _ensure_manual_skills(resolved_role: str) -> None:
        if not skills:
            return
        existing = {s.content for s in store.get_skills(resolved_role) if s.source == "manual"}
        for text in skills:
            if text not in existing:
                store.add_manual_skill(resolved_role, text)

    def decorator(func: Callable) -> Callable:
        is_method = _looks_like_method(func)

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            resolved_role, prompt_idx, original_prompt = _resolve_call(
                func, role, is_method, args,
            )
            _ensure_manual_skills(resolved_role)

            augmented = _inject_skills(store, resolved_role, original_prompt)
            args_list = list(args)
            args_list[prompt_idx] = augmented

            try:
                result = await func(*args_list, **kwargs)
            except Exception as exc:
                _learn_from_failure(
                    resolved_role, original_prompt, exc, store,
                )
                raise

            if learn_when is not None and learn_when(original_prompt, result):
                _learn_from_output(
                    resolved_role, original_prompt, result, store,
                )

            return result

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            resolved_role, prompt_idx, original_prompt = _resolve_call(
                func, role, is_method, args,
            )
            _ensure_manual_skills(resolved_role)

            augmented = _inject_skills(store, resolved_role, original_prompt)
            args_list = list(args)
            args_list[prompt_idx] = augmented

            try:
                result = func(*args_list, **kwargs)
            except Exception as exc:
                _learn_from_failure(
                    resolved_role, original_prompt, exc, store,
                )
                raise

            if learn_when is not None and learn_when(original_prompt, result):
                _learn_from_output(
                    resolved_role, original_prompt, result, store,
                )

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


def _resolve_call(
    func: Callable,
    explicit_role: str | None,
    is_method: bool,
    args: tuple,
) -> tuple[str, int, str]:
    """Return (role, prompt_arg_index, original_prompt)."""
    prompt_idx = 1 if is_method else 0

    if explicit_role:
        resolved_role = explicit_role
    elif is_method and args:
        resolved_role = type(args[0]).__name__
    else:
        resolved_role = _infer_role(func)

    original_prompt = args[prompt_idx] if len(args) > prompt_idx else ""
    return resolved_role, prompt_idx, original_prompt


def _inject_skills(store: SkillStore, role: str, prompt: str) -> str:
    """Prepend stored skills to the prompt."""
    all_skills = store.get_skills(role)
    block = _build_skill_block(all_skills)
    return block + prompt


def _learn_from_failure(
    role: str, prompt: str, exc: Exception, store: SkillStore,
) -> None:
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    failure_text = "".join(tb)
    try:
        synthesize_skill(role, prompt, failure_text, store)
    except Exception:
        pass  # don't mask the original exception


def _learn_from_output(
    role: str, prompt: str, output: Any, store: SkillStore,
) -> None:
    try:
        synthesize_skill(role, prompt, str(output), store)
    except Exception:
        pass
