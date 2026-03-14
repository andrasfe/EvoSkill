"""OpenAI-based skill synthesis from failures."""

from __future__ import annotations

from openai import OpenAI

from .config import get_api_key, get_model
from .skill import Skill
from .store import SkillStore

_SYSTEM_PROMPT = """\
You are a concise skill extractor. Given an agent role, the input it received, \
the failure it produced, and its existing skills, produce ONE new skill \
(1-3 sentences) that would help the agent avoid this kind of failure in the future. \
Output ONLY the skill text — no preamble, no bullet points, no quotes."""

_USER_TEMPLATE = """\
Role: {role}

Input prompt:
{input_prompt}

Failure:
{failure}

Existing skills:
{existing_skills}

Produce a brief, actionable skill (1-3 sentences)."""


def synthesize_skill(
    role: str,
    input_prompt: str,
    failure: str,
    store: SkillStore,
    *,
    model: str | None = None,
) -> Skill:
    """Call OpenAI to synthesize a new skill and persist it."""
    existing = store.get_skills(role)
    existing_text = (
        "\n".join(f"- {s.content}" for s in existing) if existing else "(none)"
    )

    client = OpenAI(api_key=get_api_key())
    response = client.chat.completions.create(
        model=model or get_model(),
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _USER_TEMPLATE.format(
                    role=role,
                    input_prompt=input_prompt,
                    failure=failure,
                    existing_skills=existing_text,
                ),
            },
        ],
        temperature=0.3,
        max_tokens=256,
    )

    content = response.choices[0].message.content or ""
    content = content.strip()

    skill = Skill(role=role, content=content, source="learned")
    store.add_skill(skill)
    return skill
