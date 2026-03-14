# EvoSkill

**EvoSkill** is a lightweight Python library that enables any AI agent to learn skills at runtime. Wrap your agent function with the `@evoskill` decorator and the library will automatically capture failures, synthesize concise skills via OpenAI, and inject them into future calls — so the agent improves over time without any manual prompt engineering.

## Installation

```bash
pip install evoskill
```

Or install from source:

```bash
pip install -e .
```

## Quick Start

```python
from evoskill import evoskill

@evoskill(role="data_analyst")
async def analyze(prompt: str) -> str:
    # your agent logic here
    ...
```

Every time `analyze` raises an exception, EvoSkill will call OpenAI to synthesize a short skill describing how to avoid that failure. On subsequent calls, all learned skills are automatically prepended to the prompt.

## Configuration

Create a `.env` file (or set environment variables):

```env
EVOSKILL_API_SKILL=sk-...        # OpenAI API key (required for learning)
EVOSKILL_STORAGE_PATH=./skills   # where to persist skills (default: ./evoskill_data)
EVOSKILL_MODEL=gpt-4o-mini       # model for synthesis (default: gpt-4o-mini)
```

## API Reference

### `@evoskill(...)` decorator

```python
@evoskill(
    role="data_analyst",        # optional — defaults to function/class name
    learn_when=my_callback,     # optional — callback(input, output) -> bool
    skills=["always do X"],     # optional — manual skills (list of strings)
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `role` | `str \| None` | Agent role identifier. Defaults to function name, or class name for methods. |
| `learn_when` | `Callable[[str, Any], bool] \| None` | Custom callback; if it returns `True` for `(input, output)`, skill synthesis is triggered. |
| `skills` | `list[str] \| None` | Manual skills to always include for this role. |

**Behavior:**
- Works with both `async` and `sync` functions, and with methods.
- Prepends loaded skills to the first positional `str` argument.
- On exception → synthesizes a skill via OpenAI.
- If `learn_when` returns `True` → synthesizes a skill.
- Otherwise → no learning.

### `SkillStore`

```python
from evoskill import SkillStore

store = SkillStore()
store.add_manual_skill(role="analyst", content="always validate input")
store.get_skills("analyst")   # -> [Skill(...)]
store.list_roles()            # -> ["analyst"]
```

| Method | Description |
|--------|-------------|
| `get_skills(role) -> list[Skill]` | Return all skills for the given role. |
| `add_skill(skill)` | Persist a `Skill` instance. |
| `add_manual_skill(role, content)` | Add a manual skill (shorthand). |
| `list_roles() -> list[str]` | List all roles with stored skills. |

### `Skill` model

```python
from evoskill import Skill

Skill(
    role="analyst",
    content="Always validate JSON before parsing.",
    source="learned",       # 'manual' | 'learned'
    created_at=datetime,    # auto-set to now (UTC)
)
```

## Manual Skills

Specify skills directly on the decorator:

```python
@evoskill(skills=["always validate input", "use UTC timestamps"])
def my_agent(prompt: str) -> str:
    ...
```

Or add them programmatically:

```python
from evoskill import SkillStore

store = SkillStore()
store.add_manual_skill(role="analyst", content="always validate input")
```

## Skill Injection Format

When skills are prepended to the prompt, the format is:

```
[EvoSkill] Learned skills for this role:
- skill 1 text
- skill 2 text

<original prompt here>
```

## License

MIT
