# EvoSkill

**EvoSkill** is a lightweight Python library that enables any AI agent to learn skills at runtime. Wrap your agent function with the `@evoskill` decorator and the library will automatically capture failures, synthesize concise skills via any LLM you provide, and inject them into future calls — so the agent improves over time without manual prompt engineering. Bring your own LLM, use structured inputs, learn from cross-agent reviews, and scope skills by tags.

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

Every time `analyze` raises an exception, EvoSkill will synthesize a short skill describing how to avoid that failure. On subsequent calls, all learned skills are automatically prepended to the prompt.

## Bring Your Own LLM

EvoSkill doesn't force you to use OpenAI. Pass any callable that accepts chat-style messages and returns a string:

```python
from anthropic import Anthropic

client = Anthropic()

def my_llm(messages: list[dict[str, str]]) -> str:
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        system=messages[0]["content"],
        messages=[{"role": "user", "content": messages[1]["content"]}],
    )
    return resp.content[0].text

@evoskill(role="writer", llm=my_llm)
def write_report(prompt: str) -> str:
    ...
```

If you don't pass `llm`, EvoSkill falls back to the built-in OpenAI adapter using `EVOSKILL_API_SKILL` from your `.env`.

## Configuration

Create a `.env` file (or set environment variables):

```env
EVOSKILL_API_SKILL=sk-...        # OpenAI API key (only needed if using the default adapter)
EVOSKILL_STORAGE_PATH=./skills   # where to persist skills (default: ./evoskill_data)
EVOSKILL_MODEL=gpt-4o-mini       # model for default adapter (default: gpt-4o-mini)
```

## API Reference

### `@evoskill(...)` decorator

```python
@evoskill(
    role="data_analyst",
    learn_when=my_callback,
    skills=["always do X"],
    inject_skills=my_injector,
    llm=my_llm,
    tags=["python", "data"],
    max_skills=10,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `role` | `str \| None` | Agent role identifier. Defaults to function name, or class name for methods. |
| `learn_when` | `Callable \| None` | Callback `(input, output) -> bool`. Triggers skill synthesis when `True`. |
| `skills` | `list[str] \| None` | Manual skills to always include for this role. |
| `inject_skills` | `Callable \| None` | Custom callback `(args, kwargs, skills_text) -> (args, kwargs)` to control where skills are injected. Default: prepend to first `str` arg. |
| `llm` | `Callable \| None` | LLM callable `(messages) -> str` for synthesis. Default: built-in OpenAI adapter. |
| `tags` | `list[str] \| None` | Tags for scoping skill retrieval and storage. |
| `max_skills` | `int \| None` | Max number of skills to inject (most recent). |

### `SkillStore`

```python
from evoskill import SkillStore

store = SkillStore()

# Basic CRUD
store.add_manual_skill("analyst", "always validate input", tags=["data"])
store.get_skills("analyst")                     # all enabled skills
store.get_skills("analyst", tags=["data"])       # filtered by tag
store.list_roles()                               # ["analyst"]

# Lifecycle
store.disable_skill("analyst", "some skill")     # soft-delete
store.enable_skill("analyst", "some skill")      # re-enable
store.remove_skill("analyst", "old skill")       # hard-delete

# Consolidation (deduplicate / merge via LLM)
store.consolidate("analyst", llm=my_llm, max_skills=10)

# Cross-agent feedback learning
store.learn_from_feedback(
    role="writer",
    llm=my_llm,
    input_prompt="write a summary",
    agent_output="The report shows...",
    reviewer_feedback="Too vague — include specific numbers",
    tags=["review"],
)
```

### `Skill` model

```python
from evoskill import Skill

Skill(
    role="analyst",
    content="Always validate JSON before parsing.",
    source="learned",       # 'manual' | 'learned'
    tags=["python"],        # scoping tags
    enabled=True,           # can be disabled
    created_at=datetime,    # auto-set to now (UTC)
)
```

## Structured Inputs

If your agent takes structured input (not a plain string), use `inject_skills` to control where skills land:

```python
def my_injector(args, kwargs, skills_text):
    # Put skills into the system_prompt kwarg
    kwargs = {**kwargs, "system_prompt": skills_text + kwargs.get("system_prompt", "")}
    return args, kwargs

@evoskill(role="planner", inject_skills=my_injector)
def plan(task: dict, system_prompt: str = "") -> str:
    ...
```

## Cross-Agent Feedback

When Agent B reviews Agent A's work, feed the structured feedback back as a learning signal:

```python
from evoskill import SkillStore

store = SkillStore()
store.learn_from_feedback(
    role="writer",
    llm=my_llm,
    input_prompt="Summarize Q4 results",
    agent_output="Revenue was good this quarter.",
    reviewer_feedback="Too vague. Include revenue figures and YoY comparison.",
    tags=["finance", "review"],
)
```

## Skill Lifecycle

Prevent skill bloat with consolidation, disabling, and removal:

```python
store = SkillStore()

# Merge/deduplicate all skills for a role
store.consolidate("analyst", llm=my_llm, max_skills=10)

# Disable a skill that isn't helping
store.disable_skill("analyst", "outdated advice")

# Hard-remove
store.remove_skill("analyst", "bad skill")
```

## Tag-Based Scoping

Scope skills so they only apply in the right context:

```python
@evoskill(role="writer", tags=["finance"])
def finance_writer(prompt: str) -> str:
    ...

@evoskill(role="writer", tags=["marketing"])
def marketing_writer(prompt: str) -> str:
    ...
```

Skills tagged `["finance"]` won't leak into marketing calls, even though both share the `"writer"` role.

## Skill Injection Format

```
[EvoSkill] Learned skills for this role:
- skill 1 text
- skill 2 text

<original prompt here>
```

## License

MIT
