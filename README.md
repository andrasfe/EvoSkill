# EvoSkill

**EvoSkill** is a lightweight Python library that enables any AI agent to learn skills at runtime. Use the `SkillStore` directly for full control, or wrap your agent function with the `@evoskill` decorator for automatic skill injection. The library captures failures, synthesizes concise skills via any LLM you provide, and injects them into future calls — so the agent improves over time without manual prompt engineering.

## Installation

```bash
pip install evoskill
```

Or install from source:

```bash
pip install -e .
```

## Quick Start

### Using `SkillStore` (recommended for structured-input agents)

```python
from evoskill import SkillStore

store = SkillStore()

# Get the formatted skill block ready to paste into any prompt
skills_text = store.get_skills_text("analyst", tags=["finance"], max_skills=10)
prompt = skills_text + my_actual_prompt

# Learn from cross-agent feedback
store.learn_from_feedback(
    role="analyst",
    llm=my_llm,
    input_prompt="Summarize Q4",
    agent_output="Revenue was good.",
    reviewer_feedback="Too vague — include numbers.",
)
```

### Using the `@evoskill` decorator

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

Async LLMs are also supported — the decorator's async wrapper will `await` them instead of blocking:

```python
async def my_async_llm(messages: list[dict[str, str]]) -> str:
    ...

@evoskill(role="writer", llm=my_async_llm)
async def write_report(prompt: str) -> str:
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

### `SkillStore` — primary API

```python
from evoskill import SkillStore

store = SkillStore()                          # uses FileBackend by default
store = SkillStore(backend=my_redis_backend)  # or bring your own backend
```

#### Skill retrieval

```python
store.get_skills("analyst")                     # all enabled skills
store.get_skills("analyst", tags=["data"])       # filtered by tag
store.get_skills_text("analyst", tags=["data"], max_skills=10)  # formatted block
store.list_roles()                               # ["analyst"]
```

`get_skills_text()` returns the formatted skill block ready to paste into a prompt section — no need to reimplement formatting.

#### Adding skills

```python
store.add_manual_skill("analyst", "always validate input", tags=["data"])
store.add_skill(Skill(role="analyst", content="...", source="learned"))
```

#### Lifecycle

```python
store.disable_skill("analyst", "some skill")     # soft-delete
store.enable_skill("analyst", "some skill")      # re-enable
store.remove_skill("analyst", "old skill")       # hard-delete
```

#### Cross-agent feedback learning

```python
store.learn_from_feedback(
    role="writer",
    llm=my_llm,
    input_prompt="write a summary",
    agent_output="The report shows...",
    reviewer_feedback="Too vague — include specific numbers",
    tags=["review"],
    system_prompt="You are a documentation expert.",   # optional override
    user_template="Custom template: {role} {feedback}", # optional override
)
```

#### Batch feedback ingestion

Send multiple feedback items in one LLM call for lower latency and cross-issue context:

```python
items = [
    {"input_prompt": "...", "agent_output": "...", "reviewer_feedback": "..."},
    {"input_prompt": "...", "agent_output": "...", "reviewer_feedback": "..."},
]
skills = store.learn_from_feedback_batch(role="writer", llm=my_llm, items=items)
```

Async variant: `await store.alearn_from_feedback_batch(...)`.

#### Skill effectiveness tracking

Track which skills actually help and prune the ones that don't:

```python
store.mark_hit("writer", "Always include numbers in summaries.")   # skill helped
store.mark_miss("writer", "Always include numbers in summaries.")  # skill didn't help

# Consolidate and drop zero-hit skills
store.consolidate("writer", llm=my_llm, max_skills=10, drop_zero_hit=True)
```

Each `Skill` has a `hit_rate` property: `skill.hit_rate  # 0.0 – 1.0`.

#### Consolidation

```python
store.consolidate("analyst", llm=my_llm, max_skills=10, drop_zero_hit=True)
```

Merges duplicates, removes contradictions, and prefers skills with higher hit rates.

#### Export / import

Ship a baseline skill set with a deployment, seed from a previous run, or sync across environments:

```python
data = store.export_skills("analyst")         # list[dict]
store.import_skills("analyst", data)          # appends to existing
```

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
    system_prompt="You are a code reviewer.",
    user_template="Custom: {role} {feedback}",
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
| `system_prompt` | `str \| None` | Override the system prompt used during skill synthesis. |
| `user_template` | `str \| None` | Override the user template used during skill synthesis. |

### `Skill` model

```python
from evoskill import Skill

Skill(
    role="analyst",
    content="Always validate JSON before parsing.",
    source="learned",       # 'manual' | 'learned'
    tags=["python"],        # scoping tags
    enabled=True,           # can be disabled
    hit_count=0,            # effectiveness tracking
    miss_count=0,
    created_at=datetime,    # auto-set to now (UTC)
)

skill.hit_rate  # float: 0.0 – 1.0
```

### Storage backend protocol

The default `FileBackend` persists skills as JSON files with `filelock` for process safety. Provide your own backend for SQLite, Redis, or any other storage:

```python
from evoskill import StorageBackend, FileBackend, SkillStore

class RedisBackend:
    def read(self, role: str) -> list[Skill]: ...
    def write(self, role: str, skills: list[Skill]) -> None: ...
    def lock(self, role: str) -> ContextManager: ...
    def list_roles(self) -> list[str]: ...

store = SkillStore(backend=RedisBackend())
```

### Pluggable synthesis prompts

Different agent roles need different synthesis framing. Override the system prompt and/or user template on any synthesis call:

```python
store.learn_from_feedback(
    role="doc_writer",
    llm=my_llm,
    input_prompt="...",
    agent_output="...",
    reviewer_feedback="...",
    system_prompt="You extract documentation-writing best practices.",
    user_template="Role: {role}\nFeedback: {feedback}\nExisting: {existing_skills}",
)
```

Defaults are available as `DEFAULT_SYSTEM_PROMPT` and `DEFAULT_USER_TEMPLATE` for reference.

## Structured Inputs

If your agent takes structured input (not a plain string), use `inject_skills` to control where skills land:

```python
def my_injector(args, kwargs, skills_text):
    kwargs = {**kwargs, "system_prompt": skills_text + kwargs.get("system_prompt", "")}
    return args, kwargs

@evoskill(role="planner", inject_skills=my_injector)
def plan(task: dict, system_prompt: str = "") -> str:
    ...
```

Or skip the decorator entirely and use `SkillStore.get_skills_text()` directly:

```python
store = SkillStore()
skills_text = store.get_skills_text("planner", tags=["infra"])
my_agent.run(system_prompt=skills_text + base_prompt)
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
