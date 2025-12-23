# Resources Analysis v3
> Verified API summary and existing asset inventory

## 1. Existing Project Assets

### 1.1 OfficeQA Data

**Location**: `/data/officeqa/`

| File | Content | Status |
|------|---------|--------|
| officeqa.csv | 580 questions dataset | ✅ Exists |
| reward.py | Scoring function | ✅ Exists, complete |
| treasury_bulletins_parsed/ | Treasury data | ✅ Exists |

### 1.2 Data Format (officeqa.csv)

| Column | Type | Example |
|--------|------|---------|
| uid | str | "UID0001" |
| question | str | "What were the total expenditures..." |
| answer | str | "2602" |
| source_docs | str | URL reference |
| difficulty | str | "easy" or "hard" |

### 1.3 Scoring Function (reward.py)

**Main Function**: `score_answer(ground_truth, predicted, tolerance=0.0) -> float`

**Capabilities**:
| Feature | Example |
|---------|---------|
| Percentage normalization | 4.5% ↔ 0.045 |
| Magnitude words | million, billion, trillion |
| Numeric tolerance | Optional float tolerance |
| Text matching | "March 1977" |
| FINAL_ANSWER extraction | Parses `<FINAL_ANSWER>` tags |
| Fuzzy matching | With rationale |

**Return**: 1.0 (correct) or 0.0 (incorrect)

**Helper Functions**:
- `fuzzy_match_answer(ground_truth, predicted, tolerance) -> Tuple[bool, str]`
- `extract_final_answer(text) -> str`

---

## 2. GEPA (gepa-ai/gepa)

### 2.1 Verified API

| Function | Signature |
|----------|-----------|
| optimize() | `optimize(seed_candidate, trainset, valset, adapter, reflection_lm, max_metric_calls)` |

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| seed_candidate | dict | Initial candidate (e.g., `{"system_prompt": "..."}`) |
| trainset | List[Any] | Training examples |
| valset | List[Any] | Validation examples |
| adapter | GEPAAdapter | Custom adapter implementation |
| reflection_lm | Callable | LLM for reflection (dspy.LM) |
| max_metric_calls | int | Max evaluations (default 2000) |

**Return**: `result.best_candidate` (dict)

### 2.2 GEPAAdapter Interface

| Method | Signature |
|--------|-----------|
| evaluate | `(candidate: List[str], minibatch: List[Any]) -> Tuple[float, List[Any]]` |
| extract_traces_for_reflection | `(traces: List[Any], component_name: str) -> List[str]` |

---

## 3. Deep Agents (langchain-ai/deepagents)

### 3.1 Core API

| Function | Purpose |
|----------|---------|
| create_deep_agent() | Create main agent with subagents |

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| model | str | Model identifier |
| tools | List[Callable] | Agent tools |
| system_prompt | str | Agent instructions |
| middleware | List | Middleware stack |
| subagents | List[dict] | Subagent definitions |
| backend | FilesystemBackend | Persistence backend |

### 3.2 Subagent Definition

| Field | Required | Description |
|-------|----------|-------------|
| name | Yes | Unique identifier |
| description | Yes | Used for routing |
| system_prompt | Yes | Subagent instructions |
| tools | No | Subagent-specific tools |
| model | No | Override default model |

### 3.3 Key Components

| Component | Purpose |
|-----------|---------|
| SubAgentMiddleware | Enable subagent delegation |
| FilesystemMiddleware | File-based context tools |
| FilesystemBackend | Persist to disk |
| CompiledSubAgent | Wrap custom LangGraph |

### 3.4 ReAct Behavior

The main agent operates as a ReAct loop:
- Reasons about task state
- Calls subagents as tools
- Observes results
- Iterates until task complete
- Natural retry via re-calling subagents

---

## 4. Claude Agent SDK

### 4.1 Core Components

| Component | Purpose |
|-----------|---------|
| ClaudeSDKClient | Main client for Claude interaction |
| ClaudeAgentOptions | Configuration object |
| @tool decorator | Define MCP tools |
| create_sdk_mcp_server() | Package tools as MCP server |

### 4.2 ClaudeAgentOptions

| Parameter | Type | Description |
|-----------|------|-------------|
| model | str | Claude model identifier |
| system_prompt | str | System instructions |
| mcp_servers | dict | Server name → server mapping |
| allowed_tools | List[str] | Permitted tool identifiers |
| max_turns | int | Max conversation turns |
| hooks | dict | PreToolUse/PostToolUse hooks |

### 4.3 @tool Decorator Pattern

```python
@tool("name", "description", {"param": type})
async def tool_func(args: dict[str, Any]) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": result}]}
```

### 4.4 create_sdk_mcp_server()

| Parameter | Type | Description |
|-----------|------|-------------|
| name | str | Server identifier |
| version | str | Semantic version |
| tools | List[Callable] | @tool decorated functions |

---

## 5. pctx-sandbox

### 5.1 Core API

**Decorator**: `@sandbox(dependencies=[...])`

| Parameter | Type | Description |
|-----------|------|-------------|
| dependencies | List[str] | pip packages to install |

### 5.2 Requirements

| Requirement | Value |
|-------------|-------|
| Python | 3.10, 3.11, 3.12 |
| Container Runtime | Podman (rootless) |
| License | MIT |

### 5.3 Features

| Feature | Description |
|---------|-------------|
| OCI isolation | Rootless Podman containers |
| Dependency caching | Per unique combination |
| Worker pools | Warm workers for fast execution |
| Auto-rotation | 100 jobs or 1 hour |

---

## 6. MLflow

### 6.1 Anthropic Autolog

**Enable**: `mlflow.anthropic.autolog()`

**What's Traced**:
| Item | Traced |
|------|--------|
| ClaudeSDKClient calls | ✅ |
| Raw Anthropic SDK | ✅ |
| Tool executions | ✅ |
| Token usage | ✅ |
| Response content | ✅ |

### 6.2 Token Usage Access

```python
trace = mlflow.get_trace(trace_id)
usage = trace.info.token_usage
# usage['input_tokens'], usage['output_tokens'], usage['total_tokens']
```

### 6.3 Run Management

| Function | Purpose |
|----------|---------|
| mlflow.start_run() | Begin tracked run |
| mlflow.log_metric() | Log numeric metrics |
| mlflow.log_artifact() | Log files |
| mlflow.set_experiment() | Set experiment name |

---

## 7. LangGraph (Extension Only)

### 7.1 When to Use

Use LangGraph only when extending Deep Agents with custom state graphs:
- Wrap with CompiledSubAgent
- Use for complex state machines beyond ReAct

### 7.2 Core API

| Component | Purpose |
|-----------|---------|
| StateGraph | Define state machine |
| START, END | Graph terminals |
| add_node() | Add processing node |
| add_edge() | Add transition |
| add_conditional_edges() | Add routing |

---

## 8. Dependencies

### 8.1 Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| gepa | ≥0.1.0 | Optimization framework |
| deepagents | ≥0.1.0 | Agent framework |
| claude-agent-sdk | ≥0.1.0 | Claude integration |
| pctx-sandbox | ≥0.1.0 | Code sandboxing |
| mlflow | ≥2.10.0 | Experiment tracking |
| dspy-ai | ≥2.5.0 | GEPA reflection LLM |
| anthropic | ≥0.30.0 | Claude API client |
| nest-asyncio | ≥1.6.0 | Async/sync bridge |

### 8.2 Development Dependencies

| Package | Purpose |
|---------|---------|
| pytest | Testing |
| pandas | Data loading |
| pyyaml | Config parsing |
| pydantic | Config validation |

### 8.3 External Requirements

| Requirement | Purpose | Installation |
|-------------|---------|--------------|
| Podman | pctx-sandbox | OS package manager |
| ANTHROPIC_API_KEY | Claude API | Environment variable |

---

## 9. Version Compatibility

| Package | Min Version | Python | Notes |
|---------|-------------|--------|-------|
| gepa | 0.1.0 | 3.10+ | GEPA optimizer |
| deepagents | 0.1.0 | 3.10+ | Uses LangGraph |
| claude-agent-sdk | 0.1.0 | 3.10+ | Async API |
| pctx-sandbox | 0.1.0 | 3.10-3.12 | Requires Podman |
| mlflow | 2.10.0 | 3.8+ | Anthropic autolog |

**Recommended Python**: 3.11 (best compatibility)
