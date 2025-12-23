# Tool Crafter Design v3
> Deep Agent with ReAct orchestration for autonomous multi-tool generation

## 1. Architecture Overview

### 1.1 Design Philosophy

The Tool Crafter is a **Deep Agent orchestrator** using the ReAct pattern. The main agent reasons about the task, calls subagents as tools, observes results, and iterates until the task is complete.

**Key Insight (Addresses Critical Issue 1.7)**: No explicit retry mechanism is needed. The ReAct loop inherently handles retries - if a subagent's output is unsatisfactory, the main agent reasons about the failure and calls the subagent again with corrective instructions.

### 1.2 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TOOL CRAFTER (Main Agent)                            │
│                              ReAct Loop                                      │
│                                                                              │
│   Thought → Action → Observation → Thought → Action → ... → Final Answer    │
│                                                                              │
│   The main agent:                                                            │
│   • Receives failure traces from GEPA                                        │
│   • Plans using write_todos tool                                            │
│   • Delegates to subagents as needed                                        │
│   • Observes subagent outputs                                               │
│   • Retries by calling subagent again if unsatisfied                        │
│   • Outputs validated MCP tool package                                       │
└────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌──────────────┐           ┌──────────────┐           ┌──────────────┐
│   Analyzer   │           │   Designer   │           │    Coder     │
│  Subagent    │           │  Subagent    │           │   Subagent   │
│              │           │              │           │              │
│ Identify     │           │ Spec tool    │           │ Generate     │
│ capability   │           │ interface    │           │ Claude SDK   │
│ gaps         │           │ and schema   │           │ compatible   │
└──────────────┘           └──────────────┘           │ code         │
                                                      └──────────────┘
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌──────────────┐           ┌──────────────┐           ┌──────────────┐
│   Tester     │           │   Packager   │           │  Filesystem  │
│  Subagent    │           │  Subagent    │           │  Middleware  │
│              │           │   (NEW)      │           │              │
│ Validate in  │           │              │           │ Context mgmt │
│ pctx-sandbox │           │ Create MCP   │           │ via files    │
│              │           │ server pkg   │           │              │
└──────────────┘           └──────────────┘           └──────────────┘
```

## 2. ReAct Loop Behavior

### 2.1 How ReAct Provides Retry

The ReAct (Reasoning + Acting) pattern works as follows:

```
1. THOUGHT: "I need to analyze the failure traces to identify capability gaps"
2. ACTION: Call Analyzer subagent with failure traces
3. OBSERVATION: Analyzer returns analysis identifying 2 missing capabilities
4. THOUGHT: "The analysis is complete. Now I need tool specifications"
5. ACTION: Call Designer subagent with the analysis
6. OBSERVATION: Designer returns tool spec
7. THOUGHT: "Spec looks good. Now generate the code"
8. ACTION: Call Coder subagent with the spec
9. OBSERVATION: Coder returns implementation
10. THOUGHT: "Code generated. Must test before packaging"
11. ACTION: Call Tester subagent with the code
12. OBSERVATION: Tester reports 2 test failures
13. THOUGHT: "Tests failed. I need to fix the code. Let me call Coder again with the error details"
14. ACTION: Call Coder subagent with original spec + error details  <-- RETRY
15. OBSERVATION: Coder returns fixed implementation
16. THOUGHT: "Try testing again"
17. ACTION: Call Tester subagent with fixed code
18. OBSERVATION: All tests pass
19. THOUGHT: "Tests pass. Now package as MCP server"
20. ACTION: Call Packager subagent
21. OBSERVATION: MCP server package created
22. FINAL ANSWER: Return path to MCP server package
```

### 2.2 ReAct Advantages

| Advantage | Description |
|-----------|-------------|
| Natural retry | Main agent decides when to retry based on observation |
| Context preservation | Each retry includes accumulated context |
| Flexible recovery | Agent can try different approaches on failure |
| Reasoning transparency | Thought steps explain decision making |
| No hardcoded loops | Retry count emerges from task complexity |

### 2.3 Maximum Iterations

| Parameter | Value | Purpose |
|-----------|-------|---------|
| max_turns | 20 | Main agent reasoning steps |
| max_subagent_retries | 3 | Emergent from prompt guidance |

The main agent's system prompt includes guidance: "If a subagent fails after 3 attempts, report the failure and stop."

## 3. Multi-Tool Generation

### 3.1 Design Principle

Tool Crafter can generate **multiple tools in a single invocation** when the Analyzer identifies multiple capability gaps. This is more efficient than invoking Tool Crafter multiple times.

### 3.2 Multi-Tool Workflow

```
1. GEPA Orchestrator detects plateau, invokes Tool Crafter with failure traces

2. Analyzer identifies N capability gaps (prioritized by impact)
   Output: /analysis/gaps.md with list of gaps

3. Main Agent loops through gaps (up to max_tools_per_invocation):

   For gap[1]:
   ├── Designer creates spec → /specs/tool_1_spec.json
   ├── Coder generates code → /generated/tool_1.py
   ├── Tester validates → retry Coder if fails (max 3)
   └── Packager creates package → /output/tool_1/

   For gap[2]:
   ├── Designer creates spec → /specs/tool_2_spec.json
   ├── Coder generates code → /generated/tool_2.py
   ├── Tester validates → retry Coder if fails (max 3)
   └── Packager creates package → /output/tool_2/

   ... (up to N tools)

4. Return list of all generated package paths
```

### 3.3 Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| max_tools_per_invocation | 3 | Maximum tools to generate per run |
| parallel_tool_generation | false | Sequential for reliability |
| stop_on_first_failure | false | Continue with other tools if one fails |

### 3.4 Multi-Tool ReAct Example

```
1. THOUGHT: "Analyzer found 2 capability gaps: historical_rates and inflation_calculator"
2. ACTION: Call Designer for gap 1 (historical_rates)
3. OBSERVATION: Spec created at /specs/tool_1_spec.json
4. THOUGHT: "Spec ready, now generate code"
5. ACTION: Call Coder for tool 1
6. OBSERVATION: Code at /generated/tool_1.py
7. ACTION: Call Tester for tool 1
8. OBSERVATION: All tests pass
9. ACTION: Call Packager for tool 1
10. OBSERVATION: Package at /output/historical_rates/
11. THOUGHT: "Tool 1 complete. Now process gap 2 (inflation_calculator)"
12. ACTION: Call Designer for gap 2
... (continue for tool 2)
22. FINAL ANSWER: Generated 2 tools: [/output/historical_rates/, /output/inflation_calculator/]
```

### 3.5 Handling Partial Success

| Scenario | Behavior |
|----------|----------|
| All tools succeed | Return list of all package paths |
| Some tools fail | Return successful tools, log failures |
| All tools fail | Return empty list, log all failures |
| Analyzer finds 0 gaps | Return immediately with "no tools needed" |

### 3.6 Output Format

Tool Crafter returns a structured result:

```
{
  "success": true,
  "tools_generated": 2,
  "packages": [
    "/output/historical_rates/",
    "/output/inflation_calculator/"
  ],
  "failures": [],
  "summary": "Generated 2 of 2 identified tools"
}
```

## 4. Subagent Specifications

### 4.1 Analyzer Subagent

**Purpose**: Extract actionable insights from GEPA failure traces

**Input**: Formatted failure traces from `extract_traces_for_reflection()`

**Output**: Structured analysis written to `/analysis/gaps.md`

| Output Section | Content |
|----------------|---------|
| Failure Patterns | Categorized failure types |
| Missing Capabilities | Specific tool opportunities |
| Priority Ranking | Impact-based ordering |
| Recommendations | Concrete tool suggestions |

**Tools**: Filesystem tools (read input, write output)

**Model**: claude-sonnet (good reasoning)

### 4.2 Designer Subagent

**Purpose**: Create complete tool specifications

**Input**: Analysis from Analyzer (reads `/analysis/gaps.md`)

**Output**: JSON specification written to `/specs/tool_spec.json`

| Spec Field | Content |
|------------|---------|
| tool_name | Snake_case identifier |
| description | Clear tool purpose |
| parameters | JSON schema with types |
| return_type | Expected output schema |
| error_cases | Edge case handling |
| examples | Input/output pairs |

**Tools**: Filesystem tools

**Model**: claude-sonnet (schema design)

### 4.3 Coder Subagent

**Purpose**: Generate implementation code

**Input**: Tool spec (reads `/specs/tool_spec.json`)

**Output**: Python code written to `/generated/tool.py`

**Critical Requirements**:
| Requirement | Rationale |
|-------------|-----------|
| Async function | Claude SDK tools are async |
| @tool decorator | Claude Agent SDK pattern |
| Proper return format | `{"content": [{"type": "text", "text": result}]}` |
| Error handling | Return `is_error: True` on failure |
| Type hints | Required for tool schema |
| Docstring | Becomes tool description |

**Tools**: Filesystem tools (write_file, edit_file)

**Model**: claude-sonnet (code generation)

### 4.4 Tester Subagent

**Purpose**: Validate generated code in sandbox

**Input**: Generated code (reads `/generated/tool.py`)

**Output**: Test report written to `/test_results/report.md`

**Test Categories**:
| Category | Examples |
|----------|----------|
| Normal cases | Expected inputs produce expected outputs |
| Edge cases | Empty inputs, boundary values |
| Error cases | Invalid inputs return error properly |
| Type validation | Parameters match schema |

**Tools**:
- `execute_in_sandbox(code, dependencies)` - Run arbitrary code safely
- `run_tests_in_sandbox(test_code, tool_code, deps)` - Execute pytest
- Filesystem tools (read code, write results)

**Model**: claude-haiku (fast iteration)

### 4.5 Packager Subagent

**Purpose**: Create deployable MCP server package

**Input**: Validated tool code (reads `/generated/tool.py`)

**Output**: Complete MCP server package in `/output/mcp_server/`

**Package Structure**:
```
/output/mcp_server/{tool_name}/
├── __init__.py          # Exports: tool_function, server
├── tool.py              # @tool decorated function
├── server.py            # create_sdk_mcp_server() setup
├── requirements.txt     # Runtime dependencies
└── README.md            # Usage documentation
```

**Key Responsibilities**:
| Responsibility | Output |
|----------------|--------|
| Create __init__.py | Export tool and server |
| Create server.py | MCP server with create_sdk_mcp_server() |
| Generate requirements.txt | Dependencies from code analysis |
| Generate README.md | Usage docs with examples |

**Tools**: Filesystem tools

**Model**: claude-haiku (templated output)

## 5. Sandbox Tools Implementation

### 5.1 execute_in_sandbox Tool

**Purpose**: Execute arbitrary Python code safely in Podman container

**Interface**:
| Parameter | Type | Description |
|-----------|------|-------------|
| code | str | Python code to execute |
| dependencies | List[str] | pip packages to install |
| timeout | int | Max execution seconds (default 30) |

**Return**:
| Field | Type | Description |
|-------|------|-------------|
| stdout | str | Captured output |
| stderr | str | Error output |
| return_code | int | Exit code (0 = success) |
| error | Optional[str] | Exception message if any |

**Implementation Approach**:
- Wrap code in pctx-sandbox `@sandbox` decorator
- Pass dependencies to decorator
- Capture stdout/stderr via subprocess
- Return structured result

### 5.2 run_tests_in_sandbox Tool

**Purpose**: Run pytest on generated code safely

**Interface**:
| Parameter | Type | Description |
|-----------|------|-------------|
| test_code | str | Pytest test code |
| tool_code | str | Tool implementation to test |
| dependencies | List[str] | Required packages including pytest |

**Return**:
| Field | Type | Description |
|-------|------|-------------|
| passed | bool | All tests passed |
| total_tests | int | Number of tests |
| failures | List[str] | Failed test names |
| output | str | Full pytest output |

**Implementation Approach**:
- Create temporary directory with both files
- Install dependencies in sandbox
- Run pytest with captured output
- Parse results and return structured response

### 5.3 Sandbox Prerequisites

| Requirement | Purpose | Verification |
|-------------|---------|--------------|
| Podman installed | Container runtime | `podman --version` |
| Rootless mode | Security | `podman info \| grep rootless` |
| Network access | Dependency install | `podman pull python:3.11` |

## 6. Context Management

### 6.1 Filesystem-Based Context

**Problem**: Generated code can be large, overflowing context

**Solution**: Write intermediate outputs to filesystem, pass file paths

**Workflow**:
```
Analyzer → writes to /analysis/gaps.md
           Main agent tells Designer to read /analysis/gaps.md
Designer → writes to /specs/tool_spec.json
           Main agent tells Coder to read /specs/tool_spec.json
Coder    → writes to /generated/tool.py
           Main agent tells Tester to test /generated/tool.py
Tester   → writes to /test_results/report.md
           Main agent reads results, decides next action
Packager → writes to /output/mcp_server/
           Main agent returns package path
```

### 6.2 FilesystemBackend Configuration

| Setting | Value | Purpose |
|---------|-------|---------|
| root_dir | ./tool_crafter_workspace | Isolated workspace |
| persistence | Per-invocation | Clean slate each run |
| cleanup | After successful packaging | Remove intermediate files |

## 7. GEPA Integration

### 7.1 Input: Failure Traces

Tool Crafter receives formatted strings from `extract_traces_for_reflection()`:

```
FAILED EVALUATION:
- Question ID: UID0042
- Question: What was the treasury rate in March 1977?
- Expected Answer: 5.2%
- Model's Answer: I don't have access to that information
- Score: 0.0
- Tools Used: query_treasury
- Error: None
```

### 7.2 Output: MCP Servers (Multiple)

Tool Crafter returns path to generated MCP server:

```
/generated_tools/treasury_historical_rates/
├── __init__.py
├── tool.py
├── server.py
├── requirements.txt
└── README.md
```

### 7.3 Integration Flow

```
Orchestrator detects plateau
        │
        ▼
Collect persistent failures from adapter traces
        │
        ▼
Invoke Tool Crafter with failure traces
        │
        ▼
Tool Crafter returns package path (or failure)
        │
        ├── If success:
        │   ├── Import generated server
        │   ├── Add to adapter.mcp_servers
        │   ├── Rebuild allowed_tools
        │   └── Resume GEPA optimization
        │
        └── If failure:
            ├── Log failure
            └── Continue GEPA without new tool
```

## 8. Error Handling

### 8.1 Subagent Failures

| Failure | Main Agent Response |
|---------|---------------------|
| Analyzer can't identify gaps | Report "no tool opportunities found" |
| Designer produces invalid schema | Call Designer again with validation errors |
| Coder produces non-runnable code | Call Coder again with error message |
| Tester times out | Retry with shorter test suite |
| Packager can't create package | Log error, return partial result |

### 8.2 Sandbox Failures

| Failure | Handling |
|---------|----------|
| Podman not available | Fallback to local execution with warning |
| Dependency install fails | Try with reduced dependencies |
| Execution timeout | Reduce code complexity or timeout |
| Container crash | Retry once, then fail gracefully |

### 8.3 Maximum Iterations

If main agent reaches 20 turns without completion:
- Log current state
- Return partial result or failure
- Orchestrator continues without new tool

## 9. Model Selection

| Component | Model | Rationale |
|-----------|-------|-----------|
| Main Agent | claude-sonnet | Complex reasoning for orchestration |
| Analyzer | claude-sonnet | Pattern recognition in failures |
| Designer | claude-sonnet | Schema design accuracy |
| Coder | claude-sonnet | Code generation quality |
| Tester | claude-haiku | Fast, simple logic |
| Packager | claude-haiku | Templated output |

## 10. Configuration

### 10.1 Tool Crafter Configuration

```yaml
tool_crafter:
  main_agent:
    model: "anthropic:claude-sonnet-4-20250514"
    max_turns: 20

  subagents:
    analyzer:
      model: "anthropic:claude-sonnet-4-20250514"
    designer:
      model: "anthropic:claude-sonnet-4-20250514"
    coder:
      model: "anthropic:claude-sonnet-4-20250514"
    tester:
      model: "anthropic:claude-3-5-haiku-20241022"
    packager:
      model: "anthropic:claude-3-5-haiku-20241022"

  sandbox:
    timeout_seconds: 30
    max_retries: 2

  workspace:
    root_dir: "./tool_crafter_workspace"
    output_dir: "./generated_tools"
```

### 10.2 Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| ANTHROPIC_API_KEY | Yes | All Claude calls |

## 11. Testing Strategy

### 11.1 Unit Tests

| Test | Purpose |
|------|---------|
| test_analyzer_output_format | Verify analysis structure |
| test_designer_schema_validity | JSON schema validation |
| test_coder_tool_pattern | Verify @tool decorator usage |
| test_packager_structure | Verify package completeness |

### 11.2 Integration Tests

| Test | Purpose |
|------|---------|
| test_end_to_end_tool_generation | Full workflow test |
| test_react_retry_on_failure | Verify retry behavior |
| test_sandbox_execution | Verify pctx-sandbox integration |
| test_gepa_tool_injection | Verify generated tool works in GEPA |

## 12. Key Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| ReAct loop doesn't converge | Medium | High | Max turns limit, fallback path |
| Generated code quality | Medium | Medium | Multiple test iterations |
| Sandbox unavailable | Low | High | Local fallback with warning |
| Context overflow | Low | Medium | Filesystem-based context |
| Infinite retry | Low | High | Max 3 retries per subagent |
