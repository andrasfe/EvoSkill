# System Architecture Design v3
> End-to-end architecture with orchestration layer and resolved integration points

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MAIN ORCHESTRATOR                               │
│                   (Python script or Deep Agent wrapper)                      │
│                                                                              │
│   Responsibilities:                                                          │
│   • Run GEPA optimization for N iterations                                   │
│   • Detect optimization plateau (no improvement for K iterations)            │
│   • Invoke Tool Crafter when plateau detected                               │
│   • Inject new tools into adapter                                           │
│   • Resume GEPA optimization                                                │
│   • Manage MLflow run lifecycle                                             │
└────────────────────────────────────────────────────────────────────────────┘
         │                                                    ▲
         │ Invokes                                            │ Returns best candidate
         ▼                                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GEPA OPTIMIZER                                  │
│                         (optimize() function)                                │
│                                                                              │
│   Provides: Optimization loop, Pareto selection, mutation, reflection        │
│   We Provide: GEPAAdapter implementation                                     │
└────────────────────────────────────────────────────────────────────────────┘
         │                                                    ▲
         │ Calls evaluate()                                   │ Returns (score, traces)
         ▼                                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│                            OfficeQAAdapter                                   │
│                     (GEPAAdapter implementation)                             │
│                                                                              │
│   • Loads data from /data/officeqa/officeqa.csv                             │
│   • Scores with /data/officeqa/reward.py score_answer()                     │
│   • Executes questions via Claude Agent SDK                                 │
│   • Manages MCP tools (built-in + dynamically generated)                    │
└────────────────────────────────────────────────────────────────────────────┘
         │                                                    ▲
         │ On plateau                                         │ Returns MCP server
         ▼                                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TOOL CRAFTER                                      │
│                     (Deep Agent with ReAct loop)                             │
│                                                                              │
│   Subagents: Analyzer → Designer → Coder → Tester → Packager                │
│   ReAct Loop: Main agent retries subagents until success or max iterations  │
└────────────────────────────────────────────────────────────────────────────┘
```

## 2. Orchestrator Design (NEW - Addresses Critical Issue 1.1)

### 2.1 Orchestrator Responsibilities

| Responsibility | Description |
|----------------|-------------|
| GEPA Lifecycle | Start GEPA, monitor progress, detect completion |
| Plateau Detection | Track score history, identify when improvement stalls |
| Tool Crafter Invocation | Call Tool Crafter with failure traces |
| Tool Injection | Add new MCP servers to adapter |
| MLflow Management | Start/end runs, log metrics, manage artifacts |
| Checkpointing | Save state for resume capability |

### 2.2 Plateau Detection Strategy

**Trigger Condition**: No improvement in validation score for K consecutive iterations

| Parameter | Default | Description |
|-----------|---------|-------------|
| `plateau_window` | 50 | Number of iterations to check |
| `min_improvement` | 0.01 | Minimum score improvement threshold |
| `max_tool_crafter_invocations` | 3 | Total Tool Crafter runs per optimization |

### 2.3 Orchestration Flow

```
1. Initialize
   ├── Load config
   ├── Start MLflow run
   ├── Load OfficeQA data
   └── Create OfficeQAAdapter with initial tools

2. Run GEPA Phase (iterations 1 to N)
   ├── Call optimize() with max_metric_calls=N
   ├── Collect best_candidate and scores
   └── Save checkpoint

3. Check Plateau
   ├── If no plateau → Continue to next phase
   └── If plateau detected:
       ├── Extract failure traces
       ├── Invoke Tool Crafter
       ├── If tool generated:
       │   ├── Inject into adapter
       │   └── Reset plateau counter
       └── If max_invocations reached → Stop

4. Final Evaluation
   ├── Evaluate best_candidate on test set
   ├── Log final metrics to MLflow
   └── End MLflow run (properly closed)
```

## 3. Component Interactions

### 3.1 Data Flow Summary

| From | To | Data | Format |
|------|-----|------|--------|
| CSV File | Adapter | Questions | `List[dict]` with uid, question, answer, difficulty |
| Adapter | GEPA | Evaluation result | `Tuple[float, List[OfficeQATrace]]` |
| Adapter | Tool Crafter | Failure traces | Formatted strings from `extract_traces_for_reflection()` |
| Tool Crafter | Adapter | New tool | MCP server via `create_sdk_mcp_server()` |
| All components | MLflow | Traces | Automatic via `mlflow.anthropic.autolog()` |

### 3.2 Candidate Format (Addresses Critical Issue 1.2)

**Clarification**: GEPA internally converts between formats

| Context | Format | Example |
|---------|--------|---------|
| optimize() `seed_candidate` | `dict` | `{"system_prompt": "You are..."}` |
| evaluate() `candidate` | `List[str]` | `["You are..."]` |
| Internal storage | `dict` | `{"system_prompt": "You are..."}` |

**Adapter Handling**:
```
def evaluate(self, candidate: List[str], minibatch):
    system_prompt = candidate[0]  # Extract from list
    ...
```

## 4. Existing Assets Integration

### 4.1 Data Loader (Addresses Critical Issue 1.4)

**Location**: `/data/officeqa/officeqa.csv`

| Column | Type | Description |
|--------|------|-------------|
| uid | string | Unique identifier (e.g., "UID0001") |
| question | string | The question text |
| answer | string | Ground truth answer |
| source_docs | string | URL or reference |
| difficulty | string | "easy" or "hard" |

**Loading Strategy**:
- Use pandas to load CSV
- Split by difficulty for stratified sampling
- Use 70/15/15 split for train/val/test

### 4.2 Scoring Function (Addresses Critical Issue 1.5)

**Location**: `/data/officeqa/reward.py`

**Existing Implementation**:
- `score_answer(ground_truth, predicted, tolerance=0.0) -> float`
- Returns 1.0 (correct) or 0.0 (incorrect)
- Handles percentage normalization (4.5% → 0.045)
- Handles magnitude words (million, billion, trillion)
- Handles date/text matching ("March 1977")
- Extracts from `<FINAL_ANSWER>` tags if present

**No implementation needed** - directly import and use existing function.

### 4.3 Treasury Data

**Location**: `/data/officeqa/treasury_bulletins_parsed/`

- Parsed treasury bulletin data for MCP tool use
- Provides context for answering OfficeQA questions

### 4.4 Treasury MCP Server Specification

**Purpose**: Provide Claude access to historical US Treasury data for answering OfficeQA questions

**Data Source**: `/data/officeqa/treasury_bulletins_parsed/transformed/`
- Files: `treasury_bulletin_{year}_{month}.txt` (~150KB-220KB each)
- Date range: 1939 to present

**Tool Interface**:

| Tool | Parameters | Returns |
|------|------------|---------|
| `search_treasury` | query: str, start_year: int, end_year: int, limit: int | List of relevant excerpts with dates and source references |
| `get_bulletin` | year: int, month: int | Full bulletin text for specified month |
| `get_expenditure` | category: str, year: int, month: int | Specific expenditure value with units |

**Implementation Guidelines**:
- Index text files for efficient semantic search
- Support date range filtering for temporal queries
- Return structured excerpts with source references (year, month, section)
- Handle questions about expenditures, interest rates, claims, receipts
- Include pagination for large result sets

**Example Usage**:
```
Question: "What was total public debt in March 1977?"

Tool call: search_treasury(query="total public debt", start_year=1977, end_year=1977)

Response: {
  "results": [
    {
      "year": 1977,
      "month": 3,
      "section": "Public Debt",
      "excerpt": "Total public debt outstanding: $653,544 million...",
      "source": "treasury_bulletin_1977_03.txt"
    }
  ]
}
```

### 4.5 Seed Prompt Specification

**Purpose**: Initial system prompt for GEPA optimization starting point

**Seed Prompt**:
```
You are a financial analyst assistant specialized in US Treasury historical data.

You have access to tools that can search and retrieve data from US Treasury Bulletins
spanning from 1939 to present.

When answering questions:
1. Use the treasury search tools to find relevant historical data
2. Show your reasoning and calculations step by step
3. Verify your numbers against the source data
4. Format your final answer inside <FINAL_ANSWER></FINAL_ANSWER> tags

For numerical answers:
- Include units (millions, billions, percentages)
- Round to the precision requested in the question
- For percentage differences, show the calculation

If you cannot find the specific data requested, explain what you found and what's missing.
```

**Rationale**:
| Element | Purpose |
|---------|---------|
| Domain expertise | Establishes context for treasury questions |
| Tool usage guidance | Encourages proper tool invocation |
| Step-by-step reasoning | Improves answer quality and traceability |
| FINAL_ANSWER tags | Enables reward.py `extract_final_answer()` |
| Numerical formatting | Aligns with scoring criteria |
| Failure acknowledgment | Reduces hallucination on missing data |

## 5. MLflow Integration (Addresses Critical Issue 1.8)

### 5.1 Run Lifecycle Management

**Problem**: MLflow runs must be properly closed to avoid leaks

**Solution**: Context manager pattern in Orchestrator

```
Orchestrator.run():
    with mlflow.start_run(run_name="optimization_run") as run:
        # All GEPA iterations inside this block
        # Run automatically closed on exit (normal or exception)
```

### 5.2 Experiment Structure

```
MLflow Experiment: officeqa_gepa_optimization
│
├── Run: main_optimization_YYYYMMDD_HHMMSS
│   │
│   ├── Params: seed_prompt, max_iterations, plateau_window
│   │
│   ├── Metrics (logged per iteration):
│   │   ├── batch_score
│   │   ├── val_score
│   │   └── failed_count
│   │
│   ├── Metrics (final):
│   │   ├── test_score
│   │   ├── total_iterations
│   │   └── tools_generated
│   │
│   └── Artifacts:
│       ├── best_candidate.json
│       ├── generated_tools/
│       └── failure_analysis.md
│
└── Nested Runs (per Tool Crafter invocation):
    └── Run: tool_crafter_1
        ├── Artifacts: tool_spec.json, tool_code.py
        └── Metrics: tests_passed, generation_attempts
```

### 5.3 Token/Cost Tracking

MLflow autolog automatically captures:
- `trace.info.token_usage.input_tokens`
- `trace.info.token_usage.output_tokens`
- `trace.info.token_usage.total_tokens`

Orchestrator aggregates and logs total cost estimate.

## 6. Error Handling Strategy

### 6.1 Evaluation Errors

| Error Type | Handling | Impact on Score |
|------------|----------|-----------------|
| Claude API timeout | Retry 3x with backoff, then score as 0 | 0.0 |
| Rate limiting (429) | Exponential backoff, retry | Wait and retry |
| Invalid response | Log for analysis, score as 0 | 0.0 |
| Tool execution error | Capture in trace, continue | Based on answer |

### 6.2 Timeout Configuration

| Operation | Default Timeout | Retry Strategy |
|-----------|----------------|----------------|
| Single question evaluation | 60 seconds | 3 retries with 2x backoff |
| Tool Crafter subagent | 120 seconds | Part of ReAct loop |
| Sandbox execution | 30 seconds | 2 retries |

### 6.3 Checkpointing Strategy (Addresses Medium Issue 2.10)

**Checkpoint Contents**:
- Current GEPA state (best candidates, scores)
- Number of Tool Crafter invocations
- Generated tools list
- MLflow run ID

**Checkpoint Frequency**: Every 50 iterations

**Resume**: Load checkpoint, recreate adapter with tools, continue optimization

## 7. Configuration Management

### 7.1 Configuration Schema

```yaml
# config.yaml
optimization:
  max_metric_calls: 500
  plateau_window: 50
  min_improvement: 0.01
  max_tool_crafter_invocations: 3

adapter:
  claude_model: "claude-sonnet-4-20250514"
  max_turns: 10
  timeout_seconds: 60
  retry_attempts: 3

tool_crafter:
  max_subagent_iterations: 3
  sandbox_timeout_seconds: 30
  output_directory: "./generated_tools"

mlflow:
  experiment_name: "officeqa_gepa_optimization"
  tracking_uri: "sqlite:///mlflow.db"
  log_every_n_iterations: 10

data:
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  random_seed: 42
```

### 7.2 Configuration Validation (Addresses Minor Issue 3.7)

- Use pydantic for config validation
- Fail fast on invalid configuration
- Provide clear error messages

## 8. Directory Structure

```
deep_agents/
├── src/
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── main.py              # Main orchestration logic
│   │   ├── plateau_detector.py  # Score tracking and plateau detection
│   │   └── checkpointing.py     # State save/restore
│   │
│   ├── adapters/
│   │   ├── __init__.py
│   │   └── officeqa_adapter.py  # GEPAAdapter implementation
│   │
│   ├── tool_crafter/
│   │   ├── __init__.py
│   │   ├── agent.py             # Main Tool Crafter agent
│   │   └── subagents/           # Subagent prompts and configs
│   │
│   └── utils/
│       ├── __init__.py
│       ├── async_bridge.py      # nest_asyncio handling
│       └── config.py            # Config loading and validation
│
├── data/
│   └── officeqa/
│       ├── officeqa.csv         # [EXISTS] Questions dataset
│       ├── reward.py            # [EXISTS] Scoring function
│       ├── loader.py            # Data loading utilities
│       └── treasury_bulletins_parsed/  # [EXISTS] Treasury data
│
├── generated_tools/             # Tool Crafter output
│
├── config/
│   └── config.yaml
│
└── tests/
    ├── test_adapter.py
    ├── test_orchestrator.py
    └── test_tool_crafter.py
```

## 9. Baseline Evaluation (Addresses Medium Issue 2.12)

**Phase 0**: Before any optimization

1. Load test set (15% of data)
2. Evaluate seed prompt on test set
3. Log baseline score to MLflow
4. This provides comparison baseline for optimization improvement

## 10. Cost Estimation (Addresses Medium Issue 2.11)

### 10.1 Per-Operation Costs (Approximate)

| Operation | Est. Tokens | Est. Cost (Sonnet) |
|-----------|-------------|-------------------|
| Single question eval | ~2,000 | ~$0.006 |
| GEPA reflection | ~1,500 | ~$0.0045 |
| Tool Crafter full run | ~20,000 | ~$0.06 |

### 10.2 Budget Controls

- Set `max_metric_calls` based on budget
- Log cumulative cost in MLflow
- Alert when approaching budget threshold

## 11. Risk Matrix (Updated)

| Risk | Probability | Impact | Mitigation | Status |
|------|-------------|--------|------------|--------|
| Missing orchestrator | ~~Certain~~ | ~~Blocker~~ | Orchestrator designed | ✅ Addressed |
| Undefined data loader | ~~Certain~~ | ~~Blocker~~ | CSV exists, loader spec added | ✅ Addressed |
| score_answer undefined | ~~Certain~~ | ~~Blocker~~ | Exists in reward.py | ✅ Addressed |
| Candidate format | ~~High~~ | ~~High~~ | Documented conversion | ✅ Addressed |
| MLflow lifecycle | ~~High~~ | ~~Medium~~ | Context manager pattern | ✅ Addressed |
| No timeout | ~~Medium~~ | ~~Medium~~ | Timeout config added | ✅ Addressed |
| No retry | ~~Medium~~ | ~~Medium~~ | Retry strategy defined | ✅ Addressed |
| No checkpointing | ~~Medium~~ | ~~Medium~~ | Checkpoint strategy added | ✅ Addressed |
| Async/sync conflict | Medium | Medium | nest_asyncio | ✅ Addressed |
| pctx-sandbox issues | Medium | Medium | Podman prereq documented | ⚠️ Runtime risk |
