# GEPA Integration Design v3
> OfficeQAAdapter implementation guidelines with existing assets integration

## 1. Architecture Overview

### 1.1 Adapter Role

The `OfficeQAAdapter` bridges GEPA's optimization loop with Claude Agent SDK execution:

```
GEPA optimize()
      │
      ├── Calls evaluate(candidate, minibatch)
      │         │
      │         ├── For each question:
      │         │   ├── Create ClaudeSDKClient with candidate prompt
      │         │   ├── Query Claude with question
      │         │   ├── Extract answer from response
      │         │   ├── Score with reward.py score_answer()
      │         │   └── Build trace
      │         │
      │         └── Return (average_score, traces)
      │
      └── Calls extract_traces_for_reflection(traces)
                └── Format failures for GEPA's reflection LLM
```

### 1.2 Key Design Principles

| Principle | Rationale |
|-----------|-----------|
| Use existing reward.py | Already handles all scoring edge cases |
| Sequential evaluation | Rate limit safety, simpler debugging |
| Async/sync bridge | GEPA is sync, Claude SDK is async |
| MLflow autolog | Native tracing, no custom instrumentation |
| Trace-based feedback | Rich failure information for reflection |

## 2. Data Integration

### 2.1 Loading OfficeQA Data

**Source**: `/data/officeqa/officeqa.csv`

**Data Structure**:
| Field | Type | Description |
|-------|------|-------------|
| uid | str | Unique ID like "UID0001" |
| question | str | Question text |
| answer | str | Ground truth answer |
| source_docs | str | Reference URL |
| difficulty | str | "easy" or "hard" |

**Loading Strategy**:
- Load CSV with pandas
- Create train/val/test splits (70/15/15)
- Stratify by difficulty to ensure balanced splits
- Each split is `List[dict]` where each dict has all columns

### 2.2 Scoring Integration

**Source**: `/data/officeqa/reward.py`

**Function**: `score_answer(ground_truth: str, predicted: str, tolerance: float = 0.0) -> float`

**What It Handles**:
- Percentage normalization (4.5% ↔ 0.045)
- Magnitude words (million, billion, trillion)
- Numeric tolerance for floating point
- Text matching for dates/names
- `<FINAL_ANSWER>` tag extraction
- Fuzzy matching with rationale

**Adapter Usage**:
```
from data.officeqa.reward import score_answer

score = score_answer(
    ground_truth=example["answer"],
    predicted=extracted_answer,
    tolerance=0.0
)  # Returns 1.0 or 0.0
```

## 3. GEPAAdapter Interface

### 3.1 Required Methods

| Method | Signature | Purpose |
|--------|-----------|---------|
| evaluate | `(candidate: List[str], minibatch: List[Any]) -> Tuple[float, List[Any]]` | Evaluate candidate prompt on batch |
| extract_traces_for_reflection | `(traces: List[Any], component_name: str) -> List[str]` | Format failures for GEPA reflection |

### 3.2 Candidate Format Handling

**GEPA's Internal Behavior**:
- `optimize()` receives `seed_candidate: dict` (e.g., `{"system_prompt": "..."}`)
- Internally converts to `List[str]` for `evaluate()` calls
- First element of list is the optimizable component

**Adapter Extraction**:
```
def evaluate(self, candidate: List[str], minibatch):
    system_prompt = candidate[0]  # First element is the system prompt
```

## 4. Claude SDK Integration

### 4.1 Client Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| model | claude-sonnet-4-20250514 | Balance of capability and cost |
| max_turns | 10 | Allow multi-step tool use |
| timeout | 60 seconds | Prevent hung requests |
| mcp_servers | Dynamic dict | Built-in + generated tools |

### 4.2 Response Processing (Addresses Medium Issue 2.1)

**Problem**: Multiple TextBlocks may exist; which contains the answer?

**Solution**: Look for explicit answer markers first, then fall back to last text

**Extraction Priority**:
1. Text containing `<FINAL_ANSWER>` tags (reward.py handles extraction)
2. Text containing explicit answer patterns ("The answer is...", "Result:")
3. Last TextBlock content as fallback

**Guidelines**:
- Concatenate all TextBlock content for scoring (reward.py extracts answer)
- Log full response for debugging
- Include extraction method in trace

### 4.3 Tool Routing

**allowed_tools Pattern**:
- Use explicit tool names, not wildcards (Addresses Medium Issue 2.4)
- Build list dynamically from registered MCP servers

```
allowed_tools = []
for server_name, server in self.mcp_servers.items():
    for tool in server.tools:
        allowed_tools.append(f"mcp__{server_name}__{tool.name}")
```

## 5. Async/Sync Bridge

### 5.1 Problem

| Component | Concurrency Model |
|-----------|-------------------|
| GEPA optimize() | Synchronous |
| GEPAAdapter.evaluate() | Must be synchronous |
| ClaudeSDKClient | Async (async with, await) |

### 5.2 Solution: nest_asyncio

**When to Apply**:
- Always apply at adapter initialization
- Safe even if not in nested loop context

**Pattern**:
```
import nest_asyncio
nest_asyncio.apply()

def evaluate(self, candidate, minibatch):
    return asyncio.run(self._evaluate_async(candidate, minibatch))
```

### 5.3 Event Loop Safety

| Environment | Behavior |
|-------------|----------|
| Script | asyncio.run() works normally |
| Jupyter | nest_asyncio allows nested loop |
| Existing loop | nest_asyncio patches to allow nesting |

## 6. Timeout and Retry Strategy (Addresses Issues 2.2, 2.3)

### 6.1 Timeout Configuration

| Level | Timeout | Purpose |
|-------|---------|---------|
| Per-question | 60s | Single Claude query |
| Per-batch | 600s | Full minibatch evaluation |
| Per-tool | 30s | Individual tool execution |

### 6.2 Retry Strategy

| Error Type | Retry? | Strategy |
|------------|--------|----------|
| 429 Rate Limit | Yes | Exponential backoff: 2s, 4s, 8s |
| 503 Service Unavailable | Yes | Fixed delay: 5s, 3 attempts |
| 500 Internal Error | Yes | 1 attempt after 2s |
| 400 Bad Request | No | Log and score as 0 |
| Timeout | Yes | 2 attempts with same timeout |
| Connection Error | Yes | 3 attempts with backoff |

### 6.3 Failure Scoring

When all retries exhausted:
- Score as 0.0
- Include error type in trace
- Continue with next question

## 7. Trace Data Structure

### 7.1 OfficeQATrace Fields

| Field | Type | Description |
|-------|------|-------------|
| question_id | str | UID from dataset |
| question | str | Question text |
| ground_truth | str | Expected answer |
| predicted | str | Model's answer |
| score | float | 0.0 or 1.0 |
| tool_calls | List[dict] | Tools invoked and results |
| response_time_ms | int | Total query time |
| error | Optional[str] | Error message if failed |
| extraction_method | str | How answer was extracted |

### 7.2 Tool Call Capture

Each tool call record:
| Field | Type | Description |
|-------|------|-------------|
| tool_name | str | MCP tool identifier |
| input | dict | Arguments passed |
| output | str | Tool result (truncated if large) |
| duration_ms | int | Execution time |
| error | Optional[str] | Tool error if any |

## 8. Reflection Formatting

### 8.1 extract_traces_for_reflection()

**Purpose**: Format failures for GEPA's reflection LLM to analyze

**Output Format** (per failed trace):
```
FAILED EVALUATION:
- Question ID: {uid}
- Question: {question}
- Expected Answer: {ground_truth}
- Model's Answer: {predicted}
- Score: 0.0
- Tools Used: {comma-separated tool names}
- Tool Calls: {count}
- Error: {error message or "None"}
- Extraction Method: {how answer was found}
```

### 8.2 Filtering

Only include traces where `score < 1.0` - successes don't need reflection.

### 8.3 component_name Parameter

GEPA uses this to identify which component to reflect on. For our case:
- Always "system_prompt" since that's what we're optimizing
- Reflection focuses on how to improve the prompt based on failure patterns

## 9. MCP Server Management

### 9.1 Built-in Tools

| Server Name | Purpose | Tools |
|-------------|---------|-------|
| treasury | Access treasury bulletin data | query_treasury, get_bulletin |

### 9.2 Dynamic Tool Injection

**When**: After Tool Crafter generates a new tool

**Process**:
1. Tool Crafter outputs MCP server package
2. Import server from generated code
3. Add to adapter's mcp_servers dict
4. Rebuild allowed_tools list
5. Next evaluate() uses new tools

### 9.3 Hot-Reload Consideration (Addresses Medium Issue 2.8)

**Current Design**: Recreate ClaudeSDKClient per evaluation
- Each evaluate() call creates fresh client with current tools
- No hot-reload needed since client is recreated
- Slight performance cost, but simpler and safer

## 10. MLflow Integration

### 10.1 Autolog Setup

**Once at startup**:
```
import mlflow.anthropic
mlflow.anthropic.autolog()
```

**What's Automatically Traced**:
- All ClaudeSDKClient API calls
- Token usage per call
- Response content
- Tool invocations (within spans)

### 10.2 Custom Metrics Logging

In addition to autolog, adapter logs:
- Batch score (average of minibatch)
- Failed question count
- Total tool calls in batch
- Cumulative token usage

### 10.3 Run Context

Adapter operates within MLflow run started by Orchestrator:
- Adapter doesn't start its own run
- All traces nested under Orchestrator's run
- Token usage aggregates at run level

## 11. Error Handling

### 11.1 Exception Hierarchy

| Exception | Handling |
|-----------|----------|
| anthropic.RateLimitError | Retry with backoff |
| anthropic.APITimeoutError | Retry once |
| anthropic.APIError | Log, score 0, continue |
| asyncio.TimeoutError | Score 0, log timeout |
| Exception (any) | Log, score 0, continue batch |

### 11.2 Batch Resilience

One question failure should NOT fail the entire batch:
- Try/except per question
- Failed questions scored as 0
- Trace includes error information
- Batch continues with remaining questions

## 12. Performance Considerations

### 12.1 Sequential vs Parallel

**Current Design**: Sequential (one question at a time)

**Rationale**:
- Respects Claude API rate limits
- Simpler debugging and tracing
- Predictable resource usage
- MLflow traces are cleaner

**Future Option**: Semaphore-controlled parallelism
- Limit concurrent requests to N (e.g., 3)
- Requires careful trace management

### 12.2 Caching

No response caching by default:
- Each evaluation should reflect current system prompt
- Deterministic evaluation more important than speed

## 13. Testing Strategy

### 13.1 Unit Tests

| Test | Purpose |
|------|---------|
| test_score_integration | Verify reward.py scoring works |
| test_candidate_extraction | Verify List[str] → prompt conversion |
| test_response_extraction | Verify answer extraction logic |
| test_trace_formatting | Verify reflection format |

### 13.2 Integration Tests

| Test | Purpose |
|------|---------|
| test_single_question | End-to-end single question evaluation |
| test_batch_evaluation | Minibatch with mix of easy/hard |
| test_tool_injection | Add tool and verify usage |
| test_failure_handling | Verify graceful degradation |

## 14. Configuration

### 14.1 Adapter Configuration

```yaml
adapter:
  claude_model: "claude-sonnet-4-20250514"
  max_turns: 10
  timeout_seconds: 60
  retry:
    max_attempts: 3
    base_delay_seconds: 2
    max_delay_seconds: 30
  batch_size: 10  # Questions per minibatch
```

### 14.2 Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| ANTHROPIC_API_KEY | Yes | Claude API authentication |

## 15. Dependency on dspy-ai (Addresses Medium Issue 2.9)

### 15.1 Why dspy-ai?

GEPA uses `dspy.LM()` as the reflection LLM interface.

### 15.2 Minimal Usage

Only used for:
```
reflection_lm = dspy.LM(
    model="anthropic/claude-sonnet-4-20250514",
    api_key=os.environ["ANTHROPIC_API_KEY"]
)
```

### 15.3 Alternative Investigation

Could potentially use anthropic SDK directly if GEPA supports raw Anthropic client. Recommend testing with GEPA to confirm interface flexibility.
