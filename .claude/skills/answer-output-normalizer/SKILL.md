---
name: answer-output-normalizer
description: >
  Final answer formatting and source-document alignment for quantitative questions. MUST be triggered
  as the LAST step before returning any numeric answer. Use when: (1) returning percentages, ratios, or
  monetary values, (2) answering questions that reference a SPECIFIC chart/table/page, (3) any question
  asking "what percentage" or "what rate". Ensures answers match expected formats (bare numbers vs. units)
  and that document-specific classifications take precedence over external standards.
---

# Answer Output Normalizer

**CRITICAL: Apply this skill's rules AFTER computation but BEFORE returning the final answer.**

## Core Principle

**Correct calculation + wrong format = wrong answer.** This skill ensures the final output matches what the question expects.

## Numeric Formatting Rules

### Default: Strip Unit Symbols

Unless explicitly requested, return bare numeric values:

| Question Pattern | Correct Output | Wrong Output |
|-----------------|----------------|--------------|
| "What percentage of..." | `39.31` | `39.31%` |
| "What is the rate..." | `4.25` | `4.25%` |
| "How much in millions..." | `156.7` | `$156.7M` |
| "What fraction..." | `0.75` | `75%` or `3/4` |

### When to Include Units

ONLY include unit symbols if the question explicitly says:
- "include the unit"
- "express as a percentage with the % sign"
- "format as currency"
- Similar explicit formatting instructions

### Decimal Precision

| Value Type | Default Precision | Example |
|------------|-------------------|---------|
| Percentages | 2 decimal places | `39.31` |
| Ratios | Match source data precision | `0.3931` |
| Monetary (millions) | 0-1 decimal places | `156` or `156.7` |
| Monetary (exact) | 2 decimal places | `1234.56` |

Round using standard rules (half up).

## Source Document Primacy

### When Questions Reference Specific Sources

**CRITICAL:** When a question references a specific chart, table, or page:

```
"Based on the chart on page 21..."
"According to Table FO-1..."
"From the pie chart showing..."
```

**Use that document's own classification scheme**, not external standards.

### Decision Process

```
Does the question reference a SPECIFIC source (chart/table/page)?
├── YES → Read the source's headers, legends, footnotes
│   ├── Use the source's category definitions
│   ├── Do NOT impose external frameworks (OMB, Treasury, etc.)
│   └── If source groups items differently, follow source grouping
└── NO → Apply standard domain classifications
    └── (e.g., OMB A-11 for federal accounting)
```

### Example: Classification Conflict

**Question:** "Based on the chart on page 21, what percentage is service-related?"

**Source chart categories:**
- Personnel Services: $100M
- Program Operations: $200M
- Administrative: $50M

**Wrong approach:** Apply OMB A-11 object classes 10-25 definition
**Correct approach:** Use how the chart itself defines "service-related" (check legend, footnotes)

If the chart has a "Service-Related" segment showing 69%, the answer is `69`, not a recalculated value using external definitions.

## Pre-Output Checklist

Before returning ANY numeric answer:

1. **Format check**: Did I strip unnecessary unit symbols?
2. **Source check**: If question references a specific source, did I use that source's definitions?
3. **Precision check**: Is decimal precision appropriate?
4. **Range check**: Does the answer make sense? (percentages 0-100, ratios 0-1, etc.)
5. **Type match**: Does answer type match question type? (don't return % if absolute value asked)

## Quick Reference

| Question Type | Output Format | Example |
|--------------|---------------|---------|
| "What percentage..." | Bare number | `39.31` |
| "What is X as a percentage of Y" | Bare number | `25.5` |
| "What rate..." | Bare number | `4.75` |
| "How many..." | Integer or decimal | `42` or `42.5` |
| "What is the total in millions" | Number | `156` |
| "From chart X, what is..." | Match chart's representation | varies |

## Confidence Flag

If format is genuinely ambiguous (no clear expectation), note:
`[Format uncertain: returning bare numeric value as default]`

This signals that the format choice was a judgment call.
