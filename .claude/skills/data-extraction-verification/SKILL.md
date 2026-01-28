---
name: data-extraction-verification
description: ALWAYS USE THIS SKILL when extracting numerical values from Treasury Bulletin tables. Systematic verification protocol to prevent common extraction errors like reading wrong tables, wrong metrics (outstanding vs sales vs redemptions), wrong time periods, or misread digits. Use BEFORE finalizing any extracted value.
---

# Data Extraction Verification Protocol

## Purpose

Prevent extraction errors that cause incorrect answers despite correct calculations. Common failure modes:
- Reading "amount outstanding" when question asks for "sales" or "redemptions"
- Extracting from wrong time period column (preliminary vs revised vs final)
- Misreading digits (3↔8, 1↔7, 6↔0)
- Regional aggregates vs individual country values
- Order of magnitude errors (261 vs 67,000)
- **Confusing STOCK metrics (balances) with FLOW metrics (transactions)**

## Stock vs Flow Metric Distinction (Critical)

This addresses 257x errors where "amount outstanding" (STOCK) is confused with "sales and redemptions" (FLOW).

**STOCK metrics** (point-in-time balances):
- Keywords: "amount outstanding", "balance", "holdings", "total debt", "position"
- Magnitude: Tens of thousands to hundreds of thousands (large absolute values)

**FLOW metrics** (period transactions):
- Keywords: "sales", "redemptions", "issues", "purchases", "maturities", "net change"
- Magnitude: Hundreds to low thousands (activity during a period)

**Parsing ambiguous phrases:**
- "sales and redemptions during [period]" → FLOW
- "outstanding amount of [type]" → STOCK
- "sales and redemptions outstanding during Q3" → FLOW (the transactions that occurred)
  - "sales and redemptions" = subject (metric type)
  - "outstanding during Q3" = time qualifier

**Verification rule:** If question contains "sales" or "redemptions", extract FLOW data, NOT outstanding balance—even if phrase includes "outstanding" as a modifier.

## Table Section Header Verification

Treasury Bulletin tables often have multiple SECTIONS (e.g., Table SB-2 has both "Sales and Redemptions" section AND "Amount Outstanding" section).

**Mandatory verification:**
- State: "Reading from Table [X], Section: [SECTION HEADER]"
- Verify SECTION header matches question's metric type (stock vs flow)
- If table has multiple sections, explicitly confirm you're in the correct one

**Anti-pattern:** Reading Table SB-2's "Amount Outstanding" section when question asks for "sales and redemptions" → should read from "Sales and Redemptions" section. The section header IS the metric type indicator.

## Verification Checklist

Before reporting ANY extracted value, verify each item:

### 1. Question Parsing Protocol

**Before extracting, decompose the question's metric phrase:**
1. Identify ALL metric keywords: [list each: "sales", "redemptions", "outstanding", etc.]
2. Determine primary metric type: Is this asking for STOCK or FLOW?
3. For compound phrases, identify:
   - Subject (the metric type to extract)
   - Modifiers/qualifiers (time period, conditions)
4. Match primary metric to table SECTION, not just table name

| Question keyword | Metric Type | Verify NOT reading |
|-----------------|-------------|-------------------|
| "sales" | FLOW | outstanding, redemptions |
| "redemptions" | FLOW | outstanding, sales |
| "outstanding" (alone) | STOCK | sales, redemptions |
| "issued" | FLOW | outstanding, redeemed |
| "yield" | RATE | price, return |
| "rate" | RATE | yield, price, level |

### 2. Table and Section Alignment
- State explicitly: "Reading from Table [X]: [table title]"
- State explicitly: "Section: [SECTION HEADER]"
- Confirm table subject matches question subject
- Confirm SECTION matches metric type (STOCK vs FLOW)
- If asked about "redemptions," verify you're NOT reading "amount outstanding" section

### 3. Time Period Verification
- State: "Reading column: [full header path]"
- For multi-level headers, trace full path (e.g., "1982 > Q3 > September > Interest-bearing")
- Watch for column markers: p = preliminary, r = revised
- Verify fiscal vs calendar year alignment

### 4. Coordinate Lock
State explicitly:
- Row header: "[exact row label]"
- Column header: "[exact column label]"
- Cell value: "[extracted value]"

### 5. Magnitude Sanity Check

**Red flag magnitudes:**
| If question asks for... | Expected magnitude |
|------------------------|-------------------|
| Monthly sales/redemptions (FLOW) | Hundreds to low thousands |
| Amount outstanding (STOCK) | Tens of thousands to hundreds of thousands |
| Yields/rates | Single digits with decimals |

Before finalizing:
- Does this magnitude match the expected category for the metric type?
- If expecting FLOW (~hundreds) and got STOCK (~tens of thousands), STOP and re-verify section
- Compare against adjacent values for plausibility

### 6. Cross-Reference (when available)
- Check if extracted value + siblings = stated total
- Compare against adjacent time periods
- Flag anomalies before proceeding

## Failure Pattern Recognition

**257x error pattern (Failure 2 type):** Extracted 67,238 when answer was 261
- Root cause: Read "Amount Outstanding" section instead of "Sales and Redemptions" section
- The agent extracted STOCK data when question asked for FLOW data
- Prevention: Parse question for primary metric type; verify table SECTION matches

**Digit misread pattern (Failure 1 type):** Extracted 103,235 when answer was 103,375
- Root cause: OCR artifact or adjacent column read
- Prevention: State exact row/column coordinates; verify against totals

**Wrong basis pattern (Failure 3 type):** Calculated 0.06 when answer was 0.24
- Root cause: Read wrong years, wrong bond type, or wrong yield metric
- Prevention: Verify all coordinates before calculating

## Quick Protocol

```
BEFORE extracting, complete ALL:
□ Question parsed: Primary metric = [keyword], Type = [STOCK/FLOW]
□ Table: [name/number] - matches question topic?
□ SECTION: [section header] - matches metric type?
□ Stock vs Flow: Question asks for [STOCK/FLOW], extracting [STOCK/FLOW]?
□ Row: [label]
□ Column: [label with full path]
□ Value: [number]
□ Magnitude check: Value [X] is [plausible/suspicious] for [STOCK/FLOW] metric
```

Only proceed with calculations after completing this checklist.
