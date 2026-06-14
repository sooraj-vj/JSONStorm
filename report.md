# JSONStorm Benchmark Report

**Experiment:** `5ad66d05`  
**Started:** 2026-05-21T00:13:25.095099  
**Status:** completed  

## Parameters

| Parameter | Value |
|-----------|-------|
| Db | `mathstackexchange_dev` |
| Prompt | `unknown` |
| Queries | `C:\Users\soora\Desktop\sooraj\uni\JSONStorm-experiments\JSONStorm\queries.jsonl` |
| Timeout Ms | `5000` |

## Query Outcomes

| Status  | Count | % | Bar |
|---------|------:|--:|-----|
| ✅ Success | 1 | 100.0% | `████████████████████` |
| ⏱️ Timeout | 0 | 0.0% | `░░░░░░░░░░░░░░░░░░░░` |
| ❌ Error   | 0   | 0.0% | `░░░░░░░░░░░░░░░░░░░░` |
| **Total** | **1** | | |

## Performance

| Metric | Value |
|--------|------:|
| Mean wall time | 50.0 ms |
| Max wall time | 50.0 ms |
| p50 (median) | 50.0 ms |
| p95 | 50.0 ms |
| p99 | 50.0 ms |
| Avg docs examined | 75,849 |
| Avg keys examined | 0 |

## Complexity Distribution

| Class | Count | % | Bar |
|-------|------:|--:|-----|
| 🟢 Low       | 0    | 0.0%    | `░░░░░░░░░░░░░░░░░░░░` |
| 🟡 Medium  | 0 | 0.0% | `░░░░░░░░░░░░░░░░░░░░` |
| 🔴 High      | 1   | 100.0%   | `████████████████████` |

> **Complexity mismatches** (static != execution): 0 (0.0%)

## Per-Query Results

| ID | Type | Status | Wall time | Docs examined | Complexity | Mismatch |
|----|------|--------|----------:|--------------:|------------|----------|
| Q1 | FIND | ✅ | 50.0 ms | 75,849 | 🔴 high |  |
