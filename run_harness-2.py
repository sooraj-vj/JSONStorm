"""
run_harness.py
Runs MongoDB find AND aggregate queries defined in a JSONL file, records
execution times, explain stats, and execution-based complexity classification.
Results are saved to a JSONL output file, and tracked as a Yanex experiment.

Usage (standalone — no tracking):
    python run_harness.py --queries queries.jsonl --db mathstackexchange

What gets tracked in yanex:
    Parameters : db, queries file, timeout, prompt label
    Per-query metrics (step=query_index):
        wall_time_ms, docs_examined, keys_examined, n_returned
    End-of-run summary metrics:
        success_rate, timeout_rate, error_rate,
        p50/p95/p99 wall time, complexity distribution
    Artifacts:
        results JSONL (full raw output)
        queries JSONL (the query definitions that were run)
        queries_failed JSONL (query definitions that failed or timed out)
        schema.txt (the schema used to generate the queries, if present)
"""

import argparse
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 stdout on Windows so unicode in print() never raises UnicodeEncodeError
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import yanex
from pymongo import MongoClient
from pymongo.errors import OperationFailure, ExecutionTimeout

MAX_RESULTS_TO_STORE = 10


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------

def make_json_safe(obj):
    """Recursively convert datetimes and ObjectIds to JSON-serialisable types."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    type_name = type(obj).__name__
    if type_name in ("ObjectId", "Decimal128", "Binary", "Code"):
        return str(obj)
    return obj


def parse_extended_json(obj):
    """Convert {"$date": "..."} values in filter/sort dicts to datetime objects."""
    if isinstance(obj, dict):
        if "$date" in obj and len(obj) == 1:
            date_val = obj["$date"]
            if isinstance(date_val, str):
                return datetime.fromisoformat(date_val.replace("Z", "+00:00"))
            if isinstance(date_val, (int, float)):
                return datetime.fromtimestamp(date_val / 1000, tz=timezone.utc)
        return {k: parse_extended_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [parse_extended_json(item) for item in obj]
    return obj


# ---------------------------------------------------------------------------
# Stage walking — extracts a quantitative feature vector
# ---------------------------------------------------------------------------

def walk_execution_stages(stage: dict, depth: int = 0) -> dict:
    """
    Recursively walk MongoDB's explain() stage tree and collect
    quantitative features — counts of every operator type, predicate
    counts, expression depth, memory, etc.
    """
    f = {
        "n_collscan":           0,
        "n_ixscan":             0,
        "n_fetch":              0,
        "n_lookup":             0,
        "n_unwind":             0,
        "n_group":              0,
        "n_sort":               0,
        "n_window":             0,
        "n_facet":              0,
        "n_graphlookup":        0,
        "n_union":              0,
        "n_project":            0,
        "n_addfields":          0,
        "n_match":              0,
        "n_limit":              0,
        "n_skip":               0,
        "total_operators":      0,
        "max_depth":            depth,
        "pipeline_stages":      [],
        "n_predicates":         0,
        "n_or_clauses":         0,
        "n_and_clauses":        0,
        "n_regex_predicates":   0,
        "n_expr_predicates":    0,
        "n_elemMatch":          0,
        "n_projected_fields":   0,
        "n_computed_fields":    0,
        "n_lookup_pipelines":   0,
        "n_lookup_conditions":  0,
        "n_sort_keys":          0,
        "n_window_functions":   0,
        "n_window_partitions":  0,
        "n_facet_branches":     0,
        "total_docs_examined":  0,
        "total_keys_examined":  0,
        "max_memory_bytes":     0,
    }

    stage_name = (
        stage.get("stage") or stage.get("stageName") or "UNKNOWN"
    ).upper()
    f["pipeline_stages"].append(stage_name)
    f["total_operators"] += 1

    operator_map = {
        "COLLSCAN":         "n_collscan",
        "IXSCAN":           "n_ixscan",
        "IDHACK":           "n_ixscan",
        "FETCH":            "n_fetch",
        "EQ_LOOKUP":        "n_lookup",
        "$LOOKUP":          "n_lookup",
        "$UNWIND":          "n_unwind",
        "$GROUP":           "n_group",
        "SORT":             "n_sort",
        "$SORT":            "n_sort",
        "$SETWINDOWFIELDS": "n_window",
        "$FACET":           "n_facet",
        "$GRAPHLOOKUP":     "n_graphlookup",
        "$UNIONWITH":       "n_union",
        "PROJECTION":       "n_project",
        "$PROJECT":         "n_project",
        "$ADDFIELDS":       "n_addfields",
        "$MATCH":           "n_match",
        "MATCH":            "n_match",
        "$LIMIT":           "n_limit",
        "LIMIT":            "n_limit",
        "$SKIP":            "n_skip",
        "SKIP":             "n_skip",
    }
    for marker, field in operator_map.items():
        if marker in stage_name:
            f[field] += 1

    for filter_key in ("filter", "query"):
        if filter_key in stage:
            _count_predicates(stage[filter_key], f)

    if "SORT" in stage_name:
        sort_pattern = stage.get("sortPattern", {})
        f["n_sort_keys"] += len(sort_pattern) if isinstance(sort_pattern, dict) else 0

    if "SETWINDOW" in stage_name:
        output = stage.get("output", {})
        f["n_window_functions"] += len(output) if isinstance(output, dict) else 0
        if stage.get("partitionBy") is not None:
            f["n_window_partitions"] += 1

    if "FACET" in stage_name:
        facet_doc = stage.get("facets", stage.get("$facet", {}))
        if isinstance(facet_doc, dict):
            f["n_facet_branches"] += len(facet_doc)

    if "LOOKUP" in stage_name:
        if "pipeline" in stage:
            f["n_lookup_pipelines"] += 1
        if "let" in stage:
            f["n_lookup_conditions"] += len(stage.get("let", {}))

    if "PROJECT" in stage_name or "ADDFIELDS" in stage_name:
        spec = stage.get("transformBy", stage.get("fields", {}))
        if isinstance(spec, dict):
            f["n_projected_fields"] += len(spec)
            f["n_computed_fields"]  += sum(
                1 for v in spec.values() if isinstance(v, dict)
            )

    f["total_docs_examined"] += stage.get("docsExamined", 0)
    f["total_keys_examined"] += stage.get("keysExamined", 0)
    f["max_memory_bytes"] = max(
        f["max_memory_bytes"],
        stage.get("memUsage", 0),
        stage.get("maxMemoryUsageBytes", 0),
        stage.get("spilledDataStorageSize", 0),
    )

    children = []
    if "inputStage" in stage:
        children = [stage["inputStage"]]
    elif "inputStages" in stage:
        children = stage["inputStages"]
    for key in ("stages", "innerStage", "outerStage"):
        if key in stage:
            val = stage[key]
            children += val if isinstance(val, list) else [val]

    for child in children:
        if not isinstance(child, dict):
            continue
        child_f = walk_execution_stages(child, depth + 1)
        for key in f:
            if key == "pipeline_stages":
                f["pipeline_stages"] += child_f["pipeline_stages"]
            elif key == "max_depth":
                f["max_depth"] = max(f["max_depth"], child_f["max_depth"])
            elif key == "max_memory_bytes":
                f["max_memory_bytes"] = max(
                    f["max_memory_bytes"], child_f["max_memory_bytes"]
                )
            elif isinstance(f[key], (int, float)):
                f[key] += child_f.get(key, 0)

    return f


def _count_predicates(filter_doc: dict, f: dict) -> None:
    """
    Recursively count predicate complexity inside a $match filter document.

    Counts:
      n_predicates       — total leaf conditions
      n_or_clauses       — number of branches in $or expressions
      n_and_clauses      — number of branches in $and expressions
      n_regex_predicates — $regex usages
      n_expr_predicates  — $expr usages
      n_elemMatch        — $elemMatch usages
    """
    if not isinstance(filter_doc, dict):
        return

    for key, value in filter_doc.items():
        if key == "$or":
            f["n_or_clauses"] += len(value) if isinstance(value, list) else 1
            for branch in (value if isinstance(value, list) else [value]):
                _count_predicates(branch, f)
        elif key == "$and":
            f["n_and_clauses"] += len(value) if isinstance(value, list) else 1
            for branch in (value if isinstance(value, list) else [value]):
                _count_predicates(branch, f)
        elif key == "$nor":
            for branch in (value if isinstance(value, list) else [value]):
                _count_predicates(branch, f)
        elif key == "$regex":
            f["n_regex_predicates"] += 1
            f["n_predicates"]       += 1
        elif key == "$expr":
            f["n_expr_predicates"] += 1
            f["n_predicates"]      += 1
        elif key == "$elemMatch":
            f["n_elemMatch"]   += 1
            f["n_predicates"]  += 1
            _count_predicates(value, f)
        elif key.startswith("$"):
            f["n_predicates"] += 1
        else:
            if isinstance(value, dict):
                _count_predicates(value, f)
            else:
                f["n_predicates"] += 1


# ---------------------------------------------------------------------------
# Complexity thresholds
# ---------------------------------------------------------------------------

COMPLEXITY_THRESHOLDS = {
    "high": {
        "n_window":             1,
        "n_facet":              1,
        "n_graphlookup":        1,
        "n_union":              1,
        "n_lookup":             4,
        "n_group":              4,
        "total_operators":     15,
        "n_predicates":        20,
        "max_memory_bytes":    100 * 1024 * 1024,
    },
    "medium": {
        "n_lookup":             1,
        "n_group":              1,
        "n_unwind":             1,
        "total_operators":      5,
        "n_predicates":         5,
        "n_computed_fields":    3,
    },
}

HIGH_COMBINATIONS = [
    ("n_lookup",  "n_group"),
    ("n_lookup",  "n_window"),
    ("n_unwind",  "n_group"),
    ("n_group",   "n_sort"),
]


# ---------------------------------------------------------------------------
# Classification — derived from the quantitative feature vector
# ---------------------------------------------------------------------------

def classify_from_execution(
    explain_result: dict,
    wall_time_ms:   float,
    query_type:     str,
) -> tuple[str, list[str], dict]:
    """
    Derive complexity class from a quantitative feature vector.
    Returns (class, reasons, features).
    """
    exec_stats = explain_result.get("executionStats", {})

    if query_type == "aggregate":
        raw_stages = explain_result.get("stages", [])
        root_stage = raw_stages[0] if raw_stages else {}
    else:
        root_stage = exec_stats.get("executionStages", {})

    f = walk_execution_stages(root_stage)
    f["wall_time_ms"] = wall_time_ms

    f["total_docs_examined"] = max(
        f["total_docs_examined"],
        exec_stats.get("totalDocsExamined", 0),
    )
    f["total_keys_examined"] = max(
        f["total_keys_examined"],
        exec_stats.get("totalKeysExamined", 0),
    )

    reasons = []

    for feature, threshold in COMPLEXITY_THRESHOLDS["high"].items():
        val = f.get(feature, 0)
        if val >= threshold:
            reasons.append(f"{feature}={val} >= {threshold} (high threshold)")

    for feat_a, feat_b in HIGH_COMBINATIONS:
        if f.get(feat_a, 0) > 0 and f.get(feat_b, 0) > 0:
            reasons.append(
                f"{feat_a}={f[feat_a]} + {feat_b}={f[feat_b]} (dangerous combination)"
            )

    if reasons:
        return "high", reasons, f

    for feature, threshold in COMPLEXITY_THRESHOLDS["medium"].items():
        val = f.get(feature, 0)
        if val >= threshold:
            reasons.append(f"{feature}={val} >= {threshold} (medium threshold)")

    if reasons:
        return "medium", reasons, f

    return "low", [
        f"basic query — operators={f['total_operators']}, "
        f"predicates={f['n_predicates']}, lookups={f['n_lookup']}"
    ], f


# ---------------------------------------------------------------------------
# Explain stat extraction (flat summary for quick analysis)
# ---------------------------------------------------------------------------

def extract_flat_stats(explain_result: dict, query_type: str) -> dict:
    stats = {
        "executionTimeMillis": None,
        "totalDocsExamined":   None,
        "totalKeysExamined":   None,
        "nReturned":           None,
        "winningPlanStage":    None,
        "indexUsed":           None,
    }

    if query_type == "find":
        es = explain_result.get("executionStats", {})
        stats["executionTimeMillis"] = es.get("executionTimeMillis")
        stats["totalDocsExamined"]   = es.get("totalDocsExamined")
        stats["totalKeysExamined"]   = es.get("totalKeysExamined")
        stats["nReturned"]           = es.get("nReturned")

        winning_plan = explain_result.get("queryPlanner", {}).get("winningPlan", {})
        stats["winningPlanStage"] = winning_plan.get("stage")

        def find_index(plan):
            if isinstance(plan, dict):
                if plan.get("stage") == "IXSCAN":
                    return plan.get("indexName")
                for v in plan.values():
                    r = find_index(v)
                    if r:
                        return r
            return None

        stats["indexUsed"] = find_index(winning_plan)

    else:
        es = explain_result.get("executionStats", {})
        stats["executionTimeMillis"] = es.get("executionTimeMillis")
        stats["nReturned"]           = es.get("nReturned")
        raw_stages = explain_result.get("stages", [])
        if raw_stages:
            first = raw_stages[0]
            stats["winningPlanStage"] = (
                first.get("$cursor", {})
                     .get("queryPlanner", {})
                     .get("winningPlan", {})
                     .get("stage")
            )

    return stats


# ---------------------------------------------------------------------------
# Per-query runners
# ---------------------------------------------------------------------------

def run_find_query(collection, query_def: dict, timeout_ms: int):
    if "filter" not in query_def:
        return None, None, "error", "find query is missing 'filter' field", []

    filter_doc = parse_extended_json(query_def.get("filter", {}))
    projection = parse_extended_json(query_def.get("projection")) or None
    sort       = list(query_def["sort"].items()) if "sort" in query_def else None
    limit      = query_def.get("limit", 0)

    def make_cursor():
        cur = collection.find(filter_doc, projection)
        if sort:
            cur = cur.sort(sort)
        if limit:
            cur = cur.limit(limit)
        if timeout_ms:
            cur = cur.max_time_ms(timeout_ms)
        return cur

    try:
        cmd = {"find": collection.name, "filter": filter_doc}
        if projection:
            cmd["projection"] = projection
        if sort:
            cmd["sort"] = dict(sort)
        if limit:
            cmd["limit"] = limit
        if timeout_ms:
            cmd["maxTimeMS"] = timeout_ms

        explain_result = collection.database.command(
            "explain", cmd, verbosity="executionStats"
        )
    except OperationFailure as e:
        if "exceeded" in str(e).lower() or e.code == 50:
            return None, None, "timeout", f"Exceeded {timeout_ms}ms during explain", []
        return None, None, "error", str(e), []

    flat_stats = extract_flat_stats(explain_result, "find")

    try:
        t0   = time.perf_counter()
        docs = list(make_cursor())
        wall = round((time.perf_counter() - t0) * 1000, 3)
    except ExecutionTimeout:
        return explain_result, flat_stats, "timeout", f"Exceeded {timeout_ms}ms during execution", []
    except OperationFailure as e:
        return explain_result, flat_stats, "error", str(e), []

    flat_stats["nReturned"] = len(docs)

    sample = [make_json_safe({**d, "_id": str(d["_id"])} if "_id" in d else d)
              for d in docs[:MAX_RESULTS_TO_STORE]]

    return explain_result, flat_stats, "success", wall, sample


def run_aggregate_query(collection, query_def: dict, timeout_ms: int):
    if "pipeline" not in query_def:
        return None, None, "error", "aggregate query is missing 'pipeline' field", []
    if not isinstance(query_def["pipeline"], list):
        return None, None, "error", f"'pipeline' must be a list, got {type(query_def['pipeline']).__name__}", []

    pipeline = parse_extended_json(query_def.get("pipeline", []))

    options = {}
    if timeout_ms:
        options["maxTimeMS"] = timeout_ms
    if query_def.get("allowDiskUse", False):
        options["allowDiskUse"] = True

    try:
        explain_result = collection.database.command(
            "aggregate",
            collection.name,
            pipeline=pipeline,
            explain=True,
            **options,
        )
    except OperationFailure as e:
        if "exceeded" in str(e).lower() or e.code == 50:
            return None, None, "timeout", f"Exceeded {timeout_ms}ms during explain", []
        return None, None, "error", str(e), []

    flat_stats = extract_flat_stats(explain_result, "aggregate")

    try:
        t0   = time.perf_counter()
        docs = list(collection.aggregate(pipeline, **options))
        wall = round((time.perf_counter() - t0) * 1000, 3)
    except OperationFailure as e:
        if "exceeded" in str(e).lower() or e.code == 50:
            return explain_result, flat_stats, "timeout", f"Exceeded {timeout_ms}ms during execution", []
        return explain_result, flat_stats, "error", str(e), []

    flat_stats["nReturned"] = len(docs)

    sample = [make_json_safe({**d, "_id": str(d["_id"])} if "_id" in d else d)
              for d in docs[:MAX_RESULTS_TO_STORE]]

    return explain_result, flat_stats, "success", wall, sample


# ---------------------------------------------------------------------------
# Unified result builder
# ---------------------------------------------------------------------------

def run_query(collection, query_def: dict, timeout_ms: int) -> dict:
    """
    Dispatch to find or aggregate runner, then attach execution classification.
    Any unexpected exception is caught and logged as an error result —
    the harness will never crash due to a single bad query.
    """
    query_type = "aggregate" if "pipeline" in query_def else "find"

    result = {
        "id":                           query_def.get("id", "UNKNOWN"),
        "description":                  query_def.get("description", ""),
        "collection":                   query_def.get("collection", ""),
        "query_type":                   query_type,
        "status":                       None,
        "wall_time_ms":                 None,
        "executionTimeMillis":          None,
        "totalDocsExamined":            None,
        "totalKeysExamined":            None,
        "nReturned":                    None,
        "winningPlanStage":             None,
        "indexUsed":                    None,
        "static_complexity":            query_def.get("complexity"),
        "execution_complexity":         None,
        "execution_complexity_reasons": None,
        "execution_features":           None,
        "complexity_mismatch":          None,
        "error":                        None,
        "sample_results":               [],
        "timestamp":                    datetime.now(tz=timezone.utc).isoformat(),
    }

    try:
        if query_type == "find":
            outcome = run_find_query(collection, query_def, timeout_ms)
        else:
            outcome = run_aggregate_query(collection, query_def, timeout_ms)

        explain_result, flat_stats, status, *rest = outcome
        result["status"] = status

        if status == "success":
            wall_time_ms, sample = rest
            result["wall_time_ms"]   = wall_time_ms
            result["sample_results"] = sample
            if flat_stats:
                result.update(flat_stats)

            if explain_result:
                exec_class, exec_reasons, exec_features = classify_from_execution(
                    explain_result, wall_time_ms, query_type
                )
                result["execution_complexity"]         = exec_class
                result["execution_complexity_reasons"] = exec_reasons
                result["execution_features"] = {
                    k: v for k, v in exec_features.items()
                    if not isinstance(v, set)
                }
                if result["static_complexity"] and exec_class != result["static_complexity"]:
                    result["complexity_mismatch"] = True
        else:
            result["error"] = rest[0]

    except Exception as e:
        result["status"] = "error"
        result["error"]  = f"Unhandled exception: {type(e).__name__}: {e}"
        print(f"\n[ERROR] Unexpected crash on query {result['id']}: {e}")

    return result


# ---------------------------------------------------------------------------
# JSONL loader — handles concatenated objects, blank lines, malformed lines
# ---------------------------------------------------------------------------

def load_queries(queries_file: str) -> list[dict]:
    query_defs = []
    skipped    = 0

    with open(queries_file, encoding="utf-8") as f:
        raw = f.read()

    depth  = 0
    start  = None
    in_str = False
    escape = False

    for i, ch in enumerate(raw):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                chunk = raw[start : i + 1]
                try:
                    query_defs.append(json.loads(chunk))
                except json.JSONDecodeError as e:
                    skipped += 1
                    print(f"[WARN] Skipping malformed JSON object "
                          f"(chars {start}-{i}): {e}")
                start = None

    if skipped:
        print(f"[WARN] Skipped {skipped} malformed object(s) in {queries_file}")

    return query_defs


# ---------------------------------------------------------------------------
# Numeric features list (for summary metrics)
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "total_operators",
    "n_collscan", "n_ixscan", "n_fetch",
    "n_lookup", "n_unwind", "n_group", "n_sort",
    "n_window", "n_facet", "n_graphlookup", "n_union",
    "n_project", "n_addfields", "n_match", "n_limit", "n_skip",
    "n_predicates", "n_or_clauses", "n_and_clauses",
    "n_regex_predicates", "n_expr_predicates", "n_elemMatch",
    "n_projected_fields", "n_computed_fields",
    "n_sort_keys", "n_window_functions", "n_window_partitions",
    "n_facet_branches", "n_lookup_pipelines", "n_lookup_conditions",
    "total_docs_examined", "total_keys_examined",
    "max_depth", "max_memory_bytes",
]


def _percentile(sorted_values: list, pct: float) -> float:
    if not sorted_values:
        return 0.0
    k  = (len(sorted_values) - 1) * pct / 100
    lo = int(k)
    hi = min(int(k) + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


# ---------------------------------------------------------------------------
# Summary metrics — per-class feature averages (SQLStorm Table 6 equivalent)
# ---------------------------------------------------------------------------

def compute_summary_metrics(all_results: list) -> dict:
    total = len(all_results)
    if total == 0:
        return {}

    counts       = {"success": 0, "timeout": 0, "error": 0}
    class_counts = {"low": 0, "medium": 0, "high": 0}
    mismatches   = 0
    wall_times   = []

    classes = ("low", "medium", "high", "all")
    sums    = {cls: {feat: 0.0 for feat in NUMERIC_FEATURES} for cls in classes}
    ns      = {cls: 0 for cls in classes}

    for r in all_results:
        status = r.get("status", "error")
        counts[status] = counts.get(status, 0) + 1

        if status != "success":
            continue

        ec = r.get("execution_complexity") or "low"
        class_counts[ec] = class_counts.get(ec, 0) + 1

        if r.get("complexity_mismatch"):
            mismatches += 1
        if r.get("wall_time_ms") is not None:
            wall_times.append(r["wall_time_ms"])

        feats = r.get("execution_features") or {}
        if feats:
            for feat in NUMERIC_FEATURES:
                val = feats.get(feat) or 0
                sums[ec][feat]    += val
                sums["all"][feat] += val
            ns[ec]    += 1
            ns["all"] += 1

    wall_times.sort()

    summary = {
        "success_rate":        counts["success"]  / total,
        "timeout_rate":        counts["timeout"]  / total,
        "error_rate":          counts["error"]    / total,
        "mismatch_rate":       mismatches / max(counts["success"], 1),
        "n_total":             total,
        "n_success":           counts["success"],
        "n_timeout":           counts["timeout"],
        "n_error":             counts["error"],
        "n_mismatches":        mismatches,
        "n_complexity_low":    class_counts.get("low",    0),
        "n_complexity_medium": class_counts.get("medium", 0),
        "n_complexity_high":   class_counts.get("high",   0),
        "wall_time_p50":       _percentile(wall_times, 50),
        "wall_time_p95":       _percentile(wall_times, 95),
        "wall_time_p99":       _percentile(wall_times, 99),
        "wall_time_mean":      sum(wall_times) / len(wall_times) if wall_times else 0.0,
    }

    # Per-class and overall feature averages — equivalent to SQLStorm Table 6
    for cls in classes:
        n      = ns[cls]
        suffix = f"_{cls}" if cls != "all" else ""
        for feat in NUMERIC_FEATURES:
            summary[f"avg_{feat}{suffix}"] = sums[cls][feat] / n if n > 0 else 0.0

    return summary


# ---------------------------------------------------------------------------
# Harness loop
# ---------------------------------------------------------------------------

def run_harness(queries_file, db_name, uri, output_file, failed_output_file, timeout_ms, prompt_label):
    print(f"Connecting to {uri} ...")
    client = MongoClient(uri)
    db     = client[db_name]
    print(f"Database     : {db_name}")
    print(f"Queries      : {queries_file}")
    print(f"Output       : {output_file}")
    print(f"Failed output: {failed_output_file}")
    print(f"Timeout      : {timeout_ms} ms\n")

    yanex.log_metrics({
        "param_db":          db_name,
        "param_queries":     str(queries_file),
        "param_timeout_ms":  timeout_ms,
        "param_prompt":      prompt_label,
    })

    query_defs = load_queries(queries_file)
    print(f"Loaded {len(query_defs)} queries\n")

    os.makedirs(
        os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
        exist_ok=True,
    )
    os.makedirs(
        os.path.dirname(failed_output_file) if os.path.dirname(failed_output_file) else ".",
        exist_ok=True,
    )

    all_results       = []
    counts            = {"success": 0, "timeout": 0, "error": 0}
    complexity_counts = {"low": 0, "medium": 0, "high": 0}
    mismatches        = 0

    with open(output_file, "w", encoding="utf-8") as out_f, \
         open(failed_output_file, "w", encoding="utf-8") as failed_f:

        for i, query_def in enumerate(query_defs, 1):
            if not query_def.get("collection"):
                print(f"  [{i}/{len(query_defs)}] SKIP {query_def.get('id')} — no collection")
                continue

            collection  = db[query_def["collection"]]
            qtype_label = "AGG " if "pipeline" in query_def else "FIND"
            print(
                f"  [{i}/{len(query_defs)}] {query_def['id']:8s} {qtype_label} "
                f"{query_def.get('description', '')[:45]}",
                end="  ",
            )

            result = run_query(collection, query_def, timeout_ms)
            counts[result["status"]] += 1
            all_results.append(result)

            if result["status"] == "success":
                ec = result["execution_complexity"] or "?"
                complexity_counts[ec] = complexity_counts.get(ec, 0) + 1
                mismatch_flag = " !" if result.get("complexity_mismatch") else "  "
                if result.get("complexity_mismatch"):
                    mismatches += 1
                feats = result.get("execution_features") or {}
                print(
                    f"OK  {result['wall_time_ms']:>8.1f} ms  "
                    f"returned={result['nReturned']}  "
                    f"examined={result['totalDocsExamined']}  "
                    f"ops={feats.get('total_operators', '?')}  "
                    f"lookups={feats.get('n_lookup', 0)}  "
                    f"groups={feats.get('n_group', 0)}  "
                    f"preds={feats.get('n_predicates', 0)}  "
                    f"complexity={ec}{mismatch_flag}"
                )
                yanex.log_metrics({
                    "wall_time_ms":             result["wall_time_ms"],
                    "docs_examined":            result.get("totalDocsExamined") or 0,
                    "keys_examined":            result.get("totalKeysExamined") or 0,
                    "n_returned":               result.get("nReturned") or 0,
                    "execution_complexity_num": {"low": 1, "medium": 2, "high": 3}.get(
                                                    result.get("execution_complexity"), 0),
                    "is_mismatch":              int(bool(result.get("complexity_mismatch"))),
                    "total_operators":          feats.get("total_operators", 0),
                    "n_lookup":                 feats.get("n_lookup", 0),
                    "n_group":                  feats.get("n_group", 0),
                    "n_unwind":                 feats.get("n_unwind", 0),
                    "n_sort":                   feats.get("n_sort", 0),
                    "n_window":                 feats.get("n_window", 0),
                    "n_facet":                  feats.get("n_facet", 0),
                    "n_predicates":             feats.get("n_predicates", 0),
                    "n_or_clauses":             feats.get("n_or_clauses", 0),
                    "n_computed_fields":        feats.get("n_computed_fields", 0),
                    "max_depth":                feats.get("max_depth", 0),
                }, step=i)

            elif result["status"] == "timeout":
                print("TIMEOUT")
                yanex.log_metrics({"timeout": 1}, step=i)
                failed_entry = {
                    **query_def,
                    "failure_reason": "timeout",
                    "error":          result.get("error"),
                }
                failed_f.write(json.dumps(failed_entry) + "\n")
                failed_f.flush()

            else:
                print(f"ERROR  {result['error']}")
                yanex.log_metrics({"error": 1}, step=i)
                failed_entry = {
                    **query_def,
                    "failure_reason": "error",
                    "error":          result.get("error"),
                }
                failed_f.write(json.dumps(failed_entry) + "\n")
                failed_f.flush()

            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

    summary = compute_summary_metrics(all_results)
    yanex.log_metrics(summary)

    print(f"\n{'=' * 65}")
    print(f"Results   : {counts['success']} success, "
          f"{counts['timeout']} timeout, {counts['error']} error")
    print(f"Complexity: {complexity_counts}")
    print(f"Mismatches: {mismatches} queries where static != execution class")
    print(f"p50/p95/p99 wall time: "
          f"{summary['wall_time_p50']:.1f} / "
          f"{summary['wall_time_p95']:.1f} / "
          f"{summary['wall_time_p99']:.1f} ms")
    print(f"Saved to  : {output_file}")
    print(f"Failed to : {failed_output_file}")

    yanex.copy_artifact(output_file,        "results.jsonl")
    yanex.copy_artifact(queries_file,       "queries.jsonl")
    yanex.copy_artifact(failed_output_file, "queries_failed.jsonl")

    schema_path = Path("schema.txt")
    if schema_path.exists():
        yanex.copy_artifact(str(schema_path), "schema.txt")

    client.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    params = yanex.get_params()

    parser = argparse.ArgumentParser(description="MongoDB query benchmark harness")
    parser.add_argument("--queries",
                        default=params.get("queries", "queries.jsonl"))
    parser.add_argument("--db",
                        default=params.get("db", "mathstackexchange"))
    parser.add_argument("--uri",
                        default=params.get("uri", "mongodb://localhost:27017"))
    parser.add_argument("--out",
                        default=params.get("out", "results/results.jsonl"))
    parser.add_argument("--failed-out",
                        default=params.get("failed_out", "results/queries_failed.jsonl"),
                        help="Path to write failed query definitions to")
    parser.add_argument("--timeout",
                        type=int,
                        default=int(params.get("timeout", 10_000)),
                        help="Per-query timeout in milliseconds")
    parser.add_argument("--prompt",
                        default=params.get("prompt", "unknown"),
                        help="Label for the prompt set used (e.g. P1, P2) — stored as metadata")
    args = parser.parse_args()

    run_harness(
        queries_file=args.queries,
        db_name=args.db,
        uri=args.uri,
        output_file=args.out,
        failed_output_file=args.failed_out,
        timeout_ms=args.timeout,
        prompt_label=args.prompt,
    )


if __name__ == "__main__":
    main()
