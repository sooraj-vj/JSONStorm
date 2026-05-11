"""
run_harness.py
Runs a set of MongoDB find queries defined in a JSONL file, records execution
times and explain stats, and saves results to a JSONL output file.

Usage:
    python run_harness.py --queries queries.jsonl --db mathstackexchange
    python run_harness.py --queries queries.jsonl --db mathstackexchange --out results/run1.jsonl
    python run_harness.py --queries queries.jsonl --db mathstackexchange --timeout 10
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from unittest import result

from pymongo import MongoClient
from pymongo.errors import OperationFailure, ExecutionTimeout

MAX_RESULTS_TO_STORE = 10

def make_json_safe(obj):
    """
    Recursively convert datetimes in a document to ISO strings
    so it can be safely serialized to JSON.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    else:
        return obj

# ---------------------------------------------------------------------------
# Extended JSON date handling (same logic as import_data.py)
# ---------------------------------------------------------------------------

def parse_extended_json(obj):
    """Convert {"$date": "..."} values in filter/sort dicts to datetime objects."""
    if isinstance(obj, dict):
        if "$date" in obj and len(obj) == 1:
            date_val = obj["$date"]
            if isinstance(date_val, str):
                date_val = date_val.replace("Z", "+00:00")
                return datetime.fromisoformat(date_val)
            elif isinstance(date_val, (int, float)):
                return datetime.fromtimestamp(date_val / 1000, tz=timezone.utc)
        return {k: parse_extended_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [parse_extended_json(item) for item in obj]
    return obj


# ---------------------------------------------------------------------------
# Explain stat extraction
# ---------------------------------------------------------------------------

def extract_execution_stats(explain_result):
    """
    Pull the most useful fields out of a MongoDB executionStats explain result.
    Returns a flat dict so results are easy to analyse later.
    """
    stats = {}

    # Top-level execution stats
    es = explain_result.get("executionStats", {})
    stats["executionTimeMillis"]   = es.get("executionTimeMillis")
    stats["totalDocsExamined"]     = es.get("totalDocsExamined")
    stats["totalKeysExamined"]     = es.get("totalKeysExamined")
    stats["nReturned"]             = es.get("nReturned")

    # Winning plan info
    qp = explain_result.get("queryPlanner", {})
    winning_plan = qp.get("winningPlan", {})
    stats["winningPlanStage"]      = winning_plan.get("stage")

    # Index used (if any) — walk into IXSCAN stage
    stats["indexUsed"] = None
    def find_index(plan):
        if isinstance(plan, dict):
            if plan.get("stage") == "IXSCAN":
                return plan.get("indexName")
            for v in plan.values():
                result = find_index(v)
                if result:
                    return result
        return None
    stats["indexUsed"] = find_index(winning_plan)

    return stats


# ---------------------------------------------------------------------------
# Core harness
# ---------------------------------------------------------------------------

def run_query(collection, query_def, timeout_ms):
    """
    Execute one query definition. Returns a result dict with:
      - wall_time_ms: end-to-end latency measured in Python
      - explain stats from MongoDB's executionStats
      - error info if the query failed
    """
    filter_doc     = parse_extended_json(query_def.get("filter", {}))
    projection     = parse_extended_json(query_def.get("projection")) or None
    sort           = list(query_def["sort"].items()) if "sort" in query_def else None
    limit          = query_def.get("limit", 0)

    result = {
        "id":          query_def["id"],
        "description": query_def.get("description", ""),
        "collection":  query_def["collection"],
        "status":      None,    # "success" | "timeout" | "error"
        "wall_time_ms": None,
        "executionTimeMillis":  None,
        "totalDocsExamined":    None,
        "totalKeysExamined":    None,
        "nReturned":            None,
        "winningPlanStage":     None,
        "indexUsed":            None,
        "error":                None,
        "timestamp":   datetime.now(tz=timezone.utc).isoformat(),
    }

    sample_results = []

    # --- 1. Run explain("executionStats") to capture MongoDB's internal stats ---
    try:
        cursor = collection.find(filter_doc, projection)
        if sort:
            cursor = cursor.sort(sort)
        if limit:
            cursor = cursor.limit(limit)
        if timeout_ms:
            cursor = cursor.max_time_ms(timeout_ms)

        explain_result = cursor.explain()
        stats = extract_execution_stats(explain_result)
        result.update(stats)

    except ExecutionTimeout:
        result["status"] = "timeout"
        result["error"]  = f"Exceeded {timeout_ms}ms timeout during explain"
        return result
    except OperationFailure as e:
        result["status"] = "error"
        result["error"]  = str(e)
        return result

    # --- 2. Re-run the actual query to measure wall-clock time ---
    try:
        cursor = collection.find(filter_doc, projection)
        if sort:
            cursor = cursor.sort(sort)
        if limit:
            cursor = cursor.limit(limit)
        if timeout_ms:
            cursor = cursor.max_time_ms(timeout_ms)

        t_start = time.perf_counter()
        docs = list(cursor)   # fully materialise the cursor
        t_end   = time.perf_counter()

        result["wall_time_ms"] = round((t_end - t_start) * 1000, 3)
        result["nReturned"]    = len(docs)   # override with actual count
        result["status"]       = "success"

        cursor = collection.find(
            filter_doc,
            projection=projection
        )

        if sort:
            cursor = cursor.sort(sort)

        if limit:
            cursor = cursor.limit(limit)

        for doc in cursor.limit(MAX_RESULTS_TO_STORE):
            doc = make_json_safe(doc)
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])
            sample_results.append(doc)


    except ExecutionTimeout:
        result["status"] = "timeout"
        result["error"]  = f"Exceeded {timeout_ms}ms timeout during execution"
    except OperationFailure as e:
        result["status"] = "error"
        result["error"]  = str(e)

    result["sample_results"] = sample_results
    return result


def run_harness(queries_file, db_name, uri, output_file, timeout_ms):
    print(f"Connecting to {uri} ...")
    client = MongoClient(uri)
    db = client[db_name]
    print(f"Database: '{db_name}'")
    print(f"Queries:  {queries_file}")
    print(f"Output:   {output_file}")
    print(f"Timeout:  {timeout_ms}ms\n")

    # Load query definitions
    query_defs = []
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                query_defs.append(json.loads(line))

    print(f"Loaded {len(query_defs)} queries\n")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)

    success_count = 0
    error_count   = 0
    timeout_count = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        for i, query_def in enumerate(query_defs, 1):
            collection_name = query_def.get("collection")
            if not collection_name:
                print(f"  [{i}/{len(query_defs)}] SKIP {query_def.get('id')} — no collection specified")
                continue

            collection = db[collection_name]
            print(f"  [{i}/{len(query_defs)}] {query_def['id']:8s}  {query_def.get('description', '')[:50]}", end="  ")

            result = run_query(collection, query_def, timeout_ms)

            # Print one-line summary
            if result["status"] == "success":
                success_count += 1
                print(f"OK  {result['wall_time_ms']:>8.1f}ms  "
                      f"returned={result['nReturned']}  "
                      f"examined={result['totalDocsExamined']}  "
                      f"plan={result['winningPlanStage']}")
            elif result["status"] == "timeout":
                timeout_count += 1
                print(f"TIMEOUT")
            else:
                error_count += 1
                print(f"ERROR  {result['error']}")

            # Write result to output file
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

    print(f"\n{'='*60}")
    print(f"Results: {success_count} success, {timeout_count} timeout, {error_count} error")
    print(f"Saved to: {output_file}")
    client.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MongoDB query benchmark harness")
    parser.add_argument("--queries",  default="queries.jsonl",           help="Path to query definitions JSONL file")
    parser.add_argument("--db",       default="mathstackexchange",        help="MongoDB database name")
    parser.add_argument("--uri",      default="mongodb://localhost:27017", help="MongoDB connection URI")
    parser.add_argument("--out",      default="results/results.jsonl",   help="Output file for results")
    parser.add_argument("--timeout",  type=int, default=10000,           help="Per-query timeout in milliseconds")
    args = parser.parse_args()

    run_harness(
        queries_file=args.queries,
        db_name=args.db,
        uri=args.uri,
        output_file=args.out,
        timeout_ms=args.timeout,
    )


if __name__ == "__main__":
    main()
