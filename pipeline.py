import subprocess
import sys
import yanex
from pathlib import Path
import json
from statistics import mean

PYTHON = sys.executable
PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPT_ROOT = PROJECT_ROOT


def run_step(name, cmd):
    print(f"\n=== {name} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    # -------------------------------
    # Parameters (tracked by Yanex)
    # -------------------------------
    params = yanex.get_params()

    db_name      = params.get("DB", "mathstackexchange_dev")
    sample_n   = int(params.get("SAMPLE_N", 50))
    sample_mode = params.get("SAMPLE_MODE", "nth")
    timeout_ms = int(params.get("TIMEOUT_MS", 5000))
    prompt_choice = yanex.get_param("PROMPT", default="p1")
    num_queries = int(params.get("NUM_QUERIES", 1))

    print("YANEX PARAMS:", params)

    print(f"[PIPELINE] Using sample_n={sample_n}, sample_mode={sample_mode}")
    print(f"[PIPELINE] num_queries={num_queries}, prompt={prompt_choice}")

    # Paths
    data_dir     = Path("data")
    sample_dir   = Path("sample_data")
    schema_file  = Path("schema.txt")
    queries_file = Path("queries.jsonl")
    results_file = Path("results/results.jsonl")

    # -------------------------------
    # Pipeline steps
    # -------------------------------

    run_step(
        "Sampling data",
        [
            PYTHON, "sample_data.py",
            "--data-dir", str(data_dir),
            "--out-dir", str(sample_dir),
            "--mode", sample_mode,
            "--n", str(sample_n),
        ],
    )

    run_step(
        "Importing data into MongoDB",
        [
            PYTHON, "import_data.py",
            "--data-dir", str(sample_dir),
            "--db", db_name,
            "--drop",
        ],
    )

    run_step(
        "Inferring schema",
        [
            PYTHON, "infer_schema.py",
            "--db", db_name,
            "--out", str(schema_file),
        ],
    )

    
    # Reset queries file before generating
    queries_file = SCRIPT_ROOT / "queries.jsonl"
    queries_file.write_text("")

    # -------------------------------
    # Gemini query generation
    # -------------------------------
    
    prompt_type = params.get("PROMPT", "P1")

    run_step(
        "Generating queries (Gemini)",
        [
            PYTHON,
            "generate_queries.py",
            "--prompt", prompt_type,
            "--num-queries", str(num_queries),
        ]
    )


    run_step(
        "Running benchmark harness",
        [
            PYTHON, "run_harness.py",
            "--queries", str(queries_file),
            "--db", db_name,
            "--out", str(results_file),
            "--timeout", str(timeout_ms),
        ],
    )

    # -------------------------------
    # Log artifacts to Yanex
    # -------------------------------
    yanex.copy_artifact(schema_file, "schema.txt")
    yanex.copy_artifact(queries_file, "queries.jsonl")
    yanex.copy_artifact(results_file, "results.jsonl")

    # -------------------------------
    # Log metrics to Yanex
    # -------------------------------

    # yanex.log_metrics({
    #     "sample_n": sample_n,
    #     "timeout_ms": timeout_ms,
    # })

    results_path = SCRIPT_ROOT / "results" / "results.jsonl"
    results = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))

    successful = [r for r in results if r["status"] == "success"]
    timeouts   = [r for r in results if r["status"] == "timeout"]
    errors     = [r for r in results if r["status"] == "error"]

    
    metrics = {
        # Query outcomes
        "queries_total": len(results),
        "queries_success": len(successful),
        "queries_timeout": len(timeouts),
        "queries_error": len(errors),
    }

    if successful:
        metrics.update({
            "avg_wall_time_ms": mean(r["wall_time_ms"] for r in successful),
            "max_wall_time_ms": max(r["wall_time_ms"] for r in successful),
            "avg_docs_examined": mean(r["totalDocsExamined"] for r in successful),
            "avg_keys_examined": mean(r["totalKeysExamined"] for r in successful),
        })
    
    yanex.log_metrics(metrics)
    



    print("\nExperiment completed successfully")


if __name__ == "__main__":
    main()