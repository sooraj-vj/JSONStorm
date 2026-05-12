import subprocess
import sys
import yanex
from pathlib import Path

PYTHON = sys.executable


def run_step(name, cmd):
    print(f"\n=== {name} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    # -------------------------------
    # Parameters (tracked by Yanex)
    # -------------------------------
    params = yanex.get_params()

    db_name      = params.get("db", "mathstackexchange_dev")
    sample_mode  = params.get("sample_mode", "nth")
    sample_n     = params.get("sample_n", 50)
    timeout_ms   = params.get("timeout_ms", 5000)

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

    # Optional: Gemini query generation
    # run_step("Generating queries", [PYTHON, "generate_queries.py"])

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

    # Optional: log a simple metric
    yanex.log_metrics({
        "sample_n": sample_n,
        "timeout_ms": timeout_ms,
    })

    print("\nExperiment completed successfully")


if __name__ == "__main__":
    main()