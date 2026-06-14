"""
report.py
Generates a Markdown report from a yanex experiment.

Usage:
    # Latest experiment
    python report.py

    # Specific experiment by ID
    python report.py --exp exp_20240101_120000

    # Compare multiple experiments
    python report.py --exp exp1 exp2 exp3

    # Save to file
    python report.py --out report.md
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median

import yanex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(value: float, max_value: float, width: int = 20) -> str:
    """Simple ASCII progress bar."""
    if max_value == 0:
        filled = 0
    else:
        filled = round((value / max_value) * width)
    return "█" * filled + "░" * (width - filled)


def _complexity_emoji(level: str) -> str:
    return {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(level, "⚪")


def _status_emoji(status: str) -> str:
    return {"success": "✅", "timeout": "⏱️", "error": "❌"}.get(status, "❓")


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def section_params(metrics: dict) -> str:
    param_keys = [k for k in metrics if k.startswith("param_")]
    if not param_keys:
        return ""

    lines = ["## Parameters", "", "| Parameter | Value |", "|-----------|-------|"]
    for k in sorted(param_keys):
        label = k.replace("param_", "").replace("_", " ").title()
        lines.append(f"| {label} | `{metrics[k]}` |")
    lines.append("")
    return "\n".join(lines)


def section_outcomes(metrics: dict) -> str:
    total   = metrics.get("queries_total",   0)
    success = metrics.get("queries_success", 0)
    timeout = metrics.get("queries_timeout", 0)
    error   = metrics.get("queries_error",   0)

    if total == 0:
        return ""

    s_pct = success / total * 100
    t_pct = timeout / total * 100
    e_pct = error   / total * 100

    lines = [
        "## Query Outcomes",
        "",
        f"| Status  | Count | % | Bar |",
        f"|---------|------:|--:|-----|",
        f"| {_status_emoji('success')} Success | {success} | {s_pct:.1f}% | `{_bar(success, total)}` |",
        f"| {_status_emoji('timeout')} Timeout | {timeout} | {t_pct:.1f}% | `{_bar(timeout, total)}` |",
        f"| {_status_emoji('error')} Error   | {error}   | {e_pct:.1f}% | `{_bar(error,   total)}` |",
        f"| **Total** | **{total}** | | |",
        "",
    ]
    return "\n".join(lines)


def section_performance(metrics: dict) -> str:
    keys = ["avg_wall_time_ms", "max_wall_time_ms",
            "wall_time_p50", "wall_time_p95", "wall_time_p99"]
    if not any(k in metrics for k in keys):
        return ""

    lines = [
        "## Performance",
        "",
        "| Metric | Value |",
        "|--------|------:|",
    ]

    def row(label, key, fmt=".1f"):
        val = metrics.get(key)
        if val is not None:
            lines.append(f"| {label} | {val:{fmt}} ms |")

    row("Mean wall time",  "avg_wall_time_ms")
    row("Max wall time",   "max_wall_time_ms")
    row("p50 (median)",    "wall_time_p50")
    row("p95",             "wall_time_p95")
    row("p99",             "wall_time_p99")

    if "avg_docs_examined" in metrics:
        lines.append(f"| Avg docs examined | {metrics['avg_docs_examined']:,.0f} |")
    if "avg_keys_examined" in metrics:
        lines.append(f"| Avg keys examined | {metrics['avg_keys_examined']:,.0f} |")

    lines.append("")
    return "\n".join(lines)


def section_complexity(metrics: dict) -> str:
    low    = metrics.get("n_complexity_low",    metrics.get("complexity_low",    0))
    medium = metrics.get("n_complexity_medium", metrics.get("complexity_medium", 0))
    high   = metrics.get("n_complexity_high",   metrics.get("complexity_high",   0))
    total  = low + medium + high

    if total == 0:
        return ""

    mismatches  = metrics.get("n_mismatches", 0)
    mismatch_rt = metrics.get("mismatch_rate", 0)

    lines = [
        "## Complexity Distribution",
        "",
        "| Class | Count | % | Bar |",
        "|-------|------:|--:|-----|",
        f"| {_complexity_emoji('low')} Low       | {low}    | {low/total*100:.1f}%    | `{_bar(low,    total)}` |",
        f"| {_complexity_emoji('medium')} Medium  | {medium} | {medium/total*100:.1f}% | `{_bar(medium, total)}` |",
        f"| {_complexity_emoji('high')} High      | {high}   | {high/total*100:.1f}%   | `{_bar(high,   total)}` |",
        "",
        f"> **Complexity mismatches** (static != execution): "
        f"{mismatches} ({mismatch_rt*100:.1f}%)",
        "",
    ]
    return "\n".join(lines)


def section_query_table(results_jsonl: Path) -> str:
    """Per-query detail table, read from the results artifact."""
    if not results_jsonl.exists():
        return ""

    rows = []
    with open(results_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        return ""

    lines = [
        "## Per-Query Results",
        "",
        "| ID | Type | Status | Wall time | Docs examined | Complexity | Mismatch |",
        "|----|------|--------|----------:|--------------:|------------|----------|",
    ]

    for r in rows:
        qid      = r.get("id", "?")
        qtype    = r.get("query_type", "?").upper()
        status   = _status_emoji(r.get("status", ""))
        wall     = f"{r['wall_time_ms']:.1f} ms" if r.get("wall_time_ms") else "—"
        examined = f"{r['totalDocsExamined']:,}" if r.get("totalDocsExamined") is not None else "—"
        ec       = r.get("execution_complexity", "?")
        ec_icon  = _complexity_emoji(ec) + " " + ec if ec != "?" else "?"
        mismatch = "⚠️" if r.get("complexity_mismatch") else ""
        lines.append(f"| {qid} | {qtype} | {status} | {wall} | {examined} | {ec_icon} | {mismatch} |")

    lines.append("")
    return "\n".join(lines)


def section_comparison(experiments: list) -> str:
    """Cross-experiment comparison table when multiple IDs are given."""
    if len(experiments) < 2:
        return ""

    lines = [
        "## Experiment Comparison",
        "",
        "| Experiment | Prompt | Query type | Success % | p50 ms | p95 ms | High % |",
        "|------------|--------|------------|----------:|-------:|-------:|-------:|",
    ]

    for exp in experiments:
        m       = exp["metrics"] if isinstance(exp, dict) else {}
        eid     = exp["id"] if isinstance(exp, dict) else exp.id
        prompt  = m.get("param_prompt", "?")
        qtype   = m.get("param_query_type", "?")
        total   = m.get("queries_total", 0)
        success = m.get("queries_success", 0)
        s_pct   = f"{success/total*100:.1f}%" if total else "?"
        p50     = f"{m['wall_time_p50']:.1f}" if "wall_time_p50" in m else "?"
        p95     = f"{m['wall_time_p95']:.1f}" if "wall_time_p95" in m else "?"
        high    = m.get("n_complexity_high", m.get("complexity_high", 0))
        h_pct   = f"{high/max(success,1)*100:.1f}%" if success else "?"
        lines.append(
            f"| `{eid}` | {prompt} | {qtype} | {s_pct} | {p50} | {p95} | {h_pct} |"
        )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_experiments(exp_ids: list) -> list[dict]:
    """
    Locate yanex experiment directories under ~/.yanex/experiments/.
    Reads metadata.json (id, started_at, status), metrics.json (all logged
    metrics), and artifacts/results.jsonl (per-query detail).
    """
    base = Path.home() / ".yanex" / "experiments"
    if not base.exists():
        print(f"No yanex experiments directory found at {base}")
        sys.exit(1)

    # Sort by directory modification time — most recent first
    all_dirs = sorted(
        [d for d in base.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )

    if not all_dirs:
        print("No experiments found. Run the pipeline first.")
        sys.exit(1)

    # Filter to requested IDs, or take the most recent
    if exp_ids:
        selected_dirs = [d for d in all_dirs if d.name in exp_ids]
        if not selected_dirs:
            print(f"No experiments found matching: {exp_ids}")
            print(f"Available: {[d.name for d in all_dirs[:5]]}")
            sys.exit(1)
    else:
        selected_dirs = [all_dirs[0]]

    enriched = []
    for exp_dir in selected_dirs:

        # --- metadata.json: id, status, started_at, tags, etc. ---
        metadata = {}
        meta_path = exp_dir / "metadata.json"
        if meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        # --- metrics.json: flat dict of all logged metrics ---
        # yanex stores metrics.json as either a flat dict or a list of
        # {step, data} entries depending on version — handle both.
        metrics = {}
        metrics_path = exp_dir / "metrics.json"
        if metrics_path.exists():
            try:
                raw = json.loads(metrics_path.read_text(encoding="utf-8"))
                # metrics.json is a list of step objects, each with metric keys
                # plus "step" and "timestamp" — merge all into one flat dict.
                skip_keys = {"step", "timestamp"}
                for entry in raw:
                    if isinstance(entry, dict):
                        metrics.update({
                            k: v for k, v in entry.items()
                            if k not in skip_keys
                        })
            except Exception as e:
                print(f"[WARN] Could not parse metrics.json: {e}")

        # --- artifacts/results.jsonl: per-query detail rows ---
        results_file = exp_dir / "artifacts" / "results.jsonl"
        if not results_file.exists():
            results_file = None

        enriched.append({
            "id":           metadata.get("id",         exp_dir.name),
            "started_at":   metadata.get("started_at", metadata.get("created_at", "unknown")),
            "status":       metadata.get("status",     "unknown"),
            "metrics":      metrics,
            "dir":          exp_dir,
            "results_file": results_file,
        })

    return enriched


def section_header(exp: dict) -> str:
    lines = [
        f"# JSONStorm Benchmark Report",
        f"",
        f"**Experiment:** `{exp['id']}`  ",
        f"**Started:** {exp['started_at']}  ",
        f"**Status:** {exp['status']}  ",
        f"",
    ]
    return "\n".join(lines)


def build_report(exp_ids: list) -> str:
    experiments = find_experiments(exp_ids)

    sections = []

    primary = experiments[0]
    metrics = primary["metrics"]

    sections.append(section_header(primary))
    sections.append(section_params(metrics))
    sections.append(section_outcomes(metrics))
    sections.append(section_performance(metrics))
    sections.append(section_complexity(metrics))

    if primary["results_file"]:
        sections.append(section_query_table(primary["results_file"]))

    sections.append(section_comparison(experiments))

    return "\n".join(s for s in sections if s)


def main():
    parser = argparse.ArgumentParser(description="Generate a Markdown report from yanex experiments")
    parser.add_argument("--exp", nargs="*", default=[],
                        help="Experiment ID(s) to report on. Defaults to most recent.")
    parser.add_argument("--out", default=None,
                        help="Output file. Prints to stdout if not set.")
    args = parser.parse_args()

    report = build_report(args.exp)

    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"Report written to {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()