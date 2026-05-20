# infer_schema.py
# Connects to your DB, samples documents from each collection,
# and produces a TypeScript-style schema description for use as an LLM prompt suffix.
#
# Improvements over v1:
#   - Presence rate: tracks what % of sampled docs contain each field
#   - Better array type inference: samples multiple elements, detects heterogeneous arrays
#   - Multiple example values: collects up to 3 distinct values per field
#   - Value ranges: reports min/max for numbers and dates, distinct count for strings
#   - TypeScript-style output: easier for LLMs to parse than custom tabular format
#   - Cardinality hints: flags high/low cardinality fields automatically
#   - Dot-notation paths: explicitly listed for all nested fields

import json
import argparse
import statistics
from collections import defaultdict
from datetime import datetime
from pymongo import MongoClient


# ---------------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------------

def infer_type(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, datetime):
        return "Date"
    if isinstance(value, list):
        return infer_array_type(value)
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def infer_array_type(lst):
    """
    Sample up to 10 elements to determine the array's element type.
    Returns e.g. "string[]", "object[]", "(string | int)[]" for heterogeneous arrays.
    """
    if not lst:
        return "unknown[]"
    sample = lst[:10]
    types = {infer_type(v) for v in sample if v is not None}
    types.discard("null")
    if not types:
        return "null[]"
    if len(types) == 1:
        return f"{types.pop()}[]"
    return f"({' | '.join(sorted(types))})[]"


# ---------------------------------------------------------------------------
# Per-field statistics accumulator
# ---------------------------------------------------------------------------

class FieldStats:
    """Accumulates statistics for a single dot-notation field path."""

    MAX_SAMPLES = 5      # distinct example values to keep
    MAX_NUMERIC = 1000   # cap numeric samples kept for min/max/mean

    def __init__(self):
        self.count       = 0          # docs where this field is present
        self.types       = set()      # all observed type strings
        self.samples     = []         # up to MAX_SAMPLES distinct repr values
        self.sample_set  = set()      # for fast dedup
        self.numerics    = []         # numeric values for range stats
        self.dates       = []         # datetime values for range stats
        self.str_uniq    = set()      # unique string values (capped at 200)
        self.str_capped  = False      # True once we stop tracking uniqueness

    def observe(self, value):
        self.count += 1
        t = infer_type(value)
        self.types.add(t)

        # Collect sample values (distinct)
        rep = _short_repr(value)
        if rep not in self.sample_set and len(self.samples) < self.MAX_SAMPLES:
            self.samples.append(rep)
            self.sample_set.add(rep)

        # Numeric range tracking
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if len(self.numerics) < self.MAX_NUMERIC:
                self.numerics.append(value)

        # Date range tracking
        if isinstance(value, datetime):
            self.dates.append(value)

        # String cardinality tracking (capped)
        if isinstance(value, str) and not self.str_capped:
            self.str_uniq.add(value)
            if len(self.str_uniq) >= 200:
                self.str_capped = True

    def presence(self, total_docs):
        return self.count / total_docs if total_docs else 0

    def type_str(self):
        """Return a clean type annotation, collapsing null into optional marker."""
        types = self.types - {"null"}
        if not types:
            return "null"
        if len(types) == 1:
            t = types.pop()
        else:
            t = " | ".join(sorted(types))
        return t

    def cardinality_hint(self):
        """
        Returns a cardinality label based on observed string uniqueness.
        Only meaningful for string fields with enough samples.
        """
        if "string" not in self.types or self.count < 10:
            return None
        ratio = len(self.str_uniq) / self.count
        if self.str_capped or ratio > 0.9:
            return "high-cardinality"
        if ratio < 0.05:
            return "categorical"
        return None

    def range_annotation(self):
        """
        Returns a compact range string, e.g. "range: 0–9999" or "2020-01-01 – 2024-10-01"
        Returns None if not enough data.
        """
        if self.numerics and len(self.numerics) >= 2:
            lo, hi = min(self.numerics), max(self.numerics)
            if lo == hi:
                return None
            # Format nicely
            fmt = lambda v: f"{v:,.0f}" if isinstance(v, float) and v == int(v) else (
                f"{v:,.2f}" if isinstance(v, float) else f"{v:,}"
            )
            return f"range {fmt(lo)} – {fmt(hi)}"

        if self.dates and len(self.dates) >= 2:
            lo, hi = min(self.dates), max(self.dates)
            return f"range {lo.date()} – {hi.date()}"

        if "string" in self.types and not self.str_capped and len(self.str_uniq) <= 8:
            # Looks like an enum — list all values
            quoted = ", ".join(f'"{v}"' for v in sorted(self.str_uniq))
            return f"values: {quoted}"

        return None


def _short_repr(value, max_len=50):
    """A short, human-readable representation of a sample value."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, dict):
        return "{…}"
    if isinstance(value, list):
        if not value:
            return "[]"
        inner = _short_repr(value[0])
        return f"[{inner}, …]" if len(value) > 1 else f"[{inner}]"
    s = str(value)
    return s[:max_len] + "…" if len(s) > max_len else s


# ---------------------------------------------------------------------------
# Document walker
# ---------------------------------------------------------------------------

def walk_docs(docs, stats, prefix=""):
    """
    Recursively walk a list of documents, updating `stats` (dict of field → FieldStats).
    Tracks presence correctly: a field is present only in docs that actually contain it.
    """
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        for key, value in doc.items():
            if key == "_id":
                continue
            full_key = f"{prefix}{key}"

            if full_key not in stats:
                stats[full_key] = FieldStats()
            stats[full_key].observe(value)

            # Recurse into embedded objects
            if isinstance(value, dict):
                walk_docs([value], stats, prefix=f"{full_key}.")

            # Recurse into arrays of objects
            elif isinstance(value, list):
                obj_items = [v for v in value if isinstance(v, dict)]
                if obj_items:
                    walk_docs(obj_items, stats, prefix=f"{full_key}[].")


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_field_line(path, fstats, total_docs, indent="  "):
    """
    Produce a single TypeScript-style commented field line, e.g.:
      score?: number;            // 87% present | range 0 – 9999
      tags?: string[];           // 61% present | e.g. ["python", "mongodb"]
    """
    presence  = fstats.presence(total_docs)
    type_ann  = fstats.type_str()
    optional  = "?" if "null" in fstats.types or presence < 1.0 else ""

    # Build comment fragments
    comments = [f"{presence:.0%} present"]

    range_ann = fstats.range_annotation()
    if range_ann:
        comments.append(range_ann)
    elif fstats.samples:
        sample_str = ", ".join(fstats.samples[:3])
        comments.append(f"e.g. {sample_str}")

    card = fstats.cardinality_hint()
    if card:
        comments.append(card)

    comment = " | ".join(comments)
    decl = f"{indent}{path}{optional}: {type_ann};"
    return f"{decl:<55} // {comment}"


def describe_collection(db, collection_name, sample_size=200):
    collection = db[collection_name]
    total_count = collection.estimated_document_count()

    # Two-pass sampling: general + field-rich docs to catch sparse structures
    half = sample_size // 2
    general_docs = list(collection.aggregate([{"$sample": {"size": half}}]))

    # Field-rich pass: pick documents with the most keys at the top level
    pipeline_rich = [
        {"$sample": {"size": sample_size * 5}},   # over-sample
        {"$addFields": {"__nkeys": {"$size": {"$objectToArray": "$$ROOT"}}}},
        {"$sort": {"__nkeys": -1}},
        {"$limit": half},
        {"$unset": "__nkeys"},
    ]
    rich_docs = list(collection.aggregate(pipeline_rich))

    # Deduplicate by _id
    seen_ids = set()
    docs = []
    for d in general_docs + rich_docs:
        oid = str(d.get("_id", ""))
        if oid not in seen_ids:
            seen_ids.add(oid)
            docs.append(d)

    total_sampled = len(docs)

    # Collect stats
    stats = {}
    walk_docs(docs, stats)

    # Sort: top-level fields first (no dots), then nested, alphabetically within groups
    def sort_key(path):
        depth = path.count(".")
        return (depth, path)

    sorted_paths = sorted(stats.keys(), key=sort_key)

    # Separate top-level from nested for cleaner grouping
    top_level   = [p for p in sorted_paths if "." not in p and "[]" not in p]
    nested      = [p for p in sorted_paths if "." in p or "[]" in p]

    lines = []
    lines.append(f"// Collection: {collection_name}  (~{total_count:,} documents, {total_sampled} sampled)")
    lines.append(f"interface {collection_name.capitalize()} {{")

    # Top-level fields
    for path in top_level:
        lines.append(format_field_line(path, stats[path], total_sampled))

    # Nested fields, grouped under a blank line
    if nested:
        lines.append("")
        lines.append("  // --- Nested field paths (use dot-notation in queries) ---")
        current_parent = None
        for path in nested:
            parent = path.split(".")[0]
            if parent != current_parent:
                lines.append("")
                current_parent = parent
            lines.append(format_field_line(path, stats[path], total_sampled))

    lines.append("}")
    return "\n".join(lines), total_count


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Infer MongoDB collection schema for LLM prompts"
    )
    parser.add_argument("--db",          default="mathstackexchange_dev")
    parser.add_argument("--uri",         default="mongodb://localhost:27017")
    parser.add_argument("--sample-size", type=int, default=200,
                        help="Number of documents to sample per collection (split between "
                             "random and field-rich passes)")
    parser.add_argument("--collections", nargs="+",
                        default=["users", "posts", "postHistory"],
                        help="Collections to sample")
    parser.add_argument("--out",         default="schema.txt",
                        help="Output file for the schema description")
    args = parser.parse_args()

    client = MongoClient(args.uri)
    db = client[args.db]

    available = set(db.list_collection_names())
    all_descriptions = []

    for name in args.collections:
        if name not in available:
            print(f"SKIP: '{name}' not found in {args.db}")
            continue
        print(f"Sampling '{name}'...")
        description, count = describe_collection(db, name, args.sample_size)
        all_descriptions.append(description)
        field_count = description.count("\n  ")
        print(f"  ~{count:,} documents, {field_count} fields found")

    output = "\n\n".join(all_descriptions)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"\nSchema written to '{args.out}'")
    print("Paste the contents of this file as a suffix to your LLM prompts.")
    client.close()


if __name__ == "__main__":
    main()