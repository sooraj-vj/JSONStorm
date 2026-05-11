# infer_schema.py
# Connects to your DB, samples documents from each collection,
# and produces a human-readable schema description for use as an LLM prompt suffix.

import json
import argparse
from collections import defaultdict
from datetime import datetime
from pymongo import MongoClient


def infer_type(value):
    if isinstance(value, bool):
        return "bool"
    elif isinstance(value, int):
        return "int"
    elif isinstance(value, float):
        return "float"
    elif isinstance(value, str):
        return "string"
    elif isinstance(value, datetime):
        return "date"
    elif isinstance(value, list):
        if len(value) == 0:
            return "array"
        inner = infer_type(value[0])
        return f"array<{inner}>"
    elif isinstance(value, dict):
        return "object"
    return type(value).__name__


def infer_fields(docs, prefix=""):
    """
    Walk a list of documents and collect field names, inferred types,
    and a sample value for each field. Recurses into embedded objects
    and the first element of arrays.
    """
    field_types   = defaultdict(set)
    field_samples = {}

    for doc in docs:
        if not isinstance(doc, dict):
            continue
        for key, value in doc.items():
            if key == "_id":
                continue
            full_key = f"{prefix}{key}"
            field_types[full_key].add(infer_type(value))
            if full_key not in field_samples:
                # Store a short sample value
                if isinstance(value, datetime):
                    field_samples[full_key] = value.isoformat()
                elif isinstance(value, list) and len(value) > 0:
                    field_samples[full_key] = value[0] if not isinstance(value[0], dict) else "(object)"
                elif isinstance(value, dict):
                    field_samples[full_key] = "(object)"
                else:
                    field_samples[full_key] = value

            # Recurse into embedded objects
            if isinstance(value, dict):
                sub_docs = [value]
                sub_fields = infer_fields(sub_docs, prefix=f"{full_key}.")
                field_types.update(sub_fields[0])
                for k, v in sub_fields[1].items():
                    if k not in field_samples:
                        field_samples[k] = v

            # Recurse into first element of arrays if it's an object
            elif isinstance(value, list):
                obj_items = [v for v in value if isinstance(v, dict)]
                if obj_items:
                    sub_fields = infer_fields(obj_items, prefix=f"{full_key}[].")
                    field_types.update(sub_fields[0])
                    for k, v in sub_fields[1].items():
                        if k not in field_samples:
                            field_samples[k] = v

    return field_types, field_samples


def describe_collection(db, collection_name, sample_size=200):
    collection = db[collection_name]
    count = collection.estimated_document_count()

    # Use $sample to get a random spread rather than just the first N docs
    docs = list(collection.aggregate([{"$sample": {"size": sample_size}}]))

    field_types, field_samples = infer_fields(docs)

    lines = []
    lines.append(f"Collection: {collection_name}  (~{count:,} documents)")
    lines.append("Fields:")
    for field, types in sorted(field_types.items()):
        type_str   = " | ".join(sorted(types))
        sample_val = field_samples.get(field, "")
        # Truncate long sample values
        sample_str = str(sample_val)
        if len(sample_str) > 60:
            sample_str = sample_str[:57] + "..."
        lines.append(f"  {field:<40} {type_str:<20} e.g. {sample_str}")

    return "\n".join(lines), count


def main():
    parser = argparse.ArgumentParser(description="Infer MongoDB collection schema for LLM prompts")
    parser.add_argument("--db",          default="mathstackexchange_dev")
    parser.add_argument("--uri",         default="mongodb://localhost:27017")
    parser.add_argument("--sample-size", type=int, default=200,
                        help="Number of documents to sample per collection")
    parser.add_argument("--out",         default="schema.txt",
                        help="Output file for the schema description")
    args = parser.parse_args()

    client = MongoClient(args.uri)
    db = client[args.db]

    collections = ["users", "posts", "postHistory"]
    all_descriptions = []

    for name in collections:
        if name not in db.list_collection_names():
            print(f"SKIP: '{name}' not found in {args.db}")
            continue
        print(f"Sampling {name}...")
        description, count = describe_collection(db, name, args.sample_size)
        all_descriptions.append(description)
        print(f"  {count:,} documents, {description.count(chr(10))} fields found")

    output = "\n\n".join(all_descriptions)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"\nSchema description written to {args.out}")
    print("Paste the contents of this file as a suffix to your LLM prompts.")
    client.close()


if __name__ == "__main__":
    main()