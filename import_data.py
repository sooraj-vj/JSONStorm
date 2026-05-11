"""
import_data.py
Loads users.jsonl, posts.jsonl, and postHistory.jsonl into MongoDB.
Handles MongoDB Extended JSON ($date fields) correctly.

Usage:
    python import_data.py --data-dir ./data --db mathstackexchange
    python import_data.py --data-dir ./data --db mathstackexchange --drop  # drop collections first
"""

import argparse
import json
import os
from datetime import datetime, timezone

from pymongo import MongoClient
from pymongo.errors import BulkWriteError


def parse_extended_json(obj):
    """
    Recursively walk a parsed JSON object and convert MongoDB Extended JSON
    types to native Python types. Currently handles:
      - {"$date": "ISO8601 string"}  ->  datetime
    """
    if isinstance(obj, dict):
        # Check for $date conversion
        if "$date" in obj and len(obj) == 1:
            date_val = obj["$date"]
            if isinstance(date_val, str):
                # Handle ISO 8601 with Z suffix
                date_val = date_val.replace("Z", "+00:00")
                return datetime.fromisoformat(date_val)
            elif isinstance(date_val, (int, float)):
                # Milliseconds since epoch
                return datetime.fromtimestamp(date_val / 1000, tz=timezone.utc)
        return {k: parse_extended_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [parse_extended_json(item) for item in obj]
    return obj


def load_jsonl(filepath):
    """Read a .jsonl file and return a list of parsed documents."""
    documents = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                doc = parse_extended_json(raw)
                documents.append(doc)
            except json.JSONDecodeError as e:
                print(f"  WARNING: skipping malformed line {line_num} in {filepath}: {e}")
    return documents


def import_collection(db, collection_name, filepath, drop=False, batch_size=1000):
    """Import a JSONL file into a MongoDB collection in batches."""
    if not os.path.exists(filepath):
        print(f"  SKIP: {filepath} not found")
        return 0

    collection = db[collection_name]

    if drop:
        collection.drop()
        print(f"  Dropped existing collection '{collection_name}'")

    print(f"  Reading {filepath} ...")
    documents = load_jsonl(filepath)
    total = len(documents)
    print(f"  Parsed {total:,} documents")

    if total == 0:
        return 0

    inserted = 0
    for i in range(0, total, batch_size):
        batch = documents[i : i + batch_size]
        try:
            result = collection.insert_many(batch, ordered=False)
            inserted += len(result.inserted_ids)
        except BulkWriteError as e:
            # Report duplicate key errors etc. but continue
            inserted += e.details.get("nInserted", 0)
            print(f"  WARNING: BulkWriteError on batch {i//batch_size + 1}: "
                  f"{len(e.details.get('writeErrors', []))} errors")

        progress = min(i + batch_size, total)
        print(f"  Progress: {progress:,}/{total:,}", end="\r")

    print(f"  Inserted {inserted:,} documents into '{collection_name}'          ")
    return inserted


def main():
    parser = argparse.ArgumentParser(description="Import JSONL data into MongoDB")
    parser.add_argument("--data-dir", default=".", help="Directory containing the .jsonl files")
    parser.add_argument("--db", default="mathstackexchange", help="MongoDB database name")
    parser.add_argument("--uri", default="mongodb://localhost:27017", help="MongoDB connection URI")
    parser.add_argument("--drop", action="store_true", help="Drop collections before importing")
    parser.add_argument("--batch-size", type=int, default=1000, help="Insert batch size")
    args = parser.parse_args()

    # Collections to import: (collection_name, filename)
    collections = [
        ("users",       "users.jsonl"),
        ("posts",       "posts.jsonl"),
        ("postHistory", "postHistory.jsonl"),
    ]

    print(f"Connecting to {args.uri} ...")
    client = MongoClient(args.uri)
    db = client[args.db]
    print(f"Using database: '{args.db}'\n")

    for collection_name, filename in collections:
        filepath = os.path.join(args.data_dir, filename)
        print(f"[{collection_name}] <- {filepath}")
        count = import_collection(db, collection_name, filepath, drop=args.drop, batch_size=args.batch_size)
        print()

    print("Done.")
    client.close()


if __name__ == "__main__":
    main()
