"""
sample_data.py
Creates a smaller subset of each JSONL file for development purposes.

Two sampling modes:
  head  — first N lines (fast, but may not be representative if data is ordered)
  nth   — every Nth line (spreads sample across the whole file)

Usage:
    python sample_data.py --data-dir ./data --out-dir ./sample_data --mode head --n 20000
    python sample_data.py --data-dir ./data --out-dir ./sample_data --mode nth --n 50
"""

import argparse
import os


def sample_head(input_path, output_path, n):
    """Take the first N lines."""
    written = 0
    with open(input_path, encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for i, line in enumerate(f_in):
            if i >= n:
                break
            if line.strip():
                f_out.write(line)
                written += 1
    print(f"  Wrote {written:,} lines to {output_path}")


def sample_every_nth(input_path, output_path, n):
    """Take every Nth line — spreads sample across the whole file."""
    written = 0
    with open(input_path, encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for i, line in enumerate(f_in):
            if i % n == 0 and line.strip():
                f_out.write(line)
                written += 1
    print(f"  Wrote {written:,} lines to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Sample JSONL files for development")
    parser.add_argument("--data-dir", default=".",            help="Directory containing source .jsonl files")
    parser.add_argument("--out-dir",  default="./sample_data", help="Directory to write sampled files")
    parser.add_argument("--mode",     choices=["head", "nth"], default="head",
                        help="'head' = first N lines, 'nth' = every Nth line")
    parser.add_argument("--n",        type=int, default=20000,
                        help="Number of lines (head mode) or stride (nth mode)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = ["users.jsonl", "posts.jsonl", "postHistory.jsonl"]

    print(f"Mode: {args.mode}, n={args.n}")
    print(f"Source:      {args.data_dir}")
    print(f"Destination: {args.out_dir}\n")

    for filename in files:
        input_path  = os.path.join(args.data_dir, filename)
        output_path = os.path.join(args.out_dir, filename)

        if not os.path.exists(input_path):
            print(f"  SKIP: {input_path} not found")
            continue

        print(f"  {filename}")
        if args.mode == "head":
            sample_head(input_path, output_path, args.n)
        else:
            sample_every_nth(input_path, output_path, args.n)

    print("\nDone. To import the sample:")
    print(f"  python import_data.py --data-dir {args.out_dir} --db mathstackexchange_dev --drop")


if __name__ == "__main__":
    main()
