import yanex
import os
import json
import time
from pathlib import Path
from prompts import PROMPTS
import argparse
import os

import google.generativeai as genai


# -------------------------------------------------
# Configuration
# -------------------------------------------------

MODEL_NAME = "gemini-2.5-flash"
SCHEMA_PATH = Path("schema.txt")
OUTPUT_PATH = Path("queries.jsonl")

MAX_RETRIES = 3
SLEEP_SECONDS = 2


# -------------------------------------------------
# Gemini setup
# -------------------------------------------------

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable not set. "
        "Run: setx GEMINI_API_KEY \"your_key_here\""
    )

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)


# -------------------------------------------------
# Prompt
# -------------------------------------------------

def build_prompt(schema_text: str, query_id: int, base_prompt: str) -> str:
    return f"""
You are an expert at generating MongoDB queries using the find method for performance benchmarking.

{base_prompt}

Use ONLY the collections and fields described below.
Do NOT invent fields.

{schema_text}

Return ONE query as valid JSON with EXACTLY these fields:
- id (string, e.g. "Q{query_id}")
- description (short sentence)
- collection (string)
- filter
- projection (optional)
- sort (optional)
- limit (optional)

Rules:
- Output JSON only (no markdown, no explanation)
- Query must be realistic
- Prefer aggregation pipelines when appropriate
""".strip()


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def load_schema() -> str:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"{SCHEMA_PATH} not found. Run infer_schema.py first."
        )
    return SCHEMA_PATH.read_text(encoding="utf-8")


def next_query_id() -> int:
    if not OUTPUT_PATH.exists():
        return 1
    with OUTPUT_PATH.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f) + 1


def generate_query(schema_text: str, query_id: int, base_prompt: str):
    prompt = build_prompt(schema_text, query_id, base_prompt)

    for attempt in range(1, MAX_RETRIES + 1):
        text = ""   # ALWAYS initialize first

        try:
            response = model.generate_content(prompt)

            # --- Extract text robustly ---
            text = getattr(response, "text", None)

            if not text:
                try:
                    text = response.candidates[0].content.parts[0].text
                except:
                    text = ""

            text = text.strip()

            if not text:
                raise ValueError("Empty response from Gemini")

            # --- Clean markdown ---
            if "```" in text:
                parts = text.split("```")
                if len(parts) >= 2:
                    text = parts[1].strip()

            if text.lower().startswith("json"):
                text = text[4:].strip()

            first_brace = text.find("{")
            if first_brace != -1:
                text = text[first_brace:]

            
            last_brace = text.rfind("}")
            if last_brace != -1:
                text = text[:last_brace + 1]


            # --- Parse JSON ---
            try:
                obj = json.loads(text)
            except Exception as e:
                print("[ERROR] Invalid JSON after cleaning:")
                print(text)
                raise e

            # --- Validate ---
            if "filter" not in obj and "pipeline" not in obj:
                raise ValueError("Missing filter or pipeline")

            return obj

        except Exception as e:
            print(f"[WARN] Attempt {attempt} failed: {e}")
            print("RAW TEXT:", repr(text))  #  now always safe
            time.sleep(SLEEP_SECONDS)

    return None



# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="P1")
    parser.add_argument("--num-queries", type=int, default=1)
    args = parser.parse_args()

    schema_text = load_schema()

    prompt_key = args.prompt.lower()
    if prompt_key not in PROMPTS:
        raise ValueError(f"Invalid prompt: {args.prompt}")

    base_prompt = PROMPTS[prompt_key]

    num_queries = args.num_queries  # <-- passed from pipeline.py
    generated = 0
    attempted = 0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    while generated < num_queries:
        qid = next_query_id()
        attempted += 1

        query = generate_query(schema_text, qid, base_prompt)

        if query is None:
            print(f"[WARN] Generation failed (attempt {attempted}), skipping")
            continue

        # Defensive JSON validation
        json.dumps(query)

        with OUTPUT_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(query) + "\n")

        generated += 1
        print(f"[OK] Generated {query['id']} ({generated}/{num_queries})")

    print(f"[DONE] Generated {generated} queries using prompt '{prompt_key}'")


if __name__ == "__main__":
    main()