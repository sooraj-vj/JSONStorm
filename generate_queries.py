import os
import json
import time
from pathlib import Path
from prompts import PROMPTS
import argparse

import yanex
import google.generativeai as genai


# -------------------------------------------------
# Configuration
# -------------------------------------------------

MODEL_NAME = "gemini-2.5-flash-lite"
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
# Prompts
# -------------------------------------------------

def build_prompt(schema_text: str, query_id: int, base_prompt: str, query_type: str) -> str:

    if query_type == "find":
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
- filter (required — a MongoDB filter document, e.g. {{}})
- projection (optional)
- sort (optional)
- limit (optional)

Rules:
- Output JSON only (no markdown, no explanation)
- Do NOT wrap output in ``` or ```json
- Query must use find() style — do NOT include a "pipeline" field
- Query must be realistic
""".strip()

    elif query_type == "aggregate":
        return f"""
You are an expert at generating MongoDB aggregation queries for analytical performance benchmarking.

{base_prompt}

Use ONLY the collections and fields described below.
Do NOT invent fields.

{schema_text}

Return ONE query as valid JSON with EXACTLY these fields:
- id (string, e.g. "Q{query_id}")
- description (short sentence)
- collection (string)
- pipeline (required — array of MongoDB aggregation stages)

Allowed pipeline stages include:
- $match, $group, $lookup, $unwind, $sort, $limit, $setWindowFields, $facet

Rules:
- Output STRICT JSON only (no markdown, no explanation, no comments)
- Do NOT wrap output in ``` or ```json
- Do NOT include any // or /* */ style comments anywhere in the JSON
- Do NOT include a "filter" field — use $match inside the pipeline instead
- Query must be realistic and non-trivial
- Prefer multi-stage pipelines over single-stage queries
""".strip()

    else:
        raise ValueError(f"Unknown query_type: {query_type!r}")


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


def validate_query(obj: dict, query_type: str) -> str | None:
    """
    Returns an error string if the query is invalid for its type, else None.
    Ensures find queries have 'filter' and aggregate queries have 'pipeline',
    and that neither has bled into the wrong type.
    """
    if query_type == "find":
        if "pipeline" in obj:
            return "find query must not contain a 'pipeline' field"
        if "filter" not in obj:
            return "find query is missing required 'filter' field"
    elif query_type == "aggregate":
        if "filter" in obj and "pipeline" not in obj:
            return "aggregate query has 'filter' but no 'pipeline' — looks like a find query"
        if "pipeline" not in obj:
            return "aggregate query is missing required 'pipeline' field"
        if not isinstance(obj["pipeline"], list):
            return f"'pipeline' must be a list, got {type(obj['pipeline']).__name__}"
    return None


def _count_tokens(response) -> tuple[int, int]:
    """
    Extract input and output token counts from a Gemini response object.
    Returns (input_tokens, output_tokens). Falls back to 0 if unavailable.
    """
    try:
        usage = response.usage_metadata
        return usage.prompt_token_count or 0, usage.candidates_token_count or 0
    except Exception:
        return 0, 0


def generate_query(
    schema_text: str,
    query_id:    int,
    base_prompt: str,
    query_type:  str,
    call_index:  int,
) -> dict | None:
    """
    Generate one query, logging per-call API metrics to yanex:
      - api_latency_ms     : wall-clock time for the Gemini API call
      - api_input_tokens   : prompt token count reported by the API
      - api_output_tokens  : completion token count reported by the API
      - api_total_tokens   : sum of the above
      - api_attempt        : which attempt succeeded (1, 2, or 3)
      - api_success        : 1 if the call produced a valid query, 0 otherwise
      - api_validation_fail: 1 if the call returned JSON but failed validation
    """
    prompt = build_prompt(schema_text, query_id, base_prompt, query_type)

    for attempt in range(1, MAX_RETRIES + 1):
        text = ""
        api_latency_ms    = 0.0
        input_tokens      = 0
        output_tokens     = 0
        validation_failed = 0

        try:
            # --- Timed API call ---
            t0       = time.perf_counter()
            response = model.generate_content(prompt)
            api_latency_ms = round((time.perf_counter() - t0) * 1000, 3)

            input_tokens, output_tokens = _count_tokens(response)

            # --- Extract text ---
            text = getattr(response, "text", None)
            if not text:
                try:
                    text = response.candidates[0].content.parts[0].text
                except Exception:
                    text = ""
            text = text.strip()

            if not text:
                raise ValueError("Empty response from Gemini")

            # --- Clean markdown fences ---
            if "```" in text:
                parts = text.split("```")
                if len(parts) >= 2:
                    text = parts[1].strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()

            # --- Extract JSON object ---
            first_brace = text.find("{")
            last_brace  = text.rfind("}")
            if first_brace == -1 or last_brace == -1:
                raise ValueError("No JSON object found in response")
            text = text[first_brace : last_brace + 1]

            # --- Parse ---
            obj = json.loads(text)

            # --- Type-specific validation ---
            error = validate_query(obj, query_type)
            if error:
                validation_failed = 1
                raise ValueError(f"Validation failed: {error}")

            # --- Log successful call metrics ---
            yanex.log_metrics({
                "api_latency_ms":      api_latency_ms,
                "api_input_tokens":    input_tokens,
                "api_output_tokens":   output_tokens,
                "api_total_tokens":    input_tokens + output_tokens,
                "api_attempt":         attempt,
                "api_success":         1,
                "api_validation_fail": 0,
            }, step=call_index)

            return obj

        except Exception as e:
            print(f"[WARN] Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            print("RAW TEXT:", repr(text))

            # Log the failed attempt — only emit latency if we actually got
            # a response back (latency > 0), otherwise it was a network error
            if api_latency_ms > 0:
                yanex.log_metrics({
                    "api_latency_ms":      api_latency_ms,
                    "api_input_tokens":    input_tokens,
                    "api_output_tokens":   output_tokens,
                    "api_total_tokens":    input_tokens + output_tokens,
                    "api_attempt":         attempt,
                    "api_success":         0,
                    "api_validation_fail": validation_failed,
                }, step=call_index)

            time.sleep(SLEEP_SECONDS)

    # All retries exhausted — log a final failure marker
    yanex.log_metrics({
        "api_latency_ms":      0,
        "api_input_tokens":    0,
        "api_output_tokens":   0,
        "api_total_tokens":    0,
        "api_attempt":         MAX_RETRIES,
        "api_success":         0,
        "api_validation_fail": 0,
    }, step=call_index)

    return None


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt",      default="P1")
    parser.add_argument("--num-queries", type=int, default=1)
    parser.add_argument("--query-type",  default="find", choices=["find", "aggregate"])
    args = parser.parse_args()

    schema_text = load_schema()

    prompt_key = args.prompt.lower()
    if prompt_key not in PROMPTS:
        raise ValueError(f"Invalid prompt key: {args.prompt!r}. Available: {list(PROMPTS)}")

    base_prompt = PROMPTS[prompt_key]
    num_queries = args.num_queries
    generated   = 0
    attempted   = 0
    failed      = 0

    # Log run-level parameters so every generation run is reproducible
    yanex.log_metrics({
        "param_model":      MODEL_NAME,
        "param_prompt":     prompt_key,
        "param_query_type": args.query_type,
        "param_num_queries": num_queries,
    })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Accumulators for end-of-run summary
    total_latency_ms   = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    total_api_calls    = 0

    while generated < num_queries:
        qid = next_query_id()
        attempted   += 1
        call_index   = attempted  # used as the yanex step

        query = generate_query(
            schema_text, qid, base_prompt, args.query_type, call_index
        )

        if query is None:
            failed += 1
            print(f"[WARN] Generation failed after {MAX_RETRIES} attempts "
                  f"(query #{attempted}), skipping")
            continue

        # Final safety check before writing
        try:
            json.dumps(query)
        except (TypeError, ValueError) as e:
            failed += 1
            print(f"[WARN] Query #{attempted} not JSON-serialisable, skipping: {e}")
            continue

        with OUTPUT_PATH.open("a", encoding="utf-8") as f:
            f.write("\n" + json.dumps(query) + "\n")

        generated += 1
        print(f"[OK] Generated {query.get('id', f'Q{qid}')} ({generated}/{num_queries})")

    # ------------------------------------------------------------------
    # End-of-run summary — logged once so yanex shows headline numbers
    # ------------------------------------------------------------------
    print(f"\n[DONE] Generated {generated}/{num_queries} queries "
          f"using prompt '{prompt_key}' ({failed} failed attempts)")


if __name__ == "__main__":
    main()