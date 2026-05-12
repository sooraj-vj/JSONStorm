import yanex
from JSONStorm.pipeline.prompts import PROMPTS
from time import sleep, perf_counter
from random import random

def query_llm(prompt):

    # Placeholder for LLM query logic
    # In a real implementation, this would involve sending the prompt to an LLM API and returning the response

    # simulating delay and variability in LLM response
    sleep(2 + random() * 3)  # Simulate response time between 2 to 5 seconds

    return [
        {"filter": {"foo": {"$gt": 4}, "bar": {"$eq": True}}, "skip": 10, "limit": 5},
        {"filter": {"baz": {"$lt": 100}}, "projection": {"baz": 1}, "limit": 20},
        {"filter": {"qux": {"$ne": "example"}}, "sort": {"qux": -1}, "limit": 15}
    ]

if __name__ == "__main__":

    # yanex params
    prompt_choice = yanex.get_param("prompt", default="default")

    print(f"Using Prompt {prompt_choice}: {PROMPTS[prompt_choice]}")

    # Start the timer
    start = perf_counter()
    queries = query_llm(PROMPTS[prompt_choice])
    end = perf_counter()
    elapsed = end - start

    print(f"LLM Query Time: {elapsed:.2f} seconds")

    # log metrics
    yanex.log_metrics({"llm_query_time": elapsed})

    print(f"Generated {len(queries)} MongoDB Queries:")
    for i, query in enumerate(queries, 1):
        print(f"   Query {i}: {query}")

    # save artifact
    yanex.save_artifact(queries, "llm_queries.jsonl")