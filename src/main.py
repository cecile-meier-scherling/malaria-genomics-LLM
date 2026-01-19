import pandas as pd
from typing import Dict, Any, Tuple
from dotenv import load_dotenv
import os
import json

from . import config, query_parser, summarizer, narrative
from .data_loader import load_data

def answer_question(df: pd.DataFrame, question: str) -> Tuple[Dict[str, Any], str]:
    """
    High-level pipeline:
      question -> query -> filtered data -> summary -> LLM narrative
    Returns (query_dict, answer_text).
    """
    print(f"\nUser question: {question}\n")

    # 1) natural language -> structured query
    query = query_parser.llm_question_to_query(question)
    print("Parsed query:", query)

    # 2) filter data
    subset = summarizer.filter_data(df, query)
    print(f"Subset size: {len(subset)} rows")

    # 3) summarize
    summary = summarizer.summarize_prevalence(subset)

    # 4) LLM narrative
    answer = narrative.llm_generate_answer(question, query, summary)
    return query, answer


if __name__ == "__main__":
    # Load your data once
    df_all = load_data(config.DATA_PATH)
    print("Data loaded. You can now ask questions about the uploaded prevalence data.\n")

    print("Type a question (or 'quit', 'exit', or just press Enter to stop).")

    while True:
        try:
            question = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if question.lower() in {"", "quit", "exit"}:
            print("Goodbye!")
            break

        # Run your full pipeline for this question
        try:
            query_dict, answer_text = answer_question(df_all, question)
        except Exception as e:
            print(f"\n[ERROR] Something went wrong: {e}")
            continue

        print("\n=== Structured query ===")
        print(json.dumps(query_dict, indent=2))

        print("\n=== LLM answer ===\n")
        print(answer_text)