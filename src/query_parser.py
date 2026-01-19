import json
from typing import Dict, Any

from src import config

def llm_question_to_query(question: str) -> Dict[str, Any]:
    """
    Ask the local LLM to turn a free-text question into a JSON query spec.

    Returns a dict with keys:
      country (str or null)
      mutation (str or null)
      year_min (int or null)
      year_max (int or null)
    """
    system_msg = (
        "You convert user questions about malaria genomic prevalence "
        "into a strict JSON query specification.\n"
        "Supported fields: country (string or null), mutation (string or null), "
        "year_min (integer or null), year_max (integer or null).\n"
        "If a field is not specified in the question, set it to null.\n"
        "Return ONLY valid JSON, no extra text."
    )

    raw = config.call_llm(system_msg, question)

    try:
        query = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError(f"LLM did not return valid JSON:\n{raw}")

    # Basic normalization / defaults
    query.setdefault("country", None)
    query.setdefault("mutation", None)
    query.setdefault("year_min", None)
    query.setdefault("year_max", None)

    return query