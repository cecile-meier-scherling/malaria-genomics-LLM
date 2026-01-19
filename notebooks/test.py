import os
import json
from typing import Dict, Any, Optional, Tuple

import pandas as pd
from openai import OpenAI

# =========================
# 0. CONFIG
# =========================

# Path to your prevalence + study metadata table
DATA_PATH = "prevalence_view.csv"  # or .parquet

# Choose an LLM model
LLM_MODEL = "gpt-4o-mini"  # or another chat-capable model

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# =========================
# 1. LOAD & PREP DATA
# =========================

def load_data(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        raise ValueError("Unsupported file type; use .csv or .parquet")
    
    # Extract gene and mutation
    df[["gene", "position", "aa"]] = df["mutation"].str.split(":", expand=True)
    df["mutation"] = df["position"] + df["aa"]
    
    # Rename certain columns
    df = df.rename(columns={
        "country_name": "country",
        "site_name": "site",
        "denominator": "n_samples",
        "publication_year": "year_pub"
    })

    # Minimal sanity checks / renaming (adjust to your real column names)
    expected_cols = {
        "country_name", "site_name", "year", "gene", "mutation",
        "prevalence", "n_samples", "study_id",
        "authors", "year_pub", "url"
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")

    # Ensure types
    df["year"] = df["year"].astype(int)
    df["year_pub"] = df["year_pub"].astype(int)
    df["prevalence"] = df["prevalence"].astype(float)
    df["n_samples"] = df["n_samples"].astype(int)

    return df

# =========================
# 2. LLM: NL QUESTION → STRUCTURED QUERY
# =========================

def llm_question_to_query(question: str) -> Dict[str, Any]:
    """
    Ask the LLM to turn a free-text question into a JSON query spec.

    Returns a dict with keys:
      country (str or null)
      mutation (str or null)
      year_min (int or null)
      year_max (int or null)
    You can extend this schema as needed.
    """
    system_msg = {
        "role": "system",
        "content": (
            "You convert user questions about malaria genomic prevalence "
            "into a strict JSON query specification. "
            "Supported fields: country (string or null), mutation (string or null), "
            "year_min (integer or null), year_max (integer or null). "
            "If a field is not specified in the question, set it to null. "
            "Return ONLY valid JSON, no extra text."
        ),
    }
    user_msg = {
        "role": "user",
        "content": question,
    }

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[system_msg, user_msg],
        temperature=0.0,
    )

    raw = resp.choices[0].message.content.strip()
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


# =========================
# 3. APPLY QUERY TO YOUR DATA
# =========================

def filter_data(df: pd.DataFrame, query: Dict[str, Any]) -> pd.DataFrame:
    subset = df.copy()

    if query.get("country"):
        subset = subset[subset["country"].str.lower() ==
                        query["country"].lower()]

    if query.get("mutation"):
        # Assuming mutation column has strings like "K13_561H"
        subset = subset[subset["mutation"].str.lower() ==
                        query["mutation"].lower()]

    if query.get("year_min") is not None:
        subset = subset[subset["year"] >= int(query["year_min"])]

    if query.get("year_max") is not None:
        subset = subset[subset["year"] <= int(query["year_max"])]

    return subset


def summarize_prevalence(subset: pd.DataFrame) -> Dict[str, Any]:
    """
    Produce a compact summary of the numeric + geographic patterns
    to feed into the LLM.
    """
    if subset.empty:
        return {"has_data": False}

    # Yearly prevalence summary
    year_grp = (
        subset.groupby("year")
        .apply(
            lambda g: pd.Series(
                {
                    "n_samples": int(g["n_samples"].sum()),
                    "mean_prevalence": float(
                        (g["prevalence"] * g["n_samples"]).sum()
                        / max(g["n_samples"].sum(), 1)
                    ),
                }
            )
        )
        .reset_index()
        .sort_values("year")
    )

    # Regional/site summary (if you have region column, swap that in)
    site_grp = (
        subset.groupby("site")
        .apply(
            lambda g: pd.Series(
                {
                    "n_samples": int(g["n_samples"].sum()),
                    "mean_prevalence": float(
                        (g["prevalence"] * g["n_samples"]).sum()
                        / max(g["n_samples"].sum(), 1)
                    ),
                    "first_year": int(g["year"].min()),
                    "last_year": int(g["year"].max()),
                }
            )
        )
        .reset_index()
        .sort_values("mean_prevalence", ascending=False)
    )

    # Study-level metadata
    study_grp = (
        subset.groupby(["study_id", "authors", "year_pub"])
        .agg(
            n_samples=("n_samples", "sum"),
            year_min=("year", "min"),
            year_max=("year", "max"),
            mean_prev=("prevalence", "mean"),
        )
        .reset_index()
        .sort_values("year_pub")
    )

    return {
        "has_data": True,
        "yearly": year_grp.to_dict(orient="records"),
        "by_site": site_grp.to_dict(orient="records"),
        "by_study": study_grp.to_dict(orient="records"),
    }


# =========================
# 4. LLM: DATA SUMMARY → NARRATIVE ANSWER
# =========================

def llm_generate_answer(
    question: str,
    query: Dict[str, Any],
    summary: Dict[str, Any],
) -> str:
    """
    Ask the LLM to produce a scientific-style narrative based ONLY
    on the provided summary.
    """
    system_msg = {
        "role": "system",
        "content": (
            "You are an assistant helping a malaria genomicist interpret pre-computed "
            "prevalence data and study metadata. You must ONLY use the numeric summaries "
            "and study information provided. Do not invent new data or new studies. "
            "If data are sparse or missing, say so explicitly. "
            "Write clearly in a scientific but concise style."
        ),
    }

    # Provide structured summary as JSON so the model can 'see' the numbers
    context = {
        "original_question": question,
        "parsed_query": query,
        "data_summary": summary,
    }

    user_msg = {
        "role": "user",
        "content": (
            "Here is the structured data for the user's question:\n\n"
            + json.dumps(context, indent=2)
            + "\n\n"
            "1. Summarize the temporal trend (if any).\n"
            "2. Comment on geographic/site differences.\n"
            "3. Cite studies as 'Authors year_pub' where relevant.\n"
            "4. Mention important caveats (e.g. few samples, uneven sites).\n"
        ),
    }

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[system_msg, user_msg],
        temperature=0.2,
    )

    answer = resp.choices[0].message.content.strip()
    return answer


# =========================
# 5. MAIN ENTRYPOINT
# =========================

def answer_question(df: pd.DataFrame, question: str) -> Tuple[Dict[str, Any], str]:
    """
    High-level pipeline:
      question -> query -> filtered data -> summary -> LLM narrative
    Returns (query_dict, answer_text).
    """
    print(f"\nUser question: {question}\n")

    # 1) natural language -> structured query
    query = llm_question_to_query(question)
    print("Parsed query:", query)

    # 2) filter data
    subset = filter_data(df, query)
    print(f"Subset size: {len(subset)} rows")

    # 3) summarize
    summary = summarize_prevalence(subset)

    # 4) LLM narrative
    answer = llm_generate_answer(question, query, summary)
    return query, answer


if __name__ == "__main__":
    # Load your data once
    df_all = load_data(DATA_PATH)

    # Example question – replace with anything you like
    question = (
        "What is the trend of K13 561H in Uganda after 2012, and how consistent "
        "are the findings across study sites and studies?"
    )

    query_dict, answer_text = answer_question(df_all, question)

    print("\n=== Structured query ===")
    print(json.dumps(query_dict, indent=2))

    print("\n=== LLM answer ===\n")
    print(answer_text)