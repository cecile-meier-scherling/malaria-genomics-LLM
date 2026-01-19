import pandas as pd
from typing import Dict, Any

def filter_data(df: pd.DataFrame, query: Dict[str, Any]) -> pd.DataFrame:
    subset = df.copy()

    if query.get("country"):
        subset = subset[subset["country"].str.lower() ==
                        query["country"].lower()]

    if query.get("mutation"):
        norm = normalize_mutation_label(query["mutation"])
        # df["mutation"] is like '675V'
        subset = subset[subset["mutation"].str.upper() == norm]

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
        subset.groupby(["site", "country"])
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

def normalize_mutation_label(label: str) -> str:
    """
    Normalize a mutation string like:
      - 'K13 675V'
      - 'k13:675:V'
      - '675V'
    into a common form matching df['mutation'], e.g. '675V'.
    """
    if label is None:
        return None

    s = label.strip()
    # Replace separators with spaces
    s = s.replace("_", " ")
    s = s.replace(":", " ")
    parts = s.split()

    # Look for piece like '675V'
    for p in parts:
        if len(p) >= 2 and p[:-1].isdigit() and p[-1].isalpha():
            return p.upper()

    # Fallback: just return stripped, uppercased
    return s.upper()