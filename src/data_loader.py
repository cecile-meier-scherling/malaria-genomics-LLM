import pandas as pd

from . import config

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
        "country", "site", "year", "gene", "mutation",
        "prevalence", "n_samples", "study_id",
        "authors", "year_pub", "url"
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")

    # Ensure types
    df["year"] = df["year"].astype(int)
    # df["year_pub"] = df["year_pub"].astype(int)
    df["prevalence"] = df["prevalence"].astype(float)
    df["n_samples"] = df["n_samples"].astype(int)

    return df
