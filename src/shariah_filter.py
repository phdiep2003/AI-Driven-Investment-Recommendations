# src/shariah_filter.py

import pandas as pd
from typing import Optional


def load_shariah_labels(path: str) -> pd.DataFrame:
    """
    Load Shariah labels from CSV.
    Expected columns: ticker, is_shariah, reason
    - Normalize tickers to uppercase.
    - Convert is_shariah to bool.
    """
    df = pd.read_csv(path)

    # Normalize ticker
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    # Convert is_shariah to bool
    df["is_shariah"] = df["is_shariah"].astype(str).str.lower().isin(["1", "true", "yes"])

    # Ensure reason exists
    if "reason" not in df.columns:
        df["reason"] = ""

    return df[["ticker", "is_shariah", "reason"]]


def merge_shariah_labels(prices: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """
    From price universe + label table,
    return a DataFrame indexed by ticker with columns:
        sector (optional placeholder)
        is_shariah
        reason
        avg_volume (placeholder)
        compliance_score

    - If ticker is not covered in labels, assign is_shariah=False, reason="unknown".
    """

    tickers = prices.columns
    tmp = pd.DataFrame({"ticker": tickers})
    tmp["ticker"] = tmp["ticker"].str.upper().str.strip()

    # Merge
    merged = tmp.merge(labels, how="left", on="ticker")

    # Default values for missing
    merged["is_shariah"] = merged["is_shariah"].fillna(False)
    merged["reason"] = merged["reason"].fillna("unknown")

    # Placeholder sector + avg_volume
    merged["sector"] = None
    merged["avg_volume"] = None

    # Compliance score
    merged["compliance_score"] = merged["is_shariah"].astype(int)

    # Use ticker as index
    merged = merged.set_index("ticker")

    return merged[["sector", "is_shariah", "reason", "avg_volume", "compliance_score"]]


def filter_shariah_universe(
    universe: pd.DataFrame,
    strict: bool = False
) -> pd.DataFrame:
    """
    Return subset of universe that is Shariah-compliant.

    universe: output of merge_shariah_labels
    strict=False → keep “unknown” tickers (but they have score=0)
    strict=True → remove unknown tickers

    Returns a filtered DataFrame.
    """
    universe = universe.copy()

    if strict:
        # Keep only those labeled compliant (1)
        return universe[universe["is_shariah"] == True]

    # Non-strict: keep all
    return universe
