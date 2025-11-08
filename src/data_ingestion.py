# src/data_ingestion.py

from __future__ import annotations
from typing import List
import pandas as pd
import yfinance as yf
import os
from datetime import datetime
import json
import requests

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _log(msg: str) -> None:
    """Simple logger."""
    print(f"[data_ingestion] {msg}")


# ------------------------------------------------------------
# Download Functions
# ------------------------------------------------------------

def download_prices(tickers: List[str], period: str = "5y") -> pd.DataFrame:
    """
    Download Adjusted Close prices from Yahoo Finance.

    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols.
    period : str, optional
        Period to fetch (default is "5y").

    Returns
    -------
    pd.DataFrame
        DataFrame with Date as index and tickers as columns.
    """
    if not isinstance(tickers, list) or len(tickers) == 0:
        _log("No tickers provided.")
        return pd.DataFrame()

    _log(f"Downloading prices for: {tickers} over period={period}")

    data = {}
    for t in tickers:
        try:
            ticker_obj = yf.Ticker(t)
            hist = ticker_obj.history(period=period)
            if hist.empty:
                _log(f"WARNING: No data for ticker {t}")
                continue
            data[t] = hist["Close"] if "Close" in hist else hist.iloc[:, -1]
        except Exception as e:
            _log(f"ERROR: Failed to download {t}: {e}")

    if not data:
        _log("No valid tickers downloaded.")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index)
    return df


def download_sp500(period: str = "5y") -> pd.DataFrame:
    """
    Download S&P 500 ticker list from Wikipedia, then price history.
    If fetching the ticker list fails, use fallback list.

    Parameters
    ----------
    period : str, optional
        Yahoo Finance period interval.

    Returns
    -------
    pd.DataFrame
        Adjusted close price DataFrame.
    """
    _log("Fetching S&P 500 tickers from Wikipedia...")
    try:
        wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(wiki_url)
        sp = tables[0]
        tickers = sp["Symbol"].tolist()
    except Exception as e:
        _log(f"ERROR: Unable to fetch S&P 500 from Wikipedia: {e}")
        _log("Using fallback ticker list.")
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

    return download_prices(tickers, period)


# ------------------------------------------------------------
# Cleaning
# ------------------------------------------------------------

def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean price DataFrame:
    - Drop tickers with >10% missing
    - Forward-fill then back-fill missing dates

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    if df.empty:
        _log("WARNING: Input DataFrame is empty.")
        return df

    # Remove tickers with >10% missing
    missing_frac = df.isna().mean()
    keep = missing_frac[missing_frac <= 0.10].index
    dropped = set(df.columns) - set(keep)
    if dropped:
        _log(f"Dropping tickers with >10% missing: {dropped}")

    df_clean = df[keep]

    # Forward-fill then back-fill
    df_clean = df_clean.ffill().bfill()

    return df_clean


# ------------------------------------------------------------
# Save / Load
# ------------------------------------------------------------

def save_prices(df: pd.DataFrame, path: str) -> None:
    """
    Save prices to CSV.

    Parameters
    ----------
    df : pd.DataFrame
    path : str
    """
    try:
        df.to_csv(path)
        _log(f"Saved prices to {path}")
    except Exception as e:
        _log(f"ERROR: Could not save to {path}: {e}")


def load_prices(path: str) -> pd.DataFrame:
    """
    Load prices from CSV.

    Parameters
    ----------
    path : str

    Returns
    -------
    pd.DataFrame
    """
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        _log(f"Loaded prices from {path}")
        return df
    except Exception as e:
        _log(f"ERROR: Could not load {path}: {e}")
        return pd.DataFrame()
