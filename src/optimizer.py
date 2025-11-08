"""
Portfolio optimizer — maximize Sharpe or minimize risk under constraints.
"""

import pandas as pd
from typing import Dict

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns.
    """
    return prices.pct_change().dropna()


def optimize_portfolio(filtered_assets: pd.DataFrame, risk_profile: str) -> Dict:
    """
    Return dict: {ticker: weight}
    """
    # TODO: PyPortfolioOpt / cvxpy
    return {}
