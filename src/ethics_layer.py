"""
Apply CFA + Shariah constraints to the universe of assets.
"""

import pandas as pd
from typing import Dict

def apply_constraints(
    universe: pd.DataFrame,
    rag_rules: Dict,
    user_constraints: Dict = None
) -> pd.DataFrame:
    """
    Remove assets that violate rules.
    """
    filtered = universe.copy()
    # TODO: filter logic
    return filtered
