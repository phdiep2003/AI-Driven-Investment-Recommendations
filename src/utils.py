"""
General helper utilities.
"""

import pandas as pd

def normalize_weights(weights: dict) -> dict:
    """
    Ensure weights sum to 1.
    """
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}
