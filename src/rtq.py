"""
Risk Tolerance Questionnaire → numeric score → profile bucket.
"""


from typing import Dict, List, Tuple

# ---- Move RTQ_QUESTIONS from UI to backend ----
RTQ_QUESTIONS = [
    {
        "id": "q1",
        "title": "Investment Objective and Return Preference",
        "purpose": "Measures trade-off between short-term volatility and long-term return.",
        "choices": [
            {"label": "I aim for high long-term returns even with short-term swings.", "score": 3},
            {"label": "I prefer balanced growth and moderate stability.", "score": 2},
            {"label": "I prefer steady, stable growth with minimal risk.", "score": 1},
        ],
    },
    {
        "id": "q2",
        "title": "Response to Portfolio Losses",
        "purpose": "Evaluates emotional reaction to loss.",
        "choices": [
            {"label": "Stay invested; downturns are temporary.", "score": 3},
            {"label": "Reconsider the portfolio but remain invested.", "score": 2},
            {"label": "Sell to avoid further losses.", "score": 1},
        ],
    },
    # TODO: Add more from your teammate’s file...
]


# ---- Aggregation ----
def score_rtq(form_response: Dict) -> int:
    """
    Compute raw RTQ score based on form responses.
    """
    total = 0
    for q in RTQ_QUESTIONS:
        val = form_response.get(q["id"])
        total += int(val) if val is not None else 0
    return total


# ---- Interpretation buckets ----
RISK_BUCKETS = [
    {"range": (0, 7), "label": "Conservative", "desc": "Low tolerance for volatility."},
    {"range": (8, 14), "label": "Moderate", "desc": "Balanced investing approach."},
    {"range": (15, 21), "label": "Aggressive", "desc": "Seeks higher returns despite volatility."},
]


def categorize_risk(score: int) -> Tuple[str, str]:
    """
    Return (label, description)
    """
    for b in RISK_BUCKETS:
        lo, hi = b["range"]
        if lo <= score <= hi:
            return b["label"], b["desc"]

    return "Unclassified", "Score outside expected range."
