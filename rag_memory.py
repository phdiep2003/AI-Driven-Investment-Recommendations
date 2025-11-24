# ============================================================================
# FIXED / IMPROVED RAG MEMORY ENGINE
# ============================================================================

import json
import uuid
import numpy as np
from typing import List, Dict

RAG_MEMORY_FILE = "rag_memory.jsonl"

# ----------------------------------------------------------------------
# Write RAG entry
# ----------------------------------------------------------------------
def rag_add_example(profile, a1_output, qout, eval_port, correct: bool, notes: str):
    """
    Save a portfolio evaluation case to the RAG memory file with
    structured fields supporting LLM safety.
    """

    entry = {
        "id": str(uuid.uuid4()),
        "profile": profile.model_dump(),
        "portfolio": a1_output.model_dump(),
        "quant": qout.model_dump(),
        "evaluation": eval_port.model_dump(),
        "correct": correct,
        "notes": notes,
        "mistakes": eval_port.mistakes if correct else [],
        "positive_patterns": [] if not correct else [
            "Balanced diversification",
            "Satisfies constraints",
            "Approved by PM"
        ],
    }

    with open(RAG_MEMORY_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ----------------------------------------------------------------------
# Load all memory
# ----------------------------------------------------------------------
def rag_load_all() -> List[dict]:
    try:
        with open(RAG_MEMORY_FILE, "r") as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


# ----------------------------------------------------------------------
# Scoring utility
# ----------------------------------------------------------------------
def _get_profile_vector(profile: dict) -> np.ndarray:
    """Creates a standardized numeric vector from the current CustomerProfile."""
    return np.array([
        {"low": 0, "medium": 1, "high": 2}.get(profile.get("risk_tolerance", "medium"), 1),
        len(profile.get("esg_flags", [])),
        {"Low": 0, "Moderate": 1, "High": 2}.get(profile.get("concentration_tolerance", "Moderate"), 1),
        profile.get("desired_liquid_portion_pct", 20),
    ], dtype=float)

def _score_profile_similarity(profile_vec_now: np.ndarray, entry: dict) -> float:
    """
    Calculates the Euclidean distance between the current profile vector
    and the stored entry's profile vector. Lower score is better.
    """
    prof = entry["profile"]
    prof_vec = _get_profile_vector(prof) # Use the same consistent vector creation logic
    return np.linalg.norm(profile_vec_now - prof_vec)

# ----------------------------------------------------------------------
# Agent 1 Retrieval (before tickers)
# ----------------------------------------------------------------------
def rag_retrieve_similar_for_agent1(profile, top_k=3) -> List[Dict]:
    """
    Retrieves the top_k most similar past portfolio evaluations that resulted
    in mistakes, providing Agent 1 with specific errors to avoid.
    """
    all_entries = rag_load_all()
    if not all_entries:
        return []

    # 1. Calculate the vector for the current profile
    # Ensure profile is converted to a dict if it's a Pydantic object
    profile_dict = profile.model_dump()
    profile_vec_now = _get_profile_vector(profile_dict)

    # 2. Score and filter entries
    scored_mistake_entries = []
    for entry in all_entries:
        # **Crucial filter**: Only consider entries that actually contained mistakes
        if entry.get("mistakes"):
            similarity_score = _score_profile_similarity(profile_vec_now, entry)
            scored_mistake_entries.append({
                "score": similarity_score,
                "mistakes": entry["mistakes"],
                "evaluation_summary": entry["evaluation"].get("risk_exposure_summary", "N/A"),
            })

    # 3. Sort by score (ascending: most similar first)
    scored_mistake_entries.sort(key=lambda x: x["score"])

    # 4. Structure the output, taking only the top_k
    retrieved_data = []
    for entry in scored_mistake_entries:
        retrieved_data.append({
            "similarity_score": round(entry["score"], 4),
            "past_mistakes_to_avoid": entry["mistakes"], # Renamed key for clarity
            "evaluation_summary_of_failed_case": entry["evaluation_summary"], # Renamed key for clarity
        })
        if len(retrieved_data) >= top_k:
            break

    return retrieved_data