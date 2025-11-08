"""
Log system inputs/outputs for transparency + auditability.
"""

import json
from datetime import datetime

def log_session(data: dict, path: str = "logs/") -> None:
    """
    Save JSON with timestamp.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"{path}/session_{timestamp}.json", "w") as f:
        json.dump(data, f, indent=2)
