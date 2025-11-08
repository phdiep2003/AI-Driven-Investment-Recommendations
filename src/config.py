"""
Global configuration settings and paths.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# API KEYS
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Data paths
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Vector Store
VECTOR_DB_PATH = "data/vector_store"

# Default benchmark
BENCHMARK = "^GSPC"  # S&P 500
