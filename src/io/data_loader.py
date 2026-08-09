"""
Responsible for loading the Stack Overflow temporal network dataset.
"""

import pandas as pd
from config import DEBUG_MODE, DEBUG_MAX_ROWS


def load_dataset(file_path: str) -> pd.DataFrame:

    if DEBUG_MODE:

        print(f"DEBUG MODE: Loading first {DEBUG_MAX_ROWS:,} rows...")

        df = pd.read_csv(
            file_path,
            sep=" ",
            header=None,
            names=["source", "target", "timestamp"],
            nrows=DEBUG_MAX_ROWS)

    else:

        print("PRODUCTION MODE: Loading complete dataset...")

        df = pd.read_csv(
            file_path,
            sep=" ",
            header=None,
            names=["source", "target", "timestamp"])

    return df