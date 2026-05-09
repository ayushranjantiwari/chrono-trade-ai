import pandas as pd
import os


def load_raw_data(filename="RELIANCE_NS.csv"):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    path = os.path.join(base_dir, "data", "raw", filename)

    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")

    # =====================
    # 1. Sort by date
    # =====================
    df = df.sort_index()

    # =====================
    # 2. Remove duplicates
    # =====================
    df = df[~df.index.duplicated(keep="first")]

    # =====================
    # 3. Validate columns
    # =====================
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # =====================
    # 4. Handle missing values
    # =====================
    df = df.ffill()

    return df