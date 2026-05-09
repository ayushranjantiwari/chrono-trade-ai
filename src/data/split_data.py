import pandas as pd

def train_test_split_time(df, split_date="2025-01-01"):

    # =====================
    # 1. Ensure datetime index
    # =====================
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex")

    # =====================
    # 2. Convert split_date
    # =====================
    split_date = pd.to_datetime(split_date)

    # =====================
    # 3. Perform split
    # =====================
    train = df[df.index < split_date]
    test = df[df.index >= split_date]

    # =====================
    # 4. Ensure frequency
    # =====================
    train = train.asfreq('B').ffill()
    test = test.asfreq('B').ffill()

    # =====================
    # 5. Safety checks
    # =====================
    if len(train) == 0:
        raise ValueError("Train set is empty. Check split_date.")

    if len(test) == 0:
        raise ValueError("Test set is empty. Check split_date.")

    # =====================
    # 6. Debug info
    # =====================
    print(f"Train: {train.index.min()} → {train.index.max()} | {len(train)} rows")
    print(f"Test:  {test.index.min()} → {test.index.max()} | {len(test)} rows")

    return train, test