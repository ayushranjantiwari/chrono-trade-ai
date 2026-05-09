import pandas as pd

SPLIT_DATE = pd.Timestamp("2024-01-01")

TARGET = "target"

EXOG_FEATURES = [
    "rolling_std_5",
    "hl_spread",
    "volume_change",
    "volatility_10",
    "ma_ratio",
    "price_momentum_5"
]