import numpy as np
import pandas as pd


def create_features(df):
    df = df.copy()

    # =========================
    # TARGET (LOG PRICE)
    # =========================
    df["target"] = np.log(df["Close"])

    # =========================
    # LAG FEATURES
    # =========================
    for lag in range(1, 6):
        df[f"lag_{lag}"] = df["target"].shift(lag)

    # =========================
    # ROLLING FEATURES
    # =========================
    df["rolling_std_5"] = df["target"].rolling(5).std()
    df["rolling_std_10"] = df["target"].rolling(10).std()

    # =========================
    # PRICE FEATURES
    # =========================
    df["hl_spread"] = (df["High"] - df["Low"]) / df["Close"]
    df["oc_spread"] = (df["Open"] - df["Close"]) / df["Close"]

    # =========================
    # VOLUME FEATURES
    # =========================
    df["volume_change"] = df["Volume"].pct_change()

    # =========================
    # TREND
    # =========================
    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_10"] = df["Close"].rolling(10).mean()
    df["ma_ratio"] = df["ma_5"] / df["ma_10"]

    # =========================
    # MOMENTUM
    # =========================
    df["price_momentum_5"] = df["Close"] / df["Close"].shift(5)

    # =========================
    # 🔥 FINAL CLEAN (MOST IMPORTANT)
    # =========================
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 🔥 Prevent overflow later
    df["target"] = df["target"].clip(-10, 10)

    df = df.dropna()

    return df