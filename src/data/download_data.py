import os
import pandas as pd
import yfinance as yf
from datetime import datetime
import time


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures columns are a flat Index with standard OHLCV names.
    """
    df = df.copy()

    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    rename_map = {
        "Adj Close": "Adj_Close",
        "adj close": "Adj_Close",
        "close": "Close",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "volume": "Volume",
    }

    df.columns = [rename_map.get(str(c), str(c)) for c in df.columns]

    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep]

    # Ensure datetime index
    df.index = pd.to_datetime(df.index)

    # Remove timezone if present
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = df.sort_index()

    # Remove duplicates
    df = df[~df.index.duplicated(keep="first")]

    return df


def download_stock_data(tickers, start, end=None, interval="1d"):

    os.makedirs(RAW_DIR, exist_ok=True)

    if isinstance(tickers, str):
        tickers = [tickers]

    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    for t in tickers:
        print(f"\nDownloading {t} ...")

        # Retry mechanism
        for attempt in range(3):
            try:
                df = yf.download(
                    t,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=True,  
                    progress=False
                )
                break
            except Exception as e:
                print(f"Retry {attempt+1} failed: {e}")
                time.sleep(2)
        else:
            print(f"❌ Failed to download {t}")
            continue

        if df.empty:
            print(f"⚠️ No data for {t}")
            continue

        df = _clean_columns(df)

        df["Ticker"] = t

        save_path = os.path.join(RAW_DIR, f"{t.replace('.', '_')}.csv")
        df.to_csv(save_path)

        print(f"Saved → {save_path}")
        print(f"Date Range: {df.index.min()} → {df.index.max()} | Rows: {len(df)}")


if __name__ == "__main__":
    download_stock_data(
        tickers=["RELIANCE.NS"],
        start="2015-01-01",
        end=None,
        interval="1d"
    )