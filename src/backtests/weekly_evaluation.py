import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.data.load_data import load_raw_data
from src.features.build_features import create_features
from src.evaluation.metrics import regression_metrics, direction_accuracy


def evaluate_last_week():

    # ----------------------
    # Load predictions
    # ----------------------
    file_path = "results/future_predictions.csv"

    if not os.path.exists(file_path):
        print("No predictions found. Skipping evaluation.")
        return

    preds = pd.read_csv(file_path)

    if preds.empty:
        print("No predictions available.")
        return

    preds["Date"] = pd.to_datetime(preds["Date"])
    preds["Run_Date"] = pd.to_datetime(preds["Run_Date"])

    # ----------------------
    # Get last run
    # ----------------------
    last_run_date = preds["Run_Date"].max()
    last_preds = preds[preds["Run_Date"] == last_run_date].copy()

    print(f"\nEvaluating predictions for Run_Date: {last_run_date.date()}")

    # ----------------------
    # Load actual data
    # ----------------------
    df = load_raw_data()
    df = create_features(df)
    df = df.sort_index()

    df = df.reset_index()
    df.rename(columns={"index": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])

    # 🔥 FIX: use Close (NOT returns)
    actual_df = df[["Date", "Close"]].copy()

    # ----------------------
    # Merge predictions with actuals
    # ----------------------
    merged = pd.merge(
        last_preds,
        actual_df,
        on="Date",
        how="inner"
    )

    if merged.empty:
        print("⚠ No overlapping dates between predictions and actuals.")
        return

    actual = merged["Close"].values

    # ----------------------
    # Model list
    # ----------------------
    model_list = [
        "ARIMA",
        "SARIMA",
        "SARIMAX",
        "Prophet",
        "RandomForest",
        "XGBoost"
    ]

    results = []

    # ----------------------
    # Evaluate models
    # ----------------------
    for model in model_list:

        if model not in merged.columns:
            print(f"Skipping {model} (not found)")
            continue

        pred = merged[model].values

        if len(pred) != len(actual):
            print(f"Skipping {model} (length mismatch)")
            continue

        metrics = regression_metrics(actual, pred, is_log=False)

        results.append({
            "Model": model,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "Direction_Accuracy": direction_accuracy(actual, pred, is_log=False),
            "Run_Date": last_run_date.date()
        })

    result_df = pd.DataFrame(results)

    print("\nWeekly Evaluation Results:\n")
    print(result_df)

    # ----------------------
    # Save results
    # ----------------------
    os.makedirs("results", exist_ok=True)

    output_path = "results/weekly_performance.csv"
    file_exists = os.path.exists(output_path)

    result_df.to_csv(
        output_path,
        mode='a',
        header=not file_exists,
        index=False
    )

    print("\nWeekly evaluation saved.")

    return result_df