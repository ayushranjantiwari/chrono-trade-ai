import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import sys, os
 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from datetime import datetime

from src.data.load_data import load_raw_data
from src.data.split_data import train_test_split_time
from src.features.build_features import create_features
from src.utils.model_io import save_model

from src.models.arima import ARIMAModel
from src.models.sarima import SARIMAModel
from src.models.sarimax import SARIMAXModel
from src.models.prophet_model import ProphetModel
from src.models.ml_models import RandomForestModel, XGBoostModel

from src.evaluation.metrics import regression_metrics
from src.config import SPLIT_DATE


# =====================
# BASELINE MODEL
# =====================
def baseline_predict(test):
    """
    Naive baseline:
    tomorrow ≈ today
    """
    preds = test["target"].shift(1)
    preds = preds.fillna(0)
    return preds.values


# =====================
# HELPER
# =====================
def evaluate_model(name, test, preds):

    preds = pd.Series(preds).reset_index(drop=True)
    actual = test["target"].reset_index(drop=True)

    # 🔥 SANITY CHECKS
    if np.isnan(preds).any() or np.isinf(preds).any():
        raise ValueError(f"{name} predictions contain NaN/Inf")

    preds = preds.clip(-10, 10)
    actual = actual.clip(-10, 10)

    preds_price = np.exp(preds)
    actual_price = np.exp(actual)

    min_len = min(len(preds_price), len(actual_price))
    preds_price = preds_price.iloc[:min_len]
    actual_price = actual_price.iloc[:min_len]

    metrics = regression_metrics(actual_price, preds_price, is_log=False)

    direction = (preds_price.diff() > 0) == (actual_price.diff() > 0)
    direction_acc = direction.mean()

    return {
        "Model": name,
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "Direction_Accuracy": direction_acc
    }


# =====================
# MAIN PIPELINE
# =====================
def run_pipeline():

    df = load_raw_data()
    df = create_features(df)
    df = df.sort_index().dropna()

    train, test = train_test_split_time(df, SPLIT_DATE)

    results = []
    trained_models = {}
    
    # 🔥 NEW: store predictions for ensemble
    preds_store = {}

    print("\n=== Phase 1: Train/Test Evaluation ===")

    models = {
        "ARIMA": ARIMAModel(),
        "SARIMA": SARIMAModel(),
        "SARIMAX": SARIMAXModel(),
        "Prophet": ProphetModel(),
        "RandomForest": RandomForestModel(),
        "XGBoost": XGBoostModel()
    }

    # =====================
    # BASELINE
    # =====================
    print("\nRunning BASELINE...")

    try:
        baseline_preds = baseline_predict(test)

        print(f"BASELINE → preds_len: {len(baseline_preds)}, test_len: {len(test)}")

        results.append(evaluate_model("BASELINE", test, baseline_preds))

    except Exception as e:
        print(f"❌ Error in BASELINE: {e}")

    # =====================
    # MODELS
    # =====================
    for name, model in models.items():

        print(f"\nRunning {name}...")

        try:
            model.fit(train)
            trained_models[name] = model

            if name == "SARIMAX":
                preds = model.predict(test=test)

            elif name == "Prophet":
                preds = model.predict(test=test)

            elif name in ["ARIMA", "SARIMA"]:
                preds = model.predict(len(test))

            else:
                preds = model.predict(test)

            # CLEAN
            preds = np.asarray(preds, dtype=float)
            preds = np.nan_to_num(preds)
            preds = np.clip(preds, -10, 10)

            print(f"{name} → preds_len: {len(preds)}, test_len: {len(test)}")

            results.append(evaluate_model(name, test, preds))

            # 🔥 STORE FOR ENSEMBLE
            preds_store[name] = preds

        except Exception as e:
            print(f"❌ Error in {name}: {e}")

    # =====================
    # 🔥 ENSEMBLE MODEL (NEW)
    # =====================
    print("\nRunning ENSEMBLE...")

    try:
        preds_df = pd.DataFrame(preds_store)

        # simple average
        ensemble_preds = preds_df.mean(axis=1).values

        print(f"ENSEMBLE → preds_len: {len(ensemble_preds)}, test_len: {len(test)}")

        results.append(evaluate_model("ENSEMBLE", test, ensemble_preds))

    except Exception as e:
        print(f"❌ Error in ENSEMBLE: {e}")

    # =====================
    # RESULTS
    # =====================
    results_df = pd.DataFrame(results)

    print("\nModel Performance (TEST):\n")
    print(results_df)

    best_model = results_df.sort_values("RMSE").iloc[0]
    print("\nBest Model on TEST:", best_model["Model"])

    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/model_performance.csv", index=False)

    # =====================
    # FINAL TRAINING
    # =====================
    print("\n=== Phase 2: Training FINAL models on FULL data ===")

    for name, model in models.items():

        print(f"Training FINAL {name}...")

        try:
            model.fit(df)
            save_model(model, f"{name}_final", overwrite=True)

        except Exception as e:
            print(f"❌ Error saving final {name}: {e}")

    print("\n✅ All final models saved.")

    return results_df