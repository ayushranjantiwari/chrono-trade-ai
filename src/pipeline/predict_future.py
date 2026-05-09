import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from datetime import datetime

from src.data.load_data import load_raw_data
from src.features.build_features import create_features

from src.models.arima import ARIMAModel
from src.models.sarima import SARIMAModel
from src.models.sarimax import SARIMAXModel
from src.models.prophet_model import ProphetModel
from src.models.ml_models import RandomForestModel, XGBoostModel

from src.utils.model_io import save_model  


def predict_next_5_days():

    # =====================
    # LOAD + FEATURES
    # =====================
    df = load_raw_data()
    df = create_features(df)
    df = df.sort_index().dropna()

    predictions = {}

    # =====================
    # MODEL REGISTRY (SCALABLE)
    # =====================
    models = {
        "ARIMA": ARIMAModel(),
        "SARIMA": SARIMAModel(),
        "SARIMAX": SARIMAXModel(),
        "Prophet": ProphetModel(),
        "RandomForest": RandomForestModel(),
        "XGBoost": XGBoostModel()
    }

    # =====================
    # TRAIN ON FULL DATA + SAVE FINAL + PREDICT
    # =====================
    for name, model in models.items():

        print(f"\nRunning {name} (FINAL TRAINING)...")

        try:
            # 🔥 Train on FULL data
            model.fit(df)

            # 🔥 Save FINAL model separately
            save_model(model, f"{name}_final")

            # =====================
            # PREDICTION HANDLING
            # =====================
            if name == "SARIMAX":
                preds = model.predict(test=df.tail(5), steps=5)

            elif name == "Prophet":
                preds = model.predict(test=df.tail(5), steps=5)

            elif name in ["ARIMA", "SARIMA"]:
                preds = model.predict(5)

            else:
                preds = model.predict_future(df, steps=5)

            # =====================
            # CLEAN OUTPUT (CRITICAL)
            # =====================
            preds = np.asarray(preds, dtype=float)
            preds = np.nan_to_num(preds)
            preds = np.clip(preds, -10, 10)

            predictions[name] = preds[:5]

        except Exception as e:
            print(f"❌ Error in {name}: {e}")

    # =====================
    # CREATE DATAFRAME
    # =====================
    pred_df = pd.DataFrame(predictions)

    model_cols = list(models.keys())
    pred_df = pred_df[model_cols]

    # =====================
    # DATE CREATION
    # =====================
    pred_df["Date"] = pd.date_range(
        start=df.index[-1],
        periods=6,
        freq="B"
    )[1:]

    pred_df["Run_Date"] = datetime.today().date()

    pred_df = pred_df[["Date"] + model_cols + ["Run_Date"]]

    print("\nFuture Predictions Preview:\n")
    print(pred_df.head())

    # =====================
    # SAVE RESULTS
    # =====================
    os.makedirs("results", exist_ok=True)

    file_path = "results/future_predictions.csv"

    if os.path.exists(file_path):
        try:
            existing = pd.read_csv(file_path)

            if list(existing.columns) != list(pred_df.columns):
                print("⚠ Column mismatch detected → overwriting file")
                pred_df.to_csv(file_path, index=False)
            else:
                pred_df.to_csv(file_path, mode='a', header=False, index=False)

        except Exception:
            print("⚠ Corrupted CSV detected → recreating file")
            pred_df.to_csv(file_path, index=False)

    else:
        pred_df.to_csv(file_path, index=False)

    print("\nFuture predictions saved.")

    return pred_df