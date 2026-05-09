import warnings
warnings.filterwarnings('ignore')

from prophet import Prophet
import itertools
import pandas as pd
import numpy as np


class ProphetModel:

    def __init__(self):
        self.model = None
        self.best_params = None

    # =====================
    # CLEAN DATA
    # =====================
    def _prepare_df(self, df):
        df = df.copy().reset_index()

        if "Date" not in df.columns:
            raise ValueError("Date column missing")

        df["Date"] = pd.to_datetime(df["Date"])

        df = df[["Date", "target"]]
        df.columns = ["ds", "y"]

        # clean target
        df["y"] = df["y"].replace([np.inf, -np.inf], np.nan)
        df["y"] = df["y"].ffill().fillna(0)

        return df

    # =====================
    # HYPERPARAMETER TUNING
    # =====================
    def tune_prophet(self, train):

        param_grid = {
            'changepoint_prior_scale': [0.01, 0.1, 0.5],
            'seasonality_prior_scale': [1.0, 5.0, 10.0],
            'seasonality_mode': ['additive', 'multiplicative']
        }

        all_params = list(itertools.product(
            param_grid['changepoint_prior_scale'],
            param_grid['seasonality_prior_scale'],
            param_grid['seasonality_mode']
        ))

        best_score = float("inf")
        best_params = None

        split = int(len(train) * 0.8)
        train_part = train.iloc[:split]
        val_part = train.iloc[split:]

        train_df = self._prepare_df(train_part)
        val_df = self._prepare_df(val_part)

        for cps, sps, mode in all_params:
            try:
                model = Prophet(
                    changepoint_prior_scale=cps,
                    seasonality_prior_scale=sps,
                    seasonality_mode=mode
                )

                model.fit(train_df)

                forecast = model.predict(val_df)

                error = ((forecast['yhat'] - val_df['y']) ** 2).mean()

                if np.isnan(error) or np.isinf(error):
                    continue

                if error < best_score:
                    best_score = error
                    best_params = (cps, sps, mode)

            except Exception:
                continue

        return best_params

    # =====================
    # FIT MODEL
    # =====================
    def fit(self, train):

        best = self.tune_prophet(train)

        if best is None:
            raise ValueError("Prophet tuning failed.")

        cps, sps, mode = best
        self.best_params = best

        train_df = self._prepare_df(train)

        self.model = Prophet(
            changepoint_prior_scale=cps,
            seasonality_prior_scale=sps,
            seasonality_mode=mode
        )

        self.model.fit(train_df)

    # =====================
    # CLEAN OUTPUT
    # =====================
    def _clean_array(self, arr):
        arr = np.asarray(arr, dtype=float)
        arr = np.nan_to_num(arr)
        arr = np.clip(arr, -10, 10)
        return arr

    # =====================
    # PREDICT
    # =====================
    def predict(self, steps=None, test=None):

        if self.model is None:
            raise ValueError("Model not trained or loaded.")

        # ---------------------
        # TEST EVALUATION
        # ---------------------
        if test is not None and steps is None:

            future = test.reset_index()[["Date"]].copy()
            future["Date"] = pd.to_datetime(future["Date"])
            future.columns = ["ds"]

            forecast = self.model.predict(future)

            return self._clean_array(forecast["yhat"])

        # ---------------------
        # FUTURE FORECAST
        # ---------------------
        if steps is not None and test is not None:

            last_date = pd.to_datetime(test.index[-1])

            future_dates = pd.date_range(
                start=last_date,
                periods=steps + 1,
                freq="B"
            )[1:]

            future = pd.DataFrame({"ds": future_dates})

            forecast = self.model.predict(future)

            return self._clean_array(forecast["yhat"])

        raise ValueError(
            "Use:\n"
            "predict(test=...) OR\n"
            "predict(test=..., steps=5)"
        )