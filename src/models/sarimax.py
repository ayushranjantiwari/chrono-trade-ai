import warnings
warnings.filterwarnings('ignore')

import itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


class SARIMAXModel:

    def __init__(self):
        self.model = None
        self.order = None
        self.exog_cols = None

    # =====================
    # CLEAN SERIES
    # =====================
    def _clean_series(self, series):

        if isinstance(series, pd.Series):
            s = series.values
        else:
            s = np.asarray(series)

        s = s.astype(float)
        s = np.nan_to_num(s)
        s = np.clip(s, -10, 10)

        return s

    # =====================
    # CLEAN EXOG
    # =====================
    def _clean_exog(self, df):
        df = df.copy()
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().fillna(0)
        return df

    # =====================
    # HYPERPARAMETER TUNING
    # =====================
    def tune_sarimax(self, train):

        p = q = range(0, 3)
        d = [0, 1]

        exog_options = [
            ['rolling_std_5'],
            ['rolling_std_5', 'hl_spread'],
            ['rolling_std_5', 'hl_spread', 'volume_change'],
            ['rolling_std_5', 'hl_spread', 'volume_change', 'volatility_10'],
            ['ma_ratio', 'price_momentum_5']
        ]

        best_aic = float("inf")
        best_config = None

        for order in itertools.product(p, d, q):
            for exog_cols in exog_options:

                if not all(col in train.columns for col in exog_cols):
                    continue

                try:
                    y = self._clean_series(train["target"])
                    exog = self._clean_exog(train[exog_cols])

                    model = SARIMAX(
                        y,
                        exog=exog,
                        order=order,
                        enforce_stationarity=True,
                        enforce_invertibility=True
                    )

                    result = model.fit(disp=False, maxiter=200)

                    # convergence check
                    if not result.mle_retvals.get("converged", True):
                        continue

                    if np.isnan(result.aic) or np.isinf(result.aic):
                        continue

                    # stability check
                    if np.var(result.resid) > 10 * np.var(y):
                        continue

                    if result.aic < best_aic:
                        best_aic = result.aic
                        best_config = (order, exog_cols)

                except Exception:
                    continue

        return best_config

    # =====================
    # FIT MODEL
    # =====================
    def fit(self, train):

        best = self.tune_sarimax(train)

        if best is None:
            raise ValueError("SARIMAX tuning failed.")

        self.order, self.exog_cols = best

        print(f"SARIMAX selected order: {self.order}")
        print(f"SARIMAX exog features: {self.exog_cols}")

        y = self._clean_series(train["target"])
        exog = self._clean_exog(train[self.exog_cols])

        self.model = SARIMAX(
            y,
            exog=exog,
            order=self.order,
            enforce_stationarity=True,
            enforce_invertibility=True
        ).fit(disp=False, maxiter=200)

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
    def predict(self, test=None, steps=None):

        if self.model is None:
            raise ValueError("Model not trained or loaded.")

        if self.exog_cols is None:
            raise ValueError("Exogenous columns not set.")

        # ---------------------
        # TEST EVALUATION
        # ---------------------
        if test is not None and steps is None:

            exog = self._clean_exog(test[self.exog_cols])

            forecast = self.model.forecast(
                steps=len(test),
                exog=exog
            )

            return self._clean_array(forecast)

        # ---------------------
        # FUTURE FORECAST
        # ---------------------
        if test is not None and steps is not None:

            exog = self._clean_exog(test[self.exog_cols])

            if len(exog) != steps:
                raise ValueError(
                    f"Exog length ({len(exog)}) must match steps ({steps})"
                )

            forecast = self.model.forecast(
                steps=steps,
                exog=exog
            )

            return self._clean_array(forecast)

        raise ValueError(
            "Use:\n"
            "predict(test=...) OR\n"
            "predict(test=..., steps=5)"
        )