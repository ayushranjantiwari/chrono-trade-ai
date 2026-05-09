import warnings
warnings.filterwarnings('ignore')

import itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


class ARIMAModel:

    def __init__(self):
        self.model = None
        self.order = None

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
    # TUNING
    # =====================
    def tune_arima(self, train):

        y = self._clean_series(train["target"])

        p = q = range(0, 3)
        d = [0, 1]

        best_aic = float("inf")
        best_order = None

        for order in itertools.product(p, d, q):
            try:
                model = ARIMA(y, order=order)
                result = model.fit(method_kwargs={"maxiter": 200})

                if np.isnan(result.aic) or np.isinf(result.aic):
                    continue

                # skip unstable models
                if np.var(result.resid) > 10:
                    continue

                if result.aic < best_aic:
                    best_aic = result.aic
                    best_order = order

            except Exception:
                continue

        return best_order

    # =====================
    # FIT
    # =====================
    def fit(self, train):

        y = self._clean_series(train["target"])

        self.order = self.tune_arima(train)

        if self.order is None:
            raise ValueError("ARIMA tuning failed.")

        self.model = ARIMA(y, order=self.order).fit(
            method_kwargs={"maxiter": 200}
        )

    # =====================
    # PREDICT
    # =====================
    def predict(self, steps):

        if self.model is None:
            raise ValueError("Model not trained or loaded.")

        forecast = self.model.forecast(steps=steps)

        forecast = np.asarray(forecast, dtype=float)
        forecast = np.nan_to_num(forecast)
        forecast = np.clip(forecast, -10, 10)

        return forecast