import warnings
warnings.filterwarnings('ignore')

import itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


class SARIMAModel:

    def __init__(self):
        self.model = None
        self.order = None
        self.seasonal_order = None

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

        # prevent extreme instability
        s = np.clip(s, -10, 10)

        return s

    # =====================
    # HYPERPARAMETER TUNING
    # =====================
    def tune_sarima(self, y):

        p = q = range(0, 3)
        d = [0, 1]

        P = Q = range(0, 2)
        D = [0, 1]

        s = 5  # weekly (trading days)

        best_aic = float("inf")
        best_params = None

        for order in itertools.product(p, d, q):
            for seasonal in itertools.product(P, D, Q):

                seasonal_order = (*seasonal, s)

                try:
                    model = SARIMAX(
                        y,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=True,
                        enforce_invertibility=True
                    )

                    result = model.fit(disp=False, maxiter=200)

                    # skip non-converged
                    if not result.mle_retvals.get("converged", True):
                        continue

                    # skip invalid AIC
                    if np.isnan(result.aic) or np.isinf(result.aic):
                        continue

                    # mproved stability check
                    resid_var = np.var(result.resid)
                    data_var = np.var(y)

                    if resid_var > 10 * data_var:
                        continue

                    if result.aic < best_aic:
                        best_aic = result.aic
                        best_params = (order, seasonal_order)

                except Exception:
                    continue

        return best_params

    # =====================
    # FIT MODEL
    # =====================
    def fit(self, train):

        y = self._clean_series(train['target'])

        best = self.tune_sarima(y)

        if best is None:
            raise ValueError("SARIMA tuning failed.")

        self.order, self.seasonal_order = best

        print(f"SARIMA selected order: {self.order}, seasonal: {self.seasonal_order}")

        self.model = SARIMAX(
            y,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=True,
            enforce_invertibility=True
        ).fit(disp=False, maxiter=200)

    # =====================
    # PREDICT
    # =====================
    def predict(self, steps):

        if self.model is None:
            raise ValueError("Model not trained or loaded.")

        forecast = self.model.forecast(steps=steps)

        forecast = np.asarray(forecast, dtype=float)

        # clean output
        forecast = np.nan_to_num(forecast)
        forecast = np.clip(forecast, -10, 10)

        return forecast