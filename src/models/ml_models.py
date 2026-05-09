import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.metrics import mean_squared_error


# =====================
# BASE MODEL
# =====================
class BaseMLModel:

    def _prepare_features(self, df):
        features = [col for col in df.columns if col.startswith("lag_")]

        if not features:
            raise ValueError("No lag features found. Feature engineering issue.")

        return features

    def _clean_X(self, X):
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        return X

    def _safe_prepare_test(self, test):
        test = test.copy().reset_index(drop=True)

        # ensure features exist
        missing = [f for f in self.features if f not in test.columns]
        if missing:
            raise ValueError(f"Missing features in test data: {missing}")

        X = test[self.features]
        X = self._clean_X(X)

        X_scaled = self.scaler.transform(X)

        return X_scaled, len(test)


# =====================
# RANDOM FOREST
# =====================
class RandomForestModel(BaseMLModel):

    def __init__(self):
        self.model = None
        self.features = None
        self.scaler = StandardScaler()

    def fit(self, train):

        self.features = self._prepare_features(train)

        X = train[self.features]
        X = self._clean_X(X)

        y = train["target"]

        X_scaled = self.scaler.fit_transform(X)

        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [2, 3, 5],
            "min_samples_leaf": [1, 3, 5]
        }

        search = RandomizedSearchCV(
            RandomForestRegressor(random_state=42),
            param_distributions=param_grid,
            n_iter=5,
            cv=3,
            scoring="neg_mean_squared_error",
            random_state=42,
            n_jobs=-1
        )

        search.fit(X_scaled, y)

        self.model = search.best_estimator_

        print("\nRF Best Params:", search.best_params_)

    def predict(self, test):

        X_scaled, expected_len = self._safe_prepare_test(test)

        preds = self.model.predict(X_scaled)

        preds = np.asarray(preds, dtype=float)
        preds = np.nan_to_num(preds)
        preds = np.clip(preds, -10, 10)

        return preds[:expected_len]

    def predict_future(self, df, steps=5):

        preds = []
        temp_df = df.copy()

        lag_cols = [col for col in self.features if col.startswith("lag_")]
        max_lag = len(lag_cols)

        for _ in range(steps):

            last_row = temp_df.iloc[-1:].copy()

            X_scaled, _ = self._safe_prepare_test(last_row)

            pred = float(self.model.predict(X_scaled)[0])
            preds.append(pred)

            new_row = last_row.copy()
            new_row["target"] = pred

            for lag in range(max_lag, 1, -1):
                new_row[f"lag_{lag}"] = new_row[f"lag_{lag-1}"]

            new_row["lag_1"] = pred

            new_row = self._clean_X(new_row)

            next_index = temp_df.index[-1] + pd.offsets.BDay(1)
            new_row.index = [next_index]

            temp_df = pd.concat([temp_df, new_row])

        return np.array(preds)


# =====================
# XGBOOST
# =====================
class XGBoostModel(BaseMLModel):

    def __init__(self):
        self.model = None
        self.features = None
        self.scaler = StandardScaler()
        self.is_tuned = False
        self.best_params = None

    def _get_space(self):
        return {
            "n_estimators": hp.choice("n_estimators", [50, 100, 200]),
            "max_depth": hp.choice("max_depth", [2, 3, 5]),
            "learning_rate": hp.uniform("learning_rate", 0.01, 0.2),
            "subsample": hp.uniform("subsample", 0.7, 1.0),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.7, 1.0)
        }

    def _run_hyperopt(self, X, y):

        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        def objective(params):
            model = XGBRegressor(random_state=42, **params)
            model.fit(X_train, y_train)

            preds = model.predict(X_val)
            loss = mean_squared_error(y_val, preds)

            return {"loss": loss, "status": STATUS_OK}

        trials = Trials()

        best = fmin(
            fn=objective,
            space=self._get_space(),
            algo=tpe.suggest,
            max_evals=20,
            trials=trials,
            rstate=np.random.default_rng(42)
        )

        best_params = {
            "n_estimators": [50, 100, 200][best["n_estimators"]],
            "max_depth": [2, 3, 5][best["max_depth"]],
            "learning_rate": best["learning_rate"],
            "subsample": best["subsample"],
            "colsample_bytree": best["colsample_bytree"]
        }

        print("\nXGB Hyperopt Best Params:", best_params)

        return best_params

    def fit(self, train):

        self.features = self._prepare_features(train)

        X = train[self.features]
        X = self._clean_X(X)

        y = train["target"]

        X_scaled = self.scaler.fit_transform(X)

        if not self.is_tuned:
            self.best_params = self._run_hyperopt(X_scaled, y)
            self.is_tuned = True

        self.model = XGBRegressor(random_state=42, **self.best_params)
        self.model.fit(X_scaled, y)

    def predict(self, test):

        X_scaled, expected_len = self._safe_prepare_test(test)

        preds = self.model.predict(X_scaled)

        preds = np.asarray(preds, dtype=float)
        preds = np.nan_to_num(preds)
        preds = np.clip(preds, -10, 10)

        return preds[:expected_len]

    def predict_future(self, df, steps=5):

        preds = []
        temp_df = df.copy()

        lag_cols = [col for col in self.features if col.startswith("lag_")]
        max_lag = len(lag_cols)

        for _ in range(steps):

            last_row = temp_df.iloc[-1:].copy()

            X_scaled, _ = self._safe_prepare_test(last_row)

            pred = float(self.model.predict(X_scaled)[0])
            preds.append(pred)

            new_row = last_row.copy()
            new_row["target"] = pred

            for lag in range(max_lag, 1, -1):
                new_row[f"lag_{lag}"] = new_row[f"lag_{lag-1}"]

            new_row["lag_1"] = pred

            new_row = self._clean_X(new_row)

            next_index = temp_df.index[-1] + pd.offsets.BDay(1)
            new_row.index = [next_index]

            temp_df = pd.concat([temp_df, new_row])

        return np.array(preds)