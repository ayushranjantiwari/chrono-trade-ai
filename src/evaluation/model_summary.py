import warnings
warnings.filterwarnings('ignore')

def get_model_summary(name, model):

    summary = {"Model": name}

    # =====================
    # ARIMA / SARIMA / SARIMAX
    # =====================
    if hasattr(model, "model") and hasattr(model.model, "aic"):

        summary.update({
            "AIC": model.model.aic,
            "BIC": model.model.bic,
            "LogLik": model.model.llf
        })

        if hasattr(model, "order"):
            summary["Order"] = str(model.order)

        if hasattr(model, "seasonal_order"):
            summary["Seasonal_Order"] = str(model.seasonal_order)

        # coefficients
        if hasattr(model.model, "params"):
            for k, v in model.model.params.items():
                summary[f"coef_{k}"] = v

        if hasattr(model, "exog_cols"):
            summary["Exog_Columns"] = ", ".join(model.exog_cols)

    # =====================
    # PROPHET
    # =====================
    elif name == "Prophet":

        if hasattr(model, "best_params"):
            cps, sps, mode = model.best_params

            summary.update({
                "Prophet_cps": cps,
                "Prophet_sps": sps,
                "Prophet_mode": mode
            })

    # =====================
    # ML MODELS
    # =====================
    elif name in ["RandomForest", "XGBoost"]:

        try:
            feature_importance = model.model.feature_importances_

            summary.update({
                "Num_Features": len(model.features),
                "Top_Features": ", ".join(
                    [f"{f}:{round(i,4)}" for f, i in sorted(
                        zip(model.features, feature_importance),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]]
                )
            })

        except Exception:
            summary["Num_Features"] = len(model.features)

    return summary

"""
def get_model_summary(name, model):

    summary = {"Model": name}

    # =====================
    # ARIMA / SARIMA / SARIMAX
    # =====================
    if hasattr(model, "model"):

        try:
            summary["AIC"] = getattr(model.model, "aic", None)
            summary["BIC"] = getattr(model.model, "bic", None)
            summary["LogLik"] = getattr(model.model, "llf", None)
        except Exception:
            pass

        # Orders
        if hasattr(model, "order"):
            summary["Order"] = str(model.order)

        if hasattr(model, "seasonal_order"):
            summary["Seasonal_Order"] = str(model.seasonal_order)

        # Coefficients (safe handling)
        try:
            params = getattr(model.model, "params", None)

            if params is not None:
                if hasattr(params, "items"):  # dict-like
                    items = params.items()
                else:  # numpy array
                    items = enumerate(params)

                for k, v in items:
                    summary[f"coef_{k}"] = float(v)

        except Exception:
            pass

        # Exogenous columns
        if hasattr(model, "exog_cols") and model.exog_cols is not None:
            summary["Exog_Columns"] = ", ".join(model.exog_cols)

    # =====================
    # PROPHET (better detection)
    # =====================
    if hasattr(model, "best_params"):

        try:
            cps, sps, mode = model.best_params

            summary.update({
                "Prophet_cps": cps,
                "Prophet_sps": sps,
                "Prophet_mode": mode
            })
        except Exception:
            pass

    # =====================
    # ML MODELS
    # =====================
    if hasattr(model, "model") and hasattr(model.model, "feature_importances_"):

        try:
            fi = model.model.feature_importances_

            if hasattr(model, "features") and len(model.features) == len(fi):

                top_features = sorted(
                    zip(model.features, fi),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]

                summary["Num_Features"] = len(model.features)
                summary["Top_Features"] = ", ".join(
                    [f"{f}:{round(i,4)}" for f, i in top_features]
                )
            else:
                summary["Num_Features"] = len(getattr(model, "features", []))

        except Exception:
            summary["Num_Features"] = len(getattr(model, "features", []))

    return summary
"""