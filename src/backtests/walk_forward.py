import numpy as np
import pandas as pd


def walk_forward_validation(df, model_class, window=200, step=1):
    """
    Walk-forward validation for time series

    Args:
        df (DataFrame): full dataset
        model_class: model class (not instance)
        window (int): initial training size
        step (int): step size

    Returns:
        preds, actuals
    """

    preds = []
    actuals = []
    dates = []

    for i in range(window, len(df)-1, step):

        train = df.iloc[:i]
        test = df.iloc[i:i+1]

        try:
            model = model_class()
            model.fit(train)

            # prediction logic (handle different models)
            if hasattr(model, "predict"):
                
                if "SARIMAX" in model.__class__.__name__:
                    pred = model.predict(test=test)

                elif "Prophet" in model.__class__.__name__:
                    pred = model.predict(test=test)

                elif "ARIMA" in model.__class__.__name__ or "SARIMA" in model.__class__.__name__:
                    pred = model.predict(1)

                else:
                    pred = model.predict(test)

            else:
                continue

            pred = np.asarray(pred)[0]

            preds.append(pred)
            actuals.append(test["target"].values[0])
            dates.append(test.index[0])

        except Exception:
            continue

    results = pd.DataFrame({
        "Date": dates,
        "Pred": preds,
        "Actual": actuals
    })

    return results