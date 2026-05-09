import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_metrics(y_true, y_pred, is_log=False):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ONLY convert if explicitly log
    if is_log:
        y_true = np.exp(np.clip(y_true, -10, 10))
        y_pred = np.exp(np.clip(y_pred, -10, 10))

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {"MAE": mae, "RMSE": rmse}


def direction_accuracy(y_true, y_pred, is_log=True):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if is_log:
        y_true = np.exp(np.clip(y_true, -10, 10))
        y_pred = np.exp(np.clip(y_pred, -10, 10))

    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))

    return (true_dir == pred_dir).mean()