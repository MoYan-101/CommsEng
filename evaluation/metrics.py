"""
evaluation/metrics.py

Implements common regression metrics: MSE, MAE, R2.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# ---------- Added: mixed-metric helper ----------
from data_preprocessing.scaler_utils import inverse_transform_output

def compute_regression_metrics(y_true, y_pred):
    """
    Compute MSE, MAE, and R2 for the given true/predicted values.
    :param y_true: shape (N, output_dim)
    :param y_pred: shape (N, output_dim)
    :return: a dict with "MSE", "MAE", "R2"
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred, multioutput='uniform_average')
    return {
        "MSE": mse,
        "MAE": mae,
        "R2" : r2
    }



def compute_mixed_metrics(y_true_raw,
                          y_true_std,
                          y_pred_std,
                          scaler_y):
    """
    MSE / MAE are computed in the standardized domain;
    R² is computed in the original scale.
    """
    # Invert the output scaling.
    y_pred_raw = inverse_transform_output(y_pred_std, scaler_y)

    # Compute metrics in both domains.
    m_std = compute_regression_metrics(y_true_std, y_pred_std)
    m_raw = compute_regression_metrics(y_true_raw, y_pred_raw)

    return {
        "MSE": m_std["MSE"],
        "MAE": m_std["MAE"],
        "R2":  m_raw["R2"]
    }


if __name__ == "__main__":
    # Example true and predicted values.
    y_true = np.array([[3.5, 2.1], [4.0, 3.3], [5.2, 6.8]])
    y_pred = np.array([[3.7, 2.0], [4.1, 3.5], [5.0, 6.5]])

    # Compute regression metrics.
    metrics = compute_regression_metrics(y_true, y_pred)
    print(metrics)
