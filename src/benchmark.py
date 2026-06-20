from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


def compute_baselines(data: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Evaluate four naïve baseline models on the test set.

    Baselines: Last Value (lag_2), Yesterday Same Time (lag_48),
    Last Week Same Time (lag_336), Rolling Mean 48.

    Args:
        data: Full feature-engineered DataFrame.
        y_test: Actual demand values for the test period.

    Returns:
        DataFrame with columns [Model, MAE, RMSE] sorted by MAE.
    """
    test = data.loc[y_test.index]
    lag_2 = test["lag_2"]
    lag_48 = test["lag_48"]
    lag_336 = test["lag_336"]
    demand_avg_24h = test["demand_avg_24h"]

    results = []
    for name, pred in [
        ("Last Value", lag_2),
        ("Yesterday Same Time", lag_48),
        ("Last Week Same Time", lag_336),
        ("Avg Demand 24h", demand_avg_24h),
    ]:
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        results.append({"Model": name, "MAE": mae, "RMSE": rmse})

    results_df = pd.DataFrame(results).sort_values("MAE")
    logger.info("Baseline results:\n%s", results_df.to_string(index=False))
    return results_df


def compute_peak_mae(y_test: pd.Series, predictions: pd.Series, percentile: float = 0.95) -> float:
    """
    Compute MAE for the top percentile of demand periods.

    Args:
        y_test: Actual demand values.
        predictions: Predicted demand values (aligned with y_test).
        percentile: Threshold percentile (default 0.95 → top 5 %).

    Returns:
        MAE score restricted to peak-demand periods.
    """
    threshold = y_test.quantile(percentile)
    peak_mask = y_test >= threshold
    if peak_mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(y_test[peak_mask] - predictions[peak_mask])))


def compare_all_models(
    y_test: pd.Series,
    predictions: dict[str, np.ndarray],
    baselines: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine ML-model predictions with baselines into a single ranking.

    Args:
        y_test: Actual demand values.
        predictions: Dict mapping model name → numpy array of predictions.
        baselines: DataFrame from compute_baselines().

    Returns:
        DataFrame with columns [Model, MAE, RMSE] sorted by MAE ascending.
    """
    results = []

    for model_name, preds in predictions.items():
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        results.append({"Model": model_name, "MAE": mae, "RMSE": rmse})

    for _, row in baselines.iterrows():
        results.append({"Model": row["Model"], "MAE": row["MAE"], "RMSE": row["RMSE"]})

    results_df = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)
    logger.info("All models comparison:\n%s", results_df.to_string(index=False))
    return results_df
