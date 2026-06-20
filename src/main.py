from __future__ import (
    annotations,  # For type hints pointing to classes defined LATER in the same file.
)

import logging
import sys
from typing import Any  # allow any type (int, char etc) to by-pass

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.benchmark import compare_all_models, compute_baselines, compute_peak_mae
from src.database import save_results_to_supabase
from src.extractor import extract_data
from src.model import forecast_prophet, train_extratrees, train_lightgbm
from src.processor import prepare_data
from src.settings import END_DATE, YESTR_DATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_forecast() -> dict[str, Any]:
    """
    Run the full energy demand forecasting pipeline.

    Steps:
        1. Extract yesterday's energy data from Singapore NEMS API.
        2. Engineer time-series features (calendar, lags, rolling stats).
        3. Compute baseline models.
        4. Forecast with Prophet (Singapore holidays + solar/usep regressors).
        5. Forecast with LightGBM and ExtraTrees (skforecast recursive).
        6. Compare all models and select the best by MAE.

    Returns:
        - Dict containing predictions, baselines, metrics, and feature importance
        for database storage.
        - Returns empty dict if extraction fails.
    """
    as_of_date = pd.to_datetime(END_DATE).date()
    logger.info("Starting energy demand forecast as of %s", as_of_date)

    # 1. Extract Singapore NEMS data API
    logger.info("Extracting energy demand data...")
    df = extract_data(YESTR_DATE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df.empty:
        logger.warning("No data extracted. Exiting.")
        return {}

    # 2. Feature Engineering
    logger.info("Engineering features...")
    data_dict = prepare_data(df)
    y_test = data_dict["y_test"]

    # 3. Compute Baseline models
    logger.info("Computing baseline models...")
    baselines_df = compute_baselines(data_dict["df"], y_test)

    # 4. Get Prophet model
    logger.info("Running Prophet forecast...")
    prophet_preds, prophet_actuals, prophet_df = forecast_prophet(
        data_dict["df"], int(len(data_dict["df"]) * 0.8)
    )

    # 5. Get LightBGM & ExtraTrees models
    logger.info("Training LightGBM...")
    lgbm_preds, importance = train_lightgbm(data_dict["df"], data_dict)
    logger.info("Training ExtraTrees...")
    et_preds, _ = train_extratrees(data_dict["df"], data_dict)

    # 6. Benchmark
    predictions = {
        "Prophet": prophet_preds,
        "LightGBM": lgbm_preds,
        "ExtraTrees": et_preds,
    }
    comparison = compare_all_models(y_test, predictions, baselines_df)

    metrics = {}
    for _, row in comparison.iterrows():
        key = row["Model"].lower().replace(" ", "_")
        metrics[f"mae_{key}"] = row["MAE"]

    best_row = comparison.iloc[0]
    metrics["best_model"] = best_row["Model"]
    metrics["best_mae"] = best_row["MAE"]

    peak_mae = compute_peak_mae(y_test, lgbm_preds)
    logger.info("Peak demand (top 5%%) MAE: %.4f", peak_mae)

    # 7. Logger results
    logger.info("Energy Demand Forecast Results")
    logger.info("Date: %s", as_of_date)
    logger.info("Best Model: %s (MAE: %.4f)", metrics["best_model"], metrics["best_mae"])
    logger.info(
        "Prophet MAE: %.4f, RMSE: %.4f",
        mean_absolute_error(y_test, prophet_preds),
        np.sqrt(mean_squared_error(y_test, prophet_preds)),
    )
    logger.info(
        "LightGBM MAE: %.4f, RMSE: %.4f",
        mean_absolute_error(y_test, lgbm_preds),
        np.sqrt(mean_squared_error(y_test, lgbm_preds)),
    )
    logger.info(
        "ExtraTrees MAE: %.4f, RMSE: %.4f",
        mean_absolute_error(y_test, et_preds),
        np.sqrt(mean_squared_error(y_test, et_preds)),
    )

    test_index = data_dict["test"].index
    timestamps = test_index[: len(lgbm_preds)]

    baseline_preds = {}
    for _, row in baselines_df.iterrows():
        col_map = {
            "Last Value": "lag_2",
            "Yesterday Same Time": "lag_48",
            "Last Week Same Time": "lag_336",
            "Avg Demand 24h": "demand_avg_24h",
        }
        col = col_map.get(row["Model"])
        if col:
            baseline_preds[row["Model"]] = data_dict["test"][col].values[: len(lgbm_preds)]

    return {
        "date": as_of_date,
        "predictions": predictions,
        "baselines": baseline_preds,
        "prophet_predictions": prophet_preds,
        "lightgbm_predictions": lgbm_preds,
        "extratrees_predictions": et_preds,
        "actuals": y_test.values[: len(lgbm_preds)],
        "timestamps": timestamps,
        "metrics": metrics,
        "comparison": comparison,
        "feature_importance": importance,
    }


def main() -> None:
    """Primary CLI entry point — saves forecast results to Supabase."""
    load_dotenv()  # Retrive Supabase KEY

    try:
        result = run_forecast()
        if not result:
            logger.error("Forecast returned empty result")
            sys.exit(1)

        try:  # save result to Supabase
            save_results_to_supabase(result)
            print("Results successfully saved to Supabase")

        except Exception as f:
            logger.error("Failed to save to Supabase: %s", f)
            print("Failed to save to Supabase: %s", f, file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        logger.error("Error during forecast: %s", e)
        print("Error during forecast: %s", e, file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
