from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime
from typing import Any

from supabase import Client, create_client

from src.settings import SUPABASE_TABLE_NAME

logger = logging.getLogger(__name__)


def get_supabase_client() -> Client:
    """
    Create and return Supabase client from environment variables.

    Raises:
        ValueError: If SUPABASE_URL or SUPABASE_KEY is not set.

    Returns:
        Authenticated Supabase client.
    """
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("Supabase credentials (SUPABASE_URL or SUPABASE_KEY) not found.")
    return create_client(url, key)


def save_results_to_supabase(result: dict[str, Any]) -> None:
    """
    Save today's forecast results to Supabase.

    Each row represents one 30-min time slice with actual demand, model
    predictions (Prophet, LightGBM, ExtraTrees) and evaluation metrics.

    Args:
        result: Dict from run_forecast() containing:
            - date: as_of_date
            - timestamps: index for each forecast row
            - actuals, prophet_predictions, lightgbm_predictions,
              extratrees_predictions: aligned arrays
            - baselines: dict of baseline predictions
            - metrics: dict with best_model, mae_* values

    Raises:
        ValueError: If Supabase client cannot be created.
    """
    supabase = get_supabase_client()
    as_of_date = result.get("date")
    baselines = result.get("baselines", {})
    prophet_preds = result.get("prophet_predictions")
    lgbm_preds = result.get("lightgbm_predictions")
    et_preds = result.get("extratrees_predictions")
    actuals = result.get("actuals")
    timestamps = result.get("timestamps")
    metrics = result.get("metrics", {})

    if timestamps is None or actuals is None or len(timestamps) == 0:
        logger.warning("No forecast data to save")
        return

    last_value_arr = baselines.get("Last Value")
    yesterday_arr = baselines.get("Yesterday Same Time")
    rows: list[dict[str, Any]] = []
    for i in range(len(timestamps)):
        row: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "created_at": datetime.now().isoformat(),
            "as_of_date": as_of_date.isoformat() if as_of_date else None,
            "timestamp": timestamps[i].isoformat() if hasattr(timestamps[i], "isoformat") else str(timestamps[i]),
            "actual_demand": float(actuals[i]) if actuals is not None else None,
            "prophet_prediction": float(prophet_preds[i]) if prophet_preds is not None else None,
            "lightgbm_prediction": float(lgbm_preds[i]) if lgbm_preds is not None else None,
            "extratrees_prediction": float(et_preds[i]) if et_preds is not None else None,
            "last_value_prediction": float(last_value_arr[i]) if last_value_arr is not None else None,
            "yesterday_prediction": float(yesterday_arr[i]) if yesterday_arr is not None else None,
            "best_model": metrics.get("best_model"),
            "mae_prophet": metrics.get("mae_prophet"),
            "mae_lightgbm": metrics.get("mae_lightgbm"),
            "mae_extratrees": metrics.get("mae_extratrees"),
            "mae_last_value": metrics.get("mae_last_value"),
        }
        rows.append(row)

    logger.info("Upserting %d rows into Supabase...", len(rows))
    supabase.table(SUPABASE_TABLE_NAME).upsert(rows, on_conflict="as_of_date,timestamp").execute()
    logger.info("Successfully saved %d forecast records to Supabase", len(rows))
