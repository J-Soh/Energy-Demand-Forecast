from __future__ import annotations

import base64
import logging
import os
import uuid
from datetime import datetime
from typing import Any

import pandas as pd
from supabase import Client, create_client

from src.settings import SUPABASE_ARTIFACTS_TABLE, SUPABASE_RAW_TABLE, SUPABASE_TABLE_NAME

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
            "timestamp": timestamps[i].isoformat()
            if hasattr(timestamps[i], "isoformat")
            else str(timestamps[i]),
            "actual_demand": float(actuals[i]) if actuals is not None else None,
            "prophet_prediction": float(prophet_preds[i]) if prophet_preds is not None else None,
            "lightgbm_prediction": float(lgbm_preds[i]) if lgbm_preds is not None else None,
            "extratrees_prediction": float(et_preds[i]) if et_preds is not None else None,
            "last_value_prediction": float(last_value_arr[i])
            if last_value_arr is not None
            else None,
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


def read_raw_data_from_supabase() -> pd.DataFrame:
    """
    Read all historical raw data from the energy_raw table.

    Returns:
        DataFrame with columns: [timestamp, demand, solar, usep].
        Returns empty DataFrame if table is empty or error occurs.
    """
    try:
        supabase = get_supabase_client()
        response = (
            supabase.table(SUPABASE_RAW_TABLE)
            .select("timestamp,demand,solar,usep")
            .order("timestamp")
            .range(0, 10000)
            .execute()
        )
        if not response.data:
            logger.info("No raw data found in Supabase.")
            return pd.DataFrame()
        df = pd.DataFrame(response.data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        logger.info("Loaded %d rows from energy_raw", len(df))
        return df
    except Exception as e:
        logger.warning("Failed to read raw data from Supabase: %s", e)
        return pd.DataFrame()


def save_raw_data_to_supabase(df: pd.DataFrame) -> None:
    """
    Save today's raw API data (timestamp, demand, solar, usep) to energy_raw.

    Args:
        df: DataFrame with columns [timestamp, demand, solar, usep].
    """
    supabase = get_supabase_client()
    rows = df[["timestamp", "demand", "solar", "usep"]].copy()
    rows["timestamp"] = rows["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    records = rows.to_dict(orient="records")
    logger.info("Saving %d raw rows to energy_raw...", len(records))
    supabase.table(SUPABASE_RAW_TABLE).upsert(records, on_conflict="timestamp").execute()
    logger.info("Raw data saved to energy_raw.")


def save_artifacts_to_supabase(result: dict[str, Any]) -> None:
    """
    Save model evaluation artifacts (best model, metrics, feature importance)
    and upload the trained model as a .pkl file to Supabase Storage.

    Args:
        result: Dict from run_forecast() containing metrics and feature_importance.
    """
    supabase = get_supabase_client()
    metrics = result.get("metrics", {})
    model_pickle_b64 = result.get("model_pickle")

    model_pickle_url = None
    if model_pickle_b64:
        try:
            pickle_bytes = base64.b64decode(model_pickle_b64)
            filename = f"model_{result['date']}.pkl"
            supabase.storage.from_("models").upload(
                filename,
                pickle_bytes,
                {"content-type": "application/octet-stream", "upsert": "true"},
            )
            model_pickle_url = supabase.storage.from_("models").get_public_url(filename)
            logger.info("Model pickle uploaded to storage: %s", model_pickle_url)
        except Exception as e:
            logger.warning("Failed to upload model pickle to storage: %s", e)

    row = {
        "id": str(uuid.uuid4()),
        "created_at": datetime.now().isoformat(),
        "as_of_date": result.get("date").isoformat() if result.get("date") else None,
        "best_model": metrics.get("best_model"),
        "best_mae": metrics.get("best_mae"),
        "mae_prophet": metrics.get("mae_prophet"),
        "mae_lightgbm": metrics.get("mae_lightgbm"),
        "mae_extratrees": metrics.get("mae_extratrees"),
        "mae_last_value": metrics.get("mae_last_value"),
        "model_pickle_url": model_pickle_url,
    }

    supabase.table(SUPABASE_ARTIFACTS_TABLE).upsert(row, on_conflict="as_of_date").execute()
    logger.info("Artifacts saved to energy_artifacts for %s", row["as_of_date"])
