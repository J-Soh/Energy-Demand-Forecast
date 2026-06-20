from __future__ import annotations

import numpy as np
import pandas as pd

from src.settings import TARGET


def _set_timestamp_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")
    df = df.asfreq("30min")
    return df


def _create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df.index.hour
    df["minute"] = df.index.minute
    df["day_of_week"] = df.index.dayofweek
    df["day_of_month"] = df.index.day
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def _create_cyclical_hour(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def _create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lag_1"] = df[TARGET].shift(1)
    df["lag_2"] = df[TARGET].shift(2)
    df["lag_48"] = df[TARGET].shift(48)
    df["lag_96"] = df[TARGET].shift(96)
    df["lag_336"] = df[TARGET].shift(336)
    return df


def _create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["demand_avg_24h"] = df[TARGET].shift(1).rolling(48).mean()
    df["demand_std_24h"] = df[TARGET].shift(1).rolling(48).std()
    df["demand_avg_7d"] = df[TARGET].shift(1).rolling(336).mean()
    return df


def _create_exog_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["solar_lag_1"] = df["solar"].shift(1)
    df["usep_lag_1"] = df["usep"].shift(1)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full feature-engineering pipeline to raw energy demand data.

    Steps: set 30-min frequency index, add calendar features, cyclical hour
    encoding, lag features, rolling statistics, and exogenous-variable lags.

    Args:
        df: DataFrame with columns [timestamp, demand, solar, usep].

    Returns:
        DataFrame with timestamp index and all engineered feature columns,
        with rows containing NaN (from lags / rolling windows) dropped.
    """
    df = _set_timestamp_index(df)
    df = _create_calendar_features(df)
    df = _create_cyclical_hour(df)
    df = _create_lag_features(df)
    df = _create_rolling_features(df)
    df = _create_exog_lags(df)
    df = df.dropna()
    return df
