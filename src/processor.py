from __future__ import annotations

import pandas as pd

from src.feature_eng import engineer_features
from src.settings import FEATURE_COLS, TARGET


def prepare_data(df: pd.DataFrame, split_ratio: float = 0.8) -> dict:
    """
    Engineer features and split data chronologically into train / test sets.

    Args:
        df: Raw DataFrame with columns [timestamp, demand, solar, usep].
        split_ratio: Fraction of rows to use for training (default 0.8).

    Returns:
        Dict containing:
        - df: full feature-engineered DataFrame
        - train / test: train / test subsets
        - X_train / X_test: feature matrices
        - y_train / y_test: target vectors
        - feature_cols: list of feature column names
    """
    df = engineer_features(df)

    split_idx = int(len(df) * split_ratio)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    feature_cols = FEATURE_COLS

    X_train = train[feature_cols]
    X_test = test[feature_cols]
    y_train = train[TARGET]
    y_test = test[TARGET]

    return {
        "df": df,
        "train": train,
        "test": test,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_cols": feature_cols,
    }


def create_lag_features_for_tuning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create extended lag features (1–336) for hyper-parameter tuning.

    Args:
        df: DataFrame with timestamp index and at least a 'demand' column.

    Returns:
        DataFrame with 11 additional lag columns and NaN rows dropped.
    """
    data = df.copy()
    lags = [1, 2, 3, 4, 5, 6, 12, 24, 48, 96, 336]
    for lag in lags:
        data[f"lag_{lag}"] = data[TARGET].shift(lag)
    data = data.dropna()
    return data
