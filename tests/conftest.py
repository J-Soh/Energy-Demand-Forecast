import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def sample_raw_csv() -> str:
    return """Date,Period,Demand (MW),Solar (MW),USEP ($/MWh)
18/12/2025,00:00-00:30,5932.252,0.0,78.34
18/12/2025,00:30-01:00,5814.98,0.0,78.19
18/12/2025,01:00-01:30,5756.35,0.0,77.85
18/12/2025,01:30-02:00,5723.91,0.0,77.52
18/12/2025,02:00-02:30,5701.47,0.0,77.21
18/12/2025,02:30-03:00,5689.12,0.0,76.95
18/12/2025,03:00-03:30,5678.88,0.0,76.68
18/12/2025,03:30-04:00,5665.34,0.0,76.42
18/12/2025,04:00-04:30,5652.18,0.0,76.15
18/12/2025,04:30-05:00,5640.05,0.0,75.89
"""


@pytest.fixture(scope="session")
def raw_df(sample_raw_csv: str) -> pd.DataFrame:
    from io import StringIO
    return pd.read_csv(StringIO(sample_raw_csv))


@pytest.fixture(scope="session")
def past_data_csv() -> str:
    return "src/past_data.csv"


@pytest.fixture(scope="session")
def sample_engineered_df() -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2025-12-18")
    for i in range(400):
        rows.append({
            "demand": 5000 + i * 0.5 + np.sin(i / 48 * 2 * np.pi) * 200,
            "solar": max(0, 500 * np.sin(i / 48 * 2 * np.pi)),
            "usep": 50 + 20 * np.sin(i / 96 * 2 * np.pi),
        })
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(
        [start + pd.Timedelta(minutes=30 * i) for i in range(400)], name="timestamp"
    ))
    df.index.freq = "30min"
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["lag_1"] = df["demand"].shift(1)
    df["lag_2"] = df["demand"].shift(2)
    df["lag_48"] = df["demand"].shift(48)
    df["lag_96"] = df["demand"].shift(96)
    df["lag_336"] = df["demand"].shift(336)
    df["demand_avg_24h"] = df["demand"].shift(1).rolling(48).mean()
    df["demand_std_24h"] = df["demand"].shift(1).rolling(48).std()
    df["demand_avg_7d"] = df["demand"].shift(1).rolling(336).mean()
    df["solar_lag_1"] = df["solar"].shift(1)
    df["usep_lag_1"] = df["usep"].shift(1)
    return df.dropna()


@pytest.fixture(scope="session")
def sample_data_dict(sample_engineered_df: pd.DataFrame) -> dict:
    from src.settings import FEATURE_COLS, TARGET

    split = int(len(sample_engineered_df) * 0.8)
    train = sample_engineered_df.iloc[:split]
    test = sample_engineered_df.iloc[split:]
    return {
        "df": sample_engineered_df,
        "train": train,
        "test": test,
        "X_train": train[FEATURE_COLS],
        "X_test": test[FEATURE_COLS],
        "y_train": train[TARGET],
        "y_test": test[TARGET],
        "feature_cols": FEATURE_COLS,
    }
