from datetime import datetime, timedelta

YESTR_DATE = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
END_DATE = datetime.now().strftime("%Y-%m-%d")

SUPABASE_TABLE_NAME = "energy_forecast"
SUPABASE_RAW_TABLE = "energy_raw"
SUPABASE_ARTIFACTS_TABLE = "energy_artifacts"

FEATURE_COLS = [
    "solar",
    "usep",
    "hour",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "demand_avg_24h",
    "demand_std_24h",
    "demand_avg_7d",
]

TARGET = "demand"

SKFORECAST_LAGS = [1, 2, 48, 96, 336]

PROPHET_PARAMS = {
    "yearly_seasonality": False,
    "weekly_seasonality": True,
    "daily_seasonality": True,
}

LGBM_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 5,
    "random_state": 42,
    "verbose": -1,
}

ET_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "random_state": 42,
    "n_jobs": -1,
}
