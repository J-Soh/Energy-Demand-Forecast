from __future__ import annotations

import logging
import warnings

import holidays
import numpy as np
import pandas as pd
from prophet import Prophet

from src.settings import ET_PARAMS, FEATURE_COLS, LGBM_PARAMS, PROPHET_PARAMS, SKFORECAST_LAGS

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)


def get_singapore_holidays(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Fetch Singapore public holidays formatted for Prophet.

    Each holiday gets a ±1 day window to capture eve / after-effects.

    Args:
        start_year: First year of holiday range.
        end_year: Last year of holiday range (inclusive).

    Returns:
        DataFrame with columns [holiday, ds, lower_window, upper_window].
    """
    sg = holidays.Singapore(years=range(start_year, end_year + 1))
    rows = []
    for day, name in sg.items():
        clean_name = str(name).replace("'", "").replace(",", "").replace(" ", "_").lower()
        rows.append(
            {
                "holiday": clean_name,
                "ds": pd.Timestamp(day),
                "lower_window": -1,
                "upper_window": 1,
            }
        )
    return pd.DataFrame(rows)


def forecast_prophet(
    data: pd.DataFrame, split_idx: int
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Train Prophet with Singapore holidays and solar / usep regressors.

    Args:
        data: Feature-engineered DataFrame with timestamp index and columns
              [demand, solar, usep].
        split_idx: Row index for chronological train / test split.

    Returns:
        Tuple of (yhat_values, actual_values, forecast_df).
    """
    prophet_df = data.reset_index()
    prophet_df = prophet_df.rename(columns={"timestamp": "ds", "demand": "y"})
    prophet_df = prophet_df[["ds", "y", "solar", "usep"]]

    train = prophet_df.iloc[:split_idx]
    test = prophet_df.iloc[split_idx:]

    holidays_df = get_singapore_holidays(
        pd.to_datetime(train["ds"].min()).year,
        pd.to_datetime(train["ds"].max()).year,
    )

    model = Prophet(
        holidays=holidays_df if not holidays_df.empty else None,
        **PROPHET_PARAMS,
    )
    model.add_regressor("solar")
    model.add_regressor("usep")
    model.fit(train)

    future = test[["ds", "solar", "usep"]]
    forecast = model.predict(future)

    return forecast["yhat"].values, test["y"].values, forecast[["ds", "yhat"]]


def train_lightgbm(data: pd.DataFrame, data_dict: dict, best_params: dict | None = None):
    """
    Train and predict with a LightGBM recursive forecaster.

    Args:
        data: Full feature-engineered DataFrame (unused directly, kept for
              signature consistency).
        data_dict: Dict from prepare_data() with X_train, X_test, y_train, y_test.
        best_params: Optional dict of tuned hyper-parameters; falls back to
                     LGBM_PARAMS defaults.

    Returns:
        Tuple of (predicted_values, feature_importance_df).
    """
    from lightgbm import LGBMRegressor
    from skforecast.recursive import ForecasterRecursive

    params = (
        best_params
        if best_params
        else {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "random_state": 42,
            "verbose": -1,
        }
    )

    regressor = LGBMRegressor(**params)
    forecaster = ForecasterRecursive(regressor, lags=SKFORECAST_LAGS)
    forecaster.fit(y=data_dict["y_train"], exog=data_dict["X_train"])

    predictions = forecaster.predict(steps=len(data_dict["y_test"]), exog=data_dict["X_test"])

    importance = forecaster.get_feature_importances()
    return predictions.values, importance, forecaster


def train_extratrees(data: pd.DataFrame, data_dict: dict, best_params: dict | None = None):
    """
    Train and predict with an ExtraTrees recursive forecaster.

    Args:
        data: Full feature-engineered DataFrame (unused directly, kept for
              signature consistency).
        data_dict: Dict from prepare_data() with X_train, X_test, y_train, y_test.
        best_params: Optional dict of tuned hyper-parameters; falls back to
                     ET_PARAMS defaults.

    Returns:
        Tuple of (predicted_values, None).
    """
    from skforecast.recursive import ForecasterRecursive
    from sklearn.ensemble import ExtraTreesRegressor

    params = (
        best_params
        if best_params
        else {
            "n_estimators": 200,
            "max_depth": 10,
            "random_state": 42,
            "n_jobs": -1,
        }
    )

    regressor = ExtraTreesRegressor(**params)
    forecaster = ForecasterRecursive(regressor, lags=SKFORECAST_LAGS)
    forecaster.fit(y=data_dict["y_train"], exog=data_dict["X_train"])

    predictions = forecaster.predict(steps=len(data_dict["y_test"]), exog=data_dict["X_test"])
    return predictions.values, None, forecaster


def tune_lightgbm(X_train, y_train):
    """
    Hyper-parameter tuning for LightGBM via randomised search.

    Uses TimeSeriesSplit (5-fold) and MAE scoring.

    Args:
        X_train: Feature matrix for training.
        y_train: Target vector for training.

    Returns:
        Dict of best hyper-parameters found.
    """
    from lightgbm import LGBMRegressor
    from scipy.stats import randint, uniform
    from sklearn.metrics import make_scorer, mean_absolute_error
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

    model = LGBMRegressor(random_state=42, verbose=-1)
    param_distributions = {
        "n_estimators": randint(100, 1000),
        "learning_rate": uniform(0.01, 0.19),
        "max_depth": randint(3, 15),
        "num_leaves": randint(20, 200),
        "min_child_samples": randint(5, 100),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "reg_alpha": uniform(0, 2),
        "reg_lambda": uniform(0, 2),
    }
    tscv = TimeSeriesSplit(n_splits=5)
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=30,
        cv=tscv,
        scoring=mae_scorer,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    random_search.fit(X_train, y_train)
    logger.info("Best LightGBM CV MAE: %.4f", abs(random_search.best_score_))
    return random_search.best_params_


def tune_extratrees(X_train, y_train):
    """
    Hyper-parameter tuning for ExtraTrees via randomised search.

    Uses TimeSeriesSplit (5-fold) and MAE scoring.

    Args:
        X_train: Feature matrix for training.
        y_train: Target vector for training.

    Returns:
        Dict of best hyper-parameters found.
    """
    from scipy.stats import randint
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.metrics import make_scorer, mean_absolute_error
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

    model = ExtraTreesRegressor(random_state=42, n_jobs=1)
    param_distributions = {
        "n_estimators": randint(100, 1000),
        "max_depth": randint(3, 30),
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 20),
        "max_features": ["sqrt", "log2", 0.5, 0.7, 0.9],
        "bootstrap": [True],
    }
    tscv = TimeSeriesSplit(n_splits=5)
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=30,
        cv=tscv,
        scoring=mae_scorer,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    random_search.fit(X_train, y_train)
    logger.info("Best ExtraTrees CV MAE: %.4f", abs(random_search.best_score_))
    return random_search.best_params_


def retrain_and_forecast_lightgbm(
    df: pd.DataFrame, future_X: pd.DataFrame, n_steps: int = 48
) -> np.ndarray:
    """
    Retrain LightGBM on all data and predict n_steps ahead.

    Args:
        df: Feature-engineered DataFrame with all historical rows.
        future_X: Future exogenous features (n_steps × FEATURE_COLS).
        n_steps: Number of 30-min periods to forecast (default 48).

    Returns:
        Array of predicted demand values.
    """
    from lightgbm import LGBMRegressor
    from skforecast.recursive import ForecasterRecursive

    regressor = LGBMRegressor(**LGBM_PARAMS)
    forecaster = ForecasterRecursive(regressor, lags=SKFORECAST_LAGS)
    forecaster.fit(y=df["demand"], exog=df[FEATURE_COLS])
    predictions = forecaster.predict(steps=n_steps, exog=future_X)
    return predictions.values


def retrain_and_forecast_extratrees(
    df: pd.DataFrame, future_X: pd.DataFrame, n_steps: int = 48
) -> np.ndarray:
    """
    Retrain ExtraTrees on all data and predict n_steps ahead.

    Args:
        df: Feature-engineered DataFrame with all historical rows.
        future_X: Future exogenous features (n_steps × FEATURE_COLS).
        n_steps: Number of 30-min periods to forecast (default 48).

    Returns:
        Array of predicted demand values.
    """
    from skforecast.recursive import ForecasterRecursive
    from sklearn.ensemble import ExtraTreesRegressor

    regressor = ExtraTreesRegressor(**ET_PARAMS)
    forecaster = ForecasterRecursive(regressor, lags=SKFORECAST_LAGS)
    forecaster.fit(y=df["demand"], exog=df[FEATURE_COLS])
    predictions = forecaster.predict(steps=n_steps, exog=future_X)
    return predictions.values
