from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.model import (
    forecast_prophet,
    get_singapore_holidays,
    train_extratrees,
    train_lightgbm,
)


class TestGetSingaporeHolidays:
    def test_returns_dataframe_with_expected_structure(self):
        result = get_singapore_holidays(2026, 2026)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["holiday", "ds", "lower_window", "upper_window"]

    def test_contains_known_singapore_holiday(self):
        result = get_singapore_holidays(2026, 2026)

        assert len(result) > 0
        assert result["lower_window"].iloc[0] == -1
        assert result["upper_window"].iloc[0] == 1

    def test_handles_empty_year_range(self):
        result = get_singapore_holidays(1999, 1999)
        assert isinstance(result, pd.DataFrame)


class TestForecastProphet:
    @patch("src.model.Prophet")
    def test_returns_tuple_of_correct_types(self, mock_prophet_cls, sample_engineered_df: pd.DataFrame):
        mock_instance = MagicMock()
        mock_prophet_cls.return_value = mock_instance

        split = int(len(sample_engineered_df) * 0.8)
        n_test = len(sample_engineered_df) - split
        mock_future = pd.DataFrame({
            "ds": pd.date_range("2026-06-01", periods=n_test, freq="30min"),
            "yhat": np.arange(float(n_test)),
        })
        mock_instance.predict.return_value = mock_future

        yhat, actual, forecast_df = forecast_prophet(sample_engineered_df, split)

        assert isinstance(yhat, np.ndarray)
        assert isinstance(actual, np.ndarray)
        assert isinstance(forecast_df, pd.DataFrame)
        assert len(yhat) == len(actual)
        mock_prophet_cls.assert_called_once()
        mock_instance.fit.assert_called_once()
        mock_instance.predict.assert_called_once()

    @patch("src.model.Prophet")
    def test_fit_called_with_train_data(self, mock_prophet_cls, sample_engineered_df: pd.DataFrame):
        mock_instance = MagicMock()
        mock_prophet_cls.return_value = mock_instance
        mock_future = pd.DataFrame({"ds": pd.date_range("2026-06-01", periods=10, freq="30min"), "yhat": np.arange(10.0)})
        mock_instance.predict.return_value = mock_future

        split = int(len(sample_engineered_df) * 0.8)
        forecast_prophet(sample_engineered_df, split)

        fit_args = mock_instance.fit.call_args[0][0]
        assert "ds" in fit_args.columns
        assert "y" in fit_args.columns
        assert len(fit_args) == split


class TestTrainLightGBM:
    @patch("lightgbm.LGBMRegressor")
    @patch("skforecast.recursive.ForecasterRecursive")
    def test_returns_predictions_and_importance(
        self, mock_forecaster_cls, mock_lgbm_cls, sample_data_dict: dict
    ):
        mock_regressor = MagicMock()
        mock_lgbm_cls.return_value = mock_regressor

        mock_forecaster = MagicMock()
        mock_forecaster_cls.return_value = mock_forecaster

        n_test = len(sample_data_dict["y_test"])
        mock_predictions = MagicMock()
        mock_predictions.values = np.arange(n_test, dtype=float)
        mock_forecaster.predict.return_value = mock_predictions

        mock_importance = pd.DataFrame({"feature": ["lag_1"], "importance": [0.5]})
        mock_forecaster.get_feature_importances.return_value = mock_importance

        preds, importance, forecaster = train_lightgbm(sample_data_dict["df"], sample_data_dict)

        assert isinstance(preds, np.ndarray)
        assert len(preds) == n_test
        assert importance is not None
        assert forecaster is mock_forecaster
        mock_lgbm_cls.assert_called_once()
        mock_forecaster.fit.assert_called_once()
        mock_forecaster.predict.assert_called_once()

    @patch("lightgbm.LGBMRegressor")
    @patch("skforecast.recursive.ForecasterRecursive")
    def test_uses_custom_params_when_provided(
        self, mock_forecaster_cls, mock_lgbm_cls, sample_data_dict: dict
    ):
        mock_lgbm_cls.return_value = MagicMock()
        mock_forecaster = MagicMock()
        mock_forecaster_cls.return_value = mock_forecaster
        mock_predictions = MagicMock()
        mock_predictions.values = np.arange(len(sample_data_dict["y_test"]), dtype=float)
        mock_forecaster.predict.return_value = mock_predictions
        mock_forecaster.get_feature_importances.return_value = pd.DataFrame()

        custom_params = {"n_estimators": 50, "max_depth": 3, "random_state": 42, "verbose": -1}
        train_lightgbm(sample_data_dict["df"], sample_data_dict, best_params=custom_params)

        mock_lgbm_cls.assert_called_once_with(**custom_params)


class TestTrainExtraTrees:
    @patch("sklearn.ensemble.ExtraTreesRegressor")
    @patch("skforecast.recursive.ForecasterRecursive")
    def test_returns_predictions_and_none_importance(
        self, mock_forecaster_cls, mock_et_cls, sample_data_dict: dict
    ):
        mock_regressor = MagicMock()
        mock_et_cls.return_value = mock_regressor

        mock_forecaster = MagicMock()
        mock_forecaster_cls.return_value = mock_forecaster

        n_test = len(sample_data_dict["y_test"])
        mock_predictions = MagicMock()
        mock_predictions.values = np.arange(n_test, dtype=float)
        mock_forecaster.predict.return_value = mock_predictions

        preds, importance, forecaster = train_extratrees(sample_data_dict["df"], sample_data_dict)

        assert isinstance(preds, np.ndarray)
        assert len(preds) == n_test
        assert importance is None
        assert forecaster is mock_forecaster
        mock_et_cls.assert_called_once()
        mock_forecaster.fit.assert_called_once()
        mock_forecaster.predict.assert_called_once()
