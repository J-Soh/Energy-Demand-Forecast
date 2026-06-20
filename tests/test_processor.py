import pandas as pd
import pytest

from src.processor import create_lag_features_for_tuning, prepare_data
from src.settings import FEATURE_COLS, TARGET


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2025-12-18")
    for i in range(500):
        t = start + pd.Timedelta(minutes=30 * i)
        rows.append({"timestamp": t, "demand": 5000 + i * 0.5, "solar": 100.0, "usep": 50.0})
    return pd.DataFrame(rows)


class TestPrepareData:
    def test_returns_dict_with_expected_keys(self, sample_df: pd.DataFrame):
        result = prepare_data(sample_df)

        expected_keys = {"df", "train", "test", "X_train", "X_test", "y_train", "y_test", "feature_cols"}
        assert expected_keys.issubset(result.keys())

    def test_split_ratio_applied(self, sample_df: pd.DataFrame):
        result = prepare_data(sample_df, split_ratio=0.8)
        total = len(result["train"]) + len(result["test"])

        assert abs(len(result["train"]) / total - 0.8) < 0.01

    def test_X_train_has_feature_cols(self, sample_df: pd.DataFrame):
        result = prepare_data(sample_df)

        assert list(result["X_train"].columns) == FEATURE_COLS

    def test_y_train_has_target_column(self, sample_df: pd.DataFrame):
        result = prepare_data(sample_df)

        assert result["y_train"].name == TARGET

    def test_no_nans_in_split_data(self, sample_df: pd.DataFrame):
        result = prepare_data(sample_df)

        assert not result["X_train"].isna().any().any()
        assert not result["X_test"].isna().any().any()


class TestCreateLagFeaturesForTuning:
    def test_creates_lag_columns(self, sample_df: pd.DataFrame):
        from src.feature_eng import _set_timestamp_index
        df = _set_timestamp_index(sample_df)
        result = create_lag_features_for_tuning(df)

        for lag in [1, 2, 3, 4, 5, 6, 12, 24, 48, 96, 336]:
            assert f"lag_{lag}" in result.columns

    def test_drops_nan_rows(self, sample_df: pd.DataFrame):
        from src.feature_eng import _set_timestamp_index
        df = _set_timestamp_index(sample_df)
        result = create_lag_features_for_tuning(df)

        assert len(result) < len(df)
