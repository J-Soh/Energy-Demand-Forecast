import pandas as pd
import pytest

from src.feature_eng import (
    _create_calendar_features,
    _create_cyclical_hour,
    _create_exog_lags,
    _create_lag_features,
    _create_rolling_features,
    _set_timestamp_index,
    engineer_features,
)


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2025-12-18")
    for i in range(400):
        t = start + pd.Timedelta(minutes=30 * i)
        rows.append({"timestamp": t, "demand": 5000 + i * 0.5, "solar": 100.0, "usep": 50.0})
    return pd.DataFrame(rows)


class TestSetTimestampIndex:
    def test_sets_index_and_freq(self, sample_df: pd.DataFrame):
        result = _set_timestamp_index(sample_df)

        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.freqstr == "30min"

    def test_sorts_by_timestamp(self, sample_df: pd.DataFrame):
        shuffled = sample_df.sample(frac=1, random_state=42)
        result = _set_timestamp_index(shuffled)

        assert result.index.is_monotonic_increasing


class TestCreateCalendarFeatures:
    def test_adds_all_calendar_columns(self, sample_df: pd.DataFrame):
        df = _set_timestamp_index(sample_df)
        result = _create_calendar_features(df)

        expected = {"hour", "minute", "day_of_week", "day_of_month", "month", "quarter", "is_weekend"}
        assert expected.issubset(result.columns)

    def test_weekend_detection(self, sample_df: pd.DataFrame):
        df = _set_timestamp_index(sample_df)
        result = _create_calendar_features(df)

        saturday_idx = result.index[result["day_of_week"] == 5]
        assert len(saturday_idx) > 0
        assert (result.loc[saturday_idx, "is_weekend"] == 1).all()


class TestCreateCyclicalHour:
    def test_adds_sin_cos_columns(self, sample_df: pd.DataFrame):
        df = _set_timestamp_index(sample_df)
        df = _create_calendar_features(df)
        result = _create_cyclical_hour(df)

        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns

    def test_sin_cos_values_in_range(self, sample_df: pd.DataFrame):
        df = _set_timestamp_index(sample_df)
        df = _create_calendar_features(df)
        result = _create_cyclical_hour(df)

        assert result["hour_sin"].between(-1, 1).all()
        assert result["hour_cos"].between(-1, 1).all()


class TestCreateLagFeatures:
    def test_adds_lag_columns(self, sample_df: pd.DataFrame):
        df = _set_timestamp_index(sample_df)
        result = _create_lag_features(df)

        for lag in [1, 2, 48, 96, 336]:
            assert f"lag_{lag}" in result.columns

    def test_first_rows_are_nan(self, sample_df: pd.DataFrame):
        df = _set_timestamp_index(sample_df)
        result = _create_lag_features(df)

        assert result["lag_1"].iloc[0] is None or pd.isna(result["lag_1"].iloc[0])


class TestCreateRollingFeatures:
    def test_adds_rolling_columns(self, sample_df: pd.DataFrame):
        df = _set_timestamp_index(sample_df)
        result = _create_rolling_features(df)

        for col in ["demand_avg_24h", "demand_std_24h", "demand_avg_7d"]:
            assert col in result.columns


class TestCreateExogLags:
    def test_adds_exog_lag_columns(self, sample_df: pd.DataFrame):
        df = _set_timestamp_index(sample_df)
        result = _create_exog_lags(df)

        assert "solar_lag_1" in result.columns
        assert "usep_lag_1" in result.columns


class TestEngineerFeatures:
    def test_returns_dataframe_no_nans(self, sample_df: pd.DataFrame):
        result = engineer_features(sample_df)

        assert isinstance(result, pd.DataFrame)
        assert not result.isna().any().any()

    def test_has_all_expected_columns(self, sample_df: pd.DataFrame):
        result = engineer_features(sample_df)

        expected = {
            "demand", "solar", "usep",
            "hour", "minute", "day_of_week", "day_of_month", "month", "quarter", "is_weekend",
            "hour_sin", "hour_cos",
            "lag_1", "lag_2", "lag_48", "lag_96", "lag_336",
            "demand_avg_24h", "demand_std_24h", "demand_avg_7d",
            "solar_lag_1", "usep_lag_1",
        }
        assert expected.issubset(result.columns)

    def test_drops_leading_nan_rows(self, sample_df: pd.DataFrame):
        result = engineer_features(sample_df)

        assert len(result) < len(sample_df)
