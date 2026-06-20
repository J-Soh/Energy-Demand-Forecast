import pandas as pd
import pytest

from src.extractor import _process_dataframe, extract_data
from src.settings import YESTR_DATE


class TestProcessDataframe:
    def test_returns_dataframe_with_expected_columns(self, raw_df: pd.DataFrame):
        result = _process_dataframe(raw_df)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["timestamp", "demand", "solar", "usep"]

    def test_parses_timestamp_correctly(self, raw_df: pd.DataFrame):
        result = _process_dataframe(raw_df)

        assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])
        assert result["timestamp"].iloc[0] == pd.Timestamp("2025-12-18 00:00:00")

    def test_renames_and_orders_columns(self, raw_df: pd.DataFrame):
        result = _process_dataframe(raw_df)

        assert result["demand"].iloc[0] == 5932.252
        assert result["solar"].iloc[0] == 0.0
        assert result["usep"].iloc[0] == 78.34

    def test_sorts_by_timestamp(self, raw_df: pd.DataFrame):
        unsorted = raw_df.sample(frac=1, random_state=42)
        result = _process_dataframe(unsorted)

        assert result["timestamp"].is_monotonic_increasing

    @pytest.mark.xfail(reason="pyarrow < 19 has a kernel bug with empty string concat")
    def test_handles_empty_dataframe(self):
        empty = pd.DataFrame({c: pd.Series(dtype="object") for c in
                              ["Date", "Period", "Demand (MW)", "Solar (MW)", "USEP ($/MWh)"]})
        result = _process_dataframe(empty)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


@pytest.mark.smoke
class TestExtractDataLive:
    def test_returns_dataframe_with_correct_columns(self):
        df = extract_data(YESTR_DATE)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["timestamp", "demand", "solar", "usep"]

    def test_has_at_least_one_row(self):
        df = extract_data(YESTR_DATE)

        assert len(df) > 0

    def test_timestamps_are_datetime_and_sorted(self):
        df = extract_data(YESTR_DATE)

        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        assert df["timestamp"].is_monotonic_increasing
