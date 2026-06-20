from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.database import get_supabase_client, save_results_to_supabase
from src.settings import SUPABASE_TABLE_NAME


class TestGetSupabaseClient:
    @patch.dict("os.environ", {"SUPABASE_URL": "https://test.supabase.co", "SUPABASE_KEY": "test-key"})
    @patch("src.database.create_client")
    def test_with_credentials(self, mock_create_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client

        result = get_supabase_client()

        assert result is not None
        assert result == mock_client
        mock_create_client.assert_called_once_with("https://test.supabase.co", "test-key")

    @patch.dict("os.environ", {}, clear=True)
    def test_without_credentials(self) -> None:
        with pytest.raises(ValueError, match="Supabase credentials"):
            get_supabase_client()


class TestSaveResultsToSupabase:
    @patch("src.database.get_supabase_client")
    def test_success(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_upsert = MagicMock()
        mock_client.table.return_value = mock_table
        mock_table.upsert.return_value = mock_upsert
        mock_get_client.return_value = mock_client

        timestamps = pd.date_range("2026-06-20", periods=3, freq="30min")
        result = {
            "date": date(2026, 6, 20),
            "predictions": {"Prophet": np.array([100.0, 101.0, 102.0])},
            "baselines": {
                "Last Value": np.array([90.0, 91.0, 92.0]),
                "Yesterday Same Time": np.array([88.0, 89.0, 90.0]),
            },
            "prophet_predictions": np.array([100.0, 101.0, 102.0]),
            "lightgbm_predictions": np.array([105.0, 106.0, 107.0]),
            "extratrees_predictions": np.array([103.0, 104.0, 105.0]),
            "actuals": np.array([98.0, 99.0, 100.0]),
            "timestamps": timestamps,
            "metrics": {
                "best_model": "LightGBM",
                "mae_prophet": 2.0,
                "mae_lightgbm": 1.5,
                "mae_extratrees": 2.5,
                "mae_last_value": 3.0,
            },
        }

        save_results_to_supabase(result)

        mock_get_client.assert_called_once()
        mock_client.table.assert_called_once_with(SUPABASE_TABLE_NAME)
        mock_table.upsert.assert_called_once()
        upsert_args = mock_table.upsert.call_args[0][0]
        assert len(upsert_args) == 3
        assert upsert_args[0]["actual_demand"] == 98.0
        assert upsert_args[0]["prophet_prediction"] == 100.0
        assert upsert_args[0]["lightgbm_prediction"] == 105.0
        assert upsert_args[0]["extratrees_prediction"] == 103.0
        assert upsert_args[0]["last_value_prediction"] == 90.0
        assert upsert_args[0]["best_model"] == "LightGBM"
        assert upsert_args[0]["mae_prophet"] == 2.0

    @patch("src.database.get_supabase_client")
    def test_no_timestamps_returns_early(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        result = {
            "date": date(2026, 6, 20),
            "timestamps": [],
            "actuals": np.array([]),
        }

        save_results_to_supabase(result)

        mock_client.table.assert_not_called()

    @patch("src.database.get_supabase_client")
    def test_insert_failure_propagates(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_upsert = MagicMock()
        mock_client.table.return_value = mock_table
        mock_table.upsert.return_value = mock_upsert
        mock_upsert.execute.side_effect = Exception("Database error")
        mock_get_client.return_value = mock_client

        timestamps = pd.date_range("2026-06-20", periods=1, freq="30min")
        result = {
            "date": date(2026, 6, 20),
            "predictions": {},
            "baselines": {},
            "prophet_predictions": np.array([100.0]),
            "lightgbm_predictions": np.array([105.0]),
            "extratrees_predictions": np.array([103.0]),
            "actuals": np.array([98.0]),
            "timestamps": timestamps,
            "metrics": {},
        }

        with pytest.raises(Exception, match="Database error"):
            save_results_to_supabase(result)
