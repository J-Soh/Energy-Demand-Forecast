import numpy as np
import pandas as pd
import pytest

from src.benchmark import compare_all_models, compute_baselines, compute_peak_mae


@pytest.fixture
def y_test(engineered_data) -> pd.Series:
    return pd.Series(
        np.arange(100, 200, dtype=float),
        index=engineered_data.index,
        name="demand",
    )


@pytest.fixture
def engineered_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = {
        "lag_2": np.arange(98, 198, dtype=float) + rng.normal(0, 5, 100),
        "lag_48": np.arange(52, 152, dtype=float) + rng.normal(0, 10, 100),
        "lag_336": np.arange(-236, -136, dtype=float) + rng.normal(0, 20, 100),
        "demand_avg_24h": np.full(100, 150.0) + rng.normal(0, 5, 100),
    }
    idx = pd.date_range("2026-06-20", periods=100, freq="30min", name="timestamp")
    return pd.DataFrame(data, index=idx)


class TestComputeBaselines:
    def test_returns_dataframe_with_correct_structure(self, engineered_data, y_test):
        result = compute_baselines(engineered_data, y_test)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["Model", "MAE", "RMSE"]
        assert len(result) == 4

    def test_all_baseline_models_present(self, engineered_data, y_test):
        result = compute_baselines(engineered_data, y_test)

        models = result["Model"].tolist()
        assert "Last Value" in models
        assert "Yesterday Same Time" in models
        assert "Last Week Same Time" in models
        assert "Avg Demand 24h" in models

    def test_sorted_by_mae_ascending(self, engineered_data, y_test):
        result = compute_baselines(engineered_data, y_test)

        assert result["MAE"].is_monotonic_increasing

    def test_mae_and_rmse_are_non_negative(self, engineered_data, y_test):
        result = compute_baselines(engineered_data, y_test)

        assert (result["MAE"] >= 0).all()
        assert (result["RMSE"] >= 0).all()


class TestComputePeakMAE:
    def test_returns_float(self):
        y_test = pd.Series(np.arange(100, dtype=float))
        preds = pd.Series(np.arange(100, dtype=float) + np.random.normal(0, 2, 100))

        result = compute_peak_mae(y_test, preds)

        assert isinstance(result, float)
        assert result >= 0

    def test_peak_mae_not_zero_when_errors_exist(self):
        y_test = pd.Series(np.array([50, 60, 70, 80, 90, 100], dtype=float))
        preds = pd.Series(np.array([50, 60, 70, 80, 90, 200], dtype=float))

        result = compute_peak_mae(y_test, preds, percentile=0.5)

        assert result > 0

    def test_zero_when_all_perfect(self):
        y_test = pd.Series(np.arange(100, dtype=float))
        preds = y_test.copy()

        result = compute_peak_mae(y_test, preds)

        assert result == 0.0


class TestCompareAllModels:
    def test_returns_sorted_dataframe(self, y_test):
        predictions = {"ModelA": np.arange(100, 200, dtype=float) + 5.0}
        baselines = pd.DataFrame([
            {"Model": "Baseline", "MAE": 20.0, "RMSE": 25.0},
        ])

        result = compare_all_models(y_test, predictions, baselines)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["Model", "MAE", "RMSE"]
        assert result["MAE"].is_monotonic_increasing

    def test_combines_ml_and_baseline_models(self, y_test):
        predictions = {"ML1": np.arange(100, 200, dtype=float) + 2.0, "ML2": np.arange(100, 200, dtype=float) + 10.0}
        baselines = pd.DataFrame([
            {"Model": "B1", "MAE": 15.0, "RMSE": 20.0},
            {"Model": "B2", "MAE": 5.0, "RMSE": 8.0},
        ])

        result = compare_all_models(y_test, predictions, baselines)

        assert len(result) == 4
