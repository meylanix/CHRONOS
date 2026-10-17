"""The refactored analog model must reproduce CHRONOS v0.1 exactly (preserved functionality)."""

import math
import random

import pytest

from chronos.forecasting.models import AnalogKNN, V01LogTrend
from chronos.ingestion.user_csv import demo_batch
from chronos.timeseries import TimeSeries
from chronos.validation.backtest import BacktestConfig, walk_forward


def demo_series():
    rows = demo_batch().series[0].rows
    return TimeSeries.from_pairs([(r.year, r.value) for r in rows], name="DEMO", data_kind="synthetic")


def v01_demo_dict():
    rng = random.Random(104)
    val, out = 30.0, {}
    for year in range(1900, 2026):
        growth = 0.013 if year < 1950 else 0.024 if year < 1980 else 0.007 if year < 2000 else 0.003
        val *= math.exp(growth + rng.gauss(0, 0.007))
        out[year] = val
    return out


def test_demo_generator_identical_to_v01():
    s = demo_series()
    assert s.as_dict() == v01_demo_dict()


@pytest.mark.parametrize("origin", [1950, 1975, 2000, 2025])
def test_analog_forecast_matches_v01(legacy_v01, origin):
    data = v01_demo_dict()
    old = legacy_v01.forecast(data, origin, 10, 5, 7)
    s = demo_series().truncate(origin)
    new = AnalogKNN(lookback=5, neighbors=7).predict(s, 10)
    base = V01LogTrend(5).predict(s, 10)
    assert old is not None and new is not None
    assert new.value == pytest.approx(old["analog_scenarios"]["middle"], abs=1e-4)
    assert new.lo == pytest.approx(old["analog_scenarios"]["low"], abs=1e-4)
    assert new.hi == pytest.approx(old["analog_scenarios"]["high"], abs=1e-4)
    assert base.value == pytest.approx(old["linear_log_trend_baseline"], abs=1e-4)
    assert [a["end_year"] for a in new.details["analogs"]] == [a["year"] for a in old["analogs"]]
    assert new.details["n_candidates"] == old["n_candidates"]


def test_backtest_reproduces_v01_numbers(legacy_v01):
    old = legacy_v01.backtest(v01_demo_dict(), 10, 5, 7)
    cfg = BacktestConfig(horizons=(10,), min_history=1, first_origin=1925, min_residuals=10**9)
    res = walk_forward(demo_series(), [AnalogKNN(), V01LogTrend(5)], cfg)
    h = res["summary"]["horizons"]["10"]
    assert h["n_common_origins"] == old["n"]
    assert h["models"]["analog_knn"]["MAE"] == pytest.approx(old["analog_MAE"], abs=1e-3)
    assert h["models"]["v01_log_trend"]["MAE"] == pytest.approx(old["baseline_MAE"], abs=1e-3)
    cov = h["models"]["analog_knn"]["native_interval"]["empirical_coverage"] * 100
    assert cov == pytest.approx(old["historical_interval_coverage_percent"], abs=0.1)
    # v0.1 found the analog model loses to its baseline on this data; the new verdict must say so too.
    assert old["analogs_better_than_baseline"] is False
    assert h["verdict"]["candidates"]["analog_knn"]["beats_best_baseline"] is False
