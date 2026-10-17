import numpy as np
import pytest

from chronos.forecasting.models import AnalogKNN, Drift, Naive, build_models
from chronos.timeseries import Panel, TimeSeries
from chronos.validation import metrics as M
from chronos.validation.backtest import BacktestConfig, point_experiment, walk_forward
from chronos.validation.panel import panel_walk_forward


def rw(seed=0, n=150, drift=0.0, entity="A"):
    rng = np.random.default_rng(seed)
    return TimeSeries(np.arange(1850, 1850 + n), 100 + np.cumsum(drift + rng.normal(0, 1, n)), name="x", entity=entity)


def test_metrics_basic():
    assert M.mae([1, -1, 2]) == pytest.approx(4 / 3)
    assert M.rmse([3, 4]) == pytest.approx(np.sqrt(12.5))
    assert M.mape([100, 200], [110, 180]) == pytest.approx(10.0)
    assert M.mape([0, 1], [1, 1]) is None
    assert M.nonoverlapping_count(list(range(1900, 1950)), 10) == 5


def test_interval_score_penalises_misses():
    y = np.array([0.0, 10.0])
    good = M.interval_score(np.array([-1.0, 9.0]), np.array([1.0, 11.0]), y, 0.2)
    bad = M.interval_score(np.array([1.0, 11.0]), np.array([2.0, 12.0]), y, 0.2)
    assert bad > good


def test_dm_detects_clearly_better_model():
    rng = np.random.default_rng(1)
    e_good = rng.normal(0, 1, 200)
    e_bad = rng.normal(0, 3, 200)
    res = M.diebold_mariano(e_good, e_bad, h=1)
    assert res["statistic"] < 0 and res["p_value"] < 0.001
    same = M.diebold_mariano(rng.normal(0, 1, 200), rng.normal(0, 1, 200), h=5)
    assert same["p_value"] > 0.01


def test_on_random_walk_nothing_beats_naive_significantly():
    s = rw(3)
    res = walk_forward(s, [Naive(), Drift(window=20), AnalogKNN()], BacktestConfig(horizons=(5,)))
    h = res["summary"]["horizons"]["5"]
    # On a pure random walk the analog model must not be declared significantly better.
    assert not h["verdict"]["candidates"]["analog_knn"]["significant_at_5pct"]
    assert res["revision_leakage"]["risk"] == "PRESENT"


def test_empirical_interval_calibration_on_iid_noise():
    rng = np.random.default_rng(5)
    s = TimeSeries(np.arange(1800, 2000), 50 + rng.normal(0, 1, 200), name="x")
    res = walk_forward(s, build_models(["mean"]), BacktestConfig(horizons=(1,), min_residuals=30))
    cal = res["summary"]["horizons"]["1"]["models"]["mean_30"]["empirical_interval_calibration"]["0.8"]
    assert 0.7 <= cal["empirical_coverage"] <= 0.9


def test_point_experiment_warns_single_origin():
    s = rw(1, drift=0.5)
    res = point_experiment(s, [Naive(), Drift(window=20)], 1950, 1970)
    assert res["actual"] == pytest.approx(s.get(1970))
    assert "NOT evidence" in res["warning"]
    assert res["history_last_year"] == 1950


def test_panel_backtest_runs_and_reports_sign_test():
    panel = Panel("x", {e: rw(i, drift=0.3, entity=e) for i, e in enumerate("ABCDEFGH")})
    res = panel_walk_forward(panel, [Naive(), Drift(window=20), AnalogKNN(min_separation=10)],
                             BacktestConfig(horizons=(10,), origin_step=5))
    h = res["horizons"]["10"]
    assert h["n_entities_scored"] == 8
    assert "analog_knn_sep10" in h["candidate_vs_best_baseline"]
