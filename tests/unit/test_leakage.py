"""Future-leakage tests: a forecast made at origin o must not change when post-origin data change."""

import math

import numpy as np
import pytest

from chronos.forecasting.models import MODEL_FACTORY, build_models
from chronos.timeseries import Panel, TimeSeries
from chronos.validation.backtest import BacktestConfig, run_forecasts


def series(seed=1, n=120, entity="A"):
    rng = np.random.default_rng(seed)
    y = 100 * np.exp(np.cumsum(0.02 + rng.normal(0, 0.02, n)))
    return TimeSeries(np.arange(1900, 1900 + n), y, name="x", entity=entity)


def poisoned(s: TimeSeries, origin: int) -> TimeSeries:
    v = s.values.copy()
    v[s.years > origin] *= 1000.0
    return TimeSeries(s.years, v, name=s.name, entity=s.entity)


@pytest.mark.parametrize("name", sorted(MODEL_FACTORY))
def test_model_forecast_invariant_to_future(name):
    s = series()
    origin, h = 1990, 10
    panel = Panel("x", {"A": s, "B": series(2, entity="B"), "C": series(3, entity="C")})
    ppanel = Panel("x", {k: poisoned(v, origin) for k, v in panel.series.items()})
    m = build_models([name])[0]
    f1 = m.predict(s.truncate(origin), h, panel.truncate(origin) if m.needs_panel else None)
    f2 = m.predict(poisoned(s, origin).truncate(origin), h, ppanel.truncate(origin) if m.needs_panel else None)
    assert (f1 is None) == (f2 is None)
    if f1 is not None:
        assert f1.value == pytest.approx(f2.value, rel=1e-12)


def test_backtest_records_invariant_except_actuals():
    s = series()
    cfg = BacktestConfig(horizons=(5,), min_residuals=5)
    models = build_models(["naive", "log_drift", "analog_knn"])
    a = {(r.model, r.origin): r for r in run_forecasts(s, models, cfg)}
    # Poison everything after 1995; forecasts at origins <= 1995 must be identical.
    b = {(r.model, r.origin): r for r in run_forecasts(poisoned(s, 1995), models, cfg)}
    for key, r in a.items():
        if key[1] <= 1995:
            assert b[key].forecast == pytest.approx(r.forecast, rel=1e-12)


def test_empirical_intervals_use_only_known_errors():
    s = series()
    cfg = BacktestConfig(horizons=(10,), min_residuals=5)
    recs = [r for r in run_forecasts(s, build_models(["naive"]), cfg)]
    recs.sort(key=lambda r: r.origin)
    first_with_iv = next(r for r in recs if r.intervals)
    # needs 5 errors whose outcome year (origin + 10) <= this origin
    known = [r for r in recs if r.origin + 10 <= first_with_iv.origin]
    assert len(known) == 5


def test_truncate_physically_drops_future():
    s = series()
    t = s.truncate(1950)
    assert t.last_year == 1950 and all(y <= 1950 for y in t.years)
    assert math.isclose(t.values[-1], s.get(1950))
