"""Forecasting models: simple baselines and the historical-analog (k-NN) model.

Every model sees only `history` (already truncated at the forecast origin by the caller)
and, for cross-entity models, a panel truncated at the same origin. The horizon is in
years and must be a multiple of the series step (1 for annual, 100 for Seshat centuries).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from chronos.timeseries import Panel, TimeSeries


@dataclass(frozen=True)
class Forecast:
    value: float
    # Model-native interval (e.g. analog percentiles), with its nominal level. Optional.
    lo: float | None = None
    hi: float | None = None
    level: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


class Model:
    name = "model"
    is_baseline = False
    needs_panel = False
    description = ""

    def predict(self, history: TimeSeries, horizon: int, panel: Panel | None = None) -> Forecast | None:
        raise NotImplementedError

    def config(self) -> dict[str, Any]:
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}


def _steps(history: TimeSeries, horizon: int) -> tuple[TimeSeries, int, int] | None:
    run = history.tail_regular_run()
    st = history.step()
    if horizon % st:
        return None
    return run, st, horizon // st


class Naive(Model):
    name = "naive"
    is_baseline = True
    description = "Random walk: the last observed value persists."

    def predict(self, history: TimeSeries, horizon: int, panel: Panel | None = None) -> Forecast | None:
        return Forecast(float(history.values[-1])) if len(history) else None


class Drift(Model):
    """Random walk with drift on levels, drift estimated over the last `window` steps (None = whole run)."""

    is_baseline = True

    def __init__(self, window: int | None = None, log: bool = False) -> None:
        self.window = window
        self.log = log
        self.name = ("log_drift" if log else "drift") + (f"_{window}" if window else "")
        self.description = ("Constant growth rate" if log else "Constant absolute change") + \
            (f" estimated over the last {window} steps" if window else " over the full contiguous history")

    def predict(self, history: TimeSeries, horizon: int, panel: Panel | None = None) -> Forecast | None:
        s = _steps(history, horizon)
        if s is None:
            return None
        run, _, hs = s
        y = run.values
        if self.log:
            if np.any(y <= 0):
                return None
            y = np.log(y)
        w = min(self.window or (len(y) - 1), len(y) - 1)
        if w < 1:
            return None
        slope = (y[-1] - y[-1 - w]) / w
        out = y[-1] + slope * hs
        return Forecast(float(math.exp(out) if self.log else out))


class V01LogTrend(Model):
    """The v0.1 baseline: mean of the last lookback-1 annual log returns, extrapolated."""

    name = "v01_log_trend"
    is_baseline = True
    description = "v0.1 baseline: recent mean log growth (lookback window) continued."

    def __init__(self, lookback: int = 5) -> None:
        self.lookback = lookback

    def predict(self, history: TimeSeries, horizon: int, panel: Panel | None = None) -> Forecast | None:
        s = _steps(history, horizon)
        if s is None:
            return None
        run, _, hs = s
        if len(run) < self.lookback or np.any(run.values[-self.lookback:] <= 0):
            return None
        lr = np.diff(np.log(run.values[-self.lookback:]))
        return Forecast(float(run.values[-1] * math.exp(float(np.mean(lr)) * hs)))


class HistoricalMean(Model):
    is_baseline = True

    def __init__(self, window: int = 30) -> None:
        self.window = window
        self.name = f"mean_{window}"
        self.description = f"Mean of the last {window} observations (suits stationary indicators)."

    def predict(self, history: TimeSeries, horizon: int, panel: Panel | None = None) -> Forecast | None:
        if not len(history):
            return None
        return Forecast(float(np.mean(history.values[-self.window:])))


class LinearTrend(Model):
    is_baseline = True

    def __init__(self, window: int = 30, log: bool = True) -> None:
        self.window = window
        self.log = log
        self.name = f"{'log_' if log else ''}linear_trend_{window}"
        self.description = f"OLS trend over the last {window} steps ({'log' if log else 'level'} scale)."

    def predict(self, history: TimeSeries, horizon: int, panel: Panel | None = None) -> Forecast | None:
        s = _steps(history, horizon)
        if s is None:
            return None
        run, _, hs = s
        y = run.values[-self.window:]
        if len(y) < 5:
            return None
        if self.log:
            if np.any(y <= 0):
                return None
            y = np.log(y)
        x = np.arange(len(y), dtype=float)
        b, a = np.polyfit(x, y, 1)
        out = a + b * (len(y) - 1 + hs)
        return Forecast(float(math.exp(out) if self.log else out))


class DampedTrend(Model):
    """Holt's linear method with a damped trend (Gardner & McKenzie), log scale for positive series."""

    name = "damped_trend"
    is_baseline = False
    description = "Exponential smoothing with damped trend (statsmodels Holt, damped)."

    def __init__(self, window: int = 60) -> None:
        self.window = window

    def predict(self, history: TimeSeries, horizon: int, panel: Panel | None = None) -> Forecast | None:
        from statsmodels.tsa.holtwinters import Holt

        s = _steps(history, horizon)
        if s is None:
            return None
        run, _, hs = s
        y = run.values[-self.window:]
        if len(y) < 10:
            return None
        log = bool(np.all(y > 0))
        z = np.log(y) if log else y
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fit = Holt(z, damped_trend=True, initialization_method="estimated").fit()
                out = float(fit.forecast(hs)[-1])
            except (ValueError, np.linalg.LinAlgError):
                return None
        if not math.isfinite(out):
            return None
        return Forecast(math.exp(out) if log else out)


class AnalogKNN(Model):
    """Historical analogs: find past windows whose recent log-growth pattern resembles the current one and
    use what followed them. With min_separation=0, pool=False and level_penalty=0.007 this reproduces v0.1
    exactly (regression-tested against legacy/chronos_v01.py).

    Only analog windows whose outcome was already observed by the origin are eligible (end + h <= origin).
    """

    name = "analog_knn"
    description = "k nearest historical analogs of the recent growth pattern; median of their outcomes."

    def __init__(self, lookback: int = 5, neighbors: int = 7, level_penalty: float = 0.007,
                 min_separation: int = 0, pool: bool = False, name: str | None = None) -> None:
        if lookback < 3 or neighbors < 2:
            raise ValueError("Need lookback>=3 and neighbors>=2")
        self.lookback = lookback
        self.neighbors = neighbors
        self.level_penalty = level_penalty
        self.min_separation = min_separation
        self.pool = pool
        self.needs_panel = pool
        self.name = name or ("analog_knn_pooled" if pool else "analog_knn") + (f"_sep{min_separation}" if min_separation else "")

    def _candidates(self, s: TimeSeries, origin: int, horizon: int, current: np.ndarray, last: float,
                    entity: str) -> list[tuple[float, int, str, float]]:
        """Vectorised scan of every eligible analog end year e (window [e-lb+1, e+h] fully observed, e+h <= origin)."""
        lb = self.lookback
        first, final = int(s.years[0]), int(s.years[-1])
        e_min, e_max = first + lb - 1, min(origin - horizon, origin - 1, final - horizon)
        if e_max < e_min:
            return []
        dense = np.full(final - first + 1, np.nan)
        dense[s.years - first] = s.values
        with np.errstate(divide="ignore", invalid="ignore"):
            logv = np.where(dense > 0, np.log(np.where(dense > 0, dense, 1.0)), np.nan)
        ends = np.arange(e_min, e_max + 1)
        idx = ends - first
        valid = np.concatenate(([0], np.cumsum(np.isfinite(logv))))
        lo, hi = idx - lb + 1, idx + horizon
        ok = (valid[hi + 1] - valid[lo]) == (hi - lo + 1)
        if not ok.any():
            return []
        ends, idx = ends[ok], idx[ok]
        returns = np.diff(logv)
        windows = np.lib.stride_tricks.sliding_window_view(returns, lb - 1)[idx - lb + 1]
        gaps = np.sqrt(np.mean((windows - current) ** 2, axis=1))
        level_gap = np.abs(math.log(last) - logv[idx])
        dist = gaps + self.level_penalty * level_gap
        outcome = logv[idx + horizon] - logv[idx]
        return [(float(d), int(e), entity, float(o)) for d, e, o in zip(dist, ends, outcome, strict=True)]

    def predict(self, history: TimeSeries, horizon: int, panel: Panel | None = None) -> Forecast | None:
        if history.step() != 1 or not len(history):
            return None  # the analog window logic is annual
        origin = history.last_year
        lb = self.lookback
        vals = history.as_dict()
        if not all(vals.get(y, 0) > 0 for y in range(origin - lb + 1, origin + 1)):
            return None
        current = np.array([math.log(vals[y] / vals[y - 1]) for y in range(origin - lb + 2, origin + 1)])
        last = vals[origin]
        scored = self._candidates(history, origin, horizon, current, last, history.entity)
        if self.pool and panel is not None:
            for ent, s in panel.series.items():
                if ent == history.entity or s.last_year > origin:
                    continue
                scored.extend(self._candidates(s, origin, horizon, current, last, ent))
        if len(scored) < self.neighbors:
            return None
        chosen: list[tuple[float, int, str, float]] = []
        for cand in sorted(scored, key=lambda x: (x[0], x[1], x[2])):
            if self.min_separation and any(c[2] == cand[2] and abs(c[1] - cand[1]) < self.min_separation for c in chosen):
                continue
            chosen.append(cand)
            if len(chosen) == self.neighbors:
                break
        if len(chosen) < self.neighbors:
            return None
        outcomes = [c[3] for c in chosen]
        return Forecast(
            last * math.exp(_percentile(outcomes, 0.5)),
            lo=last * math.exp(_percentile(outcomes, 0.1)),
            hi=last * math.exp(_percentile(outcomes, 0.9)),
            level=0.8,
            details={"analogs": [{"entity": c[2], "end_year": c[1], "distance": round(c[0], 6),
                                  "observed_change_pct": round(100 * (math.exp(c[3]) - 1), 2)} for c in chosen],
                     "n_candidates": len(scored)},
        )


def _percentile(xs: list[float], fraction: float) -> float:
    """Linear-interpolation percentile identical to v0.1."""
    ordered = sorted(xs)
    pos = (len(ordered) - 1) * fraction
    i = int(pos)
    rem = pos - i
    return ordered[i] * (1 - rem) + ordered[min(i + 1, len(ordered) - 1)] * rem


def default_models(positive: bool = True, include_analog: bool = True, pooled: bool = False) -> list[Model]:
    models: list[Model] = [Naive(), Drift(window=20), V01LogTrend(5), LinearTrend(30, log=positive),
                           HistoricalMean(30), DampedTrend()]
    if positive:
        models.insert(2, Drift(window=20, log=True))
    if include_analog and positive:
        models.append(AnalogKNN())
        models.append(AnalogKNN(min_separation=10))
        if pooled:
            models.append(AnalogKNN(pool=True, min_separation=10))
    return models


MODEL_FACTORY = {
    "naive": lambda: Naive(),
    "drift": lambda: Drift(window=20),
    "drift_full": lambda: Drift(),
    "log_drift": lambda: Drift(window=20, log=True),
    "v01_log_trend": lambda: V01LogTrend(5),
    "linear_trend": lambda: LinearTrend(30, log=False),
    "log_linear_trend": lambda: LinearTrend(30, log=True),
    "mean": lambda: HistoricalMean(30),
    "damped_trend": lambda: DampedTrend(),
    "analog_knn": lambda: AnalogKNN(),
    "analog_knn_sep10": lambda: AnalogKNN(min_separation=10),
    "analog_knn_pooled": lambda: AnalogKNN(pool=True, min_separation=10),
}


def build_models(names: list[str]) -> list[Model]:
    unknown = [n for n in names if n not in MODEL_FACTORY]
    if unknown:
        raise ValueError(f"Unknown models {unknown}; available: {sorted(MODEL_FACTORY)}")
    return [MODEL_FACTORY[n]() for n in names]
