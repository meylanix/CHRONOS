"""Point and interval forecast metrics and the Diebold–Mariano test."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
from scipy import stats

ArrayLike: TypeAlias = Sequence[float] | np.ndarray


def _arr(x: ArrayLike) -> np.ndarray:
    return np.asarray(x, dtype=float)


def mae(err: ArrayLike) -> float:
    return float(np.mean(np.abs(_arr(err))))


def rmse(err: ArrayLike) -> float:
    return float(np.sqrt(np.mean(_arr(err) ** 2)))


def mape(actual: ArrayLike, pred: ArrayLike) -> float | None:
    a, p = _arr(actual), _arr(pred)
    if np.any(a <= 0):
        return None
    return float(np.mean(np.abs(a - p) / a) * 100)


def mean_abs_log_ratio(actual: ArrayLike, pred: ArrayLike) -> float | None:
    """Mean |log(pred/actual)|: scale-free error for positive quantities (0.10 ~ 10% off)."""
    a, p = _arr(actual), _arr(pred)
    if np.any(a <= 0) or np.any(p <= 0):
        return None
    return float(np.mean(np.abs(np.log(p / a))))


def interval_score(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, alpha: float) -> float:
    """Gneiting & Raftery (2007) interval score for a central (1-alpha) interval; lower is better."""
    width = hi - lo
    below = (2 / alpha) * (lo - y) * (y < lo)
    above = (2 / alpha) * (y - hi) * (y > hi)
    return float(np.mean(width + below + above))


def diebold_mariano(e1: ArrayLike, e2: ArrayLike, h: int, power: int = 1) -> dict[str, float | int | str]:
    """DM test of equal accuracy with the Harvey–Leybourne–Newbold small-sample correction.

    Loss = |e|^power. Negative statistic: model 1 more accurate. Overlapping h-step forecasts are
    autocorrelated up to lag h-1; the long-run variance uses those autocovariances (rectangular,
    falling back to Bartlett weights if the estimate is not positive).
    """
    d = np.abs(_arr(e1)) ** power - np.abs(_arr(e2)) ** power
    n = d.size
    if n < max(8, 2 * h):
        return {"n": n, "statistic": math.nan, "p_value": math.nan,
                "note": f"too few paired forecasts ({n}) for a reliable test at horizon {h} (need >= {max(8, 2 * h)})"}
    dbar = d.mean()
    dc = d - dbar
    gamma = [float(np.dot(dc[k:], dc[: n - k]) / n) for k in range(max(1, h))]
    lrv = gamma[0] + 2 * sum(gamma[1:])
    if lrv <= 0:
        lrv = gamma[0] + 2 * sum((1 - k / h) * g for k, g in enumerate(gamma[1:], start=1))
    if lrv <= 0:
        return {"n": n, "statistic": math.nan, "p_value": math.nan, "note": "degenerate loss differential"}
    dm = dbar / math.sqrt(lrv / n)
    corr = math.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    stat = dm * corr
    p = 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))
    return {"n": n, "statistic": float(stat), "p_value": float(p), "mean_loss_diff": float(dbar)}


def nonoverlapping_count(origins: Sequence[int], horizon: int) -> int:
    """Greedy count of origins whose forecast windows (o, o+h] do not overlap."""
    count, last = 0, None
    for o in sorted(origins):
        if last is None or o >= last + horizon:
            count += 1
            last = o
    return count
