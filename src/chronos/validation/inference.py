"""Dependence-robust comparison of forecast losses and multiple-testing corrections.

Forecast errors in a panel backtest are dependent in two directions: across entities (common
shocks hit all countries in the same years) and over time (h-step windows from consecutive
origins overlap). Tests that treat (entity, origin) pairs or entities as independent — a binomial
sign test across countries, a t-test on pooled pairs — are too liberal.

`origin_block_bootstrap` resamples moving blocks of consecutive origin years and keeps every
entity's loss differential for a resampled origin together, so both cross-sectional dependence
and temporal dependence up to the block length are preserved (moving-block bootstrap, Künsch 1989).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def holm(pvalues: Sequence[float | None]) -> list[float | None]:
    """Holm step-down adjusted p-values (family-wise error control). None/NaN entries are passed through."""
    idx = [i for i, p in enumerate(pvalues) if p is not None and not math.isnan(p)]
    m = len(idx)
    out: list[float | None] = list(pvalues)
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda i: pvalues[i])):  # type: ignore[arg-type,return-value]
        adj = min(1.0, (m - rank) * float(pvalues[i]))  # type: ignore[arg-type]
        running = max(running, adj)
        out[i] = running
    return out


def benjamini_hochberg(pvalues: Sequence[float | None]) -> list[float | None]:
    """Benjamini–Hochberg adjusted p-values (false-discovery-rate control)."""
    idx = [i for i, p in enumerate(pvalues) if p is not None and not math.isnan(p)]
    m = len(idx)
    out: list[float | None] = list(pvalues)
    running = 1.0
    for rank, i in reversed(list(enumerate(sorted(idx, key=lambda i: pvalues[i]), start=1))):  # type: ignore[arg-type,return-value]
        running = min(running, float(pvalues[i]) * m / rank)  # type: ignore[arg-type]
        out[i] = running
    return out


def origin_block_bootstrap(diffs_by_origin: Mapping[int, Sequence[float]], block_origins: int, n_boot: int = 2000,
                           seed: int = 0, alpha: float = 0.05) -> dict[str, Any]:
    """Moving-block bootstrap of the mean loss differential over (entity, origin) pairs.

    diffs_by_origin: origin year -> loss differentials (candidate minus comparator) of all entities.
    block_origins:   number of consecutive origins per block (use ceil(horizon / origin_step) so that
                     overlapping forecast windows stay inside one block).
    Returns the mean, a percentile CI and a two-sided bootstrap p-value for H0: mean = 0.
    """
    origins = sorted(o for o, v in diffs_by_origin.items() if len(v))
    T = len(origins)
    if T < 3:
        return {"n_origins": T, "note": "fewer than 3 origins: no bootstrap inference"}
    sums = np.array([float(np.sum(diffs_by_origin[o])) for o in origins])
    counts = np.array([len(diffs_by_origin[o]) for o in origins], dtype=float)
    mean = float(sums.sum()) / float(counts.sum())
    b = int(max(1, min(block_origins, T)))
    n_blocks = math.ceil(T / b)
    starts_max = T - b + 1
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    cs_s = np.concatenate(([0.0], np.cumsum(sums)))
    cs_c = np.concatenate(([0.0], np.cumsum(counts)))
    for r in range(n_boot):
        st = rng.integers(0, starts_max, size=n_blocks)
        s = float(np.sum(cs_s[st + b] - cs_s[st]))
        c = float(np.sum(cs_c[st + b] - cs_c[st]))
        stats[r] = s / c
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    centred = stats - stats.mean()
    p = float((np.sum(np.abs(centred) >= abs(mean)) + 1) / (n_boot + 1))
    return {"mean_difference": mean, "ci_low": float(lo), "ci_high": float(hi), "p_value": p,
            "n_origins": T, "n_pairs": int(counts.sum()), "block_origins": b, "n_boot": n_boot,
            "significant_improvement": bool(hi < 0), "significant_deterioration": bool(lo > 0)}


def paired_scaled_differences(records: Sequence[Any], candidate: str, comparator: str,
                              horizon: int) -> dict[int, list[float]]:
    """Group scaled absolute-error differences (candidate - comparator) by origin, over common (entity, origin).

    `records` are chronos.validation.backtest.Record objects carrying an `entity` attribute
    (set by the caller) and a per-origin scale (MASE denominator)."""
    by_key: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for r in records:
        if r.horizon != horizon or r.model not in (candidate, comparator) or not r.scale:
            continue
        by_key[(r.entity, r.origin)][r.model] = abs(r.actual - r.forecast) / r.scale
    out: dict[int, list[float]] = defaultdict(list)
    for (_, origin), v in by_key.items():
        if candidate in v and comparator in v:
            out[origin].append(v[candidate] - v[comparator])
    return dict(out)


def nonoverlap_origin_t_test(diffs_by_origin: Mapping[int, Sequence[float]], horizon: int, origin_step: int,
                             offset: int = 0) -> dict[str, Any]:
    """t-test on cross-sectional mean loss differentials at origins spaced >= horizon apart.

    Averaging over entities within an origin absorbs cross-sectional dependence; keeping only origins
    whose forecast windows do not overlap makes the per-origin means approximately independent.
    Simulation (common shocks with correlation 0.3-0.6, t(4) tails, overlapping 5-20 year windows)
    gave rejection rates of 4-10% at the nominal 5% level, whereas the origin-block bootstrap gave
    6-20% and the entity-level sign test 52-78%. Low power is the price of a correct size.
    """
    from scipy import stats

    origins = sorted(o for o, v in diffs_by_origin.items() if len(v))
    gap = max(1, math.ceil(horizon / max(origin_step, 1)))
    kept = origins[offset::gap]
    means = np.array([float(np.mean(diffs_by_origin[o])) for o in kept])
    n = len(means)
    if n < 3:
        return {"n_independent_origins": n, "p_value": None, "note": "fewer than 3 non-overlapping origins"}
    sd = float(means.std(ddof=1))
    if sd == 0:
        return {"n_independent_origins": n, "p_value": None, "note": "no variation"}
    t = float(means.mean() / (sd / math.sqrt(n)))
    p = float(2 * (1 - stats.t.cdf(abs(t), n - 1)))
    half = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n))
    return {"mean_difference": float(means.mean()), "ci_low": float(means.mean() - half), "ci_high": float(means.mean() + half),
            "t": t, "p_value": p, "n_independent_origins": n, "origins_used": kept,
            "significant_improvement": bool(means.mean() + half < 0), "significant_deterioration": bool(means.mean() - half > 0)}
