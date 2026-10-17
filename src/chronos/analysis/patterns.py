"""Historical pattern detection.

* Structural breaks: optimal partition of a signal (growth rates or levels) into segments
  with different means; number of breaks chosen by the modified BIC of Zhang & Siegmund (2007)
  (plain BIC, Yao 1988, over-detects; cf. Bai & Perron 2003).
* Trajectory similarity: which entity-periods moved like a given one (growth-path distance).
* Event-sequence enrichment: does category B follow category A within L years more often
  than when B's dates are randomly permuted? (Small samples => low power; reported as such.)
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from chronos.timeseries import Panel, TimeSeries


def signal_of(series: TimeSeries, kind: str = "growth") -> tuple[np.ndarray, np.ndarray]:
    """(years, values) of the signal. 'growth' = log change per year between consecutive observations."""
    run = series.longest_regular_run()
    if kind == "level":
        return run.years, run.values
    if kind == "diff":
        return run.years[1:], np.diff(run.values) / np.diff(run.years)
    if kind == "growth":
        if not run.is_positive:
            raise ValueError("growth signal needs a positive series; use kind='diff' or 'level'")
        return run.years[1:], np.diff(np.log(run.values)) / np.diff(run.years)
    raise ValueError(f"unknown signal kind {kind!r}")


def _segment_cost(cs: np.ndarray, cs2: np.ndarray, i: int, j: int) -> float:
    n = j - i
    s = cs[j] - cs[i]
    return float(cs2[j] - cs2[i] - s * s / n)


@dataclass(frozen=True)
class Segment:
    start_year: int
    end_year: int
    n: int
    mean: float
    sd: float

    def as_dict(self) -> dict[str, Any]:
        return {"start_year": self.start_year, "end_year": self.end_year, "n": self.n, "mean": self.mean, "sd": self.sd}


def structural_breaks(years: np.ndarray, x: np.ndarray, max_breaks: int = 5, min_size: int = 8,
                      model: str = "mean") -> dict[str, Any]:
    """Segmentation by dynamic programming; number of breaks chosen by modified BIC.

    model='mean'    : segments differ in mean, common variance (cost = SSE; criterion n log(SSE/n))
    model='meanvar' : segments differ in mean and variance (cost = sum n_i log var_i), suited to
                      series whose volatility changes (e.g. wars, famines); one extra parameter per break.
    """
    n = len(x)
    if n < 2 * min_size:
        return {"breaks": [], "segments": [], "note": f"series too short ({n}) for min_size={min_size}"}
    if model not in ("mean", "meanvar"):
        raise ValueError("model must be 'mean' or 'meanvar'")
    cs = np.concatenate(([0.0], np.cumsum(x)))
    cs2 = np.concatenate(([0.0], np.cumsum(x * x)))
    var_floor = max(float(np.var(x)) * 1e-4, 1e-12)

    def seg(i: int, j: int) -> float:
        sse = _segment_cost(cs, cs2, i, j)
        if model == "mean":
            return sse
        return (j - i) * math.log(max(sse / (j - i), var_floor))
    kmax = min(max_breaks, n // min_size - 1)
    inf = math.inf
    # cost[k][j]: min SSE of x[0:j] split into k+1 segments; back[k][j]: start of the last segment
    cost = np.full((kmax + 1, n + 1), inf)
    back = np.zeros((kmax + 1, n + 1), dtype=int)
    for j in range(min_size, n + 1):
        cost[0][j] = seg(0, j)
    for k in range(1, kmax + 1):
        for j in range((k + 1) * min_size, n + 1):
            best, arg = inf, -1
            for i in range(k * min_size, j - min_size + 1):
                c = cost[k - 1][i] + seg(i, j)
                if c < best:
                    best, arg = c, i
            cost[k][j], back[k][j] = best, arg
    def backtrack(k: int) -> list[int]:
        cuts, j = [], n
        for kk in range(k, 0, -1):
            i = int(back[kk][j])
            cuts.append(i)
            j = i
        return sorted(cuts)

    # Modified BIC (Zhang & Siegmund 2007): plain BIC over-detects breaks because the break location is
    # searched over; MBIC charges 3 log n per break plus a term that penalises very short segments.
    bics = []
    per_break = 3 if model == "mean" else 4
    for k in range(kmax + 1):
        fit = n * math.log(max(cost[k][n], 1e-300) / n) if model == "mean" else cost[k][n]
        bounds = [0, *backtrack(k), n]
        seg_term = sum(math.log((b - a) / n) for a, b in zip(bounds, bounds[1:], strict=False))
        bics.append(fit + per_break * k * math.log(n) - seg_term)
    k_best = int(np.argmin(bics))
    cuts = backtrack(k_best)
    bounds = [0, *cuts, n]
    segs = []
    for a, b in zip(bounds, bounds[1:], strict=False):
        chunk = x[a:b]
        segs.append(Segment(int(years[a]), int(years[b - 1]), b - a, float(chunk.mean()),
                            float(chunk.std(ddof=1)) if b - a > 1 else 0.0))
    return {
        "breaks": [int(years[c]) for c in cuts],
        "segments": [s.as_dict() for s in segs],
        "mbic_by_k": [round(b, 3) for b in bics],
        "k_selected": k_best,
        "model": model,
        "bic_gain_vs_no_break": round(bics[0] - bics[k_best], 3),
        "note": ("Break years are the first year of each new regime. Locations are estimates; serial correlation "
                 "in the signal can still produce extra breaks. Treat as descriptive, not causal."),
    }


def mean_shift_test(years: np.ndarray, x: np.ndarray, break_year: int) -> dict[str, Any]:
    """Welch test for a difference in mean of the signal before vs from `break_year` (descriptive)."""
    a, b = x[years < break_year], x[years >= break_year]
    if len(a) < 3 or len(b) < 3:
        return {"note": "not enough observations on one side"}
    t = stats.ttest_ind(b, a, equal_var=False)
    return {"before_mean": float(a.mean()), "after_mean": float(b.mean()), "difference": float(b.mean() - a.mean()),
            "welch_t": float(t.statistic), "p_value": float(t.pvalue), "n_before": len(a), "n_after": len(b),
            "caveat": "p-value assumes independent observations; autocorrelated growth rates overstate significance. "
                      "Choosing the break year after looking at the data invalidates the p-value."}


def similar_trajectories(panel: Panel, entity: str, start: int, end: int, k: int = 10, mode: str = "growth",
                         exclude_overlap_same_entity: bool = True, before_only: bool = False) -> dict[str, Any]:
    """Rank windows of equal length in all entities by distance to the query window.

    mode='growth': RMS difference of annual log growth rates (scale-free; requires positivity)
    mode='shape' : Euclidean distance of z-normalised levels (pure shape)
    before_only  : only windows ending before the query starts (what an analyst could have known)
    """
    if entity not in panel.series:
        raise ValueError(f"{entity!r} not in panel")
    q = panel.series[entity].between(start, end)
    length = end - start
    if len(q) != length + 1:
        raise ValueError(f"query window {start}-{end} is not fully observed for {entity}")

    def feat(v: np.ndarray) -> np.ndarray | None:
        if mode == "growth":
            return None if np.any(v <= 0) else np.diff(np.log(v))
        sd = v.std()
        return None if sd == 0 else (v - v.mean()) / sd

    qf = feat(q.values)
    if qf is None:
        raise ValueError("query window not usable in this mode")
    results: list[dict[str, Any]] = []
    all_d: list[float] = []
    for ent, s in panel.series.items():
        d = s.as_dict()
        for y0 in s.years:
            y0 = int(y0)
            y1 = y0 + length
            if before_only and y1 >= start:
                continue
            if ent == entity and exclude_overlap_same_entity and not (y1 < start or y0 > end):
                continue
            vals = [d.get(y) for y in range(y0, y1 + 1)]
            if any(v is None for v in vals):
                continue
            f = feat(np.array(vals, dtype=float))
            if f is None:
                continue
            dist = float(np.sqrt(np.mean((f - qf) ** 2)))
            all_d.append(dist)
            results.append({"entity": ent, "start": y0, "end": y1, "distance": dist})
    results.sort(key=lambda r: r["distance"])
    picked: list[dict[str, Any]] = []
    for r in results:  # at most one window per entity per decade, so matches are not near-duplicates
        if any(p["entity"] == r["entity"] and abs(p["start"] - r["start"]) < max(10, length // 2) for p in picked):
            continue
        picked.append(r)
        if len(picked) == k:
            break
    arr = np.array(all_d)
    for p in picked:
        p["distance_percentile"] = float((arr <= p["distance"]).mean() * 100) if arr.size else None
    return {"query": {"entity": entity, "start": start, "end": end, "mode": mode}, "n_candidate_windows": len(all_d),
            "matches": picked,
            "note": "Similar movement in one indicator is a superficial resemblance unless other dimensions "
                    "(technology, institutions, demography) also match; see the multidimensional analog engine."}


def event_sequence_enrichment(events: Sequence[tuple[str, int]], cat_a: str, cat_b: str, max_lag: int,
                              n_perm: int = 5000, seed: int = 0) -> dict[str, Any]:
    """Count A->B pairs with 0 < year_B - year_A <= max_lag; compare with B-dates permuted over the observed span."""
    a = np.array([y for c, y in events if c == cat_a])
    b = np.array([y for c, y in events if c == cat_b])
    if len(a) < 2 or len(b) < 2:
        return {"n_a": len(a), "n_b": len(b), "note": "fewer than 2 events in a category: no test possible"}
    years = np.array([y for _, y in events])
    lo, hi = years.min(), years.max()

    def count(bb: np.ndarray) -> int:
        lag = bb[None, :] - a[:, None]
        return int(np.sum((lag > 0) & (lag <= max_lag)))

    obs = count(b)
    rng = np.random.default_rng(seed)
    null = np.array([count(rng.integers(lo, hi + 1, size=len(b))) for _ in range(n_perm)])
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return {"category_a": cat_a, "category_b": cat_b, "max_lag": max_lag, "n_a": len(a), "n_b": len(b),
            "observed_pairs": obs, "null_mean_pairs": float(null.mean()), "p_value_one_sided": p,
            "caveat": "Null = B events uniformly placed within the observed span; event catalogues are curated, "
                      "incomplete and biased toward well-documented regions. Enrichment is not causation."}
