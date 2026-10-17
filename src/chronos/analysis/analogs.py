"""Multidimensional historical analogy engine.

An "analog" of (entity, year) is another (entity', year') whose *state vector* is close across
several groups of dimensions: economy, demography, energy/technology, institutions, and the
position relative to the contemporary frontier. Matching in one dimension only is flagged as
superficial. Entities from different sources are joined by ISO3 code.

The engine can also be backtested as a forecaster: does the median subsequent change of the
analogs predict an entity's next h-year change better than simple baselines? (`backtest`)
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from chronos.analysis.data import panel_frame
from chronos.errors import DataError


@dataclass(frozen=True)
class Dimension:
    key: str                  # feature name in the state table
    indicator: str            # stored indicator
    source_id: str
    group: str                # economy | demography | energy | institutions | position
    transform: str = "log"    # log | level
    trend_years: int | None = None  # if set: average annual change of the transformed value over k years
    weight: float = 1.0


DEFAULT_DIMENSIONS: tuple[Dimension, ...] = (
    Dimension("log_gdp_pc", "gdp_per_capita", "owid", "economy", "log"),
    Dimension("gdp_pc_growth_10y", "gdp_per_capita", "owid", "economy", "log", trend_years=10),
    Dimension("pop_growth_10y", "population", "owid", "demography", "log", trend_years=10),
    Dimension("life_expectancy", "e_pelifeex", "vdem", "demography", "level"),
    Dimension("log_co2_pc", "co2_per_capita", "owid", "energy", "log"),
    Dimension("electoral_democracy", "v2x_polyarchy", "vdem", "institutions", "level"),
    Dimension("rule_of_law", "v2x_rule", "vdem", "institutions", "level"),
)
FRONTIER_KEY = "gdp_pc_gap_to_frontier"  # log GDP pc minus the max log GDP pc observed that year (position group)


def build_state_table(con: sqlite3.Connection, dims: Sequence[Dimension] = DEFAULT_DIMENSIONS,
                      min_year: int = 1800, frontier: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Wide table indexed by (code, year) with one column per feature, plus 'name'."""
    frames, names, missing = [], {}, []
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    for d in dims:
        key = (d.indicator, d.source_id)
        if key not in cache:
            cache[key] = panel_frame(con, d.indicator, d.source_id)
        df = cache[key]
        if df.empty:
            missing.append(f"{d.indicator} ({d.source_id})")
            continue
        df = df[df["entity_code"].notna() & (df["year"] >= min_year - (d.trend_years or 0))].copy()
        df = df[~df["entity_code"].astype(str).str.startswith("OWID_")]  # OWID aggregates/regions
        if d.source_id == "owid":
            names.update(dict(zip(df["entity_code"], df["entity"], strict=False)))
        else:
            for c, n in zip(df["entity_code"], df["entity"], strict=False):
                names.setdefault(c, n)
        val = df["value"].astype(float)
        if d.transform == "log":
            val = np.log(val.where(val > 0))
        df["v"] = val
        s = df.set_index(["entity_code", "year"])["v"].sort_index()
        s = s[~s.index.duplicated()]
        if d.trend_years:
            k = d.trend_years
            lagged = s.copy()
            lagged.index = pd.MultiIndex.from_arrays([lagged.index.get_level_values(0), lagged.index.get_level_values(1) + k])
            s = (s - lagged.reindex(s.index)) / k
        frames.append(s.rename(d.key))
    if not frames:
        raise DataError("No dimension data found. Fetch owid co2 and vdem first (see README).")
    table = pd.concat(frames, axis=1)
    table = table[table.index.get_level_values(1) >= min_year]
    if frontier and "log_gdp_pc" in table:
        fr = table["log_gdp_pc"].groupby(level=1).transform("max")
        table[FRONTIER_KEY] = table["log_gdp_pc"] - fr
    table["name"] = [names.get(c, c) for c in table.index.get_level_values(0)]
    info = {"n_rows": len(table), "n_entities": table.index.get_level_values(0).nunique(),
            "missing_dimensions": missing, "features": [c for c in table.columns if c != "name"]}
    return table, info


def _groups(dims: Sequence[Dimension], features: Sequence[str]) -> dict[str, str]:
    g = {d.key: d.group for d in dims if d.key in features}
    if FRONTIER_KEY in features:
        g[FRONTIER_KEY] = "position"
    return g


@dataclass
class AnalogEngine:
    table: pd.DataFrame
    dims: Sequence[Dimension] = DEFAULT_DIMENSIONS
    min_coverage: float = 0.6        # share of features present for both target and candidate
    min_groups: int = 3              # distinct dimension groups that must be comparable
    exclude_same_entity_years: int = 25
    superficial_threshold: float = 1.0  # group distance (SD units) above which a group "differs substantially"

    def __post_init__(self) -> None:
        self.features = [c for c in self.table.columns if c != "name"]
        self.group_of = _groups(self.dims, self.features)
        self.weights = np.array([next((d.weight for d in self.dims if d.key == f), 1.0) for f in self.features])
        self._X = self.table[self.features].to_numpy(dtype=float)
        self._codes = self.table.index.get_level_values(0).to_numpy()
        self._years = self.table.index.get_level_values(1).to_numpy()

    def _standardizer(self, cutoff: int) -> tuple[np.ndarray, np.ndarray]:
        mask = self._years <= cutoff
        mu = np.nanmean(self._X[mask], axis=0)
        sd = np.nanstd(self._X[mask], axis=0)
        sd[~np.isfinite(sd) | (sd == 0)] = 1.0
        return mu, sd

    def find(self, code: str, year: int, k: int = 10, outcome_horizon: int | None = None,
             cutoff: int | None = None) -> dict[str, Any]:
        """Analogs of (code, year). With outcome_horizon=h, only candidates whose next h years were
        observed by `cutoff` (default: the target year) are eligible."""
        cutoff = year if cutoff is None else cutoff
        hits = np.where((self._codes == code) & (self._years == year))[0]
        if not len(hits):
            raise DataError(f"No state vector for {code} {year}")
        t = hits[0]
        mu, sd = self._standardizer(cutoff)
        Z = (self._X - mu) / sd
        zt = Z[t]
        latest = cutoff - (outcome_horizon or 0)
        cand = (self._years <= latest) & ~((self._codes == code) & (np.abs(self._years - year) < self.exclude_same_entity_years))
        diff = Z[cand] - zt
        present = np.isfinite(diff)
        w = self.weights
        cov = (present * w).sum(axis=1) / w[np.isfinite(zt)].sum() if np.isfinite(zt).any() else np.zeros(len(diff))
        groups = sorted(set(self.group_of.values()))
        gpresent = np.stack([present[:, [self.group_of[f] == g for f in self.features]].any(axis=1) for g in groups], axis=1)
        sq = np.where(present, diff ** 2 * w, 0.0)
        dist = np.sqrt(sq.sum(axis=1) / np.maximum((present * w).sum(axis=1), 1e-12))
        ok = (cov >= self.min_coverage) & (gpresent.sum(axis=1) >= self.min_groups)
        idx_all = np.where(cand)[0]
        order = np.argsort(np.where(ok, dist, np.inf), kind="stable")
        picked: list[int] = []
        seen: set[tuple[str, int]] = set()
        for o in order:
            if not ok[o]:
                break
            row = idx_all[o]
            key = (str(self._codes[row]), int(self._years[row]) // 10)
            if key in seen:
                continue
            seen.add(key)
            picked.append(int(o))
            if len(picked) == k:
                break
        out = []
        for o in picked:
            row = idx_all[o]
            gd = {}
            for g in groups:
                cols = [i for i, f in enumerate(self.features) if self.group_of[f] == g and present[o, i]]
                gd[g] = float(np.sqrt(np.mean(diff[o, cols] ** 2))) if cols else None
            feats = {f: {"target": _fin(self._X[t, i]), "analog": _fin(self._X[row, i]),
                         "z_diff": _fin(diff[o, i])} for i, f in enumerate(self.features)}
            differs = [g for g, v in gd.items() if v is not None and v > self.superficial_threshold]
            unknown = [g for g, v in gd.items() if v is None]
            if differs:
                label = f"partial: differs substantially in {', '.join(differs)}"
            else:
                label = "comparable on " + ", ".join(g for g, v in gd.items() if v is not None)
            if unknown:
                label += f"; NO DATA to compare {', '.join(unknown)}"
            out.append({"code": str(self._codes[row]), "name": str(self.table["name"].iloc[row]),
                        "year": int(self._years[row]), "distance": float(dist[o]), "coverage": float(cov[o]),
                        "group_distance": gd, "features": feats,
                        "comparability": label})
        return {"target": {"code": code, "name": str(self.table["name"].iloc[t]), "year": year,
                           "features": {f: _fin(self._X[t, i]) for i, f in enumerate(self.features)}},
                "standardised_on_years_up_to": cutoff, "eligible_until_year": latest,
                "n_candidates": int(ok.sum()), "analogs": out,
                "method": ("Weighted RMS distance of standardised features over dimensions present for both; "
                           f"≥{self.min_coverage:.0%} feature coverage and ≥{self.min_groups} dimension groups required; "
                           "one match per entity-decade.")}

    def outcomes(self, analogs: Sequence[dict[str, Any]], feature: str, horizon: int) -> list[float | None]:
        i = self.features.index(feature)
        lookup = {(c, y): v for c, y, v in zip(self._codes, self._years, self._X[:, i], strict=True) if math.isfinite(v)}
        res = []
        for a in analogs:
            v0, v1 = lookup.get((a["code"], a["year"])), lookup.get((a["code"], a["year"] + horizon))
            res.append(v1 - v0 if v0 is not None and v1 is not None else None)
        return res


def _fin(x: float) -> float | None:
    return float(x) if x is not None and math.isfinite(x) else None


def backtest(engine: AnalogEngine, horizon: int = 10, origins: Sequence[int] = range(1960, 2011, 10),
             k: int = 10, outcome: str = "log_gdp_pc", max_targets: int | None = None) -> dict[str, Any]:
    """Out-of-sample test of analog-based prediction of the next-h-year change of `outcome`.

    Baselines: no change; own trailing h-year change continued; mean change of all entities
    over the h years before the origin. All use information available at the origin only.
    Inference: loss differences are averaged within each origin year (cross-sectional errors are
    correlated through common shocks) and the origin-year means are t-tested. Origins spaced by
    >= h years make those means approximately independent.
    """
    i = engine.features.index(outcome)
    val = {(c, y): v for c, y, v in zip(engine._codes, engine._years, engine._X[:, i], strict=True) if math.isfinite(v)}
    rows = []
    for o in origins:
        codes = sorted({c for (c, y) in val if y == o and (c, o + horizon) in val})
        if max_targets:
            codes = codes[:max_targets]
        prev = [val[(c, o)] - val[(c, o - horizon)] for c in {c for (c, y) in val if y == o} if (c, o - horizon) in val]
        global_mean = float(np.mean(prev)) if prev else None
        for c in codes:
            try:
                found = engine.find(c, o, k=k, outcome_horizon=horizon, cutoff=o)
            except DataError:
                continue
            outs = [x for x in engine.outcomes(found["analogs"], outcome, horizon) if x is not None]
            if len(outs) < max(3, k // 2):
                continue
            actual = val[(c, o + horizon)] - val[(c, o)]
            own = val[(c, o)] - val[(c, o - horizon)] if (c, o - horizon) in val else None
            rows.append({"code": c, "origin": o, "actual": actual, "analog_median": float(np.median(outs)),
                         "analog_p10": float(np.quantile(outs, 0.1)), "analog_p90": float(np.quantile(outs, 0.9)),
                         "naive_no_change": 0.0, "own_trend": own, "global_mean": global_mean})
    if not rows:
        return {"n": 0, "note": "no evaluable (entity, origin) pairs; fetch more data"}
    df = pd.DataFrame(rows)
    res: dict[str, Any] = {"n_pairs": len(df), "n_origins": int(df["origin"].nunique()), "horizon": horizon,
                           "outcome": outcome, "k": k, "methods": {}}
    for m in ["analog_median", "naive_no_change", "own_trend", "global_mean"]:
        sub = df.dropna(subset=[m])
        res["methods"][m] = {"n": len(sub), "MAE": float(np.mean(np.abs(sub["actual"] - sub[m])))}
    cmp = {}
    for b in ["naive_no_change", "own_trend", "global_mean"]:
        sub = df.dropna(subset=[b]).copy()
        sub["d"] = np.abs(sub["actual"] - sub["analog_median"]) - np.abs(sub["actual"] - sub[b])
        per_origin = sub.groupby("origin")["d"].mean()
        if len(per_origin) >= 3 and per_origin.std(ddof=1) > 0:
            t = stats.ttest_1samp(per_origin, 0.0)
            p = float(t.pvalue)
        else:
            p = None
        cmp[b] = {"mean_loss_difference": float(sub["d"].mean()), "share_pairs_analog_better": float((sub["d"] < 0).mean()),
                  "origin_clustered_p_value": p, "n_origin_clusters": len(per_origin)}
    res["analog_vs"] = cmp
    inside = ((df["actual"] >= df["analog_p10"]) & (df["actual"] <= df["analog_p90"])).mean()
    res["analog_p10_p90_coverage"] = float(inside)
    best = min(res["methods"], key=lambda m: res["methods"][m]["MAE"])
    if best != "analog_median":
        verdict = f"Best method by MAE: {best}. Analogs do NOT beat the best baseline — do not use analog outcomes as forecasts."
    else:
        weak = [b for b, c in cmp.items() if c["origin_clustered_p_value"] is None or c["origin_clustered_p_value"] >= 0.05]
        verdict = ("Analogs have the lowest MAE. " + (
            "The advantage is significant against every baseline (origin-clustered test)." if not weak else
            f"The advantage is NOT statistically significant against: {', '.join(weak)} "
            f"(only {res['n_origins']} origin clusters)."))
    res["verdict"] = verdict
    res["notes"] = ["Outcome is the change of the transformed feature (for log features: log growth over h years).",
                    "Revision leakage present: current data vintages are used for all origins.",
                    f"Analog p10–p90 band covered {inside:.0%} of outcomes (nominal 80%)."]
    res["sample"] = df.head(50).to_dict(orient="records")
    return res


def dims_as_dicts(dims: Sequence[Dimension]) -> list[dict[str, Any]]:
    return [asdict(d) for d in dims]


def resolve_code(table: pd.DataFrame, name_or_code: str) -> str:
    codes = table.index.get_level_values(0)
    if name_or_code in set(codes):
        return name_or_code
    hits = sorted({c for c, n in zip(codes, table["name"], strict=True) if n.lower() == name_or_code.lower()})
    if not hits:
        raise DataError(f"Unknown entity {name_or_code!r} (use ISO3 code or OWID name)")
    return hits[0]
