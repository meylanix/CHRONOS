"""Walk-forward validation across many entities (e.g. GDP per capita for every country).

One series gives few independent forecast windows; a panel gives many. Errors are pooled on a
scale-free basis (MASE, |log ratio|) and models are compared entity by entity (win rates with a
sign test). Entities share global shocks, so the sign test p-value is optimistic.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import stats

from chronos.forecasting.models import Model
from chronos.timeseries import Panel
from chronos.validation.backtest import BacktestConfig, Record, run_forecasts


def panel_walk_forward(panel: Panel, models: Sequence[Model], cfg: BacktestConfig,
                       entities: Sequence[str] | None = None, min_obs: int = 40) -> dict[str, Any]:
    ents = [e for e in (entities or panel.entities()) if len(panel.series[e]) >= min_obs]
    per_entity: dict[str, list[Record]] = {}
    for e in ents:
        recs = run_forecasts(panel.series[e], models, cfg, panel if any(m.needs_panel for m in models) else None)
        if recs:
            per_entity[e] = recs
    names = [m.name for m in models]
    baselines = [m.name for m in models if m.is_baseline]
    out: dict[str, Any] = {"indicator": panel.indicator, "n_entities": len(per_entity), "horizons": {}}
    for h in cfg.horizons:
        pooled: dict[str, dict[str, list[float]]] = {n: defaultdict(list) for n in names}
        ent_mae: dict[str, dict[str, float]] = defaultdict(dict)
        for e, recs in per_entity.items():
            hr = [r for r in recs if r.horizon == h]
            if not hr:
                continue
            common = set.intersection(*[{r.origin for r in hr if r.model == n} for n in names])
            if len(common) < 3:
                continue
            for n in names:
                rs = [r for r in hr if r.model == n and r.origin in common]
                for r in rs:
                    if r.scale:
                        pooled[n]["scaled"].append(abs(r.actual - r.forecast) / r.scale)
                    if r.actual > 0 and r.forecast > 0:
                        pooled[n]["logr"].append(abs(np.log(r.forecast / r.actual)))
                    lv = 0.8
                    if lv in r.intervals:
                        lo, hi = r.intervals[lv]
                        pooled[n]["cov80"].append(float(lo <= r.actual <= hi))
                ent_mae[e][n] = float(np.mean([abs(r.actual - r.forecast) for r in rs]))
        hres: dict[str, Any] = {"n_entities_scored": len(ent_mae), "models": {}}
        for n in names:
            p = pooled[n]
            hres["models"][n] = {
                "n_forecasts": len(p["scaled"]) or len(p["logr"]),
                "pooled_MASE": float(np.mean(p["scaled"])) if p["scaled"] else None,
                "median_abs_log_ratio": float(np.median(p["logr"])) if p["logr"] else None,
                "empirical_80pct_interval_coverage": float(np.mean(p["cov80"])) if p["cov80"] else None,
                "baseline": n in baselines,
            }
        ranked = sorted((n for n in names if hres["models"][n]["pooled_MASE"] is not None),
                        key=lambda n: hres["models"][n]["pooled_MASE"])
        best_base = next((n for n in ranked if n in baselines), None)
        wins: dict[str, dict[str, Any]] = {}
        for n in names:
            if n in baselines or best_base is None:
                continue
            pairs = [(v[n], v[best_base]) for v in ent_mae.values() if n in v and best_base in v]
            k = sum(a < b for a, b in pairs)
            wins[n] = {"vs": best_base, "entities_better": k, "entities_compared": len(pairs),
                       "sign_test_p": float(stats.binomtest(k, len(pairs), 0.5).pvalue) if pairs else None}
        hres["ranking_by_pooled_MASE"] = ranked
        hres["best_baseline"] = best_base
        hres["candidate_vs_best_baseline"] = wins
        verdict = []
        for n, w in wins.items():
            better, tot, pv = int(w["entities_better"]), int(w["entities_compared"]), w["sign_test_p"]
            if pv is None or not tot:
                verdict.append(f"{n}: no comparison")
                continue
            direction = "BETTER" if better > tot / 2 else "WORSE" if better < tot / 2 else "TIED"
            sig = "significant" if pv < 0.05 else "not significant"
            verdict.append(f"{n} vs {w['vs']}: lower MAE in {better}/{tot} entities -> mostly {direction} than the baseline "
                           f"(two-sided sign test p={pv:.3g}, {sig})")
        hres["verdict"] = verdict
        out["horizons"][str(h)] = hres
    out["notes"] = ["Revision leakage present for all origins (current vintages).",
                    "Entities are not independent (global shocks); the sign-test p-values are optimistic.",
                    "An entity is scored only where every model produced forecasts at >= 3 common origins, so demanding "
                    "models (long-history analogs) restrict the sample to long annual series; rerun with fewer models "
                    "for wider coverage."]
    return out
