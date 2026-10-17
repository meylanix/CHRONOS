"""Execute the pre-registered protocol experiments/preregistration/holdout_v1.json.

Usage: python experiments/run_holdout_v1.py <database> <output.json>

Deviation (documented, conservative): the registered decision test (origin-block bootstrap with
block = horizon) was shown by simulation, after registration and before this run, to over-reject
at 10-20 year horizons. Both the registered test and a size-correct non-overlapping-origin t-test
are reported; a candidate is declared better only if BOTH reject after Holm correction.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from chronos.forecasting.models import AnalogKNN, DampedTrend, Drift, HistoricalMean, Model, Naive
from chronos.storage import repository as repo
from chronos.storage.db import connect
from chronos.timeseries import Panel
from chronos.validation.backtest import BacktestConfig, Record, run_forecasts
from chronos.validation.inference import (
    holm,
    nonoverlap_origin_t_test,
    origin_block_bootstrap,
    paired_scaled_differences,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "experiments" / "preregistration" / "holdout_v1.json").read_text())
COMPARATOR = "drift_20"


def models_for(annual: bool) -> list[Model]:
    base: list[Model] = [Naive(), Drift(window=20), Drift(window=20, log=True), HistoricalMean(30)]
    cands: list[Model] = [DampedTrend(60)]
    if annual:
        cands += [AnalogKNN(), AnalogKNN(min_separation=10), AnalogKNN(pool=True, min_separation=10)]
    return base + cands


def eligible(panel: Panel, annual: bool) -> Panel:
    need = 40 if annual else 15
    out = {}
    for e, s in panel.series.items():
        code = s.meta.get("entity_code")
        if annual and (not code or str(code).startswith("OWID_")):
            continue
        if len(s.tail_regular_run()) >= need:
            out[e] = s
    return Panel(panel.indicator, out)


def run_indicator(con: Any, spec: dict[str, str]) -> tuple[list[Record], dict[str, Any]]:
    panel = repo.load_panel(con, spec["indicator"], spec["source"])
    annual = next(iter(panel.series.values())).step() == 1
    panel = eligible(panel, annual)
    models = models_for(annual)
    horizons = tuple(PROTOCOL["horizons"]["annual" if annual else "century"])
    cfg = BacktestConfig(horizons=horizons, origin_step=5 if annual else 100, min_history=25 if annual else 10,
                         min_residuals=10)
    pooled = any(m.needs_panel for m in models)
    records: list[Record] = []
    for name in panel.entities():
        records += run_forecasts(panel.series[name], models, cfg, panel if pooled else None)
    meta = {"annual": annual, "n_entities_eligible": len(panel.series), "horizons": list(horizons),
            "origin_step": cfg.origin_step, "models": [m.name for m in models],
            "candidates": [m.name for m in models if not m.is_baseline]}
    return records, meta


def coverage(records: list[Record], model: str, h: int) -> float | None:
    hits = [r.intervals[0.8][0] <= r.actual <= r.intervals[0.8][1] for r in records
            if r.model == model and r.horizon == h and 0.8 in r.intervals]
    return float(np.mean(hits)) if len(hits) >= 30 else None


def main(db: str, out: str) -> None:
    con = connect(db)
    t0 = time.time()
    cells: list[dict[str, Any]] = []
    per_indicator: dict[str, Any] = {}
    for spec in PROTOCOL["holdout_indicators"]:
        recs, meta = run_indicator(con, spec)
        per_indicator[spec["indicator"]] = meta
        for h in meta["horizons"]:
            hrecs = [r for r in recs if r.horizon == h]
            cov = {m: coverage(hrecs, m, h) for m in meta["models"]}
            for cand in meta["candidates"]:
                diffs = paired_scaled_differences(hrecs, cand, COMPARATOR, h)
                n_pairs = sum(len(v) for v in diffs.values())
                ents = {r.entity for r in hrecs if r.model == cand}
                cell: dict[str, Any] = {"indicator": spec["indicator"], "horizon": h, "candidate": cand,
                                        "comparator": COMPARATOR, "n_pairs": n_pairs, "n_entities": len(ents),
                                        "coverage80_candidate": cov.get(cand), "coverage80_comparator": cov.get(COMPARATOR)}
                if len(ents) < 10 or n_pairs < 30:
                    cell["status"] = "insufficient data"
                    cells.append(cell)
                    continue
                block = max(1, math.ceil(h / meta["origin_step"]))
                cell["registered_bootstrap"] = origin_block_bootstrap(diffs, block, n_boot=2000, seed=20261017)
                cell["nonoverlap_t"] = {k: v for k, v in nonoverlap_origin_t_test(diffs, h, meta["origin_step"]).items()
                                        if k != "origins_used"}
                # descriptive: share of entities where candidate MAE < comparator MAE
                wins = tot = 0
                for e in ents:
                    a = [abs(r.actual - r.forecast) for r in hrecs if r.entity == e and r.model == cand]
                    b = [abs(r.actual - r.forecast) for r in hrecs if r.entity == e and r.model == COMPARATOR]
                    if a and b:
                        tot += 1
                        wins += float(np.mean(a)) < float(np.mean(b))
                cell["entity_win_share"] = wins / tot if tot else None
                cells.append(cell)
    tested = [c for c in cells if "registered_bootstrap" in c]
    for key in ("registered_bootstrap", "nonoverlap_t"):
        adj = holm([c[key].get("p_value") for c in tested])
        for c, a in zip(tested, adj, strict=True):
            c[key]["p_holm"] = a
    for c in tested:
        rb, nt = c["registered_bootstrap"], c["nonoverlap_t"]
        better = (rb["significant_improvement"] and (rb.get("p_holm") or 1) < 0.05
                  and nt.get("significant_improvement") and (nt.get("p_holm") or 1) < 0.05)
        worse = (rb["significant_deterioration"] and (rb.get("p_holm") or 1) < 0.05
                 and nt.get("significant_deterioration") and (nt.get("p_holm") or 1) < 0.05)
        c["status"] = "candidate BETTER" if better else "candidate WORSE" if worse else "no evidence of a difference"
    result = {"protocol": PROTOCOL["protocol_id"], "database": db, "runtime_seconds": round(time.time() - t0, 1),
              "per_indicator": per_indicator, "cells": cells,
              "summary": {s: sum(c["status"] == s for c in cells) for s in
                          ("candidate BETTER", "candidate WORSE", "no evidence of a difference", "insufficient data")}}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(result["summary"]), f"{result['runtime_seconds']}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
