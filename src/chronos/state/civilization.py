"""Civilization State Model: a structured, sourced description of the world at a reference year.

Every entry reports the latest *observed* year. Values older than the reference year are never
presented as current: status is CURRENT (<= 1 year old), LAGGED (2-4 years: "latest available
X; NOT a <ref> value") or STALE (>= 5 years). An optional NOWCAST to the reference year is
produced only by the best simple baseline of a walk-forward backtest, with its empirical
interval, and is labelled model-based. Missing indicators are reported as NO DATA together
with the command that would load them.
"""

from __future__ import annotations

import math
import sqlite3
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from chronos.analysis.data import panel_frame
from chronos.errors import ChronosError, DataError
from chronos.forecasting.models import Drift, Naive
from chronos.knowledge import store
from chronos.storage import repository as repo
from chronos.timeseries import TimeSeries
from chronos.validation.backtest import BacktestConfig, walk_forward

DOMAINS = ("economy", "energy", "technology_ai", "demography", "political_institutions",
           "international_relations", "ecology", "information_systems")


def load_catalog(path: str | Path) -> list[dict[str, Any]]:
    try:
        return tomllib.loads(Path(path).read_text(encoding="utf-8"))["indicator"]
    except (OSError, KeyError, tomllib.TOMLDecodeError) as exc:
        raise ChronosError(f"Cannot read indicator catalogue {path}: {exc}") from exc


def freshness(last_year: int, reference_year: int) -> tuple[str, str]:
    age = reference_year - last_year
    if age <= 1:
        return "CURRENT", f"latest observation {last_year}"
    if age <= 4:
        return "LAGGED", f"latest available {last_year}; NOT a {reference_year} value"
    return "STALE", f"latest available {last_year} ({age} years old); do not read as current"


def _world_population(con: sqlite3.Connection) -> pd.DataFrame:
    pop = panel_frame(con, "population", "owid")
    if pop.empty:
        raise DataError("OWID population needed for population weighting (chronos fetch owid-github --dataset co2)")
    pop = pop[pop["entity_code"].notna() & ~pop["entity_code"].astype(str).str.startswith("OWID_")]
    return pop[["entity_code", "year", "value"]].rename(columns={"value": "pop"})


def popweighted(con: sqlite3.Connection, indicator: str, source: str, threshold: float | None = None) -> TimeSeries:
    """World population-weighted mean (threshold=None) or population share with indicator >= threshold."""
    df = panel_frame(con, indicator, source)
    if df.empty:
        raise DataError(f"No {indicator} from {source}")
    pop = _world_population(con)
    world = panel_frame(con, "population", "owid")
    world = world[world["entity"] == "World"].set_index("year")["value"]
    m = df.merge(pop, on=["entity_code", "year"], how="inner")
    rows, cov = [], []
    for y, g in m.groupby("year"):
        w = g["pop"].to_numpy(float)
        v = g["value"].to_numpy(float)
        if w.sum() <= 0:
            continue
        val = float(np.sum(w * (v >= threshold)) / w.sum()) if threshold is not None else float(np.average(v, weights=w))
        rows.append((int(y), val))
        cov.append((int(y), float(w.sum() / world.get(y, np.nan)) if y in world.index else math.nan))
    if not rows:
        raise DataError(f"No overlap between {indicator} and population")
    src = con.execute("SELECT name, license FROM source WHERE source_id=?", (source,)).fetchone()
    retrieved = con.execute("""SELECT MAX(v.retrieved_utc) FROM vintage v JOIN series_vintage sv USING(vintage_id)
                               JOIN series s ON s.series_id = sv.series_id WHERE s.indicator=? AND s.source_id=?""",
                            (indicator, source)).fetchone()[0]
    s = TimeSeries.from_pairs(rows, name=f"{indicator}_{'share' if threshold is not None else 'popweighted'}",
                              entity="World (derived)", unit="share of population" if threshold is not None else None,
                              meta={"derivation": "population-weighted" if threshold is None else f"population share with {indicator} >= {threshold}",
                                    "coverage": dict(cov), "source_id": source,
                                    "source_name": f"{src['name'] if src else source} x OWID population (CHRONOS-derived)",
                                    "source_license": src["license"] if src else None, "retrieved_utc": retrieved,
                                    "source_dataset": f"{indicator} weighted by population"})
    return s


def _bounded(s: TimeSeries, kind: str) -> bool:
    """Shares and indices: report absolute change, because growth rates of bounded quantities mislead."""
    unit = (s.unit or "").lower()
    return kind in ("popweight", "popshare") or any(t in unit for t in ("%", "share", "index", "°c"))


def _change(s: TimeSeries, window: int, absolute_only: bool = False) -> dict[str, Any] | None:
    end = s.last_year
    start_candidates = [int(y) for y in s.years if y <= end - window]
    if not start_candidates:
        return None
    y0 = start_candidates[-1]
    v0, v1 = s.get(y0), s.get(end)
    if v0 is None or v1 is None:
        return None
    out: dict[str, Any] = {"from_year": y0, "to_year": end, "absolute_change": v1 - v0}
    if v0 > 0 and v1 > 0 and not absolute_only:
        out["cagr_percent"] = ((v1 / v0) ** (1 / (end - y0)) - 1) * 100
    return out


def nowcast(s: TimeSeries, reference_year: int) -> dict[str, Any] | None:
    """Extrapolate to the reference year with the best baseline at that horizon (backtested)."""
    h = reference_year - s.last_year
    if h <= 0 or s.step() != 1 or len(s.tail_regular_run()) < 40:
        return None
    models = [Naive(), Drift(window=20)] + ([Drift(window=20, log=True)] if s.is_positive else [])
    run = s.tail_regular_run()
    bt = walk_forward(run, models, BacktestConfig(horizons=(h,), min_history=20, min_residuals=10))
    hres = bt["summary"]["horizons"].get(str(h))
    if not hres or not hres["verdict"]["best_model"]:
        return None
    best = hres["verdict"]["best_model"]
    model = next(m for m in models if m.name == best)
    f = model.predict(run, h)
    recs = [r for r in bt["records"] if r["model"] == best]
    if f is None or len(recs) < 10:
        return None
    pos = s.is_positive
    errs = np.array([math.log(r["actual"] / r["forecast"]) if pos else r["actual"] - r["forecast"] for r in recs])
    lo_q, hi_q = np.quantile(errs, [0.1, 0.9])
    lo, hi = (f.value * math.exp(lo_q), f.value * math.exp(hi_q)) if pos else (f.value + lo_q, f.value + hi_q)
    cov = hres["models"][best].get("empirical_interval_calibration", {}).get("0.8", {}).get("empirical_coverage")
    label = f"NOWCAST (model-based extrapolation by {best}; not an observation)"
    if cov is not None and cov < 0.65:
        label += f"; WARNING: its 80% interval covered only {cov:.0%} in the backtest — interval too narrow"
    return {"year": reference_year, "value": f.value, "lo80": lo, "hi80": hi, "model": best, "n_backtest_windows": len(recs),
            "backtest_80pct_coverage": cov, "label": label}


def build_state(con: sqlite3.Connection, catalog: list[dict[str, Any]], reference_year: int,
                with_nowcast: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {"reference_year": reference_year, "domains": {d: [] for d in DOMAINS},
                           "principles": ["Every value carries its latest observed year and source.",
                                          "Values older than the reference year are flagged LAGGED/STALE, never presented as current.",
                                          "NOWCASTs are model-based, backtested extrapolations and are labelled as such.",
                                          "No composite 'civilization index' or probability of collapse is computed."]}
    for item in catalog:
        dom = item["domain"]
        entry: dict[str, Any] = {"key": item["key"], "label": item["label"], "kind": item["kind"]}
        try:
            if item["kind"] == "fact":
                facts = [f for t in item.get("tags", []) for f in store.list_facts(con, tag=t)]
                if not facts:
                    raise DataError("no facts")
                entry["facts"] = [{"statement": f["statement"], "value": f["value"], "unit": f["unit"], "year": f["year"],
                                   "status": freshness(f["year"], reference_year)[0] if f["year"] else "UNDATED",
                                   "reference": f"{f['ref_authors']} ({f['ref_year']}) {f['ref_title']}",
                                   "verification": f["verification"], "uncertainty": f["uncertainty"]} for f in facts]
                entry["status"] = "DOCUMENTED POINT ESTIMATES"
                out["domains"].setdefault(dom, []).append(entry)
                continue
            if item["kind"] == "series":
                s = repo.load_series(con, item["indicator"], item["entity"], item.get("source"))
            else:
                s = popweighted(con, item["indicator"], item["source"], item.get("threshold") if item["kind"] == "popshare" else None)
            status, note = freshness(s.last_year, reference_year)
            entry.update(value=float(s.values[-1]), year=s.last_year, unit=s.unit, status=status, note=note,
                         n_obs=len(s), first_year=s.first_year,
                         source={k: s.meta.get(k) for k in ("source_id", "source_name", "source_dataset", "retrieved_utc",
                                                           "source_version", "source_license", "derivation")})
            if "coverage" in s.meta:
                c = s.meta["coverage"].get(s.last_year)
                entry["population_coverage"] = c
                if c is not None and not math.isnan(c) and c < 0.8:
                    entry["note"] += f"; covers only {c:.0%} of world population"
            ch = _change(s, int(item.get("change_window", 10)), absolute_only=_bounded(s, item["kind"]))
            if ch:
                entry["change"] = ch
            if with_nowcast and status != "CURRENT" and item["kind"] == "series":
                nc = nowcast(s, reference_year)
                if nc:
                    entry["nowcast"] = nc
            entry["history"] = {"years": s.years[-60:].tolist(), "values": s.values[-60:].tolist()}
        except (DataError, ChronosError):
            entry.update(status="NO DATA", note="not loaded in this database",
                         load_hint=item.get("load_hint") or f"chronos fetch ... ({item.get('source')}: {item.get('indicator')})")
        out["domains"].setdefault(dom, []).append(entry)
    counts: dict[str, int] = {}
    for entries in out["domains"].values():
        for e in entries:
            counts[e["status"]] = counts.get(e["status"], 0) + 1
    out["status_counts"] = counts
    return out


def state_text(state: dict[str, Any]) -> str:
    from chronos.report import fmt

    lines = [f"CIVILIZATION STATE — reference year {state['reference_year']}  (status counts: {state['status_counts']})"]
    for dom, entries in state["domains"].items():
        lines.append(f"\n## {dom}")
        for e in entries:
            if e["status"] == "NO DATA":
                lines.append(f"  [NO DATA] {e['label']}  -> {e.get('load_hint')}")
            elif e["kind"] == "fact":
                lines.append(f"  [DOCUMENTED] {e['label']}")
                for f in e["facts"]:
                    lines.append(f"     · {f['statement']} [{f['year']}, {f['status']}, ref: {f['reference'][:70]}]")
            else:
                ch = e.get("change", {})
                chs = f"; {ch['from_year']}–{ch['to_year']}: {fmt(ch.get('cagr_percent'))}%/yr" if ch.get("cagr_percent") is not None \
                    else (f"; change since {ch['from_year']}: {fmt(ch['absolute_change'])}" if ch else "")
                lines.append(f"  [{e['status']}] {e['label']}: {fmt(e['value'])} {e.get('unit') or ''} ({e['note']}){chs}")
                if e.get("nowcast"):
                    n = e["nowcast"]
                    lines.append(f"     NOWCAST {n['year']}: {fmt(n['value'])} [80%: {fmt(n['lo80'])}–{fmt(n['hi80'])}] — {n['label']}")
    return "\n".join(lines)
