"""Assemble a full scenario report: assumptions, mechanisms, projections, dependencies, signposts."""

from __future__ import annotations

import copy
import sqlite3
from typing import Any

from chronos.errors import ChronosError, DataError
from chronos.forecasting.models import Drift, Naive
from chronos.knowledge import store
from chronos.scenarios import engine as E
from chronos.storage import repository as repo
from chronos.timeseries import Panel
from chronos.validation.backtest import BacktestConfig
from chronos.validation.panel import panel_walk_forward

DISCLAIMER = ("Conditional scenarios: each path is what the Kaya identity implies IF the stated assumptions hold. "
              "The spread shown inside a scenario comes from the stated parameter ranges; it is not a probability "
              "distribution. No probabilities are assigned to scenarios, and the set is not exhaustive.")


def extrapolation_error(con: sqlite3.Connection, indicator: str, source: str, horizons: list[int],
                        origin_step: int = 5, min_obs: int = 40) -> dict[str, Any]:
    """How wrong were simple extrapolations of this indicator historically, across countries, at each horizon?"""
    try:
        panel = repo.load_panel(con, indicator, source)
    except DataError:
        return {}
    countries = {k: v.tail_regular_run() for k, v in panel.series.items()
                 if v.meta.get("entity_code") and not str(v.meta["entity_code"]).startswith("OWID_")}
    panel = Panel(indicator, countries)
    models = [Naive(), Drift(window=20), Drift(window=20, log=True)]
    res = panel_walk_forward(panel, models, BacktestConfig(horizons=tuple(horizons), origin_step=origin_step,
                                                           min_history=25, min_residuals=10), min_obs=min_obs)
    out = {}
    for h, hr in res["horizons"].items():
        best = hr.get("best_baseline")
        if not best:
            continue
        m = hr["models"][best]
        out[h] = {"indicator": indicator, "best_baseline": best, "n_entities": hr["n_entities_scored"],
                  "n_forecasts": m["n_forecasts"], "median_abs_log_error": m["median_abs_log_ratio"],
                  "empirical_80pct_coverage": m["empirical_80pct_interval_coverage"],
                  "note": "Country-level extrapolations; world aggregates are smoother, so this overstates world-level error."}
    return out


def mechanism_evidence(con: sqlite3.Connection, mech_ids: list[str]) -> list[dict[str, Any]]:
    mechs = {m["mechanism_id"]: m for m in store.list_mechanisms(con)}
    claims = store.list_claims(con)
    out = []
    for mid in mech_ids:
        m = mechs.get(mid)
        cs = [c for c in claims if c["mechanism_id"] == mid]
        out.append({"mechanism_id": mid, "name": m["name"] if m else "(not in knowledge graph)",
                    "description": m["description"] if m else None,
                    "historical_claims": [{"claim_id": c["claim_id"], "statement": c["statement"],
                                           "evidence_level": c["evidence_level"], "evidence_label": c["evidence_label"],
                                           "status": c["status"]} for c in cs],
                    "max_evidence_level": max((c["evidence_level"] for c in cs), default=None)})
    return out


def build_report(con: sqlite3.Connection, spec_path: str, with_evidence: bool = True,
                 sensitivity_years: tuple[int, ...] = (2050, 2076)) -> dict[str, Any]:
    model, scenarios = E.load_spec(spec_path)
    base = E.load_base(con, model)
    ref = int(model["reference_year"])
    horizons = [int(h) for h in model["horizons"]]
    evidence: dict[str, Any] = {}
    if with_evidence:
        evidence = {"gdp_per_capita": extrapolation_error(con, "gdp_per_capita", "owid", horizons),
                    "population": extrapolation_error(con, "population", "owid", horizons)}
    report: dict[str, Any] = {
        "model": {"id": model["id"], "reference_year": ref, "end_year": model["end_year"], "horizons": horizons,
                  "samples": model.get("samples"), "tcre": model.get("tcre")},
        "disclaimer": DISCLAIMER,
        "base": {"year": base.year, "values": base.values, "notes": base.notes,
                 "provenance": {k: {f: s.meta.get(f) for f in ("source_id", "source_dataset", "source_variable",
                                                              "retrieved_utc", "source_version")} | {"last_year": s.last_year}
                                for k, s in base.series.items()}},
        "validation": {"gdp_per_capita": E.validation_status(evidence.get("gdp_per_capita"), horizons),
                       "population": E.validation_status(evidence.get("population"), horizons),
                       "energy_co2_warming": {h: {"status": "UNVALIDATED", "note": "no backtest implemented for these outputs"}
                                              for h in horizons}},
        "historical_shocks": E.historical_shocks(base),
        "scenarios": [],
    }
    for scen in scenarios:
        scen = copy.deepcopy(scen)
        try:
            E.apply_derivations(scen, base)
        except (DataError, ChronosError) as exc:
            report["scenarios"].append({"id": scen.id, "error": str(exc)})
            continue
        run = E.run_scenario(base, scen, model)
        entry = {
            "id": scen.id, "name": scen.name, "summary": scen.summary,
            "assumptions": scen.assumptions,
            "parameters": {k: {"basis": p.basis, "derived_from_data": p.derived_info,
                               "periods": [vars(x) for x in p.periods], "unit": "%/yr"} for k, p in scen.params.items()},
            "mechanisms": {"narrative": scen.mechanism_narrative, "linked": mechanism_evidence(con, scen.mechanisms)},
            "projections": E.horizon_table(run, ref, horizons),
            "path": {"years": run.years.tolist(), **{k: run.central[k].tolist() for k in E.OUTPUTS},
                     "p05": {k: _q(run, k, 0.05) for k in E.OUTPUTS}, "p95": {k: _q(run, k, 0.95) for k in E.OUTPUTS}},
            "critical_dependencies": {"stated": scen.critical_dependencies,
                                      "computed": {str(y): E.sensitivity(base, scen, model, y) for y in sensitivity_years}},
            "wildcards": scen.wildcards,
            "signposts": E.evaluate_signposts(con, scen.signposts),
            "hindcast_check": E.hindcast(base, scen, model),
        }
        report["scenarios"].append(entry)
    return report


def _q(run: E.ScenarioRun, key: str, q: float) -> list[float]:
    import numpy as np

    return np.quantile(run.samples[key], q, axis=0).tolist()


def report_text(rep: dict[str, Any], outputs: tuple[str, ...] = ("population", "gdp_per_capita", "co2", "warming")) -> str:
    from chronos.report import fmt, table

    lines = [f"SCENARIOS {rep['model']['id']}  reference year {rep['model']['reference_year']}",
             f"!! {rep['disclaimer']}", *[f"base: {n}" for n in rep["base"]["notes"]]]
    lines.append("\nValidation by horizon (GDP per person, from country-level extrapolation backtests):")
    for h, v in rep["validation"]["gdp_per_capita"].items():
        if v["status"] == "UNVALIDATED":
            lines.append(f"  {h:>3}y: UNVALIDATED — {v['note']}")
        else:
            lines.append(f"  {h:>3}y: best baseline {v['best_baseline']}, median |log error| {fmt(v['median_abs_log_error'])}, "
                         f"80% interval coverage {fmt(v['empirical_80pct_coverage'])} ({v['n_forecasts']} forecasts, {v['n_entities']} countries)")
    for s in rep["scenarios"]:
        if "error" in s:
            lines.append(f"\n### {s['id']}: ERROR {s['error']}")
            continue
        lines.append(f"\n### {s['name']} [{s['id']}]\n{s['summary']}")
        for a in s["assumptions"]:
            lines.append(f"  assumption: {a}")
        for k, p in s["parameters"].items():
            d = p["derived_from_data"]
            per = "; ".join(f"≤{x['until']}: {x['central']:.2f} [{x['low']:.2f}, {x['high']:.2f}]" for x in p["periods"])
            lines.append(f"  {k} (%/yr): {per}" + (f"  (derived: {d['variable']} CAGR {d['from_year']}–{d['to_year']} = {d['value']:.2f})" if d else ""))
        rows = []
        for r in s["projections"]:
            row: dict[str, Any] = {"year": r["year"]}
            for o in outputs:
                row[o] = f"{fmt(r[o]['central'])} [{fmt(r[o]['p05'])}–{fmt(r[o]['p95'])}]"
            rows.append(row)
        lines.append(table(rows, ["year", *outputs]))
        lines.append("  mechanisms: " + "; ".join(f"{m['name']} (max evidence level {m['max_evidence_level']})" for m in s["mechanisms"]["linked"]))
        for y, sens in s["critical_dependencies"]["computed"].items():
            top = sens["gdp_per_capita"][:2]
            lines.append(f"  most influential for GDP/person in {y}: " + ", ".join(f"{t['parameter']} (swing {fmt(t['swing'])})" for t in top))
        for sp in s["signposts"]:
            lines.append(f"  signpost [{sp['status']}] {sp['label']} ({sp['rule']}, {sp['window'][0]}–{sp['window'][1]})"
                         + (f" observed {fmt(sp.get('observed'))}" if sp.get("observed") is not None else ""))
        dev = [h for h in s["hindcast_check"] if abs(h["deviation_percent"]) > 2]
        if dev:
            worst = max(dev, key=lambda h: abs(h["deviation_percent"]))
            lines.append(f"  ASSUMPTION CHECK: the pure assumption path already misses observed data in {len(dev)} of "
                         f"{len(s['hindcast_check'])} checks (worst: {worst['variable']} {worst['year']} {worst['deviation_percent']:+.1f}%). "
                         "Paths are anchored to observations up to the latest observed year.")
        else:
            lines.append(f"  assumption check: consistent with {len(s['hindcast_check'])} observations after the base year (within 2%)")
        for y, sens in s["critical_dependencies"]["computed"].items():
            top = sens["warming"][:3]
            lines.append(f"  most influential for warming in {y}: " + ", ".join(f"{t['parameter']} (swing {fmt(t['swing'])} degC)" for t in top))
        lines.append("  wildcards (no probabilities): " + "; ".join(s["wildcards"]))
    return "\n".join(lines)
