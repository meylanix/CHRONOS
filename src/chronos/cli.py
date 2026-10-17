"""CHRONOS command line. Run `chronos --help` or `python -m chronos --help`.

All v0.1 commands keep working: init, demo, fetch --slug, import-csv, analyze, import-events,
import-claims, events.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from chronos import __version__
from chronos.config import Settings, load_settings
from chronos.errors import ChronosError
from chronos.logsetup import configure_logging, get_logger
from chronos.report import backtest_text, fmt, table
from chronos.storage import repository as repo
from chronos.storage.db import connect

log = get_logger("cli")
Handler = Callable[[argparse.Namespace, sqlite3.Connection, Settings], int]


def emit(obj: Any, args: argparse.Namespace, text: str | None = None) -> None:
    payload = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    if getattr(args, "json_out", None):
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(payload, encoding="utf-8")
        print(f"Saved: {args.json_out}", file=sys.stderr)
    if text is not None and not getattr(args, "json", False):
        print(text)
    else:
        print(payload)


def _ingest(con: sqlite3.Connection, batch: repo.IngestBatch) -> int:
    rep = repo.ingest(con, batch)
    print(f"Vintage {rep.vintage_id}: {rep.n_series} series ({rep.n_series_changed} new/changed), "
          f"{rep.n_observations_written} observations written. Source: {batch.source.name} [{batch.dataset}]")
    print(f"License: {batch.source.license}")
    for w in rep.warnings[:20]:
        print(f"  warning: {w}")
    if len(rep.warnings) > 20:
        print(f"  ... {len(rep.warnings) - 20} more warnings")
    return 0


def _split(s: str | None) -> list[str] | None:
    return [x.strip() for x in s.split(",") if x.strip()] if s else None


# ----------------------------------------------------------------------------- data commands


def cmd_init(args: argparse.Namespace, con: sqlite3.Connection, st: Settings) -> int:
    migs = [r[0] for r in con.execute("SELECT name FROM schema_migrations ORDER BY version")]
    print(f"Initialized: {st.db_path} (migrations: {', '.join(migs)})")
    return 0


def cmd_demo(args: argparse.Namespace, con: sqlite3.Connection, st: Settings) -> int:
    from chronos.ingestion.user_csv import demo_batch

    _ingest(con, demo_batch())
    print("Created SYNTHETIC observations (not historical facts, not a real-world forecast).")
    return 0


def cmd_fetch(args: argparse.Namespace, con: sqlite3.Connection, st: Settings) -> int:
    which = args.fetch_source or ("owid-chart" if args.slug else None)
    if which is None:
        raise ChronosError("Choose a source: chronos fetch {owid-github,owid-chart,worldbank,seshat,vdem,gdelt} ...")
    if which == "owid-github":
        from chronos.ingestion import owid_github

        return _ingest(con, owid_github.fetch_dataset(st, args.dataset, _split(args.columns), _split(args.entities),
                                                      args.file, args.codebook_file))
    if which == "owid-chart":
        from chronos.ingestion import owid_grapher

        return _ingest(con, owid_grapher.fetch_chart(st, args.slug, args.column, _split(args.entity), args.indicator,
                                                     getattr(args, "file", None)))
    if which == "worldbank":
        from chronos.ingestion import worldbank

        return _ingest(con, worldbank.fetch_indicator(st, args.indicator, args.country, args.date, args.local_name))
    if which == "seshat":
        from chronos.ingestion import seshat

        return _ingest(con, seshat.fetch_equinox(st, args.file))
    if which == "vdem":
        from chronos.ingestion import vdem

        return _ingest(con, vdem.fetch_vdem(st, _split(args.columns), args.file))
    if which == "gdelt":
        from chronos.ingestion import gdelt

        return _ingest(con, gdelt.fetch_timeline(st, args.query, args.start, args.end, args.local_name))
    raise ChronosError(f"unknown source {which}")


def cmd_import(args: argparse.Namespace, con: sqlite3.Connection, st: Settings) -> int:
    from chronos.ingestion import files

    fn = {"maddison": files.import_maddison, "pwt": files.import_pwt, "cow-nmc": files.import_cow_nmc}[args.kind]
    return _ingest(con, fn(args.file))


def cmd_import_csv(args: argparse.Namespace, con: sqlite3.Connection, st: Settings) -> int:
    from chronos.ingestion.user_csv import import_csv

    return _ingest(con, import_csv(args.file, args.indicator, args.entity, args.source_url, args.synthetic, args.unit))


def cmd_sources(args: argparse.Namespace, con: sqlite3.Connection, st: Settings) -> int:
    from chronos.ingestion.registry import SOURCES

    if args.check:
        from chronos.ingestion.http import fetch

        probes = {
            "owid": "https://raw.githubusercontent.com/owid/co2-data/master/README.md",
            "owid-grapher": "https://ourworldindata.org/grapher/life-expectancy.metadata.json",
            "worldbank": "https://api.worldbank.org/v2/indicator/SP.POP.TOTL?format=json",
            "seshat": "https://raw.githubusercontent.com/seshatdb/Equinox_Data/main/README.md",
            "vdem": "https://raw.githubusercontent.com/vdeminstitute/vdemdata/master/DESCRIPTION",
            "gdelt": "https://api.gdeltproject.org/api/v2/doc/doc?query=history&mode=timelinevol&format=json&timespan=1week",
        }
        rows: list[dict[str, Any]] = []
        for sid, url in probes.items():
            try:
                fr = fetch(url, st, sid, keep_raw=False)
                rows.append({"source": sid, "reachable": "yes", "detail": f"{len(fr.content)} bytes"})
            except ChronosError as exc:
                rows.append({"source": sid, "reachable": "NO", "detail": str(exc)[:90]})
        print(table(rows, ["source", "reachable", "detail"]))
        return 0
    rows = [{"id": e.source_id, "status": e.status, "access": e.access, "license": e.license[:60], "command": e.command}
            for e in SOURCES.values()]
    emit([e.as_dict() for e in SOURCES.values()], args, table(rows, ["id", "status", "access", "license", "command"]))
    return 0


def cmd_series(args: argparse.Namespace, con: sqlite3.Connection, st: Settings) -> int:
    rows = repo.catalog(con)
    if args.indicator:
        rows = [r for r in rows if r["indicator"] == args.indicator]
    if args.source:
        rows = [r for r in rows if r["source_id"] == args.source]
    if args.entity:
        rows = [r for r in rows if r["entity"] == args.entity]
    if args.summary or (not args.indicator and not args.entity):
        agg: dict[tuple[str, str], dict[str, Any]] = {}
        for r in rows:
            a = agg.setdefault((r["indicator"], r["source_id"]), {"indicator": r["indicator"], "source": r["source_id"],
                                                                   "kind": r["data_kind"], "unit": r["unit"], "entities": 0,
                                                                   "first": r["first_year"], "last": r["last_year"]})
            a["entities"] += 1
            a["first"] = min(a["first"], r["first_year"])
            a["last"] = max(a["last"], r["last_year"])
        emit(list(agg.values()), args, table(list(agg.values()), ["indicator", "source", "kind", "entities", "first", "last", "unit"]))
        return 0
    emit(rows, args, table(rows, ["series_id", "indicator", "entity", "source_id", "data_kind", "n_obs", "first_year",
                                  "last_year", "retrieved_utc"]))
    return 0


def cmd_show(args: argparse.Namespace, con: sqlite3.Connection, st: Settings) -> int:
    s = repo.load_series(con, args.indicator, args.entity, args.source)
    rows = [{"year": int(y), "value": float(v), "status": stt}
            for y, v, stt in zip(s.years, s.values, s.status or ["?"] * len(s), strict=True)]
    head = json.dumps(dict(s.meta), indent=2, default=str)
    text = f"{s.name} / {s.entity} [{s.data_kind}] unit={s.unit}\n{head}\n" + table(rows[-args.tail:], ["year", "value", "status"])
    emit({"meta": dict(s.meta), "unit": s.unit, "data_kind": s.data_kind, "observations": rows}, args, text)
    return 0


def cmd_revisions(args: argparse.Namespace, con: sqlite3.Connection, st: Settings) -> int:
    rec = repo.find_series(con, args.indicator, args.entity, args.source)
    revs = repo.series_revisions(con, rec.series_id)
    emit(revs, args, json.dumps(revs, indent=2) if revs else "Only one stored value set: no revisions observed yet.")
    return 0


# ----------------------------------------------------------------------------- analysis commands


def _models(args: argparse.Namespace, positive: bool, pooled: bool = False) -> list:
    from chronos.forecasting.models import AnalogKNN, build_models, default_models

    if getattr(args, "models", None):
        return build_models(_split(args.models) or [])
    models = default_models(positive=positive, pooled=pooled)
    lb, nb = getattr(args, "lookback", None), getattr(args, "neighbors", None)
    if lb or nb:
        models = [AnalogKNN(lookback=lb or 5, neighbors=nb or 7) if m.name == "analog_knn" else m for m in models]
    return models


def cmd_analyze(args: argparse.Namespace, con: sqlite3.Connection, st: Settings) -> int:
    """v0.1-compatible: forecasts from the latest year (+ optional walk-forward backtest at that horizon)."""
    from chronos.forecasting.models import AnalogKNN, V01LogTrend
    from chronos.validation.backtest import BacktestConfig, walk_forward

    s = repo.load_series(con, args.indicator, args.entity, args.source)
    models = _models(args, s.is_positive)
    origin = s.last_year
    forecasts = {}
    for m in models:
        f = m.predict(s, args.horizon)
        forecasts[m.name] = None if f is None else {"value": f.value, "lo": f.lo, "hi": f.hi, "native_level": f.level,
                                                     **({"analogs": f.details["analogs"]} if f.details.get("analogs") else {})}
    v01 = AnalogKNN(args.lookback, args.neighbors).predict(s, args.horizon)
    base = V01LogTrend(args.lookback).predict(s, args.horizon)
    report: dict[str, Any] = {
        "version": __version__, "indicator": args.indicator, "entity": args.entity,
        "data_kind": s.data_kind, "unit": s.unit, "provenance": dict(s.meta), "n_observations": len(s),
        "latest_observed_year": origin, "target_year": origin + args.horizon,
        "forecast": None if v01 is None else {
            "origin": origin, "target_year": origin + args.horizon, "last_value": float(s.values[-1]),
            "analog_scenarios": {"low": v01.lo, "middle": v01.value, "high": v01.hi},
            "linear_log_trend_baseline": base.value if base else None, "analogs": v01.details["analogs"],
            "n_candidates": v01.details["n_candidates"]},
        "all_models": forecasts,
        "method_notice": ("Historical analog percentiles are NOT calibrated outcome probabilities and do NOT prove causation. "
                          "Trust a model only if it beats the baselines in the backtest below."),
    }
    if s.data_kind == "synthetic":
        report["banner"] = "SYNTHETIC DATA — not a real-world forecast."
    if args.backtest:
        bt = walk_forward(s, models, BacktestConfig(horizons=(args.horizon,)))
        bt.pop("records")
        report["backtest"] = bt
    emit(report, args)
    return 0


def cmd_backtest(args: argparse.Namespace, con: sqlite3.Connection, st: Settings) -> int:
    from chronos.validation import experiments
    from chronos.validation.backtest import BacktestConfig, walk_forward

    s = repo.load_series(con, args.indicator, args.entity, args.source)
    if args.exclude_imputed:
        s = repo.load_series(con, args.indicator, args.entity, args.source, statuses=("observed", "estimated", "derived"))
    models = _models(args, s.is_positive)
    cfg = BacktestConfig(horizons=tuple(int(h) for h in args.horizons.split(",")), min_history=args.min_history,
                         origin_step=args.origin_step, first_origin=args.first_origin, last_origin=args.last_origin)
    res = walk_forward(s, models, cfg)
    if not args.keep_records:
        res.pop("records")
    if args.record:
        res["experiment_id"] = experiments.record(con, "backtest", {"indicator": args.indicator, "entity": args.entity,
                                                                    "source": args.source, "cfg": res["config"],
                                                                    "models": [m.config() | {"name": m.name} for m in models]},
                                                  repo.data_fingerprint([s]), res)
    emit(res, args, backtest_text(res) + (f"\nexperiment: {res['experiment_id']}" if args.record else ""))
    return 0


def cmd_experiment(args: argparse.Namespace, con: sqlite3.Connection, st: Settings) -> int:
    from chronos.validation.backtest import point_experiment

    s = repo.load_series(con, args.indicator, args.entity, args.source)
    res = point_experiment(s, _models(args, s.is_positive), args.origin, args.target)
    rows = [{"model": r["model"], "forecast": r.get("forecast"), "abs_%_error": r.get("abs_pct_error")} for r in res["results"]]
    text = (f"Data known by {args.origin} ({res['n_history_obs']} obs) -> forecast {args.target}; actual = {fmt(res['actual'])}\n"
            + table(rows, ["model", "forecast", "abs_%_error"]) + f"\n!! {res['warning']}\nrevision leakage: "
            + res["revision_leakage"]["risk"])
    emit(res, args, text)
    return 0


def cmd_panel_backtest(args: argparse.Namespace, con: sqlite3.Connection, st: Settings) -> int:
    from chronos.validation import experiments
    from chronos.validation.backtest import BacktestConfig
    from chronos.validation.panel import panel_walk_forward

    panel = repo.load_panel(con, args.indicator, args.source, _split(args.entities))
    positive = all(s.is_positive for s in panel.series.values())
    models = _models(args, positive, pooled=args.pooled)
    cfg = BacktestConfig(horizons=tuple(int(h) for h in args.horizons.split(",")), min_history=args.min_history,
                         origin_step=args.origin_step, first_origin=args.first_origin)
    res = panel_walk_forward(panel, models, cfg, min_obs=args.min_obs)
    if args.record:
        res["experiment_id"] = experiments.record(con, "panel_backtest", {"indicator": args.indicator, "source": args.source,
                                                                          "horizons": args.horizons, "models": [m.name for m in models],
                                                                          "origin_step": args.origin_step},
                                                  repo.data_fingerprint(panel.series.values()), res)
    lines = [f"PANEL BACKTEST {args.indicator} ({args.source}): {res['n_entities']} entities"]
    for h, hr in res["horizons"].items():
        lines.append(f"\n── horizon {h}: {hr['n_entities_scored']} entities scored")
        rows = [{"model": n, **{k: v for k, v in m.items() if k != "baseline"}} for n, m in hr["models"].items()]
        rows.sort(key=lambda r: r["pooled_MASE"] if r["pooled_MASE"] is not None else 1e9)
        lines.append(table(rows, ["model", "n_forecasts", "pooled_MASE", "median_abs_log_ratio", "empirical_80pct_interval_coverage"]))
        lines += [f"  • {v}" for v in hr["verdict"]]
    lines += [f"note: {n}" for n in res["notes"]]
    emit(res, args, "\n".join(lines))
    return 0


def cmd_breaks(args: argparse.Namespace, con: sqlite3.Connection, st: Settings) -> int:
    from chronos.analysis import patterns

    s = repo.load_series(con, args.indicator, args.entity, args.source)
    yrs, x = patterns.signal_of(s, args.signal)
    res = patterns.structural_breaks(yrs, x, args.max_breaks, args.min_size, args.model)
    res["signal"] = args.signal
    res["series"] = {"indicator": s.name, "entity": s.entity, "data_kind": s.data_kind}
    rows = [{**g, "mean": g["mean"] * (100 if args.signal == "growth" else 1)} for g in res["segments"]]
    text = (f"Structural breaks in {args.signal} of {s.name}/{s.entity}: {res['breaks']}\n"
            + table(rows, ["start_year", "end_year", "n", "mean", "sd"])
            + (" (mean in %/yr)" if args.signal == "growth" else "") + f"\n{res['note']}")
    emit(res, args, text)
    return 0


def cmd_similar(args: argparse.Namespace, con: sqlite3.Connection, st: Settings) -> int:
    from chronos.analysis import patterns

    panel = repo.load_panel(con, args.indicator, args.source)
    res = patterns.similar_trajectories(panel, args.entity, args.start, args.end, args.k, args.mode,
                                        before_only=args.before_only)
    emit(res, args, table(res["matches"], ["entity", "start", "end", "distance", "distance_percentile"]) + "\n" + res["note"])
    return 0


def cmd_analogs(args: argparse.Namespace, con: sqlite3.Connection, st: Settings) -> int:
    from chronos.analysis import analogs

    tbl, info = analogs.build_state_table(con, min_year=args.min_year)
    eng = analogs.AnalogEngine(tbl)
    if args.backtest:
        res = analogs.backtest(eng, args.horizon, range(args.first_origin, args.last_origin + 1, args.origin_step), k=args.k)
        lines = [f"ANALOG ENGINE BACKTEST: next-{args.horizon}-year change of {res.get('outcome')}; {res.get('n_pairs')} "
                 f"(entity, origin) pairs, {res.get('n_origins')} origins"]
        if res.get("methods"):
            lines.append(table([{"method": k, **v} for k, v in res["methods"].items()], ["method", "n", "MAE"]))
            lines += [f"vs {b}: share better={fmt(c['share_pairs_analog_better'])}, origin-clustered p={fmt(c['origin_clustered_p_value'])}"
                      for b, c in res["analog_vs"].items()]
            lines.append(res["verdict"])
            lines += [f"note: {n}" for n in res["notes"]]
        emit(res, args, "\n".join(lines))
        return 0
    code = analogs.resolve_code(tbl, args.entity)
    res = eng.find(code, args.year, k=args.k, outcome_horizon=args.horizon if args.horizon else None,
                   cutoff=args.year + (args.horizon or 0) if args.with_outcomes else None)
    res["data"] = info
    if args.horizon:
        outs = eng.outcomes(res["analogs"], "log_gdp_pc", args.horizon)
        for a, o in zip(res["analogs"], outs, strict=True):
            a[f"gdp_pc_change_next_{args.horizon}y_pct"] = None if o is None else (2.718281828459045 ** o - 1) * 100
    rows = [{"analog": f"{a['name']} {a['year']}", "distance": a["distance"], "coverage": a["coverage"],
             "comparability": a["comparability"][:60], **({"next_gdp_pc_%": a.get(f"gdp_pc_change_next_{args.horizon}y_pct")} if args.horizon else {})}
            for a in res["analogs"]]
    cols = ["analog", "distance", "coverage", "comparability"] + (["next_gdp_pc_%"] if args.horizon else [])
    emit(res, args, f"Analogs of {res['target']['name']} {args.year} (features: {', '.join(info['features'])})\n"
         + table(rows, cols) + f"\n{res['method']}")
    return 0


def _var(spec: str) -> Any:
    from chronos.analysis.data import VarSpec

    parts = spec.split(":")
    if len(parts) < 2:
        raise ChronosError(f"Variable spec must be indicator:source[:transform], got {spec!r}")
    ind, src = parts[0], parts[1]
    tr = parts[2] if len(parts) > 2 else "level"
    return VarSpec(f"{ind}__{tr}", ind, src, tr)


def cmd_causal(args: argparse.Namespace, con: sqlite3.Connection, st: Settings) -> int:
    from chronos.analysis import causal
    from chronos.analysis.data import merged_panel

    x, y = _var(args.x), _var(args.y)
    ctr = [_var(c) for c in (_split(args.controls) or [])]
    specs = [x, y, *ctr]
    treat_spec = None
    if args.treatment:
        tv, thr = args.treatment.rsplit(">=", 1)
        treat_spec = _var(tv)
        specs.append(treat_spec)
    outcome_es = _var(args.event_outcome) if args.event_outcome else None
    if outcome_es:
        specs.append(outcome_es)
    df = merged_panel(con, specs, min_year=args.min_year, max_year=args.max_year)
    treat = causal.first_crossing_years(df, treat_spec.name, float(thr)) if treat_spec else None
    hyp = None
    if args.register:
        spec = {"x": args.x, "y": args.y, "controls": args.controls, "treatment": args.treatment,
                "expected_sign": args.expected_sign, "years": [args.min_year, args.max_year]}
        causal.register_hypothesis(con, args.register, args.statement or f"{args.x} -> {args.y}", spec)
        hyp = args.register
    rep = causal.assess(df, x.name, y.name, [c.name for c in ctr], args.expected_sign, None, maxlag=args.maxlag)
    if treat:
        rep["T4_event_study"] = causal.event_study(df, outcome_es.name if outcome_es else y.name, treat,
                                                   (args.window_lo, args.window_hi), n_perm=args.n_perm)
        tier, reasons = causal._tier(rep, x.name, args.expected_sign)
        rep["evidence_tier"].update(tier=tier, label=causal.TIER_TEXT[tier][0], does_not_show=causal.TIER_TEXT[tier][1],
                                    reasons=reasons)
        rep["treatment_definition"] = {"rule": f"first sustained crossing {args.treatment}", "n_treated": len(treat)}
    if hyp:
        fe = (rep["T3_fixed_effects"]["with_controls"] or rep["T3_fixed_effects"]["without_controls"]).get("coefficients", {})
        c = fe.get(x.name, {})
        causal.record_test(con, hyp, "fixed_effects", c.get("p"), c.get("coef"), rep["evidence_tier"]["label"], rep)
        if treat and "post_effect_p" in rep.get("T4_event_study", {}):
            es = rep["T4_event_study"]
            causal.record_test(con, hyp, "event_study", es["post_effect_p"], es["average_post_effect"], rep["evidence_tier"]["label"], es)
    rep["T2_predictive"].pop("per_unit", None)
    et = rep["evidence_tier"]
    text = (f"HYPOTHESIS {rep['hypothesis']}  controls={rep['controls']}\n"
            f"EVIDENCE TIER {et['tier']}: {et['label']}\n  does not show: {et['does_not_show']}\n"
            + "\n".join(f"  - {r}" for r in et["reasons"]) + "\nTHREATS:\n" + "\n".join(f"  - {t}" for t in rep["threats_to_validity"]))
    emit(rep, args, text)
    return 0


def cmd_hypotheses(args: argparse.Namespace, con: sqlite3.Connection, st: Settings) -> int:
    from chronos.analysis import causal

    rows = causal.registry(con)
    emit(rows, args, table(rows, ["test_id", "hyp_id", "method", "p_value", "p_bh_adjusted", "estimate", "evidence_tier",
                                  "n_tests_on_hypothesis"]) if rows else "No hypothesis tests recorded.")
    return 0


def cmd_experiments(args: argparse.Namespace, con: sqlite3.Connection, st: Settings) -> int:
    from chronos.validation import experiments

    if args.show:
        e = experiments.get_experiment(con, args.show)
        if e is None:
            raise ChronosError(f"No experiment {args.show}")
        emit(e, args)
        return 0
    rows = experiments.list_experiments(con)
    emit(rows, args, table(rows, ["experiment_id", "kind", "created_utc", "git_commit", "results_sha256"]) if rows
         else "No experiments recorded (use --record).")
    return 0


# ----------------------------------------------------------------------------- knowledge graph


def cmd_import_events(args: argparse.Namespace, con: sqlite3.Connection, st: Settings) -> int:
    from chronos.knowledge import store

    n = store.import_events_csv_v01(con, args.file, args.kind)
    print(f"Imported {n} events ({args.kind}).")
    return 0


def cmd_import_claims(args: argparse.Namespace, con: sqlite3.Connection, st: Settings) -> int:
    from chronos.knowledge import store

    n = store.import_claims_csv_v01(con, args.file, args.kind)
    print(f"Imported {n} labeled links ({args.kind}). A documentary URL is not proof of causality.")
    return 0


def cmd_events(args: argparse.Namespace, con: sqlite3.Connection, st: Settings) -> int:
    from chronos.knowledge import store

    evs = store.list_events(con, tag=args.tag)
    lines = []
    for e in evs:
        refs = "; ".join((r["url"] or r["title"])[:70] for r in e["references"]) or "NO REFERENCE"
        unc = f"±{e['year_uncertainty']}" if e["year_uncertainty"] else ""
        lines.append(f"{e['start_year']:>6}{unc:<4} {e['event_id']} [{e['category']}, {e['data_kind']}] | {e['title']} | {refs}")
    lines.append("\nConnections (claims with curator-assessed evidence levels; NOT model-proven causes):")
    for c in store.list_claims(con, [e["event_id"] for e in evs]):
        lines.append(f"  {c['source_event_id']} -> {c['target_event_id']} | {c['relation']} | level {c['evidence_level']} "
                     f"({c['evidence_label']}) | {c['statement'][:100]} | refs={len(c['references'])} alt={len(c['alternatives'])}")
    emit({"events": evs, "claims": store.list_claims(con)}, args, "\n".join(lines))
    return 0


def cmd_kg(args: argparse.Namespace, con: sqlite3.Connection, st: Settings) -> int:
    from chronos.analysis import patterns
    from chronos.knowledge import store

    if args.kg_cmd == "import":
        counts = store.import_json(con, args.file)
        print("Imported:", ", ".join(f"{k}={v}" for k, v in counts.items()))
    elif args.kg_cmd == "audit":
        issues = store.audit(con)
        emit(issues, args, table(issues, ["claim", "event", "issue"]) if issues else "No issues found.")
    elif args.kg_cmd == "claim":
        emit(store.get_claim(con, args.claim_id), args)
    elif args.kg_cmd == "set-status":
        store.set_claim_status(con, args.claim_id, args.status, args.level, args.reason)
        print(f"Updated {args.claim_id}; change logged.")
    elif args.kg_cmd == "enrichment":
        evs = [(e["category"], e["start_year"]) for e in store.list_events(con) if e["data_kind"] == "real"]
        emit(patterns.event_sequence_enrichment(evs, args.a, args.b, args.lag), args)
    elif args.kg_cmd == "graph":
        emit(store.graph(con), args)
    return 0


# ----------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="chronos", description="CHRONOS: traceable historical data, honest backtesting, "
                                                             "conditional scenarios.")
    p.add_argument("--version", action="version", version=f"chronos {__version__}")
    p.add_argument("--db", help="SQLite database (default: config/env, else chronos.sqlite)")
    p.add_argument("--config", help="TOML config file")
    p.add_argument("--log-format", choices=["text", "json"])
    p.add_argument("--log-level")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name: str, fn: Handler, help_: str, json_out: bool = True) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_, description=help_)
        sp.set_defaults(fn=fn)
        if json_out:
            sp.add_argument("--json", action="store_true", help="print JSON instead of text")
            sp.add_argument("--json-out", help="also write JSON to this file")
        return sp

    add("init", cmd_init, "Create/migrate the database", json_out=False)
    add("demo", cmd_demo, "Load a clearly labelled SYNTHETIC series for offline tests", json_out=False)

    f = add("fetch", cmd_fetch, "Download data from a source (see `chronos sources`)", json_out=False)
    f.add_argument("--slug", help="v0.1 compatibility: OWID chart slug (same as `fetch owid-chart --slug`)")
    f.add_argument("--entity", help="v0.1 compatibility: entity (comma-separated)")
    f.add_argument("--column")
    f.add_argument("--indicator")
    fs = f.add_subparsers(dest="fetch_source")
    g = fs.add_parser("owid-github", help="Official OWID datasets on GitHub (co2, energy)")
    g.add_argument("--dataset", choices=["co2", "energy"], default="co2")
    g.add_argument("--columns", help="comma-separated columns (default: curated set)")
    g.add_argument("--entities", help="comma-separated entities (default: all)")
    g.add_argument("--file", help="use a local copy of the dataset CSV")
    g.add_argument("--codebook-file")
    g = fs.add_parser("owid-chart", help="Any OWID chart via the Chart API")
    g.add_argument("--slug", required=True)
    g.add_argument("--column")
    g.add_argument("--entity", help="comma-separated (default: all)")
    g.add_argument("--indicator", help="local indicator name (default: slug)")
    g.add_argument("--file", help="local chart CSV instead of downloading")
    g = fs.add_parser("worldbank", help="World Bank Indicators API v2")
    g.add_argument("--indicator", required=True, help="WDI code, e.g. SP.POP.TOTL")
    g.add_argument("--country", default="all", help="ISO codes separated by ';' or 'all'")
    g.add_argument("--date", help="e.g. 1960:2025")
    g.add_argument("--local-name", help="local indicator name (default: WDI code)")
    g = fs.add_parser("seshat", help="Seshat Equinox-2020 (official GitHub file)")
    g.add_argument("--file", help="local copy of the Equinox workbook")
    g = fs.add_parser("vdem", help="V-Dem Country-Year (official vdemdata GitHub data or a V-Dem CSV)")
    g.add_argument("--columns", help="comma-separated V-Dem variables")
    g.add_argument("--file", help="local V-Dem CSV/RData (CSV needs far less memory)")
    g = fs.add_parser("gdelt", help="GDELT DOC 2.0 timeline volume (contemporary news)")
    g.add_argument("--query", required=True)
    g.add_argument("--start", default="20170101000000")
    g.add_argument("--end", required=True, help="YYYYMMDDHHMMSS")
    g.add_argument("--local-name", default="gdelt_news_share")

    i = add("import", cmd_import, "Import an official file (Maddison, PWT, CoW NMC)", json_out=False)
    i.add_argument("kind", choices=["maddison", "pwt", "cow-nmc"])
    i.add_argument("--file", required=True)

    i = add("import-csv", cmd_import_csv, "Import Year,Value (or Entity,Year,Value) CSV with declared provenance", json_out=False)
    i.add_argument("--file", required=True)
    i.add_argument("--indicator", required=True)
    i.add_argument("--entity")
    i.add_argument("--source-url", required=True)
    i.add_argument("--unit")
    i.add_argument("--synthetic", action="store_true", help="flag the data as synthetic (never hidden)")

    s = add("sources", cmd_sources, "List sources with connector status and licenses")
    s.add_argument("--check", action="store_true", help="probe whether each source is reachable now")

    s = add("series", cmd_series, "List stored series")
    s.add_argument("--indicator")
    s.add_argument("--source")
    s.add_argument("--entity")
    s.add_argument("--summary", action="store_true")

    s = add("show", cmd_show, "Show a series with provenance")
    s.add_argument("--indicator", required=True)
    s.add_argument("--entity", required=True)
    s.add_argument("--source")
    s.add_argument("--tail", type=int, default=15)

    s = add("revisions", cmd_revisions, "Compare stored vintages of a series")
    s.add_argument("--indicator", required=True)
    s.add_argument("--entity", required=True)
    s.add_argument("--source")

    def series_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--indicator", required=True)
        sp.add_argument("--entity", required=True)
        sp.add_argument("--source")
        sp.add_argument("--models", help="comma-separated model names (default: all baselines + candidates)")

    a = add("analyze", cmd_analyze, "v0.1-compatible: analog scenarios from the latest year, optional backtest")
    series_args(a)
    a.add_argument("--horizon", type=int, default=10)
    a.add_argument("--lookback", type=int, default=5)
    a.add_argument("--neighbors", type=int, default=7)
    a.add_argument("--backtest", action="store_true")

    b = add("backtest", cmd_backtest, "Walk-forward validation against baselines")
    series_args(b)
    b.add_argument("--horizons", default="1,5,10,20")
    b.add_argument("--min-history", type=int, default=25)
    b.add_argument("--origin-step", type=int, default=1)
    b.add_argument("--first-origin", type=int)
    b.add_argument("--last-origin", type=int)
    b.add_argument("--exclude-imputed", action="store_true", help="use only observed/estimated/derived values")
    b.add_argument("--keep-records", action="store_true")
    b.add_argument("--record", action="store_true", help="store as a reproducible experiment")

    e = add("experiment", cmd_experiment, "Single-origin experiment, e.g. data known by 1950 -> forecast 1970")
    series_args(e)
    e.add_argument("--origin", type=int, required=True)
    e.add_argument("--target", type=int, required=True)

    pb = add("panel-backtest", cmd_panel_backtest, "Walk-forward validation across all entities of an indicator")
    pb.add_argument("--indicator", required=True)
    pb.add_argument("--source", required=True)
    pb.add_argument("--entities")
    pb.add_argument("--models")
    pb.add_argument("--horizons", default="5,10")
    pb.add_argument("--min-history", type=int, default=25)
    pb.add_argument("--origin-step", type=int, default=5)
    pb.add_argument("--first-origin", type=int)
    pb.add_argument("--min-obs", type=int, default=40)
    pb.add_argument("--pooled", action="store_true", help="add the cross-entity analog model (slow)")
    pb.add_argument("--record", action="store_true")

    br = add("breaks", cmd_breaks, "Structural breaks (regimes) in a series")
    br.add_argument("--indicator", required=True)
    br.add_argument("--entity", required=True)
    br.add_argument("--source")
    br.add_argument("--signal", choices=["growth", "level", "diff"], default="growth")
    br.add_argument("--max-breaks", type=int, default=5)
    br.add_argument("--min-size", type=int, default=8)
    br.add_argument("--model", choices=["mean", "meanvar"], default="mean",
                    help="meanvar: regimes may differ in volatility as well as mean")

    si = add("similar", cmd_similar, "Find entity-periods whose trajectory resembles a given window")
    si.add_argument("--indicator", required=True)
    si.add_argument("--source", required=True)
    si.add_argument("--entity", required=True)
    si.add_argument("--start", type=int, required=True)
    si.add_argument("--end", type=int, required=True)
    si.add_argument("--k", type=int, default=10)
    si.add_argument("--mode", choices=["growth", "shape"], default="growth")
    si.add_argument("--before-only", action="store_true")

    an = add("analogs", cmd_analogs, "Multidimensional historical analogs (economy, demography, energy, institutions)")
    an.add_argument("--entity", help="ISO3 code or OWID country name")
    an.add_argument("--year", type=int)
    an.add_argument("--k", type=int, default=10)
    an.add_argument("--horizon", type=int, default=0, help="also report what followed each analog over h years")
    an.add_argument("--with-outcomes", action="store_true", help="(with --horizon) allow outcomes up to year+h")
    an.add_argument("--min-year", type=int, default=1820)
    an.add_argument("--backtest", action="store_true", help="test analog-based prediction against baselines")
    an.add_argument("--first-origin", type=int, default=1960)
    an.add_argument("--last-origin", type=int, default=2010)
    an.add_argument("--origin-step", type=int, default=10)

    c = add("causal", cmd_causal, "Evidence ladder for a hypothesis X -> Y (association, Granger, FE, event study)")
    c.add_argument("--x", required=True, help="indicator:source[:transform], e.g. v2x_polyarchy:vdem:level")
    c.add_argument("--y", required=True, help="e.g. gdp_per_capita:owid:growth")
    c.add_argument("--controls", help="comma-separated variable specs")
    c.add_argument("--expected-sign", type=int, choices=[-1, 1])
    c.add_argument("--treatment", help="event-study treatment: spec>=threshold, e.g. v2x_regime:vdem:level>=2")
    c.add_argument("--event-outcome", help="outcome spec for the event study (default: --y)")
    c.add_argument("--window-lo", type=int, default=-5)
    c.add_argument("--window-hi", type=int, default=10)
    c.add_argument("--n-perm", type=int, default=50)
    c.add_argument("--maxlag", type=int, default=3)
    c.add_argument("--min-year", type=int)
    c.add_argument("--max-year", type=int)
    c.add_argument("--register", help="hypothesis id: pre-register the spec and log the tests")
    c.add_argument("--statement")

    add("hypotheses", cmd_hypotheses, "Hypothesis registry with Benjamini–Hochberg adjusted p-values")
    ex = add("experiments", cmd_experiments, "List or show recorded experiments")
    ex.add_argument("--show")

    ie = add("import-events", cmd_import_events, "Import events CSV (v0.1 format)", json_out=False)
    ie.add_argument("--file", required=True)
    ie.add_argument("--kind", choices=["real", "example", "synthetic"], default="real")
    ic = add("import-claims", cmd_import_claims, "Import claims CSV (v0.1 format)", json_out=False)
    ic.add_argument("--file", required=True)
    ic.add_argument("--kind", choices=["real", "example", "synthetic"], default="real")
    ev = add("events", cmd_events, "List events and claims")
    ev.add_argument("--tag")

    kg = add("kg", cmd_kg, "Knowledge graph: import bundle, audit, claims, status changes, enrichment test")
    ks = kg.add_subparsers(dest="kg_cmd", required=True)
    x = ks.add_parser("import")
    x.add_argument("file")
    ks.add_parser("audit")
    ks.add_parser("graph")
    x = ks.add_parser("claim")
    x.add_argument("claim_id")
    x = ks.add_parser("set-status")
    x.add_argument("claim_id")
    x.add_argument("--status", choices=["proposed", "under_test", "supported", "contested", "refuted"])
    x.add_argument("--level", type=int, choices=range(0, 6))
    x.add_argument("--reason", required=True)
    x = ks.add_parser("enrichment")
    x.add_argument("--a", required=True, help="category A")
    x.add_argument("--b", required=True, help="category B")
    x.add_argument("--lag", type=int, default=20)

    _add_later_phases(sub, add)
    return p


def _add_later_phases(sub: Any, add: Callable[..., argparse.ArgumentParser]) -> None:
    """Commands for state model, scenarios, research dossiers and the UI (registered if available)."""
    try:
        from chronos import cli_more
    except ImportError:
        return
    cli_more.register(add)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        st = load_settings(args.config, db_path=args.db, log_format=args.log_format, log_level=args.log_level)
    except ChronosError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    configure_logging(st.log_level, st.log_format)
    try:
        con = connect(st.db_path)
    except (sqlite3.Error, OSError) as exc:
        print(f"ERROR: cannot open database {st.db_path}: {exc}", file=sys.stderr)
        return 1
    try:
        return int(args.fn(args, con, st) or 0)
    except ChronosError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except (ValueError, sqlite3.Error, OSError) as exc:
        log.debug("error", exc_info=True)
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())
