"""CLI commands for the later phases: state model, scenarios, research dossiers, UI."""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from chronos.config import Settings


def cmd_state(args: argparse.Namespace, con: sqlite3.Connection, st: Settings) -> int:
    from chronos.cli import emit
    from chronos.state.civilization import build_state, load_catalog, state_text

    state = build_state(con, load_catalog(args.catalog or st.indicator_catalog), args.reference_year or st.reference_year,
                        with_nowcast=not args.no_nowcast)
    emit(state, args, state_text(state))
    return 0


def cmd_scenarios(args: argparse.Namespace, con: sqlite3.Connection, st: Settings) -> int:
    from chronos.cli import emit
    from chronos.scenarios.report import build_report, report_text
    from chronos.validation import experiments

    path = args.file or str(Path(st.scenario_dir) / "world_2026.toml")
    rep = build_report(con, path, with_evidence=not args.no_validation)
    if args.record:
        fp = {k: v for k, v in rep["base"]["provenance"].items()}
        rep["experiment_id"] = experiments.record(con, "scenarios", {"spec": path, "spec_text": Path(path).read_text()}, fp,
                                                  {k: v for k, v in rep.items() if k != "scenarios"} |
                                                  {"scenarios": [{"id": s["id"], "projections": s.get("projections")} for s in rep["scenarios"]]})
    emit(rep, args, report_text(rep) + (f"\nexperiment: {rep['experiment_id']}" if args.record else ""))
    return 0


def cmd_research(args: argparse.Namespace, con: sqlite3.Connection, st: Settings) -> int:
    from chronos.cli import emit
    from chronos.errors import ChronosError
    from chronos.research.transitions import build_dossier, dossier_markdown
    from chronos.scenarios.report import build_report

    scen = None
    if not args.no_scenarios:
        try:
            scen = build_report(con, args.scenarios or str(Path(st.scenario_dir) / "world_2026.toml"),
                                with_evidence=not args.no_validation)
        except ChronosError as exc:
            print(f"(scenarios not run: {exc})", file=sys.stderr)
    d = build_dossier(con, args.year, scen)
    md = dossier_markdown(d)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"Saved: {args.out}", file=sys.stderr)
    emit(d, args, md)
    return 0


def cmd_ui(args: argparse.Namespace, con: sqlite3.Connection, st: Settings) -> int:
    app = Path(__file__).resolve().parent / "ui" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port),
           "--server.headless", "true", "--", "--db", st.db_path]
    print("Starting:", " ".join(cmd))
    return subprocess.call(cmd)


def register(add: Callable[..., Any]) -> None:
    s = add("state", cmd_state, "Civilization State Model at the reference year (with freshness flags)")
    s.add_argument("--catalog", help="indicator catalogue TOML (default config/indicators.toml)")
    s.add_argument("--reference-year", type=int)
    s.add_argument("--no-nowcast", action="store_true")

    sc = add("scenarios", cmd_scenarios, "Run conditional world scenarios (Kaya identity + TCRE)")
    sc.add_argument("--file", help="scenario TOML (default scenarios/world_2026.toml)")
    sc.add_argument("--no-validation", action="store_true", help="skip the panel backtests of extrapolation error")
    sc.add_argument("--record", action="store_true")

    r = add("research", cmd_research, "Research dossier: printing vs internet vs generative AI, outlook to a year")
    r.add_argument("topic", choices=["transitions"])
    r.add_argument("--year", type=int, default=2040)
    r.add_argument("--out", help="write the Markdown dossier to this file")
    r.add_argument("--scenarios", help="scenario TOML")
    r.add_argument("--no-scenarios", action="store_true")
    r.add_argument("--no-validation", action="store_true")

    u = add("ui", cmd_ui, "Launch the Streamlit research interface", json_out=False)
    u.add_argument("--port", type=int, default=8501)
