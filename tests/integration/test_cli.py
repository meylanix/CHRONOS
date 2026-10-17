"""End-to-end CLI runs on a temporary database (offline)."""

import json

import pytest

from chronos.cli import main

ROOT_EXAMPLES = "data/examples"


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("CHRONOS_CACHE_DIR", str(tmp_path / "cache"))
    return str(tmp_path / "cli.sqlite")


def run(db, *args):
    return main(["--db", db, *args])


def test_v01_command_sequence(db, tmp_path, capsys):
    assert run(db, "init") == 0
    assert run(db, "demo") == 0
    out = tmp_path / "report.json"
    assert run(db, "analyze", "--indicator", "DEMO_SYNTHETIC_INDEX", "--entity", "Synthetic World", "--horizon", "10",
               "--backtest", "--json-out", str(out)) == 0
    rep = json.loads(out.read_text())
    assert rep["data_kind"] == "synthetic" and rep["banner"].startswith("SYNTHETIC")
    assert rep["forecast"]["analog_scenarios"]["low"] < rep["forecast"]["analog_scenarios"]["high"]
    assert run(db, "import-events", "--file", f"{ROOT_EXAMPLES}/events_example.csv", "--kind", "example") == 0
    assert run(db, "import-claims", "--file", f"{ROOT_EXAMPLES}/claims_example.csv", "--kind", "example") == 0
    capsys.readouterr()
    assert run(db, "events") == 0
    assert "NOT model-proven causes" in capsys.readouterr().out


def test_import_csv_backtest_record_and_experiments(db, tmp_path, capsys):
    csv = tmp_path / "pop.csv"
    csv.write_text("Year,Value\n" + "\n".join(f"{y},{100 + 2 * (y - 1900) + (y % 7)}" for y in range(1900, 2001)))
    assert run(db, "import-csv", "--file", str(csv), "--indicator", "toy", "--entity", "Nowhere",
               "--source-url", "https://example.org/toy", "--synthetic") == 0
    assert run(db, "backtest", "--indicator", "toy", "--entity", "Nowhere", "--horizons", "5", "--record") == 0
    text = capsys.readouterr().out
    assert "WALK-FORWARD BACKTEST" in text and "experiment:" in text
    assert run(db, "experiments", "--json") == 0
    exps = json.loads(capsys.readouterr().out)
    assert exps and exps[0]["kind"] == "backtest"
    assert run(db, "experiment", "--indicator", "toy", "--entity", "Nowhere", "--origin", "1950", "--target", "1970") == 0
    assert "NOT evidence" in capsys.readouterr().out
    assert run(db, "breaks", "--indicator", "toy", "--entity", "Nowhere", "--signal", "diff") == 0


def test_errors_are_reported_not_raised(db, capsys):
    assert run(db, "show", "--indicator", "nope", "--entity", "nowhere") == 1
    assert "ERROR" in capsys.readouterr().err


def test_sources_listing(db, capsys):
    assert run(db, "sources", "--json") == 0
    rows = json.loads(capsys.readouterr().out)
    statuses = {r["source_id"]: r["status"] for r in rows}
    assert statuses["worldbank"] == "IMPLEMENTED_UNVERIFIED" and statuses["ucdp"] == "NOT_IMPLEMENTED"
