import sqlite3

import pytest

from chronos.errors import DataError
from chronos.storage import repository as repo
from chronos.storage.db import connect
from chronos.storage.repository import IngestBatch, ObsRow, SeriesMeta, SeriesPayload, SourceInfo

SRC = SourceInfo("testsrc", "Test source", license="CC0")


def batch(values, dataset="ds", kind="real", indicator="x", entity="A", as_of=None):
    meta = SeriesMeta(indicator, entity, SRC.source_id if kind == "real" else "synthetic", dataset, "col",
                      unit="u", data_kind=kind)
    src = SRC if kind == "real" else SourceInfo("synthetic", "syn")
    return IngestBatch(src, dataset, "mem://x", "sha", [SeriesPayload(meta, [ObsRow(y, v) for y, v in values])],
                       as_of_date=as_of)


def test_migrations_are_idempotent(tmp_path):
    p = tmp_path / "db.sqlite"
    c1 = connect(p)
    n1 = c1.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
    c1.close()
    c2 = connect(p)
    assert c2.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == n1
    c2.close()


def test_negative_and_zero_values_are_kept(con):
    repo.ingest(con, batch([(2000, -1.5), (2001, 0.0), (2002, 2.0)]))
    s = repo.load_series(con, "x", "A")
    assert list(s.values) == [-1.5, 0.0, 2.0]


def test_vintages_dedupe_and_record_revisions(con):
    r1 = repo.ingest(con, batch([(2000, 1.0), (2001, 2.0)]))
    r2 = repo.ingest(con, batch([(2000, 1.0), (2001, 2.0)]))
    assert r2.n_series_changed == 0 and r2.n_observations_written == 0
    r3 = repo.ingest(con, batch([(2000, 1.1), (2001, 2.0), (2002, 3.0)]))
    assert r3.n_series_changed == 1
    s = repo.load_series(con, "x", "A")
    assert s.get(2000) == pytest.approx(1.1) and s.meta["vintage_id"] == r3.vintage_id
    rec = repo.find_series(con, "x", "A")
    revs = repo.series_revisions(con, rec.series_id)
    assert revs[0]["years_revised"] == 1 and revs[0]["added_years"] == [2002]
    # the old value set is still stored
    old = con.execute("SELECT value FROM observation WHERE vintage_id=? AND year=2000", (r1.vintage_id,)).fetchone()[0]
    assert old == 1.0


def test_as_of_vintage_selection(con):
    repo.ingest(con, batch([(1940, 1.0), (1950, 2.0)], as_of="1951-01-01"))
    repo.ingest(con, batch([(1940, 1.2), (1950, 2.4), (1960, 3.0)]))
    rt = repo.load_series(con, "x", "A", as_of_year=1955)
    assert rt.meta["real_time_vintage"] is True and rt.get(1950) == 2.0
    latest = repo.load_series(con, "x", "A")
    assert latest.get(1950) == 2.4
    fallback = repo.load_series(con, "x", "A", as_of_year=1900)
    assert fallback.meta["real_time_vintage"] is False


def test_real_and_synthetic_never_mix(con):
    repo.ingest(con, batch([(2000, 1.0)], kind="synthetic"))
    with pytest.raises(DataError, match="REAL and SYNTHETIC"):
        repo.ingest(con, batch([(2000, 1.0)], kind="real"))


def test_one_series_one_dataset(con):
    repo.ingest(con, batch([(2000, 1.0)], dataset="d1"))
    with pytest.raises(DataError, match="refusing to merge"):
        repo.ingest(con, batch([(2000, 1.0)], dataset="d2"))


def test_failed_ingest_is_rolled_back(con):
    bad = batch([(2000, 1.0)])
    bad.series[0].rows.append(ObsRow(2000, 2.0))  # duplicate year
    with pytest.raises(DataError):
        repo.ingest(con, bad)
    assert con.execute("SELECT COUNT(*) FROM vintage").fetchone()[0] == 0


def test_ambiguous_series_requires_source(con):
    repo.ingest(con, batch([(2000, 1.0)]))
    other = SourceInfo("other", "Other")
    meta = SeriesMeta("x", "A", "other", "ds", "col")
    repo.ingest(con, IngestBatch(other, "ds", "mem", None, [SeriesPayload(meta, [ObsRow(2000, 5.0)])]))
    with pytest.raises(DataError, match="several sources"):
        repo.load_series(con, "x", "A")
    assert repo.load_series(con, "x", "A", source_id="other").get(2000) == 5.0


def test_v01_database_is_migrated(tmp_path, legacy_v01):
    path = tmp_path / "old.sqlite"
    old = legacy_v01.connect(str(path))
    legacy_v01.demo(old)
    legacy_v01.store_observations(old, "life", "World", [(2000, 66.0), (2001, 66.3)], "https://example.org/x", "real")
    old.execute("INSERT INTO events VALUES ('a', 1450, 'A', 'd', 'https://e.org/a')")
    old.execute("INSERT INTO events VALUES ('b', 1500, 'B', 'd', '')")
    old.execute("INSERT INTO claims VALUES ('a', 'b', 'mech', 'hypothesis', '')")
    old.commit()
    old.close()
    con = connect(path)
    s = repo.load_series(con, "DEMO_SYNTHETIC_INDEX", "Synthetic World")
    assert s.data_kind == "synthetic" and len(s) == 126
    life = repo.load_series(con, "life", "World")
    assert life.data_kind == "real" and life.meta["location"] == "https://example.org/x"
    assert con.execute("SELECT COUNT(*) FROM event").fetchone()[0] == 2
    claim = con.execute("SELECT relation, evidence_level FROM claim").fetchone()
    assert tuple(claim) == ("causal_hypothesis", 1)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"v01_observations", "v01_events", "v01_claims"} <= tables
    con.close()


def test_connection_closes(tmp_path):
    from chronos.storage.db import open_db

    with open_db(tmp_path / "x.sqlite") as c:
        c.execute("SELECT 1")
    with pytest.raises(sqlite3.ProgrammingError):
        c.execute("SELECT 1")
