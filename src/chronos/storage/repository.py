"""Read/write access to sources, series, vintages and observations."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from chronos.errors import DataError
from chronos.storage.db import transaction, utc_now
from chronos.timeseries import Panel, TimeSeries

OBS_STATUSES = ("observed", "estimated", "imputed", "interpolated", "projected", "filler", "derived")


@dataclass(frozen=True)
class SourceInfo:
    source_id: str
    name: str
    publisher: str | None = None
    homepage: str | None = None
    license: str | None = None
    citation: str | None = None
    terms_url: str | None = None


@dataclass(frozen=True)
class SeriesMeta:
    indicator: str
    entity: str
    source_id: str
    source_dataset: str
    source_variable: str
    entity_code: str | None = None
    title: str | None = None
    description: str | None = None
    unit: str | None = None
    frequency: str = "annual"
    data_kind: str = "real"


@dataclass(frozen=True)
class ObsRow:
    year: int
    value: float
    status: str = "observed"
    note: str | None = None


@dataclass
class SeriesPayload:
    meta: SeriesMeta
    rows: list[ObsRow]


@dataclass
class IngestBatch:
    """Everything one connector run produced: one source dataset, one vintage."""

    source: SourceInfo
    dataset: str
    location: str
    content_sha256: str | None
    series: list[SeriesPayload]
    source_version: str | None = None
    as_of_date: str | None = None
    notes: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class IngestReport:
    vintage_id: int
    n_series: int
    n_series_changed: int
    n_observations_written: int
    warnings: tuple[str, ...]


def clean_rows(rows: Iterable[ObsRow]) -> tuple[list[ObsRow], int]:
    """Drop non-finite values; reject duplicate years. Returns (rows, n_dropped).

    Unlike v0.1, zero and negative values are kept: they are legitimate for many indicators
    (growth rates, balances, index values). Transformations that need positivity check it.
    """
    out: dict[int, ObsRow] = {}
    dropped = 0
    for r in rows:
        if r.status not in OBS_STATUSES:
            raise DataError(f"Unknown observation status {r.status!r}")
        if r.value is None or not math.isfinite(float(r.value)):
            dropped += 1
            continue
        if r.year in out:
            raise DataError(f"Duplicate year {r.year} in one series payload")
        out[int(r.year)] = ObsRow(int(r.year), float(r.value), r.status, r.note)
    return [out[y] for y in sorted(out)], dropped


def values_hash(rows: Sequence[ObsRow]) -> str:
    canon = json.dumps([(r.year, repr(float(r.value)), r.status) for r in rows], separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


def ensure_source(con: sqlite3.Connection, src: SourceInfo) -> None:
    con.execute(
        """INSERT INTO source(source_id,name,publisher,homepage,license,citation,terms_url)
           VALUES (?,?,?,?,?,?,?)
           ON CONFLICT(source_id) DO UPDATE SET name=excluded.name, publisher=excluded.publisher,
             homepage=excluded.homepage, license=excluded.license, citation=excluded.citation,
             terms_url=excluded.terms_url""",
        (src.source_id, src.name, src.publisher, src.homepage, src.license, src.citation, src.terms_url),
    )


def upsert_series(con: sqlite3.Connection, meta: SeriesMeta) -> int:
    if meta.data_kind not in ("real", "synthetic"):
        raise DataError(f"data_kind must be real or synthetic, got {meta.data_kind!r}")
    clash = con.execute(
        "SELECT source_id, data_kind FROM series WHERE indicator=? AND entity=? AND data_kind<>?",
        (meta.indicator, meta.entity, meta.data_kind),
    ).fetchone()
    if clash:
        raise DataError(
            f"Refusing to mix REAL and SYNTHETIC data: {meta.indicator}/{meta.entity} already exists as "
            f"{clash['data_kind']} (source {clash['source_id']}). Use a different indicator name."
        )
    row = con.execute(
        "SELECT series_id, source_dataset FROM series WHERE indicator=? AND entity=? AND source_id=?",
        (meta.indicator, meta.entity, meta.source_id),
    ).fetchone()
    if row and row["source_dataset"] != meta.source_dataset:
        raise DataError(
            f"{meta.indicator}/{meta.entity} from source {meta.source_id} already comes from dataset "
            f"{row['source_dataset']!r}; refusing to merge values from {meta.source_dataset!r} into the same "
            f"series. Choose another indicator name."
        )
    if row:
        con.execute(
            """UPDATE series SET entity_code=COALESCE(?,entity_code), source_dataset=?, source_variable=?,
               title=COALESCE(?,title), description=COALESCE(?,description), unit=COALESCE(?,unit), frequency=?
               WHERE series_id=?""",
            (meta.entity_code, meta.source_dataset, meta.source_variable, meta.title, meta.description,
             meta.unit, meta.frequency, row["series_id"]),
        )
        return int(row["series_id"])
    cur = con.execute(
        """INSERT INTO series(indicator,entity,entity_code,source_id,source_dataset,source_variable,title,
             description,unit,frequency,data_kind,created_utc) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (meta.indicator, meta.entity, meta.entity_code, meta.source_id, meta.source_dataset,
         meta.source_variable, meta.title, meta.description, meta.unit, meta.frequency, meta.data_kind,
         utc_now()),
    )
    return int(cur.lastrowid or 0)


def ingest(con: sqlite3.Connection, batch: IngestBatch) -> IngestReport:
    """Store a batch as one new vintage. Unchanged series reuse their previous rows."""
    warnings = list(batch.warnings)
    written = changed = 0
    with transaction(con):
        ensure_source(con, batch.source)
        cur = con.execute(
            """INSERT INTO vintage(source_id,source_dataset,retrieved_utc,source_version,as_of_date,location,
                 content_sha256,notes) VALUES (?,?,?,?,?,?,?,?)""",
            (batch.source.source_id, batch.dataset, utc_now(), batch.source_version, batch.as_of_date,
             batch.location, batch.content_sha256, batch.notes),
        )
        vintage_id = int(cur.lastrowid or 0)
        for payload in batch.series:
            rows, dropped = clean_rows(payload.rows)
            if dropped:
                warnings.append(f"{payload.meta.indicator}/{payload.meta.entity}: dropped {dropped} non-finite values")
            if not rows:
                warnings.append(f"{payload.meta.indicator}/{payload.meta.entity}: no usable observations, skipped")
                continue
            sid = upsert_series(con, payload.meta)
            h = values_hash(rows)
            prev = con.execute(
                """SELECT rows_vintage_id, values_sha256 FROM series_vintage WHERE series_id=?
                   ORDER BY vintage_id DESC LIMIT 1""",
                (sid,),
            ).fetchone()
            if prev and prev["values_sha256"] == h:
                rows_vid = int(prev["rows_vintage_id"])
            else:
                rows_vid = vintage_id
                changed += 1
                con.executemany(
                    "INSERT INTO observation(series_id,vintage_id,year,value,status,note) VALUES (?,?,?,?,?,?)",
                    [(sid, vintage_id, r.year, r.value, r.status, r.note) for r in rows],
                )
                written += len(rows)
            con.execute(
                """INSERT INTO series_vintage(series_id,vintage_id,rows_vintage_id,values_sha256,n_obs,first_year,last_year)
                   VALUES (?,?,?,?,?,?,?)""",
                (sid, vintage_id, rows_vid, h, len(rows), rows[0].year, rows[-1].year),
            )
        con.execute("UPDATE vintage SET notes=COALESCE(notes,'') || ? WHERE vintage_id=?",
                    ("" if not warnings else f" [warnings: {len(warnings)}]", vintage_id))
    return IngestReport(vintage_id, len(batch.series), changed, written, tuple(warnings))


# ----------------------------------------------------------------------------- queries


@dataclass(frozen=True)
class SeriesRecord:
    series_id: int
    indicator: str
    entity: str
    entity_code: str | None
    source_id: str
    source_dataset: str
    source_variable: str
    title: str | None
    unit: str | None
    frequency: str
    data_kind: str


def _record(row: sqlite3.Row) -> SeriesRecord:
    return SeriesRecord(row["series_id"], row["indicator"], row["entity"], row["entity_code"], row["source_id"],
                        row["source_dataset"], row["source_variable"], row["title"], row["unit"],
                        row["frequency"], row["data_kind"])


def find_series(con: sqlite3.Connection, indicator: str, entity: str, source_id: str | None = None) -> SeriesRecord:
    q = "SELECT * FROM series WHERE indicator=? AND entity=?"
    args: list[Any] = [indicator, entity]
    if source_id:
        q += " AND source_id=?"
        args.append(source_id)
    rows = con.execute(q, args).fetchall()
    if not rows:
        hint = con.execute("SELECT DISTINCT entity FROM series WHERE indicator=? LIMIT 8", (indicator,)).fetchall()
        extra = f" Entities available for this indicator: {[r[0] for r in hint]}" if hint else ""
        raise DataError(f"No series {indicator!r} / {entity!r}.{extra} Run `chronos fetch ...` or `chronos sources`.")
    if len(rows) > 1:
        raise DataError(
            f"{indicator}/{entity} exists in several sources {[r['source_id'] for r in rows]}; pass --source."
        )
    return _record(rows[0])


def list_series(con: sqlite3.Connection, indicator: str | None = None, source_id: str | None = None,
                entity: str | None = None) -> list[SeriesRecord]:
    q = "SELECT * FROM series WHERE 1=1"
    args: list[Any] = []
    for col, val in (("indicator", indicator), ("source_id", source_id), ("entity", entity)):
        if val:
            q += f" AND {col}=?"
            args.append(val)
    return [_record(r) for r in con.execute(q + " ORDER BY indicator, entity", args)]


@dataclass(frozen=True)
class VintageChoice:
    vintage_id: int
    rows_vintage_id: int
    retrieved_utc: str
    source_version: str | None
    as_of_date: str | None
    location: str | None
    real_time: bool  # True if the vintage's knowledge date is <= the requested as-of date


def choose_vintage(con: sqlite3.Connection, series_id: int, as_of_year: int | None = None) -> VintageChoice:
    rows = con.execute(
        """SELECT sv.vintage_id, sv.rows_vintage_id, v.retrieved_utc, v.source_version, v.as_of_date, v.location
           FROM series_vintage sv JOIN vintage v ON v.vintage_id = sv.vintage_id
           WHERE sv.series_id=? ORDER BY sv.vintage_id DESC""",
        (series_id,),
    ).fetchall()
    if not rows:
        raise DataError(f"Series {series_id} has no stored vintages")
    if as_of_year is not None:
        for r in rows:
            knowledge = r["as_of_date"] or r["retrieved_utc"]
            if int(knowledge[:4]) <= as_of_year:
                return VintageChoice(r["vintage_id"], r["rows_vintage_id"], r["retrieved_utc"], r["source_version"],
                                     r["as_of_date"], r["location"], True)
    r = rows[0]
    return VintageChoice(r["vintage_id"], r["rows_vintage_id"], r["retrieved_utc"], r["source_version"],
                         r["as_of_date"], r["location"], as_of_year is None)


def load_series_by_id(con: sqlite3.Connection, series_id: int, as_of_year: int | None = None,
                      statuses: Sequence[str] | None = None) -> TimeSeries:
    rec = _record(con.execute("SELECT * FROM series WHERE series_id=?", (series_id,)).fetchone())
    vc = choose_vintage(con, series_id, as_of_year)
    q = "SELECT year, value, status FROM observation WHERE series_id=? AND vintage_id=?"
    args: list[Any] = [series_id, vc.rows_vintage_id]
    if statuses:
        q += f" AND status IN ({','.join('?' * len(statuses))})"
        args.extend(statuses)
    rows = con.execute(q + " ORDER BY year", args).fetchall()
    src = con.execute("SELECT * FROM source WHERE source_id=?", (rec.source_id,)).fetchone()
    meta = {
        "series_id": rec.series_id,
        "indicator": rec.indicator,
        "source_id": rec.source_id,
        "source_name": src["name"] if src else rec.source_id,
        "source_license": src["license"] if src else None,
        "source_citation": src["citation"] if src else None,
        "source_dataset": rec.source_dataset,
        "source_variable": rec.source_variable,
        "title": rec.title,
        "frequency": rec.frequency,
        "vintage_id": vc.vintage_id,
        "rows_vintage_id": vc.rows_vintage_id,
        "retrieved_utc": vc.retrieved_utc,
        "source_version": vc.source_version,
        "as_of_date": vc.as_of_date,
        "location": vc.location,
        "real_time_vintage": vc.real_time,
    }
    return TimeSeries(
        np.array([r["year"] for r in rows], dtype=np.int64),
        np.array([r["value"] for r in rows], dtype=float),
        name=rec.indicator, entity=rec.entity, unit=rec.unit, data_kind=rec.data_kind, meta=meta,
        status=tuple(r["status"] for r in rows),
    )


def load_series(con: sqlite3.Connection, indicator: str, entity: str, source_id: str | None = None,
                as_of_year: int | None = None, statuses: Sequence[str] | None = None) -> TimeSeries:
    rec = find_series(con, indicator, entity, source_id)
    return load_series_by_id(con, rec.series_id, as_of_year, statuses)


def load_panel(con: sqlite3.Connection, indicator: str, source_id: str | None = None,
               entities: Sequence[str] | None = None, statuses: Sequence[str] | None = None,
               min_obs: int = 1) -> Panel:
    recs = list_series(con, indicator=indicator, source_id=source_id)
    if source_id is None and len({r.source_id for r in recs}) > 1:
        raise DataError(f"Indicator {indicator!r} exists in several sources; pass a source.")
    wanted = set(entities) if entities else None
    out: dict[str, TimeSeries] = {}
    for r in recs:
        if wanted is not None and r.entity not in wanted:
            continue
        s = load_series_by_id(con, r.series_id, statuses=statuses)
        if len(s) >= min_obs:
            out[r.entity] = s
    if not out:
        raise DataError(f"No series for indicator {indicator!r}")
    return Panel(indicator, out)


def series_revisions(con: sqlite3.Connection, series_id: int) -> list[dict[str, Any]]:
    """Compare consecutive stored value sets of one series (documented revisions)."""
    vids = [r[0] for r in con.execute(
        "SELECT DISTINCT rows_vintage_id FROM series_vintage WHERE series_id=? ORDER BY rows_vintage_id",
        (series_id,))]
    out = []
    for a, b in zip(vids, vids[1:], strict=False):
        old = dict(con.execute("SELECT year, value FROM observation WHERE series_id=? AND vintage_id=?", (series_id, a)).fetchall())
        new = dict(con.execute("SELECT year, value FROM observation WHERE series_id=? AND vintage_id=?", (series_id, b)).fetchall())
        common = sorted(set(old) & set(new))
        changed = [y for y in common if not math.isclose(old[y], new[y], rel_tol=1e-12, abs_tol=0.0)]
        rel = [abs(new[y] - old[y]) / abs(old[y]) for y in changed if old[y] != 0]
        out.append({
            "from_vintage": a, "to_vintage": b, "years_compared": len(common), "years_revised": len(changed),
            "added_years": sorted(set(new) - set(old)), "removed_years": sorted(set(old) - set(new)),
            "max_abs_relative_revision": max(rel) if rel else 0.0,
        })
    return out


def data_fingerprint(series: Iterable[TimeSeries]) -> dict[str, Any]:
    fp: dict[str, Any] = {}
    for s in series:
        key = f"{s.meta.get('source_id', s.data_kind)}:{s.name}:{s.entity}"
        h = hashlib.sha256(np.ascontiguousarray(s.years).tobytes() + np.ascontiguousarray(s.values).tobytes()).hexdigest()
        fp[key] = {"rows_vintage_id": s.meta.get("rows_vintage_id"), "retrieved_utc": s.meta.get("retrieved_utc"),
                   "values_sha256": h[:16], "n": len(s)}
    return fp


def catalog(con: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = con.execute(
        """SELECT s.series_id, s.indicator, s.entity, s.source_id, s.unit, s.data_kind, s.frequency,
                  sv.n_obs, sv.first_year, sv.last_year, v.retrieved_utc, v.source_version
           FROM series s
           JOIN series_vintage sv ON sv.series_id = s.series_id
           JOIN vintage v ON v.vintage_id = sv.vintage_id
           WHERE sv.vintage_id = (SELECT MAX(vintage_id) FROM series_vintage WHERE series_id = s.series_id)
           ORDER BY s.indicator, s.entity"""
    ).fetchall()
    return [dict(r) for r in rows]
