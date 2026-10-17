"""Migration of CHRONOS v0.1 databases (tables observations/events/claims).

v0.1 stored one row per (indicator, entity, year) without vintages or units. The data is
copied into the new schema as one 'legacy-v0.1' vintage per (source_url, import time), and
the original tables are kept, renamed to v01_*, so nothing is lost or silently rewritten.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections import defaultdict

from chronos.storage.db import utc_now

LEGACY_SOURCE = ("legacy-v0.1", "Imported from a CHRONOS v0.1 database",
                 "Provenance limited to the URL stored by v0.1; unit and source version unknown")


def _has_table(con: sqlite3.Connection, name: str) -> bool:
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def import_v01_tables(con: sqlite3.Connection) -> None:
    if not (_has_table(con, "observations") or _has_table(con, "events") or _has_table(con, "claims")):
        return
    now = utc_now()
    con.execute(
        "INSERT OR IGNORE INTO source(source_id,name,license,citation) VALUES (?,?,?,?)",
        (LEGACY_SOURCE[0], LEGACY_SOURCE[1], "UNKNOWN (see original URLs)", LEGACY_SOURCE[2]),
    )
    if _has_table(con, "observations"):
        groups: dict[tuple[str, str], list[sqlite3.Row]] = defaultdict(list)
        for row in con.execute("SELECT * FROM observations ORDER BY indicator, entity, year"):
            groups[(row["source_url"], row["imported_utc"][:19])].append(row)
        for (url, imported), rows in groups.items():
            cur = con.execute(
                "INSERT INTO vintage(source_id,source_dataset,retrieved_utc,location,notes) VALUES (?,?,?,?,?)",
                (LEGACY_SOURCE[0], "v0.1-observations", imported, url, "migrated from v0.1"),
            )
            vid = cur.lastrowid
            by_series: dict[tuple[str, str, str], list[sqlite3.Row]] = defaultdict(list)
            for r in rows:
                by_series[(r["indicator"], r["entity"], r["source_kind"])].append(r)
            for (indicator, entity, kind), srows in by_series.items():
                sid_row = con.execute(
                    "SELECT series_id FROM series WHERE indicator=? AND entity=? AND source_id=?",
                    (indicator, entity, LEGACY_SOURCE[0]),
                ).fetchone()
                if sid_row:
                    sid = sid_row[0]
                else:
                    sid = con.execute(
                        """INSERT INTO series(indicator,entity,source_id,source_dataset,source_variable,
                             description,frequency,data_kind,created_utc) VALUES (?,?,?,?,?,?,?,?,?)""",
                        (indicator, entity, LEGACY_SOURCE[0], "v0.1-observations", indicator,
                         f"Migrated from v0.1; original source: {url}", "annual", kind, now),
                    ).lastrowid
                con.executemany(
                    "INSERT INTO observation(series_id,vintage_id,year,value,status,note) VALUES (?,?,?,?,?,?)",
                    [(sid, vid, r["year"], r["value"], "observed", None) for r in srows],
                )
                payload = repr([(r["year"], repr(float(r["value"]))) for r in srows]).encode()
                con.execute(
                    "INSERT INTO series_vintage VALUES (?,?,?,?,?,?,?)",
                    (sid, vid, vid, hashlib.sha256(payload).hexdigest(), len(srows), srows[0]["year"], srows[-1]["year"]),
                )
        con.execute("ALTER TABLE observations RENAME TO v01_observations")

    if _has_table(con, "events"):
        for r in con.execute("SELECT * FROM events"):
            con.execute(
                """INSERT OR IGNORE INTO event(event_id,title,description,category,start_year,date_precision,
                     tags,data_kind,created_utc) VALUES (?,?,?,?,?,?,?,?,?)""",
                (r["event_id"], r["title"], r["description"], "other", r["year"], "year", '["legacy-v0.1"]', "real", now),
            )
            if r["evidence_url"]:
                ref_id = _url_ref(con, r["evidence_url"])
                con.execute("INSERT OR IGNORE INTO event_reference(event_id,ref_id) VALUES (?,?)", (r["event_id"], ref_id))
        con.execute("ALTER TABLE events RENAME TO v01_events")

    if _has_table(con, "claims"):
        mapping = {"hypothesis": ("causal_hypothesis", 1), "analogy": ("analogy", 0), "documented": ("documented_claim", 0)}
        for i, r in enumerate(con.execute("SELECT * FROM claims")):
            relation, level = mapping.get(r["claim_type"], ("causal_hypothesis", 0))
            claim_id = f"v01-{i + 1}-{r['cause_event_id']}-{r['effect_event_id']}"
            con.execute(
                """INSERT OR IGNORE INTO claim(claim_id,source_event_id,target_event_id,relation,statement,
                     evidence_level,curator,notes,created_utc) VALUES (?,?,?,?,?,?,?,?,?)""",
                (claim_id, r["cause_event_id"], r["effect_event_id"], relation, r["mechanism"], level,
                 "legacy-v0.1", "Migrated from v0.1; evidence level not assessed", now),
            )
            if r["evidence_url"]:
                ref_id = _url_ref(con, r["evidence_url"])
                con.execute("INSERT OR IGNORE INTO claim_reference(claim_id,ref_id,stance) VALUES (?,?,?)",
                            (claim_id, ref_id, "discusses"))
        con.execute("ALTER TABLE claims RENAME TO v01_claims")


def _url_ref(con: sqlite3.Connection, url: str) -> str:
    ref_id = "url-" + hashlib.sha1(url.encode()).hexdigest()[:10]
    con.execute(
        "INSERT OR IGNORE INTO reference(ref_id,ref_type,title,url,verification,notes) VALUES (?,?,?,?,?,?)",
        (ref_id, "web_page", url, url, "unverified", "URL imported from v0.1"),
    )
    return ref_id
