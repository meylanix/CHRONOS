"""Historical knowledge graph: events, references, mechanisms, claims, alternative explanations.

Rules enforced on every write (ValidationRuleError otherwise):
  R1  claim endpoints must exist
  R2  for causal/contributing/documented/sequence relations the target cannot start before the
      source (beyond their stated dating uncertainty)
  R3  'temporal_sequence' claims carry evidence level 0: a sequence is not evidence of causation
  R4  evidence level >= 3 needs at least one reference with stance 'supports'
  R5  evidence level >= 4 needs a supporting scholarly reference (article, working paper, book, chapter)
  R6  'documented_claim' needs a supporting reference (a URL alone is not proof of causality)
  R7  status 'supported' needs evidence level >= 3; every status/level change is logged with a reason
Evidence levels are curator assessments; CHRONOS never raises them automatically.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chronos.errors import DataError, ValidationRuleError
from chronos.storage.db import transaction, utc_now

EVIDENCE_LEVELS = {
    0: "unassessed",
    1: "speculative",
    2: "plausible argument",
    3: "supported by qualitative scholarship",
    4: "supported by quantitative association studies",
    5: "supported by identified causal designs",
}
CAUSAL_RELATIONS = {"causal_hypothesis", "contributing_condition", "documented_claim", "temporal_sequence"}
SCHOLARLY = {"journal_article", "working_paper", "book", "book_chapter"}


@dataclass
class Reference:
    ref_id: str
    ref_type: str
    title: str
    authors: str | None = None
    year: int | None = None
    venue: str | None = None
    publisher: str | None = None
    doi: str | None = None
    url: str | None = None
    accessed: str | None = None
    verification: str = "unverified"
    notes: str | None = None


@dataclass
class Event:
    event_id: str
    title: str
    category: str
    start_year: int
    description: str = ""
    end_year: int | None = None
    year_uncertainty: int = 0
    date_precision: str = "year"
    date_detail: str | None = None
    region: str | None = None
    tags: list[str] = field(default_factory=list)
    data_kind: str = "real"
    references: list[dict[str, str]] = field(default_factory=list)  # [{ref_id, locator}]


@dataclass
class Mechanism:
    mechanism_id: str
    name: str
    description: str
    domain: str


@dataclass
class Claim:
    claim_id: str
    source_event_id: str
    target_event_id: str
    relation: str
    statement: str
    mechanism_id: str | None = None
    evidence_level: int = 0
    status: str = "proposed"
    lag_min_years: int | None = None
    lag_max_years: int | None = None
    curator: str | None = None
    notes: str | None = None
    data_kind: str = "real"
    references: list[dict[str, str]] = field(default_factory=list)   # [{ref_id, stance, locator}]
    alternatives: list[dict[str, str]] = field(default_factory=list)  # [{description, ref_id}]


@dataclass
class Fact:
    fact_id: str
    subject: str
    statement: str
    ref_id: str
    value: float | None = None
    unit: str | None = None
    year: int | None = None
    locator: str | None = None
    uncertainty: str | None = None
    tags: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------- writes


def add_reference(con: sqlite3.Connection, r: Reference) -> None:
    con.execute(
        """INSERT INTO reference(ref_id,ref_type,authors,title,year,venue,publisher,doi,url,accessed,verification,notes)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(ref_id) DO UPDATE SET ref_type=excluded.ref_type, authors=excluded.authors, title=excluded.title,
             year=excluded.year, venue=excluded.venue, publisher=excluded.publisher, doi=excluded.doi, url=excluded.url,
             accessed=excluded.accessed, verification=excluded.verification, notes=excluded.notes""",
        (r.ref_id, r.ref_type, r.authors, r.title, r.year, r.venue, r.publisher, r.doi, r.url, r.accessed,
         r.verification, r.notes),
    )


def add_event(con: sqlite3.Connection, e: Event) -> None:
    con.execute(
        """INSERT INTO event(event_id,title,description,category,start_year,end_year,year_uncertainty,date_precision,
             date_detail,region,tags,data_kind,created_utc) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(event_id) DO UPDATE SET title=excluded.title, description=excluded.description,
             category=excluded.category, start_year=excluded.start_year, end_year=excluded.end_year,
             year_uncertainty=excluded.year_uncertainty, date_precision=excluded.date_precision,
             date_detail=excluded.date_detail, region=excluded.region, tags=excluded.tags, data_kind=excluded.data_kind""",
        (e.event_id, e.title, e.description, e.category, e.start_year, e.end_year, e.year_uncertainty,
         e.date_precision, e.date_detail, e.region, json.dumps(e.tags), e.data_kind, utc_now()),
    )
    for ref in e.references:
        _require_ref(con, ref["ref_id"])
        con.execute("INSERT OR REPLACE INTO event_reference(event_id,ref_id,locator) VALUES (?,?,?)",
                    (e.event_id, ref["ref_id"], ref.get("locator")))


def add_mechanism(con: sqlite3.Connection, m: Mechanism) -> None:
    con.execute(
        """INSERT INTO mechanism(mechanism_id,name,description,domain) VALUES (?,?,?,?)
           ON CONFLICT(mechanism_id) DO UPDATE SET name=excluded.name, description=excluded.description, domain=excluded.domain""",
        (m.mechanism_id, m.name, m.description, m.domain),
    )


def add_fact(con: sqlite3.Connection, f: Fact) -> None:
    _require_ref(con, f.ref_id)
    con.execute(
        """INSERT OR REPLACE INTO fact(fact_id,subject,statement,value,unit,year,ref_id,locator,uncertainty,tags)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (f.fact_id, f.subject, f.statement, f.value, f.unit, f.year, f.ref_id, f.locator, f.uncertainty, json.dumps(f.tags)),
    )


def _require_ref(con: sqlite3.Connection, ref_id: str) -> sqlite3.Row:
    row = con.execute("SELECT * FROM reference WHERE ref_id=?", (ref_id,)).fetchone()
    if row is None:
        raise ValidationRuleError(f"Unknown reference {ref_id!r}")
    return row


def _event(con: sqlite3.Connection, event_id: str) -> sqlite3.Row:
    row = con.execute("SELECT * FROM event WHERE event_id=?", (event_id,)).fetchone()
    if row is None:
        raise ValidationRuleError(f"R1: unknown event {event_id!r}")
    return row


def check_claim(con: sqlite3.Connection, c: Claim) -> None:
    src, tgt = _event(con, c.source_event_id), _event(con, c.target_event_id)
    if c.relation in CAUSAL_RELATIONS:
        slack = src["year_uncertainty"] + tgt["year_uncertainty"]
        if tgt["start_year"] + slack < src["start_year"]:
            raise ValidationRuleError(
                f"R2: {c.claim_id}: target {tgt['event_id']} ({tgt['start_year']}) starts before source "
                f"{src['event_id']} ({src['start_year']}); an effect cannot precede its cause.")
    if c.relation == "temporal_sequence" and c.evidence_level != 0:
        raise ValidationRuleError(f"R3: {c.claim_id}: a temporal sequence carries evidence level 0 (not causal evidence)")
    if not 0 <= c.evidence_level <= 5:
        raise ValidationRuleError(f"{c.claim_id}: evidence level must be 0..5")
    refs = [(r["ref_id"], r["stance"]) for r in c.references]
    types = {rid: _require_ref(con, rid)["ref_type"] for rid, _ in refs}
    supporting = [rid for rid, st in refs if st == "supports"]
    if c.evidence_level >= 3 and not supporting:
        raise ValidationRuleError(f"R4: {c.claim_id}: evidence level {c.evidence_level} needs a supporting reference")
    if c.evidence_level >= 4 and not any(types[r] in SCHOLARLY for r in supporting):
        raise ValidationRuleError(f"R5: {c.claim_id}: evidence level {c.evidence_level} needs a supporting scholarly reference")
    if c.relation == "documented_claim" and not supporting:
        raise ValidationRuleError(f"R6: {c.claim_id}: a documented claim needs a supporting reference")
    if c.status == "supported" and c.evidence_level < 3:
        raise ValidationRuleError(f"R7: {c.claim_id}: status 'supported' requires evidence level >= 3")
    if c.mechanism_id and con.execute("SELECT 1 FROM mechanism WHERE mechanism_id=?", (c.mechanism_id,)).fetchone() is None:
        raise ValidationRuleError(f"{c.claim_id}: unknown mechanism {c.mechanism_id!r}")


def add_claim(con: sqlite3.Connection, c: Claim) -> None:
    check_claim(con, c)
    prev = con.execute("SELECT status, evidence_level FROM claim WHERE claim_id=?", (c.claim_id,)).fetchone()
    con.execute(
        """INSERT INTO claim(claim_id,source_event_id,mechanism_id,target_event_id,relation,statement,evidence_level,
             status,lag_min_years,lag_max_years,curator,notes,data_kind,created_utc) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(claim_id) DO UPDATE SET source_event_id=excluded.source_event_id, mechanism_id=excluded.mechanism_id,
             target_event_id=excluded.target_event_id, relation=excluded.relation, statement=excluded.statement,
             evidence_level=excluded.evidence_level, status=excluded.status, lag_min_years=excluded.lag_min_years,
             lag_max_years=excluded.lag_max_years, curator=excluded.curator, notes=excluded.notes, data_kind=excluded.data_kind""",
        (c.claim_id, c.source_event_id, c.mechanism_id, c.target_event_id, c.relation, c.statement, c.evidence_level,
         c.status, c.lag_min_years, c.lag_max_years, c.curator, c.notes, c.data_kind, utc_now()),
    )
    if prev and (prev["status"] != c.status or prev["evidence_level"] != c.evidence_level):
        con.execute("""INSERT INTO claim_status_log(claim_id,old_status,new_status,old_level,new_level,reason,changed_utc)
                       VALUES (?,?,?,?,?,?,?)""",
                    (c.claim_id, prev["status"], c.status, prev["evidence_level"], c.evidence_level, "re-import", utc_now()))
    con.execute("DELETE FROM claim_reference WHERE claim_id=?", (c.claim_id,))
    con.execute("DELETE FROM alternative_explanation WHERE claim_id=?", (c.claim_id,))
    for r in c.references:
        con.execute("INSERT INTO claim_reference(claim_id,ref_id,stance,locator) VALUES (?,?,?,?)",
                    (c.claim_id, r["ref_id"], r["stance"], r.get("locator")))
    for a in c.alternatives:
        if a.get("ref_id"):
            _require_ref(con, a["ref_id"])
        con.execute("INSERT INTO alternative_explanation(claim_id,description,ref_id) VALUES (?,?,?)",
                    (c.claim_id, a["description"], a.get("ref_id")))


def set_claim_status(con: sqlite3.Connection, claim_id: str, status: str | None, level: int | None, reason: str) -> None:
    if not reason.strip():
        raise ValidationRuleError("A reason is required for every status or evidence-level change")
    c = get_claim(con, claim_id)
    new = Claim(**{**{k: c[k] for k in Claim.__dataclass_fields__ if k in c},
                   "status": status or c["status"], "evidence_level": c["evidence_level"] if level is None else level})
    with transaction(con):
        check_claim(con, new)
        con.execute("UPDATE claim SET status=?, evidence_level=? WHERE claim_id=?", (new.status, new.evidence_level, claim_id))
        con.execute("""INSERT INTO claim_status_log(claim_id,old_status,new_status,old_level,new_level,reason,changed_utc)
                       VALUES (?,?,?,?,?,?,?)""",
                    (claim_id, c["status"], new.status, c["evidence_level"], new.evidence_level, reason, utc_now()))


@dataclass
class OutlookHypothesis:
    oh_id: str
    statement: str
    falsified_if: str
    check_by_year: int
    mechanism_id: str | None = None
    based_on_claims: list[str] = field(default_factory=list)
    signpost: dict[str, Any] | None = None
    scope: str | None = None
    notes: str | None = None
    data_kind: str = "real"


def add_outlook_hypothesis(con: sqlite3.Connection, h: OutlookHypothesis) -> None:
    for cid in h.based_on_claims:
        if con.execute("SELECT 1 FROM claim WHERE claim_id=?", (cid,)).fetchone() is None:
            raise ValidationRuleError(f"{h.oh_id}: unknown claim {cid!r}")
    if not h.falsified_if.strip():
        raise ValidationRuleError(f"{h.oh_id}: a forward-looking hypothesis needs a falsification condition")
    con.execute(
        """INSERT OR REPLACE INTO outlook_hypothesis(oh_id,statement,mechanism_id,based_on_claims,falsified_if,check_by_year,
             signpost_json,operationalised,scope,notes,data_kind,registered_utc) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (h.oh_id, h.statement, h.mechanism_id, json.dumps(h.based_on_claims), h.falsified_if, h.check_by_year,
         json.dumps(h.signpost) if h.signpost else None, 1 if h.signpost else 0, h.scope, h.notes, h.data_kind, utc_now()))


def list_outlook_hypotheses(con: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = _rows(con, "SELECT * FROM outlook_hypothesis ORDER BY oh_id")
    for r in rows:
        r["based_on_claims"] = json.loads(r["based_on_claims"])
        r["signpost"] = json.loads(r["signpost_json"]) if r["signpost_json"] else None
        levels = [con.execute("SELECT evidence_level FROM claim WHERE claim_id=?", (c,)).fetchone()[0] for c in r["based_on_claims"]]
        r["basis_evidence_levels"] = levels
    return rows


# ----------------------------------------------------------------------------- bulk import


def import_json(con: sqlite3.Connection, path: str | Path) -> dict[str, int]:
    """Import a curated bundle {references, events, mechanisms, claims, facts} atomically."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataError(f"Cannot read knowledge bundle {path}: {exc}") from exc
    counts = {}
    with transaction(con):
        for r in doc.get("references", []):
            add_reference(con, Reference(**r))
        for m in doc.get("mechanisms", []):
            add_mechanism(con, Mechanism(**m))
        for e in doc.get("events", []):
            add_event(con, Event(**e))
        for c in doc.get("claims", []):
            add_claim(con, Claim(**c))
        for f in doc.get("facts", []):
            add_fact(con, Fact(**f))
        for h in doc.get("outlook_hypotheses", []):
            add_outlook_hypothesis(con, OutlookHypothesis(**h))
    for k in ("references", "mechanisms", "events", "claims", "facts", "outlook_hypotheses"):
        counts[k] = len(doc.get(k, []))
    return counts


def import_events_csv_v01(con: sqlite3.Connection, path: str, kind: str = "real") -> int:
    """v0.1 format: event_id,year,title,description,evidence_url."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        need = {"event_id", "year", "title", "description", "evidence_url"}
        if not need <= set(reader.fieldnames or []):
            raise DataError("Events CSV requires: " + ",".join(sorted(need)))
        rows = list(reader)
    with transaction(con):
        for r in rows:
            refs = []
            if r["evidence_url"].strip():
                ref_id = _url_reference(con, r["evidence_url"].strip())
                refs = [{"ref_id": ref_id}]
            add_event(con, Event(r["event_id"], r["title"], r.get("category") or "other", int(r["year"]),
                                 r["description"], tags=["csv-import"], data_kind=kind, references=refs))
    return len(rows)


def import_claims_csv_v01(con: sqlite3.Connection, path: str, kind: str = "real") -> int:
    """v0.1 format: cause_event_id,effect_event_id,mechanism,claim_type,evidence_url."""
    mapping = {"hypothesis": "causal_hypothesis", "analogy": "analogy", "documented": "documented_claim"}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        need = {"cause_event_id", "effect_event_id", "mechanism", "claim_type", "evidence_url"}
        if not need <= set(reader.fieldnames or []):
            raise DataError("Claims CSV requires: " + ",".join(sorted(need)))
        rows = list(reader)
    with transaction(con):
        for i, r in enumerate(rows, start=1):
            if r["claim_type"] not in mapping:
                raise DataError(f"Unknown claim_type {r['claim_type']!r} (hypothesis|analogy|documented)")
            refs = []
            url = r["evidence_url"].strip()
            if url:
                # v0.1 'documented' meant "has a URL"; the URL is recorded as supporting, level stays unassessed.
                refs = [{"ref_id": _url_reference(con, url), "stance": "supports" if r["claim_type"] == "documented" else "discusses"}]
            add_claim(con, Claim(f"csv-{r['cause_event_id']}-{r['effect_event_id']}-{i}", r["cause_event_id"],
                                 r["effect_event_id"], mapping[r["claim_type"]], r["mechanism"],
                                 evidence_level=1 if r["claim_type"] == "hypothesis" else 0, data_kind=kind,
                                 curator="csv-import", references=refs))
    return len(rows)


def _url_reference(con: sqlite3.Connection, url: str) -> str:
    import hashlib

    ref_id = "url-" + hashlib.sha1(url.encode()).hexdigest()[:10]
    if con.execute("SELECT 1 FROM reference WHERE ref_id=?", (ref_id,)).fetchone() is None:
        add_reference(con, Reference(ref_id, "web_page", url, url=url, notes="URL from CSV import (unverified)"))
    return ref_id


# ----------------------------------------------------------------------------- reads


def _rows(con: sqlite3.Connection, q: str, args: Sequence[Any] = ()) -> list[dict[str, Any]]:
    return [dict(r) for r in con.execute(q, args)]


def list_events(con: sqlite3.Connection, tag: str | None = None, kind: str | None = None) -> list[dict[str, Any]]:
    evs = _rows(con, "SELECT * FROM event ORDER BY start_year, event_id")
    out = []
    for e in evs:
        e["tags"] = json.loads(e["tags"] or "[]")
        if tag and tag not in e["tags"]:
            continue
        if kind and e["data_kind"] != kind:
            continue
        e["references"] = _rows(con, """SELECT r.*, er.locator FROM event_reference er JOIN reference r USING(ref_id)
                                        WHERE er.event_id=?""", (e["event_id"],))
        out.append(e)
    return out


def get_claim(con: sqlite3.Connection, claim_id: str) -> dict[str, Any]:
    rows = _rows(con, "SELECT * FROM claim WHERE claim_id=?", (claim_id,))
    if not rows:
        raise DataError(f"Unknown claim {claim_id!r}")
    c = rows[0]
    c["references"] = _rows(con, """SELECT cr.ref_id, cr.stance, cr.locator, r.title, r.authors, r.year, r.doi, r.url,
                                    r.ref_type, r.verification FROM claim_reference cr JOIN reference r USING(ref_id)
                                    WHERE cr.claim_id=?""", (claim_id,))
    c["alternatives"] = _rows(con, "SELECT description, ref_id FROM alternative_explanation WHERE claim_id=?", (claim_id,))
    c["evidence_label"] = EVIDENCE_LEVELS[c["evidence_level"]]
    c["status_log"] = _rows(con, "SELECT * FROM claim_status_log WHERE claim_id=? ORDER BY log_id", (claim_id,))
    return c


def list_claims(con: sqlite3.Connection, event_ids: Iterable[str] | None = None) -> list[dict[str, Any]]:
    ids = [r["claim_id"] for r in con.execute("SELECT claim_id FROM claim ORDER BY claim_id")]
    claims = [get_claim(con, i) for i in ids]
    if event_ids is not None:
        s = set(event_ids)
        claims = [c for c in claims if c["source_event_id"] in s or c["target_event_id"] in s]
    return claims


def list_mechanisms(con: sqlite3.Connection) -> list[dict[str, Any]]:
    return _rows(con, "SELECT * FROM mechanism ORDER BY mechanism_id")


def list_facts(con: sqlite3.Connection, tag: str | None = None) -> list[dict[str, Any]]:
    facts = _rows(con, """SELECT f.*, r.title AS ref_title, r.authors AS ref_authors, r.year AS ref_year, r.url AS ref_url,
                          r.doi AS ref_doi, r.verification FROM fact f JOIN reference r USING(ref_id) ORDER BY f.year""")
    for f in facts:
        f["tags"] = json.loads(f["tags"] or "[]")
    return [f for f in facts if tag is None or tag in f["tags"]]


def list_references(con: sqlite3.Connection) -> list[dict[str, Any]]:
    return _rows(con, "SELECT * FROM reference ORDER BY authors, year")


def graph(con: sqlite3.Connection) -> dict[str, Any]:
    nodes = [{"id": e["event_id"], "label": e["title"], "year": e["start_year"], "category": e["category"],
              "kind": e["data_kind"], "n_refs": len(e["references"])} for e in list_events(con)]
    edges = [{"id": c["claim_id"], "source": c["source_event_id"], "target": c["target_event_id"],
              "relation": c["relation"], "mechanism": c["mechanism_id"], "evidence_level": c["evidence_level"],
              "status": c["status"], "statement": c["statement"]} for c in list_claims(con)]
    return {"nodes": nodes, "edges": edges}


def audit(con: sqlite3.Connection) -> list[dict[str, str]]:
    """Weak spots that rules do not forbid but a researcher should see."""
    issues = []
    for c in list_claims(con):
        if c["relation"] in ("causal_hypothesis", "contributing_condition") and not c["alternatives"]:
            issues.append({"claim": c["claim_id"], "issue": "causal claim without recorded alternative explanations"})
        if c["evidence_level"] >= 3 and all(r["verification"] == "unverified" for r in c["references"] if r["stance"] == "supports"):
            issues.append({"claim": c["claim_id"], "issue": "supporting references are all unverified"})
        if c["evidence_level"] >= 3 and not any(r["stance"] in ("contradicts", "qualifies") for r in c["references"]):
            issues.append({"claim": c["claim_id"], "issue": "no contradicting or qualifying reference recorded"})
    for e in list_events(con):
        if not e["references"]:
            issues.append({"event": e["event_id"], "issue": "event has no reference"})
    return issues
