import json

import pytest

from chronos.errors import ValidationRuleError
from chronos.knowledge import store
from chronos.knowledge.store import Claim, Event, Mechanism, Reference


def seed(con):
    store.add_reference(con, Reference("r1", "journal_article", "A study", authors="X", year=2011))
    store.add_reference(con, Reference("r2", "web_page", "A page", url="https://example.org"))
    store.add_mechanism(con, Mechanism("m1", "Cheaper copies", "desc", "informational"))
    store.add_event(con, Event("a", "Cause", "invention", 1450, year_uncertainty=5, references=[{"ref_id": "r1"}]))
    store.add_event(con, Event("b", "Effect", "cultural", 1517))
    store.add_event(con, Event("early", "Earlier", "other", 1300))


def test_effect_cannot_precede_cause(con):
    seed(con)
    with pytest.raises(ValidationRuleError, match="R2"):
        store.add_claim(con, Claim("c", "a", "early", "causal_hypothesis", "a caused early"))
    # an analogy has no temporal constraint
    store.add_claim(con, Claim("c2", "a", "early", "analogy", "similar"))


def test_sequence_is_not_evidence(con):
    seed(con)
    with pytest.raises(ValidationRuleError, match="R3"):
        store.add_claim(con, Claim("c", "a", "b", "temporal_sequence", "b after a", evidence_level=2))


def test_evidence_levels_need_references(con):
    seed(con)
    with pytest.raises(ValidationRuleError, match="R4"):
        store.add_claim(con, Claim("c", "a", "b", "causal_hypothesis", "s", evidence_level=3))
    with pytest.raises(ValidationRuleError, match="R5"):
        store.add_claim(con, Claim("c", "a", "b", "causal_hypothesis", "s", evidence_level=4,
                                   references=[{"ref_id": "r2", "stance": "supports"}]))
    store.add_claim(con, Claim("c", "a", "b", "causal_hypothesis", "s", evidence_level=4, mechanism_id="m1",
                               references=[{"ref_id": "r1", "stance": "supports"}],
                               alternatives=[{"description": "other cause"}]))
    c = store.get_claim(con, "c")
    assert c["evidence_label"].startswith("supported by quantitative") and len(c["alternatives"]) == 1


def test_documented_claim_needs_support_and_status_rule(con):
    seed(con)
    with pytest.raises(ValidationRuleError, match="R6"):
        store.add_claim(con, Claim("c", "a", "b", "documented_claim", "s"))
    with pytest.raises(ValidationRuleError, match="R7"):
        store.add_claim(con, Claim("c", "a", "b", "causal_hypothesis", "s", status="supported", evidence_level=2))


def test_status_changes_are_logged_and_need_reason(con):
    seed(con)
    store.add_claim(con, Claim("c", "a", "b", "causal_hypothesis", "s", evidence_level=1))
    with pytest.raises(ValidationRuleError):
        store.set_claim_status(con, "c", "contested", None, " ")
    store.set_claim_status(con, "c", "contested", 2, "new counter-evidence")
    c = store.get_claim(con, "c")
    assert c["status"] == "contested" and c["status_log"][0]["reason"] == "new counter-evidence"
    with pytest.raises(ValidationRuleError):
        store.set_claim_status(con, "c", "supported", None, "want it")  # level 2 < 3


def test_audit_flags_missing_alternatives_and_refs(con):
    seed(con)
    store.add_claim(con, Claim("c", "a", "b", "causal_hypothesis", "s", evidence_level=1))
    issues = store.audit(con)
    assert {"claim": "c", "issue": "causal claim without recorded alternative explanations"} in issues
    assert any(i.get("event") == "b" for i in issues)


def test_json_bundle_import_is_atomic(con, tmp_path):
    bundle = {"references": [{"ref_id": "r", "ref_type": "book", "title": "T"}],
              "events": [{"event_id": "x", "title": "X", "category": "other", "start_year": 1900}],
              "claims": [{"claim_id": "bad", "source_event_id": "x", "target_event_id": "missing",
                          "relation": "analogy", "statement": "s"}]}
    p = tmp_path / "b.json"
    p.write_text(json.dumps(bundle))
    with pytest.raises(ValidationRuleError):
        store.import_json(con, p)
    assert con.execute("SELECT COUNT(*) FROM event").fetchone()[0] == 0


def test_v01_csv_formats(con, tmp_path):
    ev = tmp_path / "e.csv"
    ev.write_text("event_id,year,title,description,evidence_url\np,1450,P,d,https://a.org\nw,1989,W,d,\n")
    cl = tmp_path / "c.csv"
    cl.write_text("cause_event_id,effect_event_id,mechanism,claim_type,evidence_url\np,w,mech,documented,https://b.org\n")
    assert store.import_events_csv_v01(con, str(ev), "example") == 2
    assert store.import_claims_csv_v01(con, str(cl), "example") == 1
    c = store.list_claims(con)[0]
    assert c["relation"] == "documented_claim" and c["evidence_level"] == 0 and c["data_kind"] == "example"
