import json
from pathlib import Path

import numpy as np
import pytest

from chronos.errors import ChronosError
from chronos.knowledge import store
from chronos.research.transitions import build_dossier, dossier_markdown
from chronos.scenarios import engine as E
from chronos.scenarios.report import build_report
from chronos.state.civilization import build_state, freshness, load_catalog, popweighted
from chronos.storage import repository as repo
from chronos.storage.repository import IngestBatch, ObsRow, SeriesMeta, SeriesPayload, SourceInfo

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "data" / "curated" / "tech_transitions.json"
OWID = SourceInfo("owid", "OWID test fixture")


def put(con, indicator, entity, pairs, source=OWID, code=None, unit=None):
    meta = SeriesMeta(indicator, entity, source.source_id, "fixture", indicator, entity_code=code, unit=unit)
    repo.ingest(con, IngestBatch(source, "fixture", "mem", None, [SeriesPayload(meta, [ObsRow(y, v) for y, v in pairs])]))


def world_fixture(con):
    """Synthetic world aggregates with known growth rates (test fixture, not real data)."""
    yrs = range(1960, 2025)
    put(con, "population", "World", [(y, 3e9 * 1.015 ** (y - 1960)) for y in yrs], code="OWID_WRL", unit="people")
    put(con, "gdp", "World", [(y, 1e13 * 1.035 ** (y - 1960)) for y in range(1960, 2023)], code="OWID_WRL")
    put(con, "primary_energy_consumption", "World", [(y, 5e4 * 1.02 ** (y - 1960)) for y in yrs], code="OWID_WRL")
    put(con, "co2", "World", [(y, 1e4 * 1.01 ** (y - 1960)) for y in yrs], code="OWID_WRL")
    put(con, "cumulative_co2", "World", [(y, 5e5 + 1e4 * (y - 1960)) for y in yrs], code="OWID_WRL")
    put(con, "temperature_change_from_ghg", "World", [(y, 0.3 + 0.02 * (y - 1960)) for y in yrs], code="OWID_WRL", unit="°C")


# ----------------------------------------------------------------------------- knowledge bundle


def test_curated_bundle_imports_and_respects_rules(con):
    counts = store.import_json(con, BUNDLE)
    assert counts["claims"] >= 15 and counts["references"] >= 25
    doc = json.loads(BUNDLE.read_text())
    for c in store.list_claims(con):
        if c["evidence_level"] >= 3:
            assert any(r["stance"] == "supports" for r in c["references"]), c["claim_id"]
    for h in store.list_outlook_hypotheses(con):
        assert h["falsified_if"].strip()
    assert {r["verification"] for r in doc["references"]} <= {"unverified", "metadata_checked"}  # nothing claims expert review


# ----------------------------------------------------------------------------- state model


def test_freshness_labels():
    assert freshness(2025, 2026)[0] == "CURRENT"
    s, note = freshness(2022, 2026)
    assert s == "LAGGED" and "NOT a 2026 value" in note
    assert freshness(2000, 2026)[0] == "STALE"


def test_state_reports_no_data_and_lagged(con):
    world_fixture(con)
    state = build_state(con, load_catalog(ROOT / "config" / "indicators.toml"), 2026)
    econ = {e["key"]: e for e in state["domains"]["economy"]}
    assert econ["world_gdp"]["status"] == "LAGGED" and econ["world_gdp"]["year"] == 2022
    tech = {e["key"]: e for e in state["domains"]["technology_ai"]}
    assert tech["internet_users_wb"]["status"] == "NO DATA" and "worldbank" in tech["internet_users_wb"]["load_hint"]
    pop = {e["key"]: e for e in state["domains"]["demography"]}["world_population"]
    assert pop["nowcast"]["label"].startswith("NOWCAST") and pop["nowcast"]["year"] == 2026


def test_popweighted_mean_and_share(con):
    put(con, "population", "A", [(2000, 100.0)], code="AAA")
    put(con, "population", "B", [(2000, 300.0)], code="BBB")
    put(con, "population", "World", [(2000, 400.0)], code="OWID_WRL")
    vdem = SourceInfo("vdem", "V-Dem fixture")
    put(con, "v2x_polyarchy", "A", [(2000, 0.8)], source=vdem, code="AAA")
    put(con, "v2x_polyarchy", "B", [(2000, 0.4)], source=vdem, code="BBB")
    m = popweighted(con, "v2x_polyarchy", "vdem")
    assert m.get(2000) == pytest.approx((100 * 0.8 + 300 * 0.4) / 400)
    sh = popweighted(con, "v2x_polyarchy", "vdem", threshold=0.5)
    assert sh.get(2000) == pytest.approx(0.25)
    assert m.meta["coverage"][2000] == pytest.approx(1.0)


# ----------------------------------------------------------------------------- scenarios


SPEC = ROOT / "scenarios" / "world_2026.toml"


def test_triangular_ppf_bounds_and_monotonic():
    xs = [E._tri_ppf(u, -1.0, 0.5, 2.0) for u in np.linspace(0, 1, 11)]
    assert xs[0] == pytest.approx(-1.0) and xs[-1] == pytest.approx(2.0)
    assert all(a <= b for a, b in zip(xs, xs[1:], strict=False))


def test_lhs_stratifies():
    u = E.lhs(50, 3, np.random.default_rng(0))
    for j in range(3):
        assert sorted((u[:, j] * 50).astype(int)) == list(range(50))


def test_kaya_identity_and_anchoring(con):
    world_fixture(con)
    model, scens = E.load_spec(SPEC)
    base = E.load_base(con, model)
    assert base.year == 2022  # last year with GDP
    s = scens[0]
    E.apply_derivations(s, base)
    pop_rate = s.params["pop_growth"].periods[0].central
    assert pop_rate == pytest.approx(1.5, abs=1e-6)  # derived from the fixture's 1.5%/yr
    sim = E.simulate(base, s, 2040)
    np.testing.assert_allclose(sim["gdp"], sim["population"] * sim["gdp_per_capita"])
    # anchored: 2023 and 2024 population equal observations
    i = list(sim["year"]).index(2024)
    assert sim["population"][i] == pytest.approx(3e9 * 1.015 ** 64)
    assert sim["co2"][i] == pytest.approx(1e4 * 1.01 ** 64)
    # warming continues from the last observed value via TCRE
    assert sim["warming"][i] == pytest.approx(0.3 + 0.02 * 64)
    j = list(sim["year"]).index(2025)
    assert sim["warming"][j] == pytest.approx(sim["warming"][i] + 0.45 * sim["co2"][j] / 1e6)


def test_spec_validation_rejects_inverted_ranges(tmp_path):
    bad = SPEC.read_text().replace("{ until = 2035, low = 1.4, central = 2.0, high = 2.6 }",
                                   "{ until = 2035, low = 2.4, central = 2.0, high = 2.6 }")
    p = tmp_path / "bad.toml"
    p.write_text(bad)
    with pytest.raises(ChronosError, match="low <= central <= high"):
        E.load_spec(p)


def test_signposts_pending_consistent_inconsistent(con):
    world_fixture(con)
    sps = [
        {"label": "future", "indicator": "co2", "source": "owid", "entity": "World", "metric": "cagr", "start": 2026,
         "end": 2032, "op": "<=", "threshold": -2.0},
        {"label": "past ok", "indicator": "co2", "source": "owid", "entity": "World", "metric": "cagr", "start": 2010,
         "end": 2020, "op": "between", "low": 0.5, "high": 1.5},
        {"label": "past bad", "indicator": "co2", "source": "owid", "entity": "World", "metric": "cagr", "start": 2010,
         "end": 2020, "op": "<=", "threshold": 0.0},
        {"label": "missing", "indicator": "nothing", "source": "owid", "entity": "World", "metric": "level", "start": 2020,
         "end": 2020, "op": ">=", "threshold": 1},
    ]
    st = [r["status"] for r in E.evaluate_signposts(con, sps)]
    assert st == ["PENDING", "CONSISTENT", "INCONSISTENT", "NO DATA"]


def test_full_scenario_report_has_required_parts(con):
    world_fixture(con)
    store.import_json(con, BUNDLE)
    rep = build_report(con, str(SPEC), with_evidence=False)
    assert "No probabilities" in rep["disclaimer"]
    assert rep["validation"]["gdp_per_capita"][100]["status"] == "UNVALIDATED"
    for s in rep["scenarios"]:
        assert s["assumptions"] and s["mechanisms"]["linked"] and s["signposts"] and s["wildcards"]
        assert [p["horizon"] for p in s["projections"]] == [5, 10, 20, 50, 100]
        sens = s["critical_dependencies"]["computed"]["2050"]["warming"]
        assert sens[0]["swing"] >= sens[-1]["swing"]
        for row in s["projections"]:
            assert row["co2"]["p05"] <= row["co2"]["central"] * 1.0001 or row["co2"]["p05"] <= row["co2"]["p95"]


# ----------------------------------------------------------------------------- research dossier


def test_dossier_separates_facts_interpretations_and_gaps(con):
    store.import_json(con, BUNDLE)
    d = build_dossier(con, 2040, None)
    assert d["outlook"]["status"] == "NOT RUN"
    assert d["timelines"]["printing"] and d["timelines"]["genai"]
    assert any(m["mechanism_id"] == "M_info_cost" for m in d["similarities"]["shared_by_all_three"])
    assert any("Mechanism transfer" in a for a in d["conclusions"]["assumptions"])
    assert d["data_gaps"]
    md = dossier_markdown(d)
    for section in ("## 1–2. Timelines", "## 6. Counter-hypotheses", "## 9. References", "## 10. Data gaps"):
        assert section in md
    jl = next(h for h in d["outlook_hypotheses"] if h["oh_id"] == "oh_jcurve_lag")
    assert jl["status"] in ("PENDING", "NO DATA")


def test_dossier_requires_knowledge(con):
    with pytest.raises(Exception, match="kg import"):
        build_dossier(con, 2040, None)
