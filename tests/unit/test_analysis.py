import numpy as np
import pandas as pd
import pytest

from chronos.analysis import analogs, causal, patterns
from chronos.storage import repository as repo
from chronos.storage.repository import IngestBatch, ObsRow, SeriesMeta, SeriesPayload, SourceInfo
from chronos.timeseries import Panel, TimeSeries


def test_structural_breaks_finds_known_regime_change():
    rng = np.random.default_rng(0)
    years = np.arange(1900, 2000)
    x = np.where(years < 1950, 0.01, 0.04) + rng.normal(0, 0.005, 100)
    res = patterns.structural_breaks(years, x, max_breaks=3, min_size=8)
    assert res["breaks"] and abs(res["breaks"][0] - 1950) <= 2
    assert res["segments"][0]["mean"] == pytest.approx(0.01, abs=0.003)


def test_no_break_in_stationary_noise():
    rng = np.random.default_rng(1)
    res = patterns.structural_breaks(np.arange(1800, 1900), rng.normal(0, 1, 100), max_breaks=3, min_size=10)
    assert res["breaks"] == []


def test_similar_trajectories_finds_planted_copy():
    rng = np.random.default_rng(2)
    base = np.exp(np.cumsum(rng.normal(0.02, 0.03, 80)))
    other = np.exp(np.cumsum(rng.normal(0.02, 0.03, 80)))
    other[40:61] = other[39] * base[10:31] / base[9]  # copy of A's 1910-1930 growth path into B 1940-1960
    panel = Panel("x", {"A": TimeSeries(np.arange(1900, 1980), base, entity="A"),
                        "B": TimeSeries(np.arange(1900, 1980), other, entity="B")})
    res = patterns.similar_trajectories(panel, "A", 1910, 1930, k=3)
    assert res["matches"][0]["entity"] == "B" and res["matches"][0]["start"] == 1940
    assert res["matches"][0]["distance"] < 1e-9


def test_event_enrichment_detects_planted_sequence():
    events = [("epidemic", y) for y in range(1000, 2000, 100)] + [("economic_crisis", y + 3) for y in range(1000, 2000, 100)]
    res = patterns.event_sequence_enrichment(events, "epidemic", "economic_crisis", 5, n_perm=500)
    assert res["observed_pairs"] == 10 and res["p_value_one_sided"] < 0.01
    few = patterns.event_sequence_enrichment([("war", 1900)], "war", "epidemic", 5)
    assert "no test possible" in few["note"]


def _store(con, indicator, source, values_by_code):
    src = SourceInfo(source, source)
    payloads = [SeriesPayload(SeriesMeta(indicator, f"Country {c}", source, "ds", indicator, entity_code=c),
                              [ObsRow(y, v) for y, v in vals]) for c, vals in values_by_code.items()]
    repo.ingest(con, IngestBatch(src, "ds", "mem", None, payloads))


def test_analog_engine_joins_sources_and_respects_cutoff(con):
    rng = np.random.default_rng(3)
    codes = [f"C{i:02d}" for i in range(12)]
    gdp, pop, co2, dem, rule, life = {}, {}, {}, {}, {}, {}
    for i, c in enumerate(codes):
        yrs = range(1900, 2021)
        g = 1000 * np.exp(np.cumsum(rng.normal(0.015 + 0.002 * i, 0.03, len(yrs))))
        gdp[c] = list(zip(yrs, g, strict=True))
        pop[c] = [(y, 1e6 * (1.01 ** (y - 1900))) for y in yrs]
        co2[c] = [(y, 0.5 + i * 0.1 + (y - 1900) * 0.01) for y in yrs]
        dem[c] = [(y, min(1, 0.05 * i)) for y in yrs]
        rule[c] = [(y, min(1, 0.06 * i)) for y in yrs]
        life[c] = [(y, 40 + i + 0.2 * (y - 1900)) for y in yrs]
    _store(con, "gdp_per_capita", "owid", gdp)
    _store(con, "population", "owid", pop)
    _store(con, "co2_per_capita", "owid", co2)
    _store(con, "v2x_polyarchy", "vdem", dem)
    _store(con, "v2x_rule", "vdem", rule)
    _store(con, "e_pelifeex", "vdem", life)
    table, info = analogs.build_state_table(con, min_year=1920)
    assert info["n_entities"] == 12 and not info["missing_dimensions"]
    eng = analogs.AnalogEngine(table)
    res = eng.find("C05", 2000, k=5, outcome_horizon=10)
    assert res["analogs"] and all(a["year"] <= 1990 for a in res["analogs"])
    assert all(not (a["code"] == "C05" and abs(a["year"] - 2000) < 25) for a in res["analogs"])
    bt = analogs.backtest(eng, horizon=10, origins=range(1950, 2001, 10), k=5)
    assert bt["n_pairs"] > 20 and set(bt["methods"]) == {"analog_median", "naive_no_change", "own_trend", "global_mean"}


def _panel_with_effect(effect: float, seed: int = 0) -> tuple[pd.DataFrame, dict[str, int]]:
    rng = np.random.default_rng(seed)
    rows, treat = [], {}
    for u in range(60):
        g = 1980 + (u % 20) if u < 40 else None
        if g:
            treat[f"U{u}"] = g
        level = rng.normal(0, 0.2)
        for y in range(1960, 2011):
            level += 0.02 + rng.normal(0, 0.01)
            d = 1.0 if g and y >= g else 0.0
            rows.append({"unit": f"U{u}", "year": y, "y": level + effect * d * min(y - g + 1, 5) / 5 if g else level,
                         "x": d + rng.normal(0, 0.1), "treat": d})
    return pd.DataFrame(rows), treat


def test_event_study_recovers_planted_effect_and_rejects_null():
    df, treat = _panel_with_effect(0.15)
    es = causal.event_study(df, "y", treat, window=(-5, 8), n_perm=30)
    assert es["average_post_effect"] > 0.05 and es["post_effect_p"] < 0.01
    assert es["parallel_pretrends_not_rejected"]
    df0, treat0 = _panel_with_effect(0.0, seed=1)
    es0 = causal.event_study(df0, "y", treat0, window=(-5, 8), n_perm=30)
    assert es0["post_effect_p"] > 0.01


def test_assess_ladder_labels_and_threats():
    df, treat = _panel_with_effect(0.15)
    rep = causal.assess(df, "x", "y", expected_sign=1, treat_year=treat, n_perm=30, granger_min_obs=40)
    assert rep["evidence_tier"]["tier"] in (3, 4)
    assert any("confounders" in t for t in rep["threats_to_validity"])
    assert "does_not_show" in rep["evidence_tier"]


def test_first_crossing_excludes_always_treated():
    df = pd.DataFrame({"unit": ["a"] * 10 + ["b"] * 10, "year": list(range(10)) * 2,
                       "v": [0, 0, 0, 2, 2, 2, 2, 2, 2, 2] + [3] * 10})
    assert causal.first_crossing_years(df, "v", 2, sustain=5) == {"a": 3}


def test_hypothesis_registry_preregistration_and_bh(con):
    causal.register_hypothesis(con, "h1", "x -> y", {"x": "a", "y": "b"})
    causal.register_hypothesis(con, "h1", "x -> y", {"x": "a", "y": "b"})  # identical: OK
    with pytest.raises(Exception, match="different specification"):
        causal.register_hypothesis(con, "h1", "x -> y", {"x": "a", "y": "c"})
    for p in (0.01, 0.04, 0.03):
        causal.record_test(con, "h1", "fe", p, 0.1, "association", {})
    rows = causal.registry(con)
    assert len(rows) == 3 and all(r["n_tests_on_hypothesis"] == 3 for r in rows)
    assert max(r["p_bh_adjusted"] for r in rows) == pytest.approx(0.04)


def test_meanvar_model_detects_volatility_regime():
    rng = np.random.default_rng(4)
    years = np.arange(1900, 2000)
    x = np.concatenate([rng.normal(0.02, 0.08, 40), rng.normal(0.05, 0.01, 60)])
    res = patterns.structural_breaks(years, x, max_breaks=3, min_size=8, model="meanvar")
    assert res["breaks"] and abs(res["breaks"][0] - 1940) <= 3
