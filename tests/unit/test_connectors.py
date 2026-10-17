"""Parser tests for every connector, using small fixtures built in the test.

Fixtures follow the layouts documented in each connector module. For World Bank, OWID Chart
API and GDELT the fixtures are constructed from the documented response structure, not recorded
from a live response (those hosts were unreachable from the development sandbox).
Live tests are in tests/integration/test_live_sources.py (CHRONOS_NETWORK_TESTS=1).
"""

import hashlib
import io

import pandas as pd
import pytest

from chronos.errors import DataError, SourceError
from chronos.ingestion import files, gdelt, owid_github, owid_grapher, seshat, user_csv, worldbank
from chronos.ingestion.http import FetchResult
from chronos.storage import repository as repo


def fr(content: bytes, url: str = "mem://f.csv") -> FetchResult:
    return FetchResult(url, content, hashlib.sha256(content).hexdigest(), "2026-01-01T00:00:00+00:00", '"etag1"', None, None)


def test_owid_github_parse_with_codebook_and_derived_gdp_pc(con):
    data = b"country,year,iso_code,population,gdp,co2\nWorld,2000,OWID_WRL,6e9,6e13,25000\nWorld,2001,OWID_WRL,6.1e9,,25500\nFrance,2000,FRA,6e7,1.5e12,400\n"
    cb = b"column,title,description,unit,source\npopulation,Population,desc,people,Various\ngdp,GDP,desc,int-$,Maddison 2023\nco2,CO2,desc,Mt,GCB\n"
    b = owid_github.parse(fr(data), fr(cb), owid_github.DATASETS["co2"], ["population", "gdp", "co2"], None)
    rep = repo.ingest(con, b)
    assert rep.n_series == 8  # 3 World + 3 France + 2 derived gdp_per_capita
    s = repo.load_series(con, "gdp", "World")
    assert s.unit == "int-$" and "Maddison 2023" in con.execute("SELECT description FROM series WHERE indicator='gdp' AND entity='World'").fetchone()[0]
    pc = repo.load_series(con, "gdp_per_capita", "France")
    assert pc.get(2000) == pytest.approx(25000.0) and pc.status == ("derived",)
    assert s.meta["source_version"] == 'etag:"etag1"'


def test_owid_github_rejects_unknown_columns():
    with pytest.raises(DataError):
        owid_github.parse(fr(b"country,year,iso_code,a\nX,2000,XXX,1\n"), None, owid_github.DATASETS["co2"], ["zzz"], None)


def test_owid_grapher_requires_column_choice_when_ambiguous():
    csv = b"Entity,Code,Year,a,b\nWorld,OWID_WRL,2000,1,2\n"
    with pytest.raises(DataError, match="choose one with --column"):
        owid_grapher.parse(fr(csv), None, "x", None, None, None)
    b = owid_grapher.parse(fr(csv), {"columns": {"b": {"titleShort": "B", "unit": "years", "lastUpdated": "2024-05-01"}}},
                           "x", "b", None, "life")
    assert b.series[0].meta.unit == "years" and b.series[0].meta.indicator == "life"
    assert b.source_version == "lastUpdated:2024-05-01"


def test_worldbank_parse_pages_and_errors():
    page = [{"page": 1, "pages": 1, "per_page": 50, "total": 3, "lastupdated": "2025-07-01"},
            [{"indicator": {"id": "SP.POP.TOTL", "value": "Population, total"}, "country": {"id": "1W", "value": "World"},
              "countryiso3code": "WLD", "date": "2023", "value": 8.06e9, "unit": "", "obs_status": "", "decimal": 0},
             {"indicator": {"id": "SP.POP.TOTL", "value": "Population, total"}, "country": {"id": "1W", "value": "World"},
              "countryiso3code": "WLD", "date": "2024", "value": None, "unit": "", "obs_status": "", "decimal": 0},
             {"indicator": {"id": "SP.POP.TOTL", "value": "Population, total"}, "country": {"id": "1W", "value": "World"},
              "countryiso3code": "WLD", "date": "2022", "value": 7.99e9, "unit": "", "obs_status": "E", "decimal": 0}]]
    payloads, last = worldbank.parse_pages([page], {"unit": "", "sourceNote": "note"}, None)
    assert last == "2025-07-01"
    assert [r.year for r in payloads[0].rows] == [2022, 2023]
    assert payloads[0].rows[0].status == "estimated"
    with pytest.raises(SourceError, match="Invalid value"):
        worldbank._check([{"message": [{"id": "120", "key": "Invalid value", "value": "bad"}]}], "u")


def test_gdelt_parse_annual_means():
    payload = {"timeline": [{"series": "Volume Intensity", "data": [
        {"date": "20240101T000000Z", "value": 0.1}, {"date": "20240102T000000Z", "value": 0.3},
        {"date": "20250101T000000Z", "value": 0.5}]}]}
    p = gdelt.parse_timeline(payload, "ai_news", '"AI"')[0]
    assert [r.year for r in p.rows] == [2024, 2025]
    assert p.rows[0].value == pytest.approx(0.2) and p.rows[0].status == "estimated"


def test_seshat_parse_flags_imputed_and_filler(tmp_path):
    imp = pd.DataFrame({"NGA": ["Latium"] * 3, "PolID": ["P1"] * 3, "Time": [-300, -200, -100],
                        "Pop": [5.0, 5.0, 5.5], "Terr": [3.0] * 3, "Cap": [4.0] * 3, "Hier": [2.0] * 3, "Gov": [0.5] * 3,
                        "Infra": [0.5] * 3, "Info": [0.5] * 3, "Money": [0.5] * 3, "uniq": ["y", "n", "y"], "Dupl": ["n"] * 3})
    agg = imp.copy()
    agg.loc[2, "Pop"] = None  # -100 not coded -> imputed
    spc = pd.DataFrame({"NGA": ["Latium"] * 3, "PolID": ["P1"] * 3, "Time": [-300, -200, -100], "SPC": [3.0, 3.0, 3.5],
                        "MilTech": [10, 10, 12], "uniq": ["y", "n", "y"], "Dupl": ["n"] * 3})
    pol = pd.DataFrame({"PolID": ["P1"], "PolName": ["Roman Republic"]})
    path = tmp_path / "eq.xlsx"
    with pd.ExcelWriter(path) as w:
        imp.to_excel(w, sheet_name="ImpSCDat", index=False)
        agg.to_excel(w, sheet_name="AggrSCWarAgriRelig", index=False)
        spc.to_excel(w, sheet_name="SPC_MilTech", index=False)
        pol.to_excel(w, sheet_name="Polities", index=False)
    b = seshat.parse(fr(path.read_bytes(), str(path)))
    pop = next(p for p in b.series if p.meta.indicator == "seshat_pop")
    assert [r.status for r in pop.rows] == ["observed", "filler", "imputed"]
    raw = next(p for p in b.series if p.meta.indicator == "seshat_raw_pop")
    assert [r.year for r in raw.rows] == [-300, -200]
    assert pop.meta.frequency == "century" and "Roman Republic" in pop.rows[0].note


def test_maddison_pwt_cow_file_layouts(tmp_path):
    mad = pd.DataFrame({"countrycode": ["FRA", "FRA"], "country": ["France", "France"], "region": ["WE", "WE"],
                        "year": [1900, 1901], "gdppc": [3000.0, 3100.0], "pop": [40000.0, None]})
    p = tmp_path / "mpd.xlsx"
    with pd.ExcelWriter(p) as w:
        pd.DataFrame({"a": [1]}).to_excel(w, sheet_name="Notes", index=False)
        mad.to_excel(w, sheet_name="Full data", index=False)
    b = files.import_maddison(str(p))
    ind = {s.meta.indicator: s for s in b.series}
    assert len(ind["maddison_gdppc"].rows) == 2 and len(ind["maddison_pop"].rows) == 1
    pwt = pd.DataFrame({"countrycode": ["USA"], "country": ["United States"], "year": [2019], "rgdpe": [2e7], "hc": [3.7]})
    p2 = tmp_path / "pwt.xlsx"
    with pd.ExcelWriter(p2) as w:
        pwt.to_excel(w, sheet_name="Data", index=False)
    assert {s.meta.indicator for s in files.import_pwt(str(p2)).series} == {"pwt_rgdpe", "pwt_hc"}
    cow = pd.DataFrame({"stateabb": ["USA", "USA"], "ccode": [2, 2], "year": [1900, 1901], "milex": [-9, 100],
                        "cinc": [0.1, 0.11], "version": [2021, 2021]})
    p3 = tmp_path / "nmc.csv"
    cow.to_csv(p3, index=False)
    nb = files.import_cow_nmc(str(p3))
    milex = next(s for s in nb.series if s.meta.indicator == "cow_milex")
    assert [r.year for r in milex.rows] == [1901]  # -9 = missing
    with pytest.raises(DataError):
        bad = tmp_path / "bad.csv"
        pd.DataFrame({"x": [1]}).to_csv(bad, index=False)
        files.import_cow_nmc(str(bad))


def test_user_csv_requires_provenance_and_flags_synthetic(tmp_path, con):
    p = tmp_path / "x.csv"
    p.write_text("Year,Value\n2000,1\n2001,abc\n2002,3\n")
    with pytest.raises(DataError):
        user_csv.import_csv(str(p), "x", "W", "  ")
    b = user_csv.import_csv(str(p), "x", "W", "https://example.org/data", synthetic=True)
    repo.ingest(con, b)
    s = repo.load_series(con, "x", "W")
    assert s.data_kind == "synthetic" and len(s) == 2


def test_long_csv_with_entity_column(tmp_path):
    p = tmp_path / "x.csv"
    pd.DataFrame({"Entity": ["A", "A", "B"], "Year": [1, 2, 1], "Value": [1, 2, 3]}).to_csv(p, index=False)
    b = user_csv.import_csv(str(p), "x", None, "https://example.org")
    assert sorted(s.meta.entity for s in b.series) == ["A", "B"]


def test_owid_github_entity_filter_errors():
    data = io.BytesIO(b"country,year,iso_code,population\nWorld,2000,OWID_WRL,1\n").getvalue()
    with pytest.raises(DataError, match="Entities not in"):
        owid_github.parse(fr(data), None, owid_github.DATASETS["co2"], ["population"], ["Mars"])
