"""Live checks against real sources. Run with CHRONOS_NETWORK_TESTS=1 (skipped otherwise).

In the development sandbox only raw.githubusercontent.com was reachable, so the OWID GitHub,
Seshat and V-Dem connectors were verified live; World Bank, OWID Chart API and GDELT were not.
"""

import pytest

from chronos.config import load_settings
from chronos.ingestion import gdelt, owid_github, owid_grapher, seshat, worldbank
from chronos.storage import repository as repo

pytestmark = pytest.mark.network


@pytest.fixture()
def settings(tmp_path):
    return load_settings(cache_dir=str(tmp_path / "cache"))


def test_owid_github_co2_world(settings, con):
    b = owid_github.fetch_dataset(settings, "co2", ["population", "gdp", "co2"], ["World"])
    repo.ingest(con, b)
    s = repo.load_series(con, "population", "World")
    assert s.first_year <= 1800 and s.last_year >= 2023 and s.unit == "people"


def test_seshat_equinox(settings, con):
    repo.ingest(con, seshat.fetch_equinox(settings))
    s = repo.load_series(con, "seshat_spc", "Latium")
    assert s.step() == 100 and s.first_year < 0


def test_worldbank_population(settings, con):
    repo.ingest(con, worldbank.fetch_indicator(settings, "SP.POP.TOTL", "WLD", "1960:2025"))
    s = repo.load_series(con, "SP.POP.TOTL", "World")
    assert s.first_year == 1960 and s.values[-1] > 7e9


def test_owid_chart_life_expectancy(settings, con):
    b = owid_grapher.fetch_chart(settings, "life-expectancy", entities=["World"], indicator="life_expectancy")
    repo.ingest(con, b)
    assert repo.load_series(con, "life_expectancy", "World").last_year >= 2020


def test_gdelt_timeline(settings, con):
    b = gdelt.fetch_timeline(settings, '"artificial intelligence"', "20240101000000", "20240301000000", "gdelt_ai")
    assert b.series and b.series[0].rows
