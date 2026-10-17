"""Every Streamlit page renders without an exception on a small fixture database."""

from pathlib import Path

import pytest

from chronos.knowledge import store
from chronos.storage.db import connect

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
APP = Path(__file__).resolve().parents[2] / "src" / "chronos" / "ui" / "app.py"


@pytest.fixture()
def ui_db(tmp_path, monkeypatch):
    from tests.unit.test_state_scenarios_research import BUNDLE, world_fixture

    path = tmp_path / "ui.sqlite"
    con = connect(path)
    world_fixture(con)
    store.import_json(con, BUNDLE)
    con.close()
    monkeypatch.setenv("CHRONOS_DB", str(path))
    return path


PAGES = ["Overview & sources", "Time series", "Knowledge graph", "Backtesting", "Patterns", "Causal hypotheses",
         "Civilization state", "Scenarios", "Research query"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(ui_db, page):
    at = streamlit_testing.AppTest.from_file(str(APP), default_timeout=240)
    at.run()
    assert not at.exception
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]
