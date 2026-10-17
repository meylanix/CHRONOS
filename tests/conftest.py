import importlib.util
import os
from pathlib import Path

import pytest

from chronos.storage.db import connect

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def con(tmp_path):
    c = connect(tmp_path / "test.sqlite")
    yield c
    c.close()


@pytest.fixture(scope="session")
def legacy_v01():
    spec = importlib.util.spec_from_file_location("chronos_v01", ROOT / "legacy" / "chronos_v01.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pytest_collection_modifyitems(config, items):
    if os.environ.get("CHRONOS_NETWORK_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="network test: set CHRONOS_NETWORK_TESTS=1")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
