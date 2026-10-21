import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="session", autouse=True)
def warehouse_present():
    """The warehouse is built from real downloads (src/sources.py + src/setup_data.py)
    and committed, so tests run against the real data without downloading it."""
    path = os.path.join(ROOT, "data", "ops.duckdb")
    assert os.path.exists(path), "data/ops.duckdb is missing; run python -m src.sources && python -m src.setup_data"
    yield
