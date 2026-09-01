import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.setup_data import build_warehouse, train_credit_risk_model


@pytest.fixture(scope="session", autouse=True)
def generated_data():
    """Ensures the synthetic warehouse and risk model exist before any test
    runs, by generating them for real (no mocking of data generation)."""
    build_warehouse()
    train_credit_risk_model()
    yield
