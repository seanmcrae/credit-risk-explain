from __future__ import annotations

import pandas as pd
import pytest

from credit_ranking.schema import BILL_AMT, PAY_AMT, PAY_STATUS


def make_account(**overrides: float) -> dict[str, float]:
    """One canonical row with neutral defaults; override any column by name."""
    row: dict[str, float] = {
        "id": 1,
        "limit_bal": 100_000.0,
        "sex": 2,
        "age": 35,
        "education": 2,
        "marriage": 1,
        "default_next_month": 0,
    }
    row.update({c: 0 for c in PAY_STATUS})
    row.update({c: 10_000.0 for c in BILL_AMT})
    row.update({c: 2_000.0 for c in PAY_AMT})
    row.update(overrides)
    return row


@pytest.fixture
def account_frame() -> pd.DataFrame:
    return pd.DataFrame([make_account(id=i) for i in range(1, 4)])
