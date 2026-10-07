from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from credit_ranking.config import DEFAULT_CONFIG_PATH, load_config
from credit_ranking.pipeline import TrainResult, train
from credit_ranking.schema import BILL_AMT, PAY_AMT, PAY_STATUS
from credit_ranking.synthetic import generate_synthetic


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


@pytest.fixture(scope="session")
def fast_config_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Default config with fewer trees so integration tests stay quick."""
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    raw["models"]["lightgbm"]["n_estimators"] = 60
    path = tmp_path_factory.mktemp("cfg") / "fast.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data") / "synthetic_1500.csv"
    generate_synthetic(1_500, seed=21).to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def trained_run(
    synthetic_csv: Path, fast_config_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> TrainResult:
    out = tmp_path_factory.mktemp("run")
    return train(synthetic_csv, load_config(fast_config_path), out, data_label="test synthetic")
