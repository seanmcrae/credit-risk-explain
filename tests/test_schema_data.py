import hashlib

import pandas as pd
import pandera.errors
import pytest

from credit_ranking.data import audit_groups, normalize_uci_columns
from credit_ranking.download import verify_checksum
from credit_ranking.schema import COLUMNS, validate
from tests.conftest import make_account


def test_valid_frame_passes(account_frame: pd.DataFrame) -> None:
    out = validate(account_frame)
    assert set(out.columns) == set(COLUMNS)


def test_schema_reports_out_of_domain_values(account_frame: pd.DataFrame) -> None:
    bad = account_frame.copy()
    bad.loc[0, "sex"] = 3
    bad.loc[1, "pay_amt_1"] = -5.0
    with pytest.raises(pandera.errors.SchemaErrors) as exc:
        validate(bad)
    failed = set(exc.value.failure_cases["column"])
    assert {"sex", "pay_amt_1"} <= failed


def test_schema_rejects_unexpected_columns(account_frame: pd.DataFrame) -> None:
    with pytest.raises(pandera.errors.SchemaErrors):
        validate(account_frame.assign(extra=1))


def test_target_is_optional_for_scoring(account_frame: pd.DataFrame) -> None:
    validate(account_frame.drop(columns="default_next_month"))


def test_normalize_uci_columns_maps_pay0_to_most_recent_month() -> None:
    canonical = make_account(pay_status_1=2, pay_status_2=1)
    raw = {
        "ID": 1,
        "LIMIT_BAL": 1,
        "SEX": 1,
        "EDUCATION": 1,
        "MARRIAGE": 1,
        "AGE": 30,
        "PAY_0": 2,
        "PAY_2": 1,
        "default payment next month": 1,
    }
    raw |= {f"PAY_{m}": 0 for m in range(3, 7)}
    raw |= {f"BILL_AMT{m}": canonical[f"bill_amt_{m}"] for m in range(1, 7)}
    raw |= {f"PAY_AMT{m}": canonical[f"pay_amt_{m}"] for m in range(1, 7)}
    out = normalize_uci_columns(pd.DataFrame([raw]))
    assert out.loc[0, "pay_status_1"] == 2
    assert out.loc[0, "pay_status_2"] == 1
    assert out.loc[0, "default_next_month"] == 1


def test_normalize_rejects_foreign_table() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        normalize_uci_columns(pd.DataFrame({"a": [1]}))


def test_audit_groups_decodes_and_pools_unknown_codes() -> None:
    df = pd.DataFrame(
        [
            make_account(sex=1, age=24, education=5, marriage=0),
            make_account(sex=2, age=60, education=1, marriage=2),
        ]
    )
    groups = audit_groups(df)
    assert groups.iloc[0].tolist() == ["male", "18-24", "other/unknown", "other/unknown"]
    assert groups.iloc[1].tolist() == ["female", "55+", "graduate school", "single"]


def test_checksum_verification() -> None:
    payload = b"credit"
    verify_checksum(payload, hashlib.sha256(payload).hexdigest())
    with pytest.raises(ValueError, match="checksum"):
        verify_checksum(payload, "0" * 64)
