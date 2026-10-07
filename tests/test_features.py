import pandas as pd
import pytest

from credit_ranking.features import FEATURES, build_features
from credit_ranking.schema import PROTECTED_ATTRIBUTES
from tests.conftest import make_account


def _features(**overrides: float) -> pd.Series:
    return build_features(pd.DataFrame([make_account(**overrides)])).iloc[0]


def test_output_columns_and_no_protected_attributes() -> None:
    out = build_features(pd.DataFrame([make_account()]))
    assert tuple(out.columns) == FEATURES
    assert not set(out.columns) & set(PROTECTED_ATTRIBUTES)


def test_protected_attributes_do_not_change_features() -> None:
    a = _features(sex=1, age=22, education=3, marriage=2)
    b = _features(sex=2, age=61, education=1, marriage=1)
    pd.testing.assert_series_equal(a, b)


def test_utilization() -> None:
    f = _features(limit_bal=50_000, bill_amt_1=40_000, bill_amt_6=10_000)
    assert f["utilization_recent"] == pytest.approx(0.8)
    assert f["utilization_max"] == pytest.approx(0.8)
    assert f["utilization_trend"] == pytest.approx(0.6)
    assert f["utilization_mean"] == pytest.approx((40_000 + 4 * 10_000 + 10_000) / 6 / 50_000)


def test_payment_ratio_uses_prior_statement_and_caps_at_one() -> None:
    f = _features(pay_amt_1=2_500, bill_amt_2=10_000)
    assert f["payment_ratio_recent"] == pytest.approx(0.25)
    overpaid = _features(pay_amt_1=50_000, bill_amt_2=10_000)
    assert overpaid["payment_ratio_recent"] == 1.0


def test_nothing_owed_counts_as_fully_paid() -> None:
    f = _features(pay_amt_1=0, bill_amt_2=0)
    assert f["payment_ratio_recent"] == 1.0
    assert f["months_zero_payment"] == 0


def test_months_zero_payment_counts_unpaid_balances() -> None:
    f = _features(pay_amt_1=0, pay_amt_2=0, bill_amt_3=0)
    # pay_amt_1 settles bill_amt_2 (owed), pay_amt_2 settles bill_amt_3 (nothing owed).
    assert f["months_zero_payment"] == 1


@pytest.mark.parametrize(
    ("statuses", "streak", "count", "max_delay"),
    [
        ((2, 1, 0, 3, 0, 0), 2, 3, 3),
        ((0, 2, 2, 2, 2, 2), 0, 5, 2),
        ((1, 1, 1, 1, 1, 1), 6, 6, 1),
        ((-1, -2, 0, -1, 0, 0), 0, 0, 0),
    ],
)
def test_delinquency_features(
    statuses: tuple[int, ...], streak: int, count: int, max_delay: int
) -> None:
    f = _features(**{f"pay_status_{m}": s for m, s in enumerate(statuses, start=1)})
    assert f["delinquency_streak"] == streak
    assert f["months_delinquent"] == count
    assert f["max_delay"] == max_delay
    assert f["pay_status_recent"] == statuses[0]


def test_row_wise_features_do_not_depend_on_other_rows() -> None:
    one = build_features(pd.DataFrame([make_account(id=1, bill_amt_1=90_000)]))
    many = build_features(
        pd.DataFrame([make_account(id=1, bill_amt_1=90_000), make_account(id=2, limit_bal=5_000)])
    )
    pd.testing.assert_series_equal(one.iloc[0], many.iloc[0])
