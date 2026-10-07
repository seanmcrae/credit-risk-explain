import pandas as pd

from credit_ranking.schema import TARGET, validate
from credit_ranking.synthetic import generate_synthetic


def test_same_seed_same_frame() -> None:
    pd.testing.assert_frame_equal(generate_synthetic(500, seed=3), generate_synthetic(500, seed=3))


def test_different_seed_differs() -> None:
    assert not generate_synthetic(200, seed=1).equals(generate_synthetic(200, seed=2))


def test_matches_schema_and_plausible_base_rate() -> None:
    df = validate(generate_synthetic(5_000, seed=0))
    assert len(df) == 5_000
    assert 0.15 < df[TARGET].mean() < 0.30


def test_recent_delinquency_raises_default_rate() -> None:
    df = generate_synthetic(5_000, seed=0)
    late = df.loc[df["pay_status_1"] >= 2, TARGET].mean()
    current = df.loc[df["pay_status_1"] <= 0, TARGET].mean()
    assert late > 2 * current


def test_no_consumption_months_have_zero_bill() -> None:
    df = generate_synthetic(2_000, seed=0)
    assert (df.loc[df["pay_status_3"] == -2, "bill_amt_3"] == 0).all()
