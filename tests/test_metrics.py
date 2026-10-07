import numpy as np
import pytest

from credit_ranking.config import EconomicsConfig
from credit_ranking.metrics import (
    assign_deciles,
    brier,
    expected_value_curve,
    expected_value_from_probabilities,
    ks_statistic,
    lift_table,
    precision_at_top_percent,
    psi,
    recall_at_top_percent,
    reliability_curve,
    roc_auc,
)

# Ten accounts already in descending score order; events at ranks 1, 2 and 4.
Y = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
S = np.linspace(1.0, 0.1, 10)


def test_ks_hand_computed() -> None:
    # cum events:     1/3, 2/3, 2/3, 1,   1 ...
    # cum non-events: 0,   0,   1/7, 1/7, 2/7 ...  -> max gap at rank 4 = 1 - 1/7
    assert ks_statistic(Y, S) == pytest.approx(6 / 7)


def test_ks_does_not_split_ties() -> None:
    # All scores tied: the model separates nothing, so KS must be 0, not an artifact of order.
    assert ks_statistic(Y, np.ones(10)) == 0.0


def test_lift_table_hand_computed() -> None:
    table = lift_table(Y, S, n_bins=5)
    assert table["accounts"].tolist() == [2, 2, 2, 2, 2]
    assert table["events"].tolist() == [2, 1, 0, 0, 0]
    # base rate 0.3: bucket 1 rate 1.0 -> lift 3.33; bucket 2 rate 0.5 -> lift 1.67
    assert table["lift"].iloc[0] == pytest.approx(1.0 / 0.3)
    assert table["lift"].iloc[1] == pytest.approx(0.5 / 0.3)
    assert table["capture_rate"].tolist() == pytest.approx([2 / 3, 1, 1, 1, 1])
    assert table["cum_lift"].iloc[1] == pytest.approx((3 / 4) / 0.3)
    assert table["cum_lift"].iloc[-1] == pytest.approx(1.0)


def test_deciles_balanced_and_riskiest_first() -> None:
    scores = np.random.default_rng(0).random(103)
    d = assign_deciles(scores)
    counts = np.bincount(d)[1:]
    assert counts.max() - counts.min() <= 1
    assert d[np.argmax(scores)] == 1
    assert d[np.argmin(scores)] == 10


def test_precision_and_recall_at_top_percent() -> None:
    assert precision_at_top_percent(Y, S, 20) == pytest.approx(1.0)
    assert precision_at_top_percent(Y, S, 40) == pytest.approx(0.75)
    assert recall_at_top_percent(Y, S, 40) == pytest.approx(1.0)
    assert precision_at_top_percent(Y, S, 1) == 1.0  # at least one account is always taken


def test_expected_value_picks_capacity_hand_computed() -> None:
    econ = EconomicsConfig(cost_per_contact=10, loss_given_default=100, cure_rate_if_worked=0.5)
    choice = expected_value_curve(Y, S, econ)
    # value per true positive = 50; EV(k) = 50*TP(k) - 10k
    assert choice.ev_curve[:5].tolist() == [0, 40, 80, 70, 110]
    assert choice.optimal_capacity == 4
    assert choice.optimal_value == 110
    assert choice.value_at(10) == 150 - 100


def test_reliability_and_ece() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.05, 0.15, 0.85, 0.95])
    curve = reliability_curve(y, p, n_bins=2)
    assert curve.bin_mean_predicted.tolist() == pytest.approx([0.1, 0.9])
    assert curve.bin_observed_rate.tolist() == [0.0, 1.0]
    assert curve.ece == pytest.approx(0.1)
    assert brier(y, p) == pytest.approx((0.05**2 + 0.15**2 + 0.15**2 + 0.05**2) / 4)


def test_reliability_rejects_non_probabilities() -> None:
    with pytest.raises(ValueError):
        reliability_curve(np.array([0, 1]), np.array([0.2, 1.4]))


def test_psi_zero_for_same_distribution_and_grows_with_shift() -> None:
    rng = np.random.default_rng(1)
    ref = rng.normal(size=5_000)
    assert psi(ref, ref) == pytest.approx(0.0, abs=1e-9)
    assert psi(ref, ref + 0.1) < psi(ref, ref + 1.0)
    assert psi(ref, ref + 1.0) > 0.25


def test_auc_perfect_ranking() -> None:
    assert roc_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        ks_statistic(np.array([0, 2]), np.array([0.1, 0.2]))
    with pytest.raises(ValueError):
        ks_statistic(np.array([0, 1]), np.array([0.1]))


def test_prospective_expected_value_breaks_even_at_threshold() -> None:
    econ = EconomicsConfig(cost_per_contact=60, loss_given_default=4000, cure_rate_if_worked=0.1)
    ev = expected_value_from_probabilities(np.array([0.0, 0.15, 0.5]), econ)
    assert ev.tolist() == pytest.approx([-60.0, 0.0, 140.0])
