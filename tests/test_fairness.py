import numpy as np
import pandas as pd
import pytest

from credit_ranking.fairness import disparity_summary, prioritized_mask, slice_metrics


def _fixture() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    # 8 accounts, scores descending; capacity 3 works accounts 0, 1, 2.
    groups = pd.DataFrame({"sex": ["f", "m", "f", "m", "f", "m", "f", "m"]})
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    score = np.linspace(0.9, 0.2, 8)
    prob = np.array([0.8, 0.7, 0.4, 0.3, 0.3, 0.1, 0.1, 0.1])
    return groups, y, prob, score


def test_prioritized_mask_takes_top_capacity() -> None:
    assert prioritized_mask(np.array([0.1, 0.9, 0.5]), 2).tolist() == [False, True, True]


def test_slice_metrics_hand_computed() -> None:
    groups, y, prob, score = _fixture()
    table = slice_metrics(groups, y, prob, score, capacity=3).set_index("group")
    f, m = table.loc["f"], table.loc["m"]
    assert f["priority_rate"] == pytest.approx(2 / 4)  # accounts 0, 2
    assert m["priority_rate"] == pytest.approx(1 / 4)  # account 1
    assert f["tpr_at_capacity"] == pytest.approx(1 / 2)  # defaulters 0, 4; only 0 worked
    assert m["tpr_at_capacity"] == pytest.approx(1.0)
    assert f["fpr_at_capacity"] == pytest.approx(1 / 2)  # non-defaulters 2, 6; 2 worked
    assert f["calibration_gap"] == pytest.approx((0.8 + 0.4 + 0.3 + 0.1) / 4 - 0.5)
    assert f["small_slice"]


def test_disparity_summary_ignores_small_slices_by_default() -> None:
    groups, y, prob, score = _fixture()
    slices = slice_metrics(groups, y, prob, score, capacity=3)
    assert disparity_summary(slices).empty
    slices["small_slice"] = False
    row = disparity_summary(slices).iloc[0]
    assert row["priority_rate_ratio"] == pytest.approx(0.5)
    assert row["tpr_gap"] == pytest.approx(0.5)


def test_slice_auc_nan_when_single_class() -> None:
    groups = pd.DataFrame({"g": ["a", "a", "b", "b"]})
    y = np.array([0, 0, 0, 1])
    table = slice_metrics(groups, y, np.full(4, 0.2), np.array([0.4, 0.3, 0.2, 0.1]), 1)
    assert np.isnan(table.loc[table["group"] == "a", "auc"]).all()
    assert np.isnan(table.loc[table["group"] == "a", "tpr_at_capacity"]).all()
