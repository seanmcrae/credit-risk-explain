"""Ranking, calibration, economic and stability metrics for a scored work queue.

Conventions: ``y`` is a 0/1 outcome array, ``score`` a risk score where higher means work first.
Ties are broken by original order so every result is deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from credit_ranking.config import EconomicsConfig

FloatArray = np.ndarray


def _as_arrays(y: FloatArray, score: FloatArray) -> tuple[FloatArray, FloatArray]:
    y_arr = np.asarray(y, dtype=float)
    s_arr = np.asarray(score, dtype=float)
    if y_arr.shape != s_arr.shape or y_arr.ndim != 1:
        raise ValueError("y and score must be 1-D arrays of equal length")
    if not np.isin(y_arr, (0.0, 1.0)).all():
        raise ValueError("y must be binary 0/1")
    return y_arr, s_arr


def rank_order(score: FloatArray) -> FloatArray:
    """Indices that sort accounts from highest to lowest risk (stable on ties)."""
    return np.argsort(-np.asarray(score, dtype=float), kind="stable")


def roc_auc(y: FloatArray, score: FloatArray) -> float:
    return float(roc_auc_score(*_as_arrays(y, score)))


def pr_auc(y: FloatArray, score: FloatArray) -> float:
    return float(average_precision_score(*_as_arrays(y, score)))


def ks_statistic(y: FloatArray, score: FloatArray) -> float:
    """Max distance between the cumulative score distributions of events and non-events.

    Evaluated only at distinct score values so tied scores are never split.
    """
    y_arr, s_arr = _as_arrays(y, score)
    order = rank_order(s_arr)
    y_sorted, s_sorted = y_arr[order], s_arr[order]
    cum_pos = np.cumsum(y_sorted) / y_sorted.sum()
    cum_neg = np.cumsum(1 - y_sorted) / (1 - y_sorted).sum()
    last_of_tie = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    return float(np.max(np.abs(cum_pos - cum_neg)[last_of_tie]))


def top_k_count(n: int, percent: float) -> int:
    return max(1, math.ceil(n * percent / 100))


def precision_at_top_percent(y: FloatArray, score: FloatArray, percent: float) -> float:
    y_arr, s_arr = _as_arrays(y, score)
    k = top_k_count(len(y_arr), percent)
    return float(y_arr[rank_order(s_arr)[:k]].mean())


def recall_at_top_percent(y: FloatArray, score: FloatArray, percent: float) -> float:
    y_arr, s_arr = _as_arrays(y, score)
    k = top_k_count(len(y_arr), percent)
    return float(y_arr[rank_order(s_arr)[:k]].sum() / y_arr.sum())


def assign_deciles(score: FloatArray, n_bins: int = 10) -> FloatArray:
    """Bucket 1 holds the highest-risk accounts; buckets differ in size by at most one."""
    s_arr = np.asarray(score, dtype=float)
    ranks = np.empty(len(s_arr), dtype=int)
    ranks[rank_order(s_arr)] = np.arange(len(s_arr))
    return np.asarray(ranks * n_bins // len(s_arr) + 1)


def lift_table(y: FloatArray, score: FloatArray, n_bins: int = 10) -> pd.DataFrame:
    """Per-bucket event rate, lift and cumulative capture, ordered from riskiest bucket."""
    y_arr, s_arr = _as_arrays(y, score)
    frame = pd.DataFrame({"bucket": assign_deciles(s_arr, n_bins), "y": y_arr, "score": s_arr})
    table = frame.groupby("bucket").agg(
        accounts=("y", "size"),
        events=("y", "sum"),
        min_score=("score", "min"),
        max_score=("score", "max"),
    )
    base_rate = y_arr.mean()
    table["event_rate"] = table["events"] / table["accounts"]
    table["lift"] = table["event_rate"] / base_rate
    table["cum_accounts"] = table["accounts"].cumsum()
    table["cum_events"] = table["events"].cumsum()
    table["capture_rate"] = table["cum_events"] / y_arr.sum()
    table["cum_lift"] = (table["cum_events"] / table["cum_accounts"]) / base_rate
    return table.reset_index()


@dataclass(frozen=True)
class CapacityChoice:
    """Expected value of working the top-k accounts, and the k that maximizes it."""

    ev_curve: FloatArray  # ev_curve[k] = value of working the k riskiest accounts
    optimal_capacity: int
    optimal_value: float

    def value_at(self, capacity: int) -> float:
        return float(self.ev_curve[min(capacity, len(self.ev_curve) - 1)])


def expected_value_curve(y: FloatArray, score: FloatArray, econ: EconomicsConfig) -> CapacityChoice:
    """Realized value of working the top-k accounts for every k, under the cost/benefit matrix.

    Working an eventual defaulter avoids ``loss_given_default * cure_rate`` of loss; every
    contact costs ``cost_per_contact``. Unworked accounts contribute zero (the baseline).
    """
    y_arr, s_arr = _as_arrays(y, score)
    tp = np.r_[0.0, np.cumsum(y_arr[rank_order(s_arr)])]
    k = np.arange(len(y_arr) + 1)
    ev = tp * econ.value_per_true_positive - k * econ.cost_per_contact
    best = int(np.argmax(ev))
    return CapacityChoice(ev_curve=ev, optimal_capacity=best, optimal_value=float(ev[best]))


def expected_value_from_probabilities(prob: FloatArray, econ: EconomicsConfig) -> FloatArray:
    """Prospective value of working each account given its calibrated default probability.

    Used for unlabeled queues: ``p * value_per_true_positive - cost_per_contact`` per account.
    """
    return np.asarray(prob, dtype=float) * econ.value_per_true_positive - econ.cost_per_contact


@dataclass(frozen=True)
class ReliabilityCurve:
    bin_mean_predicted: FloatArray
    bin_observed_rate: FloatArray
    bin_counts: FloatArray

    @property
    def ece(self) -> float:
        """Expected calibration error: count-weighted mean |observed - predicted| per bin."""
        weights = self.bin_counts / self.bin_counts.sum()
        return float(np.sum(weights * np.abs(self.bin_observed_rate - self.bin_mean_predicted)))


def reliability_curve(y: FloatArray, prob: FloatArray, n_bins: int = 10) -> ReliabilityCurve:
    """Equal-width probability bins; empty bins are dropped."""
    y_arr, p_arr = _as_arrays(y, prob)
    if ((p_arr < 0) | (p_arr > 1)).any():
        raise ValueError("probabilities must lie in [0, 1]")
    bins = np.minimum((p_arr * n_bins).astype(int), n_bins - 1)
    counts = np.bincount(bins, minlength=n_bins).astype(float)
    pred_sum = np.bincount(bins, weights=p_arr, minlength=n_bins)
    obs_sum = np.bincount(bins, weights=y_arr, minlength=n_bins)
    keep = counts > 0
    return ReliabilityCurve(
        bin_mean_predicted=pred_sum[keep] / counts[keep],
        bin_observed_rate=obs_sum[keep] / counts[keep],
        bin_counts=counts[keep],
    )


def brier(y: FloatArray, prob: FloatArray) -> float:
    return float(brier_score_loss(*_as_arrays(y, prob)))


def psi(expected: FloatArray, actual: FloatArray, n_bins: int = 10, eps: float = 1e-4) -> float:
    """Population stability index of ``actual`` scores against an ``expected`` reference.

    Bin edges are reference quantiles. Rule of thumb: < 0.1 stable, 0.1-0.25 watch, > 0.25 shift.
    """
    ref = np.asarray(expected, dtype=float)
    cur = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)[1:-1]))
    ref_share = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=len(edges) + 1)
    cur_share = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=len(edges) + 1)
    p = np.clip(ref_share / len(ref), eps, None)
    q = np.clip(cur_share / len(cur), eps, None)
    return float(np.sum((q - p) * np.log(q / p)))
