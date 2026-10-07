"""Holdout evaluation of one scored model: ranking quality, calibration and queue economics."""

from __future__ import annotations

from typing import Any

import numpy as np

from credit_ranking.config import Config
from credit_ranking.metrics import (
    brier,
    expected_value_curve,
    ks_statistic,
    lift_table,
    pr_auc,
    precision_at_top_percent,
    recall_at_top_percent,
    reliability_curve,
    roc_auc,
    top_k_count,
)


def evaluate_scores(
    y: np.ndarray, raw: np.ndarray, prob: np.ndarray, cfg: Config
) -> dict[str, Any]:
    """Ranking metrics use the raw score (the queue order); calibration metrics use ``prob``."""
    n = len(y)
    capacity = top_k_count(n, cfg.ranking.eval_capacity_percent)
    lift = lift_table(y, raw)
    ev = expected_value_curve(y, raw, cfg.economics)
    return {
        "roc_auc": roc_auc(y, raw),
        "pr_auc": pr_auc(y, raw),
        "ks": ks_statistic(y, raw),
        "brier": brier(y, prob),
        "ece": reliability_curve(y, prob, cfg.calibration.n_bins).ece,
        "ece_uncalibrated": reliability_curve(y, raw, cfg.calibration.n_bins).ece,
        "precision_at_top_pct": {
            str(p): precision_at_top_percent(y, raw, p) for p in cfg.ranking.top_k_percents
        },
        "recall_at_top_pct": {
            str(p): recall_at_top_percent(y, raw, p) for p in cfg.ranking.top_k_percents
        },
        "top_decile_lift": float(lift["lift"].iloc[0]),
        "top_two_decile_capture": float(lift["capture_rate"].iloc[1]),
        "eval_capacity": capacity,
        "ev_at_eval_capacity": ev.value_at(capacity),
        "ev_optimal_capacity": ev.optimal_capacity,
        "ev_optimal_capacity_pct": 100 * ev.optimal_capacity / n,
        "ev_optimal_value": ev.optimal_value,
        "ev_work_everyone": ev.value_at(n),
    }
