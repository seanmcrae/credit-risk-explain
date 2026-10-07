"""Slice audit of the ranked queue across protected attributes.

Protected attributes never enter the model; they are joined back only here. Being prioritized
can read as adverse (more collections contact) or beneficial (earlier hardship outreach), so the
audit reports both selection-rate parity and error-rate parity rather than picking one definition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from credit_ranking.metrics import rank_order

MIN_SLICE_SIZE = 100  # slices smaller than this are reported but flagged as unreliable


def prioritized_mask(score: np.ndarray, capacity: int) -> np.ndarray:
    mask = np.zeros(len(score), dtype=bool)
    mask[rank_order(score)[:capacity]] = True
    return mask


def _slice_auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def slice_metrics(
    groups: pd.DataFrame, y: np.ndarray, prob: np.ndarray, score: np.ndarray, capacity: int
) -> pd.DataFrame:
    """One row per (attribute, group) with selection, error-rate and calibration metrics."""
    worked = prioritized_mask(score, capacity)
    rows = []
    for attribute in groups.columns:
        labels = groups[attribute].to_numpy()
        for group in sorted(pd.unique(labels)):
            m = labels == group
            y_g, worked_g = y[m], worked[m]
            events = y_g.sum()
            rows.append(
                {
                    "attribute": attribute,
                    "group": group,
                    "accounts": int(m.sum()),
                    "default_rate": float(y_g.mean()),
                    "mean_predicted": float(prob[m].mean()),
                    "calibration_gap": float(prob[m].mean() - y_g.mean()),
                    "priority_rate": float(worked_g.mean()),
                    "tpr_at_capacity": float(worked_g[y_g == 1].mean()) if events else float("nan"),
                    "fpr_at_capacity": (
                        float(worked_g[y_g == 0].mean()) if events < len(y_g) else float("nan")
                    ),
                    "auc": _slice_auc(y_g, score[m]),
                    "small_slice": bool(m.sum() < MIN_SLICE_SIZE),
                }
            )
    return pd.DataFrame(rows)


def disparity_summary(slices: pd.DataFrame) -> pd.DataFrame:
    """Per attribute: worst-to-best priority-rate ratio and largest gaps, excluding small slices."""
    reliable = slices[~slices["small_slice"]]
    summary = reliable.groupby("attribute").agg(
        groups=("group", "size"),
        priority_rate_min=("priority_rate", "min"),
        priority_rate_max=("priority_rate", "max"),
        tpr_gap=("tpr_at_capacity", lambda s: s.max() - s.min()),
        fpr_gap=("fpr_at_capacity", lambda s: s.max() - s.min()),
        max_abs_calibration_gap=("calibration_gap", lambda s: s.abs().max()),
    )
    summary["priority_rate_ratio"] = summary["priority_rate_min"] / summary["priority_rate_max"]
    return summary.reset_index()


@dataclass(frozen=True)
class TprGap:
    """The attribute whose reliable slices differ most in TPR at capacity."""

    attribute: str
    high_group: str
    low_group: str
    high_tpr: float
    low_tpr: float


def largest_tpr_gap(slices: pd.DataFrame) -> TprGap | None:
    """Widest TPR spread across reliable slices, or None if no attribute has two of them."""
    reliable = slices[~slices["small_slice"].astype(bool)].dropna(subset=["tpr_at_capacity"])
    best: TprGap | None = None
    for attribute, rows in reliable.groupby("attribute", sort=True):
        if len(rows) < 2:
            continue
        ordered = rows.sort_values("tpr_at_capacity", ascending=False, kind="stable")
        hi, lo = ordered.iloc[0], ordered.iloc[-1]
        gap = TprGap(
            str(attribute),
            str(hi["group"]),
            str(lo["group"]),
            float(hi["tpr_at_capacity"]),
            float(lo["tpr_at_capacity"]),
        )
        if best is None or gap.high_tpr - gap.low_tpr > best.high_tpr - best.low_tpr:
            best = gap
    return best


def contributions_by_group(
    contributions: pd.DataFrame, groups: pd.DataFrame, y: np.ndarray
) -> pd.DataFrame:
    """Mean SHAP contribution of each feature among each group's defaulters, in long format.

    Protected attributes are not model inputs, so any gap in how a group's defaulters are ranked
    has to travel through the features. Comparing where defaulters' log-odds come from, group by
    group, shows which features carry the gap. It is a proxy screen, not a causal decomposition.
    """
    defaulters = y == 1
    frames: list[pd.DataFrame] = []
    for attribute in groups.columns:
        labels = groups[attribute].to_numpy()[defaulters]
        means = contributions[defaulters].groupby(labels).mean()
        long = means.stack().rename("mean_contribution").reset_index()
        long.columns = pd.Index(["group", "feature", "mean_contribution"])
        long.insert(0, "attribute", attribute)
        frames.append(long)
    return pd.concat(frames, ignore_index=True)


def gap_drivers(contributions: pd.DataFrame, gap: TprGap, top: int = 4) -> pd.DataFrame:
    """Features whose mean defaulter contribution differs most between the gap's two groups."""
    rows = contributions[contributions["attribute"] == gap.attribute]
    wide = rows.pivot(index="feature", columns="group", values="mean_contribution")
    drivers = pd.DataFrame(
        {"high": wide[gap.high_group], "low": wide[gap.low_group]}, index=wide.index
    )
    drivers["difference"] = drivers["high"] - drivers["low"]
    order = drivers["difference"].abs().sort_values(ascending=False).index
    return drivers.loc[order].head(top).rename_axis("feature").reset_index()
