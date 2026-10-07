"""Matplotlib charts for the README, model card, CLI and app.

Each chart draws onto a supplied ``Axes`` (so charts compose into the dashboard) or creates its
own figure, and returns the ``Figure`` it drew on.
"""

from __future__ import annotations

from typing import cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox

from credit_ranking.config import EconomicsConfig
from credit_ranking.explain import Attribution
from credit_ranking.features import FEATURE_DESCRIPTIONS
from credit_ranking.metrics import expected_value_curve, rank_order, reliability_curve
from credit_ranking.queue import AccountExplanation

RISK_UP, RISK_DOWN = "#c0392b", "#2e86c1"


def _canvas(ax: Axes | None, figsize: tuple[float, float]) -> tuple[Figure, Axes, bool]:
    """Return (figure, axes, owns_figure)."""
    if ax is None:
        fig, new_ax = plt.subplots(figsize=figsize)
        return fig, new_ax, True
    return cast(Figure, ax.get_figure()), ax, False


def _finish(fig: Figure, owns: bool) -> Figure:
    if owns:
        fig.tight_layout()
    return fig


def cumulative_capture(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Share of accounts worked vs share of all defaulters captured, riskiest first."""
    captured = np.r_[0.0, np.cumsum(y[rank_order(score)])] / y.sum()
    worked = np.linspace(0, 1, len(y) + 1)
    return worked, captured


def lift_curve(
    y: np.ndarray,
    scores: dict[str, np.ndarray],
    capacity_share: float,
    ax: Axes | None = None,
) -> Figure:
    fig, ax, owns = _canvas(ax, (6, 4.2))
    for name, score in scores.items():
        ax.plot(*cumulative_capture(y, score), label=name, lw=2)
    ax.plot(*cumulative_capture(y, y.astype(float)), "k:", lw=1, label="perfect ranking")
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="random order")
    ax.axvline(capacity_share, color=RISK_UP, lw=1, alpha=0.6)
    ax.text(capacity_share + 0.01, 0.93, f"capacity {capacity_share:.0%}", color=RISK_UP)
    ax.set(
        xlabel="Share of accounts worked",
        ylabel="Share of defaulters captured",
        title="Cumulative capture on holdout",
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    return _finish(fig, owns)


def reliability_plot(
    y: np.ndarray, probs: dict[str, np.ndarray], n_bins: int, ax: Axes | None = None
) -> Figure:
    fig, ax, owns = _canvas(ax, (5, 4.2))
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="perfectly calibrated")
    for name, prob in probs.items():
        curve = reliability_curve(y, prob, n_bins)
        ax.plot(
            curve.bin_mean_predicted,
            curve.bin_observed_rate,
            "o-",
            label=f"{name} (ECE {curve.ece:.3f})",
        )
    ax.set(
        xlabel="Predicted default probability",
        ylabel="Observed default rate",
        title="Reliability on holdout",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    return _finish(fig, owns)


def expected_value_plot(
    y: np.ndarray, score: np.ndarray, econ: EconomicsConfig, ax: Axes | None = None
) -> Figure:
    choice = expected_value_curve(y, score, econ)
    fig, ax, owns = _canvas(ax, (6, 4))
    ax.plot(np.arange(len(choice.ev_curve)), choice.ev_curve / 1000, lw=2)
    ax.axvline(choice.optimal_capacity, color=RISK_UP, lw=1)
    ax.set(
        xlabel="Accounts worked (riskiest first)",
        ylabel="Expected value (thousands)",
        title=f"Value of working the queue (optimum: {choice.optimal_capacity:,} accounts)",
    )
    ax.grid(alpha=0.3)
    return _finish(fig, owns)


def shap_summary(attribution: Attribution, max_display: int = 14) -> Figure:
    """SHAP beeswarm; shap draws on the current figure, so this always creates its own."""
    plt.figure(figsize=(7.5, 5))
    shap.summary_plot(
        attribution.values.to_numpy(),
        attribution.data,
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    fig = plt.gcf()
    fig.axes[0].set_title("SHAP contributions on holdout (log-odds of default)")
    fig.tight_layout()
    return fig


def waterfall(
    contributions: pd.Series,
    values: pd.Series,
    base_value: float,
    top: int = 8,
    ax: Axes | None = None,
) -> Figure:
    """Per-account waterfall from the base log-odds to the account's score."""
    order = contributions.abs().sort_values(ascending=False).index
    steps = [
        (f"{FEATURE_DESCRIPTIONS[f]} = {values[f]:.2f}", contributions[f]) for f in order[:top]
    ]
    if len(order) > top:
        steps.append((f"{len(order) - top} other features", contributions[order[top:]].sum()))
    deltas = np.array([d for _, d in steps])
    starts = base_value + np.r_[0.0, np.cumsum(deltas)[:-1]]
    ypos = np.arange(len(steps))[::-1]  # largest contribution at the top
    fig, ax, owns = _canvas(ax, (8, 0.45 * len(steps) + 1.6))
    ax.barh(ypos, deltas, left=starts, color=np.where(deltas > 0, RISK_UP, RISK_DOWN))
    for y_, x0, d in zip(ypos, starts, deltas, strict=True):
        ax.text(x0 + d, y_, f" {d:+.2f} ", va="center", ha="left" if d > 0 else "right", fontsize=8)
    ax.set_yticks(ypos, [label for label, _ in steps], fontsize=8)
    ax.axvline(base_value, color="grey", ls="--", lw=1)
    ends = starts + deltas
    lo, hi = min(base_value, starts.min(), ends.min()), max(base_value, starts.max(), ends.max())
    pad = 0.25 * (hi - lo) + 0.1  # room for the value labels
    ax.set_xlim(lo - pad, hi + pad)
    ax.set(
        xlabel=f"Log-odds of default (base {base_value:.2f} -> account {ends[-1]:.2f})",
        title="Why this account is ranked here",
    )
    return _finish(fig, owns)


def queue_dashboard(
    queue: pd.DataFrame,
    account: AccountExplanation,
    y: np.ndarray,
    raw: np.ndarray,
    prob: np.ndarray,
    capacity_share: float,
    n_bins: int,
    title: str,
) -> Figure:
    """Static stand-in for the app: queue head, one account's waterfall, gains and reliability."""
    fig = plt.figure(figsize=(15, 9.5), layout="constrained")
    rows = fig.add_gridspec(2, 1, height_ratios=[1, 1.1])
    top, bottom = fig.add_subfigure(rows[0]), fig.add_subfigure(rows[1])
    table_ax = top.add_subplot()
    table_ax.axis("off")
    wf_ax, lift_ax, rel_ax = bottom.subplots(1, 3, width_ratios=[1.5, 1, 1])
    shown = queue.head(10)
    cells = [
        [
            str(r["rank"]),
            str(r["id"]),
            f"{r['raw_score']:.4f}",
            f"{r['probability']:.3f}",
            str(r["decile"]),
            str(r["reason_codes"]),
            str(r["top_reasons"]).split("; ")[0],
        ]
        for _, r in shown.iterrows()
    ]
    headers = [
        "Rank",
        "Account",
        "Model score",
        "P(default)",
        "Decile",
        "Reason codes",
        "Primary reason",
    ]
    table = table_ax.table(
        cellText=cells,
        colLabels=headers,
        cellLoc="left",
        colWidths=[0.05, 0.08, 0.09, 0.08, 0.06, 0.14, 0.50],
        bbox=Bbox.from_bounds(0.0, 0.0, 1.0, 0.95),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if row == 0:
            cell.set_facecolor("#f2f2f2")
            cell.set_text_props(weight="bold")
    table_ax.set_title("Prioritized work queue (top 10)", loc="left", fontsize=12, weight="bold")

    waterfall(account.contributions, account.feature_values, account.base_value, top=6, ax=wf_ax)
    wf_ax.set_title(
        f"Account {account.account_id}: P(default) {account.probability:.2f}, "
        f"decile {account.decile}"
    )
    lift_curve(y, {"Champion": raw}, capacity_share, ax=lift_ax)
    reliability_plot(y, {"Calibrated": prob}, n_bins, ax=rel_ax)
    fig.suptitle(title, fontsize=13, x=0.01, ha="left")
    return fig
