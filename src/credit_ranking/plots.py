"""Matplotlib charts for the README, model card and app. All functions return a Figure."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.figure import Figure

from credit_ranking.config import EconomicsConfig
from credit_ranking.explain import Attribution
from credit_ranking.features import FEATURE_DESCRIPTIONS
from credit_ranking.metrics import expected_value_curve, rank_order, reliability_curve


def cumulative_capture(y: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Share of accounts worked vs share of all defaulters captured, riskiest first."""
    captured = np.r_[0.0, np.cumsum(y[rank_order(score)])] / y.sum()
    worked = np.linspace(0, 1, len(y) + 1)
    return worked, captured


def lift_curve(y: np.ndarray, scores: dict[str, np.ndarray], capacity_share: float) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for name, score in scores.items():
        ax.plot(*cumulative_capture(y, score), label=name, lw=2)
    ax.plot(*cumulative_capture(y, y.astype(float)), "k:", lw=1, label="perfect ranking")
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="random order")
    ax.axvline(capacity_share, color="tab:red", lw=1, alpha=0.6)
    ax.text(capacity_share + 0.01, 0.05, f"capacity {capacity_share:.0%}", color="tab:red")
    ax.set(
        xlabel="Share of accounts worked",
        ylabel="Share of defaulters captured",
        title="Cumulative capture (gains) on holdout",
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def reliability_plot(y: np.ndarray, probs: dict[str, np.ndarray], n_bins: int) -> Figure:
    fig, ax = plt.subplots(figsize=(5, 4.2))
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
    fig.tight_layout()
    return fig


def expected_value_plot(y: np.ndarray, score: np.ndarray, econ: EconomicsConfig) -> Figure:
    choice = expected_value_curve(y, score, econ)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(len(choice.ev_curve)), choice.ev_curve / 1000, lw=2)
    ax.axvline(choice.optimal_capacity, color="tab:red", lw=1)
    ax.set(
        xlabel="Accounts worked (riskiest first)",
        ylabel="Expected value (thousands)",
        title=f"Value of working the queue (optimum: {choice.optimal_capacity} accounts)",
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def shap_summary(attribution: Attribution, max_display: int = 14) -> Figure:
    plt.figure(figsize=(7, 5))
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
    contributions: pd.Series, values: pd.Series, base_value: float, top: int = 8
) -> Figure:
    """Per-account waterfall from the base log-odds to the account's score."""
    order = contributions.abs().sort_values(ascending=False).index
    shown, rest = list(order[:top]), contributions[order[top:]].sum()
    steps = [(f"{FEATURE_DESCRIPTIONS[f]} = {values[f]:.2f}", contributions[f]) for f in shown]
    if len(order) > top:
        steps.append((f"{len(order) - top} other features", rest))
    deltas = np.array([d for _, d in steps])
    starts = base_value + np.r_[0.0, np.cumsum(deltas)[:-1]]
    ypos = np.arange(len(steps))[::-1]  # largest contribution at the top
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(steps) + 1.6))
    ax.barh(ypos, deltas, left=starts, color=np.where(deltas > 0, "tab:red", "tab:blue"))
    for y_, x0, d in zip(ypos, starts, deltas, strict=True):
        ax.text(x0 + d, y_, f" {d:+.2f}", va="center", ha="left" if d > 0 else "right", fontsize=8)
    ax.set_yticks(ypos, [label for label, _ in steps], fontsize=8)
    ax.axvline(base_value, color="grey", ls="--", lw=1)
    final = base_value + deltas.sum()
    ax.set(
        xlabel=f"Log-odds of default (base {base_value:.2f} -> account {final:.2f})",
        title="Why this account is ranked here",
    )
    fig.tight_layout()
    return fig
