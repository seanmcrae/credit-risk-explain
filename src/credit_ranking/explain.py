"""SHAP attributions and adverse-action-style reason codes.

Attributions are in the raw model's log-odds space, where contributions add up exactly to the
account's score. Reason codes group related features so an analyst sees "balance is high relative
to the limit" once, instead of three correlated utilization features splitting the credit.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from credit_ranking.features import FEATURES
from credit_ranking.models import ScoredModel


@dataclass(frozen=True)
class ReasonDefinition:
    code: str
    text: str
    features: tuple[str, ...]


REASONS: tuple[ReasonDefinition, ...] = (
    ReasonDefinition(
        "R01", "Most recent payment is past due", ("pay_status_recent", "pay_status_prev")
    ),
    ReasonDefinition(
        "R02",
        "Repeated or ongoing delinquency in the last six months",
        ("months_delinquent", "delinquency_streak", "max_delay"),
    ),
    ReasonDefinition(
        "R03",
        "Balance is high relative to credit limit",
        ("utilization_recent", "utilization_mean", "utilization_max"),
    ),
    ReasonDefinition("R04", "Balance has been increasing", ("utilization_trend",)),
    ReasonDefinition(
        "R05",
        "Payments are small relative to the balance owed",
        ("payment_ratio_recent", "payment_ratio_mean", "payment_to_limit_mean"),
    ),
    ReasonDefinition("R06", "Balance due with no payment made", ("months_zero_payment",)),
    ReasonDefinition("R07", "Limited credit line", ("log_limit",)),
)
FEATURE_TO_REASON = {f: r for r in REASONS for f in r.features}


@dataclass(frozen=True)
class Attribution:
    """Per-account feature contributions; ``base_value + values.sum(axis=1)`` is the log-odds."""

    values: pd.DataFrame
    base_value: float
    data: pd.DataFrame


@dataclass(frozen=True)
class ReasonCode:
    code: str
    text: str
    contribution: float  # summed log-odds contribution of the reason's features
    driver: str  # the single feature contributing most within the reason
    driver_value: float


def _linear_attribution(pipe: Pipeline, X: pd.DataFrame, background: pd.DataFrame) -> Attribution:
    """Exact SHAP for a standardized linear model with independent features."""
    scaler, lr = pipe.steps[0][1], pipe.steps[-1][1]
    z = scaler.transform(X)
    z_bg = scaler.transform(background).mean(axis=0)
    coef = lr.coef_.ravel()
    values = pd.DataFrame((z - z_bg) * coef, index=X.index, columns=X.columns)
    base = float(lr.intercept_[0] + z_bg @ coef)
    return Attribution(values=values, base_value=base, data=X)


def attribute(model: ScoredModel, X: pd.DataFrame, background: pd.DataFrame) -> Attribution:
    X = X[model.features]
    if model.name == "logistic_regression":
        assert isinstance(model.estimator, Pipeline)
        return _linear_attribution(model.estimator, X, background[model.features])
    explainer = shap.TreeExplainer(model.estimator)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*shap values output has changed.*")
        raw = explainer.shap_values(X)
    # Older shap versions return [class0, class1] for binary classifiers.
    matrix = np.asarray(raw[1] if isinstance(raw, list) else raw)
    base = np.ravel(explainer.expected_value)[-1]
    values = pd.DataFrame(matrix, index=X.index, columns=X.columns)
    return Attribution(values=values, base_value=float(base), data=X)


def global_importance(attribution: Attribution) -> pd.Series:
    """Mean absolute contribution per feature, largest first."""
    return attribution.values.abs().mean().sort_values(ascending=False)


def reason_codes(contributions: pd.Series, values: pd.Series, n: int) -> list[ReasonCode]:
    """Top ``n`` risk-increasing reasons for one account.

    Reasons whose net contribution lowers risk are never returned: an adverse-action reason must
    explain why the account was prioritized, not what counted in its favor.
    """
    codes = []
    for reason in REASONS:
        parts = contributions[list(reason.features)]
        total = float(parts.sum())
        if total <= 0:
            continue
        driver = str(parts.idxmax())
        codes.append(ReasonCode(reason.code, reason.text, total, driver, float(values[driver])))
    codes.sort(key=lambda r: r.contribution, reverse=True)
    return codes[:n]


def reason_table(attribution: Attribution, n: int) -> pd.DataFrame:
    """One row per account with its top reasons as readable text, for queues and exports."""
    rows = {}
    for (account_id, contrib), (_, values) in zip(
        attribution.values.iterrows(), attribution.data.iterrows(), strict=True
    ):
        codes = reason_codes(contrib, values, n)
        rows[account_id] = {
            "reason_codes": ";".join(c.code for c in codes),
            "top_reasons": "; ".join(c.text for c in codes),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


assert set(FEATURE_TO_REASON) == set(FEATURES), "every feature needs exactly one reason code"
