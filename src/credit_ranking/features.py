"""Leakage-safe feature engineering.

Every feature is a pure row-wise function of one account's six months of history, all observed
before the outcome month. Nothing is fitted on the dataset (no target encoding, no global
scaling), so features cannot carry information across the train/test boundary. Scaling for the
linear model happens inside its sklearn pipeline, fitted on training rows only.

Protected attributes (sex, age, education, marriage) are deliberately absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_ranking.schema import (
    BILL_AMT,
    LIMIT,
    N_MONTHS,
    PAY_AMT,
    PAY_STATUS,
    PROTECTED_ATTRIBUTES,
)

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "log_limit": "Log of the credit limit",
    "pay_status_recent": "Repayment status last month (k = months late)",
    "pay_status_prev": "Repayment status two months ago",
    "months_delinquent": "Months out of six with a late payment",
    "delinquency_streak": "Consecutive late months ending last month",
    "max_delay": "Longest delay in months over the six-month window",
    "utilization_recent": "Last statement balance divided by credit limit",
    "utilization_mean": "Average six-month balance-to-limit ratio",
    "utilization_max": "Peak six-month balance-to-limit ratio",
    "utilization_trend": "Change in balance-to-limit ratio over six months",
    "payment_ratio_recent": "Last payment as a share of the prior statement balance",
    "payment_ratio_mean": "Average payment as a share of the prior statement balance",
    "months_zero_payment": "Months with a balance due but no payment",
    "payment_to_limit_mean": "Average payment divided by credit limit",
}
FEATURES: tuple[str, ...] = tuple(FEATURE_DESCRIPTIONS)

assert not set(FEATURES) & set(PROTECTED_ATTRIBUTES), "protected attributes must not be features"


def _payment_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Payment in month m against the statement it settles (month m+1), capped at 1.

    A month with nothing owed counts as fully paid, so new or dormant accounts are not
    mistaken for non-payers.
    """
    ratios = {}
    for m in range(N_MONTHS - 1):
        owed = df[BILL_AMT[m + 1]]
        paid = df[PAY_AMT[m]]
        ratio = np.where(owed > 0, paid / owed.where(owed > 0, 1.0), 1.0)
        ratios[m] = np.clip(ratio, 0.0, 1.0)
    return pd.DataFrame(ratios, index=df.index)


def _leading_streak(late: pd.DataFrame) -> pd.Series:
    """Length of the run of True values starting at the first (most recent) column."""
    return late.cumprod(axis=1).sum(axis=1)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the model feature matrix; output columns are exactly ``FEATURES``."""
    status = df[list(PAY_STATUS)]
    late = status > 0
    limit = df[LIMIT]
    util = df[list(BILL_AMT)].div(limit, axis=0)
    ratios = _payment_ratios(df)
    owed_unpaid = pd.DataFrame(
        {m: (df[BILL_AMT[m + 1]] > 0) & (df[PAY_AMT[m]] == 0) for m in range(N_MONTHS - 1)}
    )

    out = pd.DataFrame(
        {
            "log_limit": np.log1p(limit),
            "pay_status_recent": status.iloc[:, 0],
            "pay_status_prev": status.iloc[:, 1],
            "months_delinquent": late.sum(axis=1),
            "delinquency_streak": _leading_streak(late.astype(int)),
            "max_delay": status.clip(lower=0).max(axis=1),
            "utilization_recent": util.iloc[:, 0],
            "utilization_mean": util.mean(axis=1),
            "utilization_max": util.max(axis=1),
            "utilization_trend": util.iloc[:, 0] - util.iloc[:, -1],
            "payment_ratio_recent": ratios[0],
            "payment_ratio_mean": ratios.mean(axis=1),
            "months_zero_payment": owed_unpaid.sum(axis=1),
            "payment_to_limit_mean": df[list(PAY_AMT)].div(limit, axis=0).mean(axis=1),
        },
        index=df.index,
    )
    return out[list(FEATURES)].astype(float)
