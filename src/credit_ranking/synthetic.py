"""Seeded SYNTHETIC generator with the same schema as the UCI credit default data.

Rows are fabricated. Relationships are hand-specified to be plausible, not fitted to real data:
a latent risk factor drives repayment delays, utilization and partial payments, and default
depends on that factor plus recent delinquency. Protected attributes influence credit limits
mildly (as they can through income in real portfolios) but never the default outcome directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_ranking.schema import (
    BILL_AMT,
    COLUMNS,
    ID,
    LIMIT,
    N_MONTHS,
    PAY_AMT,
    PAY_STATUS,
    TARGET,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-x)))


def generate_synthetic(n: int, seed: int = 0) -> pd.DataFrame:
    """Return ``n`` synthetic accounts; identical output for identical ``(n, seed)``."""
    rng = np.random.default_rng(seed)

    sex = rng.choice([1, 2], size=n, p=[0.4, 0.6])
    education = rng.choice(
        [1, 2, 3, 4, 5, 6, 0], size=n, p=[0.35, 0.465, 0.16, 0.005, 0.01, 0.005, 0.005]
    )
    marriage = rng.choice([1, 2, 3, 0], size=n, p=[0.45, 0.53, 0.015, 0.005])
    age = np.clip(21 + rng.gamma(shape=3.0, scale=4.5, size=n) + 4 * (marriage == 1), 21, 79)
    age = age.astype(int)

    edu_effect = np.select([education == 1, education == 2, education == 3], [0.35, 0.0, -0.3], 0.0)
    log_limit = 11.7 + edu_effect + 0.012 * (age - 35) + rng.normal(0, 0.7, n)
    limit = np.clip(np.round(np.exp(log_limit), -4), 10_000, 1_000_000)

    # Latent creditworthiness: higher means riskier. Lower limits carry slightly more risk.
    risk = rng.normal(0, 1, n) - 0.35 * (log_limit - 11.7)

    status = np.zeros((n, N_MONTHS), dtype=int)  # column 0 = most recent month
    prev = np.zeros(n, dtype=int)
    for col in reversed(range(N_MONTHS)):  # simulate oldest month first
        p_late = _sigmoid(-2.3 + 1.1 * risk + 1.6 * (prev > 0))
        late = rng.random(n) < p_late
        roll = np.where(prev > 0, prev + 1, rng.choice([1, 2], size=n, p=[0.6, 0.4]))
        cured = rng.random(n) < 0.25  # some late accounts catch up partially
        late_status = np.where(cured & (prev > 1), prev - 1, roll)
        p_full = _sigmoid(-1.4 - 1.0 * risk)
        current = np.where(rng.random(n) < p_full, -1, 0)
        current = np.where(rng.random(n) < 0.08, -2, current)  # no consumption this month
        status[:, col] = np.where(late, np.minimum(late_status, 8), current)
        prev = status[:, col]

    util_base = _sigmoid(-0.4 + 0.9 * risk + rng.normal(0, 0.6, n))
    util = np.clip(util_base[:, None] + rng.normal(0, 0.08, (n, N_MONTHS)), -0.02, 1.15)
    bills = np.round(util * limit[:, None])
    bills[status == -2] = 0.0

    # pay_amt in month m settles the statement of month m+1 (the previous month).
    prior_bill = np.concatenate([bills[:, 1:], bills[:, -1:]], axis=1)
    ratio = _sigmoid(-1.9 - 0.9 * risk[:, None] + rng.normal(0, 0.9, (n, N_MONTHS)))
    ratio = np.where(status == -1, 1.0, ratio)
    ratio = np.where(status > 0, ratio * 0.25, ratio)
    payments = np.round(np.clip(ratio * prior_bill, 0, None))

    recent = status[:, 0]
    logit = (
        -1.85
        + 0.75 * risk
        + 0.9 * (recent > 0)
        + 0.25 * np.clip(recent, 0, 4)
        + 0.6 * (util[:, 0] - 0.5)
        + rng.normal(0, 0.3, n)
    )
    default = (rng.random(n) < _sigmoid(logit)).astype(int)

    frame = pd.DataFrame(
        {
            ID: np.arange(1, n + 1),
            LIMIT: limit,
            "sex": sex,
            "age": age,
            "education": education,
            "marriage": marriage,
            **{c: status[:, i] for i, c in enumerate(PAY_STATUS)},
            **{c: bills[:, i] for i, c in enumerate(BILL_AMT)},
            **{c: payments[:, i] for i, c in enumerate(PAY_AMT)},
            TARGET: default,
        }
    )
    return frame[list(COLUMNS)]
