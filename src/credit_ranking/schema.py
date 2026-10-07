"""Canonical column names and the validation schema every input dataset must satisfy.

Column naming follows the UCI dataset with one fix: the source labels the most recent repayment
status ``PAY_0`` but the matching bill and payment columns ``*_1``. Here month 1 is the most recent
month (September 2005 in the source) for every series.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

N_MONTHS = 6
MONTHS = tuple(range(1, N_MONTHS + 1))

ID = "id"
TARGET = "default_next_month"
LIMIT = "limit_bal"
PAY_STATUS = tuple(f"pay_status_{m}" for m in MONTHS)
BILL_AMT = tuple(f"bill_amt_{m}" for m in MONTHS)
PAY_AMT = tuple(f"pay_amt_{m}" for m in MONTHS)

# Present in the data for auditing only. Never used as model features.
PROTECTED_ATTRIBUTES = ("sex", "age", "education", "marriage")

COLUMNS = (ID, LIMIT, *PROTECTED_ATTRIBUTES, *PAY_STATUS, *BILL_AMT, *PAY_AMT, TARGET)

_status = pa.Check.in_range(-2, 9)

SCHEMA = pa.DataFrameSchema(
    {
        ID: pa.Column(int, unique=True),
        LIMIT: pa.Column(float, pa.Check.gt(0), coerce=True),
        "sex": pa.Column(int, pa.Check.isin([1, 2])),
        "age": pa.Column(int, pa.Check.in_range(18, 100)),
        "education": pa.Column(int, pa.Check.in_range(0, 6)),
        "marriage": pa.Column(int, pa.Check.in_range(0, 3)),
        **{c: pa.Column(int, _status) for c in PAY_STATUS},
        **{c: pa.Column(float, coerce=True) for c in BILL_AMT},
        **{c: pa.Column(float, pa.Check.ge(0), coerce=True) for c in PAY_AMT},
        TARGET: pa.Column(int, pa.Check.isin([0, 1]), required=False),
    },
    strict=True,
    coerce=True,
    ordered=False,
)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce ``df``; raises ``pandera.errors.SchemaErrors`` listing every failure."""
    validated: pd.DataFrame = SCHEMA.validate(df, lazy=True)
    return validated
