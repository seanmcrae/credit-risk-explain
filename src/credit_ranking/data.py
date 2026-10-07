"""Loading the canonical dataset and decoding audit attributes into readable groups."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from credit_ranking.schema import BILL_AMT, ID, LIMIT, PAY_AMT, PAY_STATUS, TARGET, validate

_UCI_RENAMES: dict[str, str] = {
    "ID": ID,
    "LIMIT_BAL": LIMIT,
    "SEX": "sex",
    "EDUCATION": "education",
    "MARRIAGE": "marriage",
    "AGE": "age",
    "default payment next month": TARGET,
    **{
        src: dst
        for src, dst in zip(
            ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"), PAY_STATUS, strict=True
        )
    },
    **{f"BILL_AMT{m}": dst for m, dst in enumerate(BILL_AMT, start=1)},
    **{f"PAY_AMT{m}": dst for m, dst in enumerate(PAY_AMT, start=1)},
}

SEX_LABELS = {1: "male", 2: "female"}
EDUCATION_LABELS = {1: "graduate school", 2: "university", 3: "high school"}
MARRIAGE_LABELS = {1: "married", 2: "single"}
AGE_BANDS = (18, 25, 35, 45, 55, 101)
AGE_BAND_LABELS = ("18-24", "25-34", "35-44", "45-54", "55+")


def normalize_uci_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename the UCI spreadsheet headers to canonical names."""
    missing = set(_UCI_RENAMES) - set(raw.columns)
    if missing:
        raise ValueError(f"not a UCI credit default table; missing columns: {sorted(missing)}")
    return raw.rename(columns=_UCI_RENAMES)[list(_UCI_RENAMES.values())]


def load_dataset(path: Path) -> pd.DataFrame:
    """Read a canonical CSV and validate it against the schema."""
    return validate(pd.read_csv(path))


def audit_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Readable group labels for protected attributes, used only for fairness slicing."""
    age_band = pd.cut(df["age"], bins=list(AGE_BANDS), right=False, labels=list(AGE_BAND_LABELS))
    return pd.DataFrame(
        {
            "sex": df["sex"].map(SEX_LABELS).fillna("unknown"),
            "age_band": age_band.astype(str),
            # Codes 0, 4, 5, 6 are undocumented in the source; pool them rather than guess.
            "education": df["education"].map(EDUCATION_LABELS).fillna("other/unknown"),
            "marriage": df["marriage"].map(MARRIAGE_LABELS).fillna("other/unknown"),
        },
        index=df.index,
    )
