"""The saved model bundle and what is served from it: ranking a queue, explaining one account."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from credit_ranking.config import Config
from credit_ranking.explain import Attribution, ReasonCode, attribute, reason_codes, reason_table
from credit_ranking.features import build_features
from credit_ranking.models import ScoredModel
from credit_ranking.schema import ID, validate

BUNDLE_FILE = "model.joblib"


@dataclass
class ModelBundle:
    champion: ScoredModel
    challenger: ScoredModel
    background: pd.DataFrame  # training-feature sample used as the SHAP reference for linear models
    decile_edges: (
        np.ndarray
    )  # holdout raw-score cut points, so deciles mean the same across batches
    config: Config

    def save(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / BUNDLE_FILE
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(out_dir: Path) -> ModelBundle:
        bundle = joblib.load(out_dir / BUNDLE_FILE)
        if not isinstance(bundle, ModelBundle):
            raise TypeError(f"{out_dir / BUNDLE_FILE} does not contain a ModelBundle")
        return bundle

    def decile(self, raw: np.ndarray) -> np.ndarray:
        """Decile 1 = riskiest 10% of the holdout reference population."""
        return np.asarray(10 - np.searchsorted(self.decile_edges, raw, side="right"))

    def attribution(self, features: pd.DataFrame) -> Attribution:
        return attribute(self.champion, features, self.background)


def decile_edges(raw_holdout: np.ndarray) -> np.ndarray:
    return np.asarray(np.quantile(raw_holdout, np.linspace(0.1, 0.9, 9)))


def prepare(accounts: pd.DataFrame) -> pd.DataFrame:
    """Validate canonical rows and compute features, indexed by account id."""
    return build_features(validate(accounts).set_index(ID))


def build_queue(
    bundle: ModelBundle, accounts: pd.DataFrame, capacity: int | None = None
) -> pd.DataFrame:
    """Score accounts and return them in work order with probability, decile and top reasons."""
    if capacity is not None and capacity < 1:
        raise ValueError(f"capacity must be at least 1, got {capacity}")
    features = prepare(accounts)
    raw = bundle.champion.raw_score(features)
    order = np.argsort(-raw, kind="stable")
    if capacity is not None:
        order = order[:capacity]
    chosen = features.iloc[order]
    reasons = reason_table(bundle.attribution(chosen), bundle.config.n_reasons)
    queue = pd.DataFrame(
        {
            "rank": np.arange(1, len(order) + 1),
            "probability": bundle.champion.calibrator.transform(raw[order]),
            "raw_score": raw[order],
            "decile": bundle.decile(raw[order]),
        },
        index=chosen.index,
    ).join(reasons)
    queue.index.name = ID
    return queue.reset_index()


@dataclass(frozen=True)
class AccountExplanation:
    account_id: int
    probability: float
    decile: int
    reasons: list[ReasonCode]
    contributions: pd.Series  # log-odds contribution per feature
    feature_values: pd.Series
    base_value: float


def explain_account(
    bundle: ModelBundle, accounts: pd.DataFrame, account_id: int
) -> AccountExplanation:
    row = accounts[accounts[ID] == account_id]
    if row.empty:
        raise KeyError(f"account {account_id} not found")
    features = prepare(row)
    raw = bundle.champion.raw_score(features)
    attr = bundle.attribution(features)
    contrib, values = attr.values.iloc[0], attr.data.iloc[0]
    return AccountExplanation(
        account_id=account_id,
        probability=float(bundle.champion.calibrator.transform(raw)[0]),
        decile=int(bundle.decile(raw)[0]),
        reasons=reason_codes(contrib, values, bundle.config.n_reasons),
        contributions=contrib,
        feature_values=values,
        base_value=attr.base_value,
    )
