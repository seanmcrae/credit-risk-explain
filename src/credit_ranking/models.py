"""Data split, the two candidate models, and post-hoc probability calibration.

Ranking uses each model's raw score; the calibrator maps that score to a default probability
for display, expected-value maths and calibration audits. Isotonic calibration is a step
function, so ranking on calibrated output would create large ties at the top of the queue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from credit_ranking.config import CalibrationMethod, Config, ModelName, SplitConfig

_EPS = 1e-6


class Classifier(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Any: ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True)
class Splits:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_calib: pd.DataFrame
    y_calib: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def stratified_split(X: pd.DataFrame, y: pd.Series, cfg: SplitConfig, seed: int) -> Splits:
    """Train / calibration / test split stratified on the outcome.

    The source data is a single cross-section (one observation month), so there is no time axis
    to split on; stratification keeps the default rate equal across the three sets.
    """
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=seed
    )
    calib_share = cfg.calibration_size / (1 - cfg.test_size)
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_rest, y_rest, test_size=calib_share, stratify=y_rest, random_state=seed
    )
    return Splits(X_train, y_train, X_calib, y_calib, X_test, y_test)


def build_logistic(params: dict[str, Any], seed: int) -> Pipeline:
    return make_pipeline(StandardScaler(), LogisticRegression(random_state=seed, **params))


def build_lightgbm(
    params: dict[str, Any], features: list[str], constraints: dict[str, int], seed: int
) -> LGBMClassifier:
    params = dict(params)
    if params.pop("monotone", False):
        unknown = set(constraints) - set(features)
        if unknown:
            raise ValueError(f"monotone constraints reference unknown features: {sorted(unknown)}")
        params["monotone_constraints"] = [constraints.get(f, 0) for f in features]
    # Single-threaded deterministic mode makes runs bit-for-bit reproducible.
    return LGBMClassifier(
        random_state=seed, deterministic=True, force_row_wise=True, n_jobs=1, verbose=-1, **params
    )


class Calibrator:
    """Maps raw model probabilities to calibrated default probabilities."""

    def __init__(self, method: CalibrationMethod) -> None:
        self.method = method
        self._isotonic: IsotonicRegression | None = None
        self._platt: LogisticRegression | None = None

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, _EPS, 1 - _EPS)
        return np.asarray(np.log(p / (1 - p))).reshape(-1, 1)

    def fit(self, raw: np.ndarray, y: np.ndarray) -> Calibrator:
        if self.method == "isotonic":
            self._isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self._isotonic.fit(raw, y)
        else:
            self._platt = LogisticRegression(C=1e6).fit(self._logit(raw), y)
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        if self._isotonic is not None:
            return np.asarray(self._isotonic.predict(raw))
        if self._platt is not None:
            return np.asarray(self._platt.predict_proba(self._logit(raw))[:, 1])
        raise RuntimeError("calibrator is not fitted")


@dataclass
class ScoredModel:
    """A fitted classifier plus its calibrator; the unit that is saved and served."""

    name: ModelName
    estimator: Classifier
    calibrator: Calibrator
    features: list[str]

    def raw_score(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(X[self.features])[:, 1])

    def probability(self, X: pd.DataFrame) -> np.ndarray:
        return self.calibrator.transform(self.raw_score(X))


def train_model(name: ModelName, splits: Splits, cfg: Config) -> ScoredModel:
    features = list(splits.X_train.columns)
    estimator: Classifier
    if name == "logistic_regression":
        estimator = build_logistic(cfg.models.logistic_regression, cfg.seed)
    else:
        estimator = build_lightgbm(
            cfg.models.lightgbm, features, cfg.monotone_constraints, cfg.seed
        )
    estimator.fit(splits.X_train, splits.y_train)
    raw_calib = np.asarray(estimator.predict_proba(splits.X_calib))[:, 1]
    calibrator = Calibrator(cfg.calibration.method).fit(raw_calib, splits.y_calib.to_numpy())
    return ScoredModel(name=name, estimator=estimator, calibrator=calibrator, features=features)
