import dataclasses

import numpy as np
import pandas as pd
import pytest

from credit_ranking.config import Config, load_config
from credit_ranking.features import build_features
from credit_ranking.metrics import roc_auc
from credit_ranking.models import Calibrator, Splits, stratified_split, train_model
from credit_ranking.schema import TARGET
from credit_ranking.synthetic import generate_synthetic


@pytest.fixture(scope="module")
def cfg() -> Config:
    base = load_config()
    fast = {**base.models.lightgbm, "n_estimators": 80}
    return dataclasses.replace(base, models=dataclasses.replace(base.models, lightgbm=fast))


@pytest.fixture(scope="module")
def splits(cfg: Config) -> Splits:
    df = generate_synthetic(3_000, seed=11).set_index("id")
    return stratified_split(build_features(df), df[TARGET], cfg.split, cfg.seed)


def test_split_sizes_and_stratification(splits: Splits, cfg: Config) -> None:
    n = len(splits.X_train) + len(splits.X_calib) + len(splits.X_test)
    assert n == 3_000
    assert len(splits.X_test) == pytest.approx(0.2 * n, abs=1)
    assert len(splits.X_calib) == pytest.approx(0.15 * n, abs=1)
    rates = [s.mean() for s in (splits.y_train, splits.y_calib, splits.y_test)]
    assert max(rates) - min(rates) < 0.01
    ids = [set(x.index) for x in (splits.X_train, splits.X_calib, splits.X_test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])


@pytest.mark.parametrize("name", ["logistic_regression", "lightgbm"])
def test_training_is_deterministic_and_beats_chance(name: str, splits: Splits, cfg: Config) -> None:
    first = train_model(name, splits, cfg).probability(splits.X_test)  # type: ignore[arg-type]
    second = train_model(name, splits, cfg).probability(splits.X_test)  # type: ignore[arg-type]
    np.testing.assert_array_equal(first, second)
    assert roc_auc(splits.y_test.to_numpy(), first) > 0.7


def test_monotone_constraint_holds(splits: Splits, cfg: Config) -> None:
    model = train_model("lightgbm", splits, cfg)
    probe = pd.concat([splits.X_test.iloc[:50]] * 9, ignore_index=True)
    probe["pay_status_recent"] = np.repeat(np.arange(-1, 8), 50)
    raw = model.raw_score(probe).reshape(9, 50)
    assert (np.diff(raw, axis=0) >= -1e-12).all()


def test_unknown_monotone_feature_rejected(splits: Splits, cfg: Config) -> None:
    bad = dataclasses.replace(cfg, monotone_constraints={"not_a_feature": 1})
    with pytest.raises(ValueError, match="unknown features"):
        train_model("lightgbm", splits, bad)


@pytest.mark.parametrize("method", ["isotonic", "platt"])
def test_calibrator_fixes_systematic_overconfidence(method: str) -> None:
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.02, 0.6, 20_000)
    y = (rng.random(true_p.size) < true_p).astype(int)
    raw = np.clip(true_p * 1.5, 0, 1)  # model overstates risk by 50%
    cal = Calibrator(method).fit(raw, y)  # type: ignore[arg-type]
    out = cal.transform(raw)
    assert np.all((out >= 0) & (out <= 1))
    assert abs(out.mean() - y.mean()) < abs(raw.mean() - y.mean()) / 5


def test_unfitted_calibrator_raises() -> None:
    with pytest.raises(RuntimeError):
        Calibrator("isotonic").transform(np.array([0.5]))
