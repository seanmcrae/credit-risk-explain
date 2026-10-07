import dataclasses

import numpy as np
import pandas as pd
import pytest

from credit_ranking.config import load_config
from credit_ranking.explain import (
    REASONS,
    attribute,
    global_importance,
    reason_codes,
    reason_table,
)
from credit_ranking.features import FEATURES, build_features
from credit_ranking.models import stratified_split, train_model
from credit_ranking.schema import TARGET
from credit_ranking.synthetic import generate_synthetic


def _row(**contribs: float) -> pd.Series:
    return pd.Series({f: contribs.get(f, 0.0) for f in FEATURES})


def test_every_feature_maps_to_exactly_one_reason() -> None:
    mapped = [f for r in REASONS for f in r.features]
    assert sorted(mapped) == sorted(FEATURES)
    assert len({r.code for r in REASONS}) == len(REASONS)


def test_reason_codes_aggregate_group_and_rank() -> None:
    contrib = _row(utilization_recent=0.3, utilization_max=0.2, pay_status_recent=0.4,
                   months_zero_payment=0.1, log_limit=-0.9)  # fmt: skip
    values = _row(utilization_recent=0.95, pay_status_recent=2)
    codes = reason_codes(contrib, values, n=4)
    assert [c.code for c in codes] == ["R03", "R01", "R06"]
    assert codes[0].contribution == pytest.approx(0.5)
    assert codes[0].driver == "utilization_recent"
    assert codes[0].driver_value == pytest.approx(0.95)


def test_reason_codes_exclude_net_favourable_groups_and_respect_n() -> None:
    contrib = _row(payment_ratio_recent=0.5, payment_ratio_mean=-0.7,  # net -0.2 -> excluded
                   max_delay=0.3, utilization_trend=0.2, log_limit=0.1)  # fmt: skip
    codes = reason_codes(contrib, _row(), n=2)
    assert [c.code for c in codes] == ["R02", "R04"]


def test_no_reasons_for_uniformly_low_risk_account() -> None:
    assert reason_codes(_row(pay_status_recent=-0.5), _row(), n=4) == []


@pytest.mark.parametrize("name", ["lightgbm", "logistic_regression"])
def test_attributions_sum_to_model_log_odds(name: str) -> None:
    base = load_config()
    cfg = dataclasses.replace(
        base,
        models=dataclasses.replace(
            base.models, lightgbm={**base.models.lightgbm, "n_estimators": 60}
        ),
    )
    df = generate_synthetic(2_000, seed=5).set_index("id")
    splits = stratified_split(build_features(df), df[TARGET], cfg.split, cfg.seed)
    model = train_model(name, splits, cfg)  # type: ignore[arg-type]
    X = splits.X_test.iloc[:40]
    attr = attribute(model, X, splits.X_train)
    raw = np.clip(model.raw_score(X), 1e-12, 1 - 1e-12)
    np.testing.assert_allclose(attr.base_value + attr.values.sum(axis=1), np.log(raw / (1 - raw)),
                               atol=1e-5)  # fmt: skip
    assert global_importance(attr).index[0] in FEATURES
    table = reason_table(attr, n=3)
    assert list(table.index) == list(X.index)
    assert table["reason_codes"].str.count("R").max() <= 3
