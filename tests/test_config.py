import pytest

from credit_ranking.config import Config, EconomicsConfig, load_config


def test_default_config_loads() -> None:
    cfg = load_config()
    assert cfg.models.champion in {"lightgbm", "logistic_regression"}
    assert 0 < cfg.split.test_size < 1
    assert cfg.ranking.top_k_percents == (1, 5, 10, 20)


def test_break_even_probability() -> None:
    econ = EconomicsConfig(cost_per_contact=60, loss_given_default=4000, cure_rate_if_worked=0.1)
    assert econ.value_per_true_positive == pytest.approx(400)
    assert econ.break_even_probability == pytest.approx(0.15)


def test_invalid_monotone_direction_rejected() -> None:
    with pytest.raises(ValueError, match="monotone"):
        Config.from_dict({"monotone_constraints": {"max_delay": 2}})
