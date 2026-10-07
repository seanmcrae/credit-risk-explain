"""Typed run configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"

ModelName = Literal["logistic_regression", "lightgbm"]
CalibrationMethod = Literal["isotonic", "platt"]


@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.2
    calibration_size: float = 0.15


@dataclass(frozen=True)
class ModelsConfig:
    champion: ModelName = "lightgbm"
    logistic_regression: dict[str, Any] = field(default_factory=dict)
    lightgbm: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CalibrationConfig:
    method: CalibrationMethod = "isotonic"
    n_bins: int = 10


@dataclass(frozen=True)
class EconomicsConfig:
    """Illustrative value of working one account; every figure is a configurable assumption."""

    cost_per_contact: float = 60.0
    loss_given_default: float = 4000.0
    cure_rate_if_worked: float = 0.10

    @property
    def value_per_true_positive(self) -> float:
        """Expected loss avoided by working an account that would otherwise default."""
        return self.loss_given_default * self.cure_rate_if_worked

    @property
    def break_even_probability(self) -> float:
        """Default probability above which working an account has positive expected value."""
        return self.cost_per_contact / self.value_per_true_positive


@dataclass(frozen=True)
class RankingConfig:
    capacity: int = 1000
    top_k_percents: tuple[float, ...] = (1, 5, 10, 20)


@dataclass(frozen=True)
class Config:
    seed: int = 42
    split: SplitConfig = field(default_factory=SplitConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    monotone_constraints: dict[str, int] = field(default_factory=dict)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    economics: EconomicsConfig = field(default_factory=EconomicsConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    n_reasons: int = 4

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        ranking = dict(raw.get("ranking", {}))
        if "top_k_percents" in ranking:
            ranking["top_k_percents"] = tuple(ranking["top_k_percents"])
        constraints = {k: int(v) for k, v in raw.get("monotone_constraints", {}).items()}
        bad = {k: v for k, v in constraints.items() if v not in (-1, 1)}
        if bad:
            raise ValueError(f"monotone constraints must be -1 or +1, got {bad}")
        return cls(
            seed=int(raw.get("seed", 42)),
            split=SplitConfig(**raw.get("split", {})),
            models=ModelsConfig(**raw.get("models", {})),
            monotone_constraints=constraints,
            calibration=CalibrationConfig(**raw.get("calibration", {})),
            economics=EconomicsConfig(**raw.get("economics", {})),
            ranking=RankingConfig(**ranking),
            n_reasons=int(raw.get("explain", {}).get("n_reasons", 4)),
        )


def load_config(path: Path | None = None) -> Config:
    with (path or DEFAULT_CONFIG_PATH).open() as fh:
        return Config.from_dict(yaml.safe_load(fh) or {})
