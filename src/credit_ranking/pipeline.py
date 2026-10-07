"""End-to-end training run: validate, featurize, split, fit, calibrate, evaluate, audit, save."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from credit_ranking import plots
from credit_ranking.config import Config, ModelName
from credit_ranking.data import audit_groups, load_dataset
from credit_ranking.evaluation import evaluate_scores
from credit_ranking.explain import global_importance, reason_table
from credit_ranking.fairness import (
    contributions_by_group,
    disparity_summary,
    largest_tpr_gap,
    slice_metrics,
)
from credit_ranking.features import build_features
from credit_ranking.metrics import lift_table, psi, top_k_count
from credit_ranking.models import ScoredModel, stratified_split, train_model
from credit_ranking.queue import ModelBundle, build_queue, decile_edges, explain_account
from credit_ranking.schema import ID, TARGET

MODEL_LABELS: dict[ModelName, str] = {
    "logistic_regression": "Logistic regression",
    "lightgbm": "LightGBM",
}
HOLDOUT_FILE = "holdout_accounts.csv"
METRICS_FILE = "metrics.json"
BACKGROUND_ROWS = 500


@dataclass
class TrainResult:
    out_dir: Path
    metrics: dict[str, Any]
    bundle: ModelBundle


def _to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = json.loads(df.to_json(orient="records"))
    return records


def train(
    data_path: Path, cfg: Config, out_dir: Path, data_label: str | None = None
) -> TrainResult:
    accounts = load_dataset(data_path)
    indexed = accounts.set_index(ID)
    X, y = build_features(indexed), indexed[TARGET]
    groups = audit_groups(indexed)
    splits = stratified_split(X, y, cfg.split, cfg.seed)

    challenger_name: ModelName = (
        "logistic_regression" if cfg.models.champion == "lightgbm" else "lightgbm"
    )
    models: dict[ModelName, ScoredModel] = {
        name: train_model(name, splits, cfg) for name in (cfg.models.champion, challenger_name)
    }
    champion = models[cfg.models.champion]

    X_test, y_test = splits.X_test, splits.y_test.to_numpy()
    raw = {name: m.raw_score(X_test) for name, m in models.items()}
    prob = {name: m.probability(X_test) for name, m in models.items()}
    evaluation = {name: evaluate_scores(y_test, raw[name], prob[name], cfg) for name in models}

    bundle = ModelBundle(
        champion=champion,
        challenger=models[challenger_name],
        background=splits.X_train.sample(
            min(BACKGROUND_ROWS, len(splits.X_train)), random_state=cfg.seed
        ),
        decile_edges=decile_edges(raw[cfg.models.champion]),
        config=cfg,
    )
    attribution = bundle.attribution(X_test)
    importance = global_importance(attribution)
    capacity = top_k_count(len(y_test), cfg.ranking.eval_capacity_percent)
    champ_raw, champ_prob = raw[cfg.models.champion], prob[cfg.models.champion]
    test_groups = groups.loc[X_test.index]
    slices = slice_metrics(test_groups, y_test, champ_prob, champ_raw, capacity)
    disparities = disparity_summary(slices)
    group_contributions = contributions_by_group(attribution.values, test_groups, y_test)

    out_dir.mkdir(parents=True, exist_ok=True)
    bundle.save(out_dir)
    accounts.set_index(ID).loc[X_test.index].reset_index().to_csv(
        out_dir / HOLDOUT_FILE, index=False
    )
    scored = (
        pd.DataFrame(
            {
                "probability": champ_prob,
                "raw_score": champ_raw,
                "decile": bundle.decile(champ_raw),
                TARGET: y_test,
            },
            index=X_test.index,
        )
        .join(groups)
        .join(reason_table(attribution, cfg.n_reasons))
    )
    scored.sort_values("raw_score", ascending=False).to_csv(out_dir / "holdout_scores.csv")
    lift = lift_table(y_test, champ_raw)
    lift.to_csv(out_dir / "lift_table.csv", index=False)
    slices.to_csv(out_dir / "fairness_slices.csv", index=False)
    disparities.to_csv(out_dir / "fairness_summary.csv", index=False)
    importance.rename("mean_abs_shap").to_csv(out_dir / "global_importance.csv")

    metrics: dict[str, Any] = {
        "data": {
            "label": data_label or data_path.name,
            "path": str(data_path),
            "rows": len(accounts),
            "default_rate": float(y.mean()),
        },
        "split": {
            "train": len(splits.X_train),
            "calibration": len(splits.X_calib),
            "test": len(X_test),
        },
        "champion": cfg.models.champion,
        "models": {MODEL_LABELS[name]: ev for name, ev in evaluation.items()},
        "champion_label": MODEL_LABELS[cfg.models.champion],
        "psi_train_vs_test": psi(champion.raw_score(splits.X_train), champ_raw),
        "lift_table": _to_records(lift),
        "global_importance": {k: float(v) for k, v in importance.items()},
        "fairness_slices": _to_records(slices),
        "fairness_summary": _to_records(disparities),
        "fairness_contributions": _to_records(group_contributions),
        "config": asdict(cfg),
        "break_even_probability": cfg.economics.break_even_probability,
    }
    (out_dir / METRICS_FILE).write_text(json.dumps(metrics, indent=2, default=str))
    return TrainResult(out_dir=out_dir, metrics=metrics, bundle=bundle)


def save_figures(result: TrainResult, img_dir: Path) -> list[Path]:
    """Render the holdout charts from a finished run's saved artifacts."""
    img_dir.mkdir(parents=True, exist_ok=True)
    cfg = result.bundle.config
    holdout = pd.read_csv(result.out_dir / HOLDOUT_FILE)
    X = build_features(holdout.set_index(ID))
    y = holdout[TARGET].to_numpy()
    champ, chall = result.bundle.champion, result.bundle.challenger
    champ_label, chall_label = MODEL_LABELS[champ.name], MODEL_LABELS[chall.name]
    figures = {
        "lift_curve.png": plots.lift_curve(
            y,
            {champ_label: champ.raw_score(X), chall_label: chall.raw_score(X)},
            cfg.ranking.eval_capacity_percent / 100,
        ),
        "reliability_curve.png": plots.reliability_plot(
            y,
            {
                f"{champ_label} calibrated ({cfg.calibration.method})": champ.probability(X),
                f"{champ_label} uncalibrated": champ.raw_score(X),
            },
            cfg.calibration.n_bins,
        ),
        "expected_value.png": plots.expected_value_plot(y, champ.raw_score(X), cfg.economics),
        "shap_summary.png": plots.shap_summary(result.bundle.attribution(X)),
        "fairness.png": _fairness_figure(result),
        "queue_dashboard.png": _dashboard(result, holdout),
    }
    paths = []
    for name, fig in figures.items():
        path = img_dir / name
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
    return paths


def _fairness_figure(result: TrainResult) -> Figure:
    m = result.metrics
    slices = pd.DataFrame(m["fairness_slices"])
    contributions = pd.DataFrame(m["fairness_contributions"])
    return plots.fairness_chart(
        slices,
        contributions,
        largest_tpr_gap(slices),
        m["config"]["ranking"]["eval_capacity_percent"],
    )


def _dashboard(result: TrainResult, holdout: pd.DataFrame) -> Figure:
    bundle, cfg = result.bundle, result.bundle.config
    X = build_features(holdout.set_index(ID))
    raw = bundle.champion.raw_score(X)
    queue = build_queue(bundle, holdout, capacity=10)
    top = explain_account(bundle, holdout, int(queue[ID].iloc[0]))
    return plots.queue_dashboard(
        queue,
        top,
        holdout[TARGET].to_numpy(),
        raw,
        bundle.champion.calibrator.transform(raw),
        cfg.ranking.eval_capacity_percent / 100,
        cfg.calibration.n_bins,
        title=f"Holdout work queue - {result.metrics['data']['label']}",
    )


def load_metrics(out_dir: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = json.loads((out_dir / METRICS_FILE).read_text())
    return metrics


def load_holdout(out_dir: Path) -> pd.DataFrame:
    return pd.read_csv(out_dir / HOLDOUT_FILE)


__all__ = ["TrainResult", "load_holdout", "load_metrics", "save_figures", "train"]
