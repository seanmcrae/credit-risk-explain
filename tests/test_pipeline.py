from pathlib import Path

import numpy as np
import pandas as pd

from credit_ranking.pipeline import TrainResult, load_holdout, load_metrics, save_figures
from credit_ranking.queue import ModelBundle, build_queue, explain_account


def test_run_writes_artifacts_and_consistent_metrics(trained_run: TrainResult) -> None:
    out = trained_run.out_dir
    for name in (
        "model.joblib",
        "metrics.json",
        "holdout_accounts.csv",
        "holdout_scores.csv",
        "lift_table.csv",
        "fairness_slices.csv",
        "global_importance.csv",
    ):
        assert (out / name).exists(), name
    metrics = load_metrics(out)
    assert metrics["data"]["label"] == "test synthetic"
    assert metrics["split"]["test"] == len(load_holdout(out)) == 300
    champ = metrics["models"][metrics["champion_label"]]
    assert 0.5 < champ["roc_auc"] <= 1
    assert champ["ev_optimal_value"] >= champ["ev_at_eval_capacity"]
    assert {s["attribute"] for s in metrics["fairness_slices"]} == {
        "sex",
        "age_band",
        "education",
        "marriage",
    }


def test_bundle_round_trip_and_queue(trained_run: TrainResult) -> None:
    bundle = ModelBundle.load(trained_run.out_dir)
    holdout = load_holdout(trained_run.out_dir)
    queue = build_queue(bundle, holdout, capacity=25)
    assert len(queue) == 25
    assert queue["rank"].tolist() == list(range(1, 26))
    assert queue["raw_score"].is_monotonic_decreasing
    assert queue["decile"].between(1, 10).all()
    assert queue["decile"].iloc[0] == 1
    assert queue["top_reasons"].str.len().gt(0).all()


def test_scoring_ignores_protected_attributes(trained_run: TrainResult) -> None:
    bundle = ModelBundle.load(trained_run.out_dir)
    holdout = load_holdout(trained_run.out_dir).head(50)
    flipped = holdout.assign(sex=3 - holdout["sex"], age=holdout["age"].clip(upper=60) + 10)
    a = build_queue(bundle, holdout).set_index("id")["probability"]
    b = build_queue(bundle, flipped).set_index("id")["probability"]
    pd.testing.assert_series_equal(a.sort_index(), b.sort_index())


def test_explain_account_matches_queue(trained_run: TrainResult) -> None:
    bundle = ModelBundle.load(trained_run.out_dir)
    holdout = load_holdout(trained_run.out_dir)
    top = build_queue(bundle, holdout, capacity=1).iloc[0]
    explanation = explain_account(bundle, holdout, int(top["id"]))
    assert np.isclose(explanation.probability, top["probability"])
    assert explanation.decile == top["decile"]
    assert ";".join(r.code for r in explanation.reasons) == top["reason_codes"]


def test_figures_render(trained_run: TrainResult, tmp_path: Path) -> None:
    paths = save_figures(trained_run, tmp_path)
    assert {p.name for p in paths} == {
        "lift_curve.png",
        "reliability_curve.png",
        "expected_value.png",
        "shap_summary.png",
    }
    assert all(p.stat().st_size > 10_000 for p in paths)
