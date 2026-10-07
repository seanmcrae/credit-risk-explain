from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from credit_ranking.cli import app
from credit_ranking.pipeline import TrainResult, load_holdout

runner = CliRunner()


def test_train_writes_card_and_charts(
    synthetic_csv: Path, fast_config_path: Path, tmp_path: Path
) -> None:
    out, card, img = (
        tmp_path / "run",
        tmp_path / "docs" / "MODEL_CARD.md",
        tmp_path / "docs" / "img",
    )
    result = runner.invoke(
        app,
        [
            "train",
            "--data",
            str(synthetic_csv),
            "--out",
            str(out),
            "--config",
            str(fast_config_path),
            "--label",
            "cli synthetic",
            "--card",
            str(card),
            "--img",
            str(img),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Champion: LightGBM" in result.output
    assert "](img/lift_curve.png)" in card.read_text()
    assert (img / "shap_summary.png").exists()


def test_rank_limits_queue_to_capacity(trained_run: TrainResult, tmp_path: Path) -> None:
    out = tmp_path / "queue.csv"
    result = runner.invoke(
        app,
        [
            "rank",
            "-a",
            str(trained_run.out_dir),
            "--capacity",
            "12",
            "--out",
            str(out),
            "--show",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    queue = pd.read_csv(out)
    assert len(queue) == 12
    assert list(queue.columns[:5]) == ["id", "rank", "probability", "raw_score", "decile"]


def test_explain_known_and_unknown_account(trained_run: TrainResult, tmp_path: Path) -> None:
    account = int(load_holdout(trained_run.out_dir)["id"].iloc[0])
    plot = tmp_path / "waterfall.png"
    ok = runner.invoke(
        app, ["explain", str(account), "-a", str(trained_run.out_dir), "--plot", str(plot)]
    )
    assert ok.exit_code == 0, ok.output
    assert f"Account {account}: P(default)=" in ok.output
    assert plot.stat().st_size > 5_000
    missing = runner.invoke(app, ["explain", "999999", "-a", str(trained_run.out_dir)])
    assert missing.exit_code != 0


def test_evaluate_saved_and_new_data(trained_run: TrainResult, tmp_path: Path) -> None:
    saved = runner.invoke(app, ["evaluate", "-a", str(trained_run.out_dir)])
    assert saved.exit_code == 0, saved.output
    assert "AUC=" in saved.output
    fresh = tmp_path / "fresh.csv"
    assert (
        runner.invoke(
            app, ["synth", "--out", str(fresh), "--rows", "400", "--seed", "99"]
        ).exit_code
        == 0
    )
    scored = runner.invoke(app, ["evaluate", "-a", str(trained_run.out_dir), "--data", str(fresh)])
    assert scored.exit_code == 0, scored.output
    assert "PSI vs training holdout" in scored.output
