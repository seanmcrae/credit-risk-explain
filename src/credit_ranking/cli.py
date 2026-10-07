"""Command-line interface: ``credit-rank train | evaluate | explain | rank | model-card | ...``."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer
from matplotlib import pyplot as plt

from credit_ranking.config import DEFAULT_CONFIG_PATH, load_config
from credit_ranking.evaluation import evaluate_scores
from credit_ranking.features import FEATURE_DESCRIPTIONS
from credit_ranking.metrics import psi
from credit_ranking.model_card import render_model_card
from credit_ranking.pipeline import METRICS_FILE, load_holdout, load_metrics, save_figures, train
from credit_ranking.plots import waterfall
from credit_ranking.queue import ModelBundle, build_queue, explain_account, prepare
from credit_ranking.schema import ID, TARGET

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)

ArtifactsOpt = Annotated[Path, typer.Option("--artifacts", "-a", help="Training run directory.")]
DataOpt = Annotated[
    Path | None, typer.Option("--data", "-d", help="Canonical CSV; defaults to the run holdout.")
]


def _metric_summary(name: str, ev: dict[str, Any]) -> str:
    p = ev["precision_at_top_pct"]
    top_k = "  ".join(f"P@{k}%={v:.3f}" for k, v in p.items())
    return (
        f"{name:<20} AUC={ev['roc_auc']:.3f}  PR-AUC={ev['pr_auc']:.3f}  KS={ev['ks']:.3f}  "
        f"lift@D1={ev['top_decile_lift']:.2f}  ECE={ev['ece']:.3f}\n{'':<20} {top_k}"
    )


def _economics_summary(ev: dict[str, Any]) -> str:
    return (
        f"Expected value @ {ev['eval_capacity']} accounts: {ev['ev_at_eval_capacity']:,.0f}   "
        f"value-maximizing capacity: {ev['ev_optimal_capacity']} "
        f"({ev['ev_optimal_capacity_pct']:.1f}%) -> {ev['ev_optimal_value']:,.0f}"
    )


@app.command("train")
def train_cmd(
    data: Annotated[Path, typer.Option("--data", "-d", help="Canonical CSV with outcome.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Run directory to write.")] = Path(
        "artifacts/run"
    ),
    config: Annotated[Path, typer.Option("--config", "-c")] = DEFAULT_CONFIG_PATH,
    label: Annotated[str | None, typer.Option(help="Dataset name used in reports.")] = None,
    card: Annotated[Path | None, typer.Option(help="Also write the model card here.")] = None,
    img: Annotated[Path | None, typer.Option(help="Also write charts to this directory.")] = None,
    snapshot: Annotated[
        Path | None,
        typer.Option(help="Also copy metrics.json here (committed results for the docs site)."),
    ] = None,
) -> None:
    """Train both models, calibrate, evaluate on the holdout and save artifacts."""
    result = train(data, load_config(config), out, data_label=label)
    m = result.metrics
    typer.echo(
        f"Data: {m['data']['label']}  rows={m['data']['rows']:,}  "
        f"default rate={m['data']['default_rate']:.1%}"
    )
    typer.echo(f"Holdout: {m['split']['test']:,} accounts. Champion: {m['champion_label']}")
    for name, ev in m["models"].items():
        typer.echo(_metric_summary(name, ev))
    typer.echo(_economics_summary(m["models"][m["champion_label"]]))
    typer.echo(f"Score PSI train vs holdout: {m['psi_train_vs_test']:.4f}")
    if img:
        for path in save_figures(result, img):
            typer.echo(f"Wrote {path}")
    if card:
        img_rel = os.path.relpath(img, card.parent) if img else None
        card.parent.mkdir(parents=True, exist_ok=True)
        card.write_text(render_model_card(m, img_rel))
        typer.echo(f"Wrote {card}")
    if snapshot:
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out / METRICS_FILE, snapshot)
        typer.echo(f"Wrote {snapshot}")
    typer.echo(f"Artifacts in {out}")


@app.command("evaluate")
def evaluate_cmd(artifacts: ArtifactsOpt = Path("artifacts/run"), data: DataOpt = None) -> None:
    """Report holdout metrics, or score a new labeled file and check score stability."""
    if data is None:
        m = load_metrics(artifacts)
        for name, ev in m["models"].items():
            typer.echo(_metric_summary(name, ev))
        typer.echo(_economics_summary(m["models"][m["champion_label"]]))
        lift = pd.DataFrame(m["lift_table"])[
            ["bucket", "accounts", "events", "lift", "capture_rate"]
        ]
        typer.echo(lift.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        return
    bundle = ModelBundle.load(artifacts)
    accounts = pd.read_csv(data)
    if TARGET not in accounts:
        raise typer.BadParameter(f"{data} has no {TARGET} column to evaluate against")
    X = prepare(accounts)
    y = accounts.set_index(ID).loc[X.index, TARGET].to_numpy()
    raw = bundle.champion.raw_score(X)
    ev = evaluate_scores(y, raw, bundle.champion.calibrator.transform(raw), bundle.config)
    reference = bundle.champion.raw_score(prepare(load_holdout(artifacts)))
    typer.echo(_metric_summary(bundle.champion.name, ev))
    typer.echo(_economics_summary(ev))
    typer.echo(f"Score PSI vs training holdout: {psi(reference, raw):.4f}")


@app.command("explain")
def explain_cmd(
    account_id: Annotated[int, typer.Argument(help="Account id to explain.")],
    artifacts: ArtifactsOpt = Path("artifacts/run"),
    data: DataOpt = None,
    plot: Annotated[Path | None, typer.Option(help="Save a SHAP waterfall PNG here.")] = None,
) -> None:
    """Show one account's probability, decile, reason codes and feature contributions."""
    bundle = ModelBundle.load(artifacts)
    accounts = pd.read_csv(data) if data else load_holdout(artifacts)
    try:
        exp = explain_account(bundle, accounts, account_id)
    except KeyError as err:
        raise typer.BadParameter(str(err)) from err
    typer.echo(f"Account {exp.account_id}: P(default)={exp.probability:.3f}  decile={exp.decile}")
    typer.echo("Reasons:" if exp.reasons else "No risk-increasing reasons.")
    for r in exp.reasons:
        typer.echo(f"  {r.code} {r.text}  [{r.driver}={r.driver_value:.2f}, {r.contribution:+.2f}]")
    top = exp.contributions.reindex(exp.contributions.abs().sort_values(ascending=False).index)
    typer.echo("Largest contributions (log-odds):")
    for feature in top.index[:6]:
        name = str(feature)
        typer.echo(
            f"  {top[name]:+.3f}  {FEATURE_DESCRIPTIONS[name]} = {exp.feature_values[name]:.2f}"
        )
    if plot:
        fig = waterfall(exp.contributions, exp.feature_values, exp.base_value)
        fig.savefig(plot, dpi=120)
        plt.close(fig)
        typer.echo(f"Wrote {plot}")


@app.command("rank")
def rank_cmd(
    artifacts: ArtifactsOpt = Path("artifacts/run"),
    data: DataOpt = None,
    capacity: Annotated[
        int | None, typer.Option(min=1, help="Accounts to work (config default).")
    ] = None,
    out: Annotated[
        Path | None, typer.Option(help="CSV path (default <artifacts>/queue.csv).")
    ] = None,
    show: Annotated[int, typer.Option(help="Rows to print.")] = 10,
) -> None:
    """Build the prioritized work queue for the next ``capacity`` accounts."""
    bundle = ModelBundle.load(artifacts)
    accounts = pd.read_csv(data) if data else load_holdout(artifacts)
    cap = capacity or bundle.config.ranking.capacity
    queue = build_queue(bundle, accounts, capacity=cap)
    path = out or artifacts / "queue.csv"
    queue.to_csv(path, index=False)
    typer.echo(f"Queue of {len(queue)} of {len(accounts):,} accounts -> {path}")
    view = queue.head(show).assign(
        primary_reason=lambda q: q["top_reasons"].str.split("; ").str[0].fillna("")
    )[["rank", ID, "probability", "decile", "reason_codes", "primary_reason"]]
    typer.echo(view.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


@app.command("model-card")
def model_card_cmd(
    artifacts: ArtifactsOpt = Path("artifacts/run"),
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("docs/MODEL_CARD.md"),
    img_dir: Annotated[str | None, typer.Option(help="Chart dir relative to the card.")] = None,
) -> None:
    """Render the model card from a run's metrics."""
    out.write_text(render_model_card(load_metrics(artifacts), img_dir))
    typer.echo(f"Wrote {out}")


@app.command("synth")
def synth_cmd(
    out: Annotated[Path, typer.Option("--out", "-o")],
    rows: Annotated[int, typer.Option(min=10)] = 5_000,
    seed: int = 7,
) -> None:
    """Write a SYNTHETIC dataset with the canonical schema."""
    from credit_ranking.synthetic import generate_synthetic

    out.parent.mkdir(parents=True, exist_ok=True)
    generate_synthetic(rows, seed=seed).to_csv(out, index=False)
    typer.echo(f"Wrote {rows:,} synthetic accounts to {out}")


@app.command("download")
def download_cmd(
    dest: Annotated[Path, typer.Option()] = Path("data/raw/uci_credit_default.csv"),
) -> None:
    """Download the UCI dataset (CC BY 4.0); needs the ``data`` extra."""
    from credit_ranking.download import UCI_URL, download_uci

    typer.echo(f"Downloading {UCI_URL}")
    typer.echo(f"Wrote {download_uci(dest)}")
