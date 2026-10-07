from credit_ranking.model_card import render_model_card, table
from credit_ranking.pipeline import TrainResult


def test_table_renders_markdown() -> None:
    assert table(["a", "b"], [(1, 2)]) == "| a | b |\n|---|---|\n| 1 | 2 |"


def test_model_card_reports_run_numbers(trained_run: TrainResult) -> None:
    m = trained_run.metrics
    card = render_model_card(m, img_dir="img")
    champ = m["models"][m["champion_label"]]
    assert card.startswith("# Model card")
    assert "**test synthetic**" in card
    assert "synthetic** sample" in card  # synthetic runs must be labeled as such
    assert f"{champ['roc_auc']:.3f}" in card
    for section in ("Intended use", "Fairness audit", "Limitations", "Queue economics"):
        assert f"## {section}" in card
    assert "![Reliability curve](img/reliability_curve.png)" in card
    assert "`sex`" in card and "excluded from the features" in card


def test_real_data_card_cites_source(trained_run: TrainResult) -> None:
    m = {**trained_run.metrics, "data": {**trained_run.metrics["data"], "label": "UCI data"}}
    card = render_model_card(m)
    assert "Yeh & Lien, 2009" in card
    assert "![" not in card


def test_card_names_largest_tpr_gap(trained_run: TrainResult) -> None:
    m = trained_run.metrics
    worst = max(m["fairness_summary"], key=lambda s: s["tpr_gap"])["attribute"]
    assert f"largest TPR gap is on `{worst}`" in render_model_card(m)
