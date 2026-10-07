import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from credit_ranking.cli import app
from credit_ranking.docsite import SiteInputs, build_site
from credit_ranking.docsite.build import readme_sections, rewrite_links, slugify
from credit_ranking.docsite.flowchart import layers, parse, to_svg
from credit_ranking.model_card import pct

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results" / "uci_metrics.json"
REPO = "https://github.com/seanmcrae/credit-risk-explain"


def _readme_mermaid() -> str:
    match = re.search(r"```mermaid\n(.*?)```", (ROOT / "README.md").read_text(), re.DOTALL)
    assert match, "README must keep its architecture diagram"
    return match.group(1)


def test_readme_flowchart_parses_and_layers() -> None:
    chart = parse(_readme_mermaid())
    depth = layers(chart)
    assert min(depth.values()) == 0
    sinks = {n for n in chart.labels if all(src != n for src, _ in chart.edges)}
    assert {depth[n] for n in sinks} == {max(depth.values())}
    svg = to_svg(chart)
    assert svg.count("<rect") == len(chart.labels)
    assert svg.count('class="fc-edge"') == len(chart.edges)


def test_flowchart_labels_and_errors() -> None:
    chart = parse('flowchart LR\n  A["Load & check"] --> B\n  B --> C["Score"]')
    assert chart.labels == {"A": "Load & check", "B": "B", "C": "Score"}
    assert layers(chart) == {"A": 0, "B": 1, "C": 2}
    assert "Load &amp; check" in to_svg(chart)
    assert parse("flowchart LR\n A --> B").direction == "LR"
    assert 'd="M' in to_svg(parse("flowchart LR\n A --> B"))
    with pytest.raises(ValueError, match="flowchart TB"):
        parse("flowchart RL\n A --> B")
    with pytest.raises(ValueError, match="unsupported"):
        parse("flowchart LR\n A -.-> B")
    with pytest.raises(ValueError, match="cycle"):
        layers(parse("flowchart LR\n A --> B\n B --> A"))


def test_rewrite_links_targets_site_pages_and_repo() -> None:
    md = (
        "[card](docs/MODEL_CARD.md) ![c](docs/img/lift_curve.png) [lic](LICENSE) "
        "[ext](https://example.com) [anchor](#data)"
    )
    out = rewrite_links(md, "README.md", REPO)
    assert "[card](model-card.html)" in out
    assert "![c](img/lift_curve.png)" in out
    assert f"[lic]({REPO}/blob/main/LICENSE)" in out
    assert "(https://example.com)" in out and "(#data)" in out
    assert "![r](img/x.png)" in rewrite_links("![r](img/x.png)", "docs/MODEL_CARD.md", REPO)


def test_readme_sections_and_slugify() -> None:
    sections = readme_sections("# T\nintro\n## Quickstart\nrun it\n## Data\nUCI\n")
    assert sections == {"Quickstart": "run it", "Data": "UCI"}
    assert slugify("Fairness <code>audit</code>") == "fairness-audit"


@pytest.fixture(scope="module")
def built_site(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("site") / "site"
    demo = out.parent / "demo.txt"
    demo.write_text("Queue of 50 of 1,000 accounts\n")
    build_site(SiteInputs(ROOT, RESULTS, demo, REPO), out)
    return out


def test_site_numbers_come_from_the_results_snapshot(built_site: Path) -> None:
    m = json.loads(RESULTS.read_text())
    champ = m["models"][m["champion_label"]]
    index = (built_site / "index.html").read_text()
    assert pct(champ["recall_at_top_pct"]["20"]) in index
    assert f"{champ['roc_auc']:.3f}" in index
    assert "Fairness findings" in index and "Where the gap comes from" in index
    assert "Queue of 50 of 1,000 accounts" in index
    assert 'class="flowchart"' in index


def test_site_is_self_contained(built_site: Path) -> None:
    assert (built_site / ".nojekyll").exists()
    assert (built_site / "style.css").stat().st_size > 1_000
    for page in ("index.html", "model-card.html", "product.html"):
        html = (built_site / page).read_text()
        assert "<script" not in html
        assert not re.search(r'<link[^>]+href="https?://', html)
        for src in re.findall(r'src="([^"]+)"', html):
            assert (built_site / src).exists(), f"{page} references missing {src}"
    card = (built_site / "model-card.html").read_text()
    assert 'id="fairness-audit"' in card and 'src="img/fairness.png"' in card


def test_site_cli_writes_into_out(tmp_path: Path) -> None:
    out = tmp_path / "site"
    result = CliRunner().invoke(
        app, ["site", "--out", str(out), "--root", str(ROOT), "--results", str(RESULTS)]
    )
    assert result.exit_code == 0, result.output
    assert (out / "index.html").exists()
    assert "Example output" not in (out / "index.html").read_text()
