"""Assemble the site: landing page, rendered model card and product brief, charts."""

from __future__ import annotations

import json
import posixpath
import re
import shutil
from dataclasses import dataclass
from html import escape
from importlib.resources import files
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape
from markdown_it import MarkdownIt

from credit_ranking.docsite.flowchart import parse, to_svg
from credit_ranking.fairness import gap_drivers, largest_tpr_gap
from credit_ranking.model_card import num, pct

REPO_URL = "https://github.com/seanmcrae/credit-risk-explain"
README_SECTIONS = (
    "Features",
    "Quickstart",
    "Architecture",
    "Design decisions",
    "Data",
    "Limitations",
)
PAGES = {"docs/MODEL_CARD.md": "model-card.html", "docs/PRODUCT.md": "product.html"}

_LINK = re.compile(r"(\]\()([^)\s]+)(\))")
_MERMAID = re.compile(r"```mermaid\n(.*?)```", re.DOTALL)
_HEADING = re.compile(r"<h([23])>(.*?)</h\1>")
_FLOWCHART_TOKEN = "FLOWCHART-PLACEHOLDER"
_MD = MarkdownIt("commonmark").enable("table")


@dataclass(frozen=True)
class SiteInputs:
    root: Path
    results: Path
    demo_output: Path | None = None
    repo_url: str = REPO_URL


def slugify(text: str) -> str:
    plain = re.sub(r"<[^>]+>", "", text).lower()
    return re.sub(r"[^a-z0-9]+", "-", plain).strip("-")


def readme_sections(readme: str) -> dict[str, str]:
    """Body of each ``## `` section of a markdown document, keyed by heading text."""
    sections: dict[str, str] = {}
    for chunk in re.split(r"^## ", readme, flags=re.MULTILINE)[1:]:
        heading, _, body = chunk.partition("\n")
        sections[heading.strip()] = body.strip()
    return sections


def rewrite_links(markdown: str, source: str, repo_url: str) -> str:
    """Point relative links at the site's own pages and images, or at the repository."""
    base = posixpath.dirname(source)

    def target(match: re.Match[str]) -> str:
        link = match.group(2)
        if re.match(r"^(https?:|mailto:|#)", link):
            return match.group(0)
        path = posixpath.normpath(posixpath.join(base, link))
        if path.startswith("docs/img/"):
            new = "img/" + path.removeprefix("docs/img/")
        elif path in PAGES:
            new = PAGES[path]
        elif path == "README.md":
            new = "index.html"
        else:
            new = f"{repo_url}/blob/main/{path}"
        return f"{match.group(1)}{new}{match.group(3)}"

    return _LINK.sub(target, markdown)


def render_markdown(markdown: str, source: str, repo_url: str) -> str:
    """Markdown to HTML with anchored h2/h3 headings and site-relative links."""
    html = _MD.render(rewrite_links(markdown, source, repo_url))
    return _HEADING.sub(
        lambda m: f'<h{m.group(1)} id="{slugify(m.group(2))}">{m.group(2)}</h{m.group(1)}>', html
    )


def _architecture(body: str, repo_url: str) -> str:
    """Architecture section with the Mermaid block swapped for a pre-rendered SVG."""
    match = _MERMAID.search(body)
    if match is None:
        return render_markdown(body, "README.md", repo_url)
    svg = to_svg(parse(match.group(1)), "Pipeline architecture")
    source = escape(match.group(1).strip())
    figure = (
        f'<figure class="diagram">{svg}<details><summary>Mermaid source</summary>'
        f"<pre><code>{source}</code></pre></details></figure>"
    )
    html = render_markdown(_MERMAID.sub(_FLOWCHART_TOKEN, body), "README.md", repo_url)
    return html.replace(f"<p>{_FLOWCHART_TOKEN}</p>", figure)


def _results(m: dict[str, Any]) -> dict[str, Any]:
    champ_label = m["champion_label"]
    base_label = next(n for n in m["models"] if n != champ_label)
    champ, base = m["models"][champ_label], m["models"][base_label]
    cap = m["config"]["ranking"]["eval_capacity_percent"]
    cap_key = str(cap)
    rows = [
        ("ROC-AUC", num(champ["roc_auc"]), num(base["roc_auc"])),
        ("PR-AUC", num(champ["pr_auc"]), num(base["pr_auc"])),
        ("KS", num(champ["ks"]), num(base["ks"])),
        ("Top-decile lift", num(champ["top_decile_lift"], 2), num(base["top_decile_lift"], 2)),
        *(
            (
                f"Precision in top {k}% of queue",
                pct(champ["precision_at_top_pct"][str(k)]),
                pct(base["precision_at_top_pct"][str(k)]),
            )
            for k in m["config"]["ranking"]["top_k_percents"]
        ),
        (
            f"Defaulters captured in top {cap}%",
            pct(champ["recall_at_top_pct"][cap_key]),
            pct(base["recall_at_top_pct"][cap_key]),
        ),
        ("Calibrated ECE", num(champ["ece"]), num(base["ece"])),
        ("Brier score", num(champ["brier"]), num(base["brier"])),
        (
            f"Expected value at {cap}% capacity (illustrative)",
            f"{champ['ev_at_eval_capacity']:,.0f}",
            f"{base['ev_at_eval_capacity']:,.0f}",
        ),
    ]
    lift = [
        (
            r["bucket"],
            f"{r['accounts']:,}",
            f"{int(r['events']):,}",
            pct(r["event_rate"]),
            num(r["lift"], 2),
            pct(r["capture_rate"]),
        )
        for r in m["lift_table"]
    ]
    return {
        "champion": champ_label,
        "baseline": base_label,
        "capacity_pct": cap,
        "capture": pct(champ["recall_at_top_pct"][cap_key]),
        "capture_baseline": pct(base["recall_at_top_pct"][cap_key]),
        "auc": num(champ["roc_auc"]),
        "auc_baseline": num(base["roc_auc"]),
        "ece": num(champ["ece"]),
        "ece_uncalibrated": num(champ["ece_uncalibrated"]),
        "precision_top5": pct(champ["precision_at_top_pct"].get("5")),
        "ev_optimal_capacity": f"{champ['ev_optimal_capacity']:,}",
        "ev_optimal_pct": f"{champ['ev_optimal_capacity_pct']:.1f}%",
        "psi": num(m["psi_train_vs_test"]),
        "rows": rows,
        "lift": lift,
    }


def _fairness(m: dict[str, Any]) -> dict[str, Any]:
    slices = pd.DataFrame(m["fairness_slices"])
    gap = largest_tpr_gap(slices)
    summary = [
        (
            s["attribute"],
            num(s["priority_rate_ratio"], 2),
            num(s["tpr_gap"]),
            num(s["fpr_gap"]),
            num(s["max_abs_calibration_gap"]),
        )
        for s in m["fairness_summary"]
    ]
    drivers: list[tuple[str, str, str, str]] = []
    if gap is not None and m.get("fairness_contributions"):
        frame = gap_drivers(pd.DataFrame(m["fairness_contributions"]), gap)
        drivers = [
            (str(d["feature"]), f"{d['high']:+.3f}", f"{d['low']:+.3f}", f"{d['difference']:+.3f}")
            for _, d in frame.iterrows()
        ]
    return {
        "gap": gap,
        "gap_points": f"{100 * (gap.high_tpr - gap.low_tpr):.1f}" if gap else None,
        "high_tpr": pct(gap.high_tpr) if gap else None,
        "low_tpr": pct(gap.low_tpr) if gap else None,
        "summary": summary,
        "drivers": drivers,
        "small_slices": [
            f"{s['attribute']}: {s['group']} ({s['accounts']})"
            for s in m["fairness_slices"]
            if s["small_slice"]
        ],
    }


def build_site(inputs: SiteInputs, out: Path) -> list[Path]:
    """Write the site into ``out`` (replaced if it exists) and return the files written."""
    root, repo_url = inputs.root, inputs.repo_url
    metrics: dict[str, Any] = json.loads(inputs.results.read_text())
    sections = readme_sections((root / "README.md").read_text())
    rendered = {
        name: (
            _architecture(sections[name], repo_url)
            if name == "Architecture"
            else render_markdown(sections[name], "README.md", repo_url)
        )
        for name in README_SECTIONS
        if name in sections
    }
    demo = inputs.demo_output.read_text().rstrip() if inputs.demo_output else None

    if out.exists():
        shutil.rmtree(out)
    (out / "img").mkdir(parents=True)
    for image in sorted((root / "docs" / "img").glob("*.png")):
        shutil.copyfile(image, out / "img" / image.name)

    env = Environment(
        loader=PackageLoader("credit_ranking.docsite"),
        autoescape=select_autoescape(["html", "j2"]),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    common = {"repo_url": repo_url, "data": metrics["data"], "split": metrics["split"]}
    pages: dict[str, str] = {
        "index.html": env.get_template("index.html.j2").render(
            **common,
            page="index.html",
            sections=rendered,
            results=_results(metrics),
            fairness=_fairness(metrics),
            demo=demo,
        )
    }
    titles = {"model-card.html": "Model card", "product.html": "Product brief"}
    for source, name in PAGES.items():
        body = render_markdown((root / source).read_text(), source, repo_url)
        pages[name] = env.get_template("page.html.j2").render(
            **common, page=name, title=titles[name], body=body, source=source
        )
    written = []
    for name, html in pages.items():
        (out / name).write_text(html)
        written.append(out / name)
    css = files("credit_ranking.docsite").joinpath("templates", "style.css").read_text()
    (out / "style.css").write_text(css)
    (out / ".nojekyll").write_text("")
    written += [out / "style.css", out / ".nojekyll"]
    written += sorted((out / "img").iterdir())
    return written
