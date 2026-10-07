"""Static documentation site built from the README, the docs and a committed results snapshot.

Every number on the site is read from ``docs/results/uci_metrics.json`` (written by
``credit-rank train --snapshot``) or from the generated model card, never typed by hand. The
output is plain HTML and CSS with no external requests, so it builds and renders offline.
"""

from __future__ import annotations

from credit_ranking.docsite.build import REPO_URL, SiteInputs, build_site

__all__ = ["REPO_URL", "SiteInputs", "build_site"]
