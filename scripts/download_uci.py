"""Download the UCI credit default dataset (CC BY 4.0) to data/raw/uci_credit_default.csv.

Requires the ``data`` extra (xlrd) to read the source .xls file:
    uv sync --extra data && uv run python scripts/download_uci.py
"""

from pathlib import Path

from credit_ranking.download import UCI_URL, download_uci

if __name__ == "__main__":
    dest = Path(__file__).resolve().parents[1] / "data" / "raw" / "uci_credit_default.csv"
    print(f"Downloading {UCI_URL}")
    print(f"Wrote {download_uci(dest)}")
