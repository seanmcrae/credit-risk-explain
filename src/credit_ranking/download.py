"""Fetch the UCI "Default of Credit Card Clients" dataset and convert it to the canonical CSV.

Source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
License: CC BY 4.0. Citation: Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining
techniques for the predictive accuracy of probability of default of credit card clients.
Expert Systems with Applications, 36(2), 2473-2480.
"""

from __future__ import annotations

import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from credit_ranking.data import normalize_uci_columns
from credit_ranking.schema import validate

UCI_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
# SHA-256 of the archive as served by UCI when this repo was written (2026-10).
UCI_ZIP_SHA256 = "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"
XLS_NAME = "default of credit card clients.xls"


def verify_checksum(payload: bytes, expected: str) -> None:
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError(f"checksum mismatch: expected {expected}, got {actual}")


def extract_table(archive: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(archive)) as zf, zf.open(XLS_NAME) as fh:
        # Row 0 holds X1..X23 placeholders; the descriptive header is row 1.
        raw = pd.read_excel(fh, header=1)
    return validate(normalize_uci_columns(raw))


def download_uci(dest: Path, url: str = UCI_URL, timeout: float = 60.0) -> Path:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        archive: bytes = resp.read()
    verify_checksum(archive, UCI_ZIP_SHA256)
    df = extract_table(archive)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest
