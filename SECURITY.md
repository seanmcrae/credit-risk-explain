# Security policy

## Supported versions

Only the latest release on `main` receives fixes.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/seanmcrae/credit-risk-explain/security/advisories/new)
rather than a public issue. Include the affected version or commit, steps to reproduce and the
impact you expect. You should get an acknowledgement within a week.

## Scope notes

- The repository ships no credentials and needs none: CI and tests run fully offline.
- `credit-rank` loads model bundles with `joblib`, which can execute code on load. Only load
  bundles you trained yourself or otherwise trust.
- `scripts/download_uci.py` verifies the SHA-256 of the downloaded archive before extracting it.
- The bundled data is synthetic; the UCI data is public and anonymized. Do not commit real
  customer data to a fork.
