# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-07

### Added

- Pandera schema, UCI "Default of Credit Card Clients" loader and checksum-verified download
  script; seeded synthetic generator with a bundled 5,000-row synthetic sample.
- Fourteen leakage-safe, row-wise features; protected attributes excluded by construction.
- Logistic-regression baseline and monotone-constrained LightGBM champion with
  Laplace-smoothed isotonic or Platt calibration on a separate split.
- Ranking metrics: KS, lift and capture by decile, precision at k, expected value by queue size,
  ECE and PSI.
- SHAP attributions and seven grouped, risk-increasing reason codes.
- Fairness slice audit: priority rate, TPR and FPR at capacity, calibration gap and AUC per
  group, plus a proxy screen that traces the largest TPR gap to feature contributions.
- Model card generated from each run, with charts and a committed results snapshot.
- `credit-rank` CLI (`train`, `evaluate`, `rank`, `explain`, `model-card`, `synth`, `download`,
  `site`) and a Streamlit queue viewer.
- Static documentation site built offline from the README, docs and results snapshot, deployed
  to GitHub Pages from the `gh-pages` branch.
- CI on Python 3.11 and 3.12 (ruff, mypy strict, pytest with coverage, demo and site build),
  Dockerfile and Make targets.

[0.1.0]: https://github.com/seanmcrae/credit-risk-explain/commits/main
