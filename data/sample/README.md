# Bundled sample (SYNTHETIC)

`synthetic_credit_5k.csv` holds 5,000 fabricated accounts produced by
`credit_ranking.synthetic.generate_synthetic(5000, seed=7)` (see `scripts/make_synthetic_sample.py`).
It shares the canonical schema of the UCI "Default of Credit Card Clients" data so the full pipeline,
tests and CI run without downloading anything. No row describes a real person, and metrics computed
on it say nothing about real-world model performance.
