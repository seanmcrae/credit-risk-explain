"""Regenerate the bundled SYNTHETIC demo sample at data/sample/synthetic_credit_5k.csv."""

from pathlib import Path

from credit_ranking.synthetic import generate_synthetic

SAMPLE_ROWS = 5_000
SAMPLE_SEED = 7

if __name__ == "__main__":
    dest = Path(__file__).resolve().parents[1] / "data" / "sample" / "synthetic_credit_5k.csv"
    generate_synthetic(SAMPLE_ROWS, seed=SAMPLE_SEED).to_csv(dest, index=False)
    print(f"Wrote {dest}")
