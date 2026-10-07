.PHONY: install lint format typecheck test demo data train app all

install:
	uv sync --all-extras

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy

test:
	uv run pytest --cov --cov-report=term-missing

# End-to-end on the bundled synthetic sample: train, evaluate, rank, explain.
demo:
	uv run credit-rank train --data data/sample/synthetic_credit_5k.csv --out artifacts/demo
	uv run credit-rank rank --artifacts artifacts/demo --capacity 50 --show 10
	uv run credit-rank explain 17 --artifacts artifacts/demo

# Real UCI data (downloaded, not committed).
data:
	uv run python scripts/download_uci.py

train:
	uv run credit-rank train --data data/raw/uci_credit_default.csv --out artifacts/uci --card docs/MODEL_CARD.md --img docs/img

app:
	uv run streamlit run app/streamlit_app.py -- --artifacts artifacts/demo

all: lint typecheck test demo
