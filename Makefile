.PHONY: install lint format typecheck test check demo data train-uci app docker

install:
	uv sync --all-extras

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy

test:
	uv run pytest --cov --cov-report=term-missing

# Same gates as CI.
check: lint typecheck test

# End-to-end on the bundled SYNTHETIC sample: train, rank, explain the top account.
demo:
	uv run credit-rank train --data data/sample/synthetic_credit_5k.csv --out artifacts/demo \
		--label "bundled synthetic sample (5,000 rows)"
	uv run credit-rank rank --artifacts artifacts/demo --capacity 50 --show 10
	uv run credit-rank explain 2185 --artifacts artifacts/demo

# Real UCI data (downloaded to data/raw/, never committed).
data:
	uv run python scripts/download_uci.py

# Regenerates docs/MODEL_CARD.md and docs/img/ from the real-data run.
train-uci: data
	uv run credit-rank train --data data/raw/uci_credit_default.csv --out artifacts/uci \
		--label "UCI Default of Credit Card Clients (30,000 rows)" \
		--card docs/MODEL_CARD.md --img docs/img

app:
	uv run streamlit run app/streamlit_app.py -- --artifacts artifacts/demo

docker:
	docker build -t credit-risk-explain .
	docker run --rm -p 8501:8501 credit-risk-explain
