# Trains on the bundled synthetic sample at build time and serves the Streamlit queue viewer.
FROM python:3.12-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# uv.lock is written by uv 0.12 (lockfile revision 5); older uv cannot sync it frozen.
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

RUN useradd --create-home app
WORKDIR /home/app/credit-risk-explain
COPY --chown=app:app . .
USER app

ENV UV_LINK_MODE=copy MPLBACKEND=Agg
RUN uv sync --frozen --no-dev --extra app \
    && uv run credit-rank train --data data/sample/synthetic_credit_5k.csv --out artifacts/demo

EXPOSE 8501
CMD ["uv", "run", "--no-sync", "streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.headless=true", "--", "--artifacts", "artifacts/demo"]
