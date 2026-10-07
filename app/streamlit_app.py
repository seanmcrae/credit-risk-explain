"""Streamlit work-queue viewer.

    uv run streamlit run app/streamlit_app.py -- --artifacts artifacts/demo

Shows the prioritized queue for a chosen capacity, then one account's probability, reason codes
and SHAP waterfall. Reads a run directory written by ``credit-rank train`` (``--artifacts`` or
the ``CREDIT_RANK_ARTIFACTS`` environment variable).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from credit_ranking.metrics import expected_value_from_probabilities
from credit_ranking.pipeline import load_holdout, load_metrics
from credit_ranking.plots import waterfall
from credit_ranking.queue import ModelBundle, build_queue, explain_account


def _artifacts_dir() -> Path:
    parser = argparse.ArgumentParser()
    default = os.environ.get("CREDIT_RANK_ARTIFACTS", "artifacts/demo")
    parser.add_argument("--artifacts", type=Path, default=Path(default))
    args, _ = parser.parse_known_args()
    return Path(args.artifacts)


@st.cache_resource
def _bundle(path: str) -> ModelBundle:
    return ModelBundle.load(Path(path))


@st.cache_data
def _accounts(path: str) -> pd.DataFrame:
    return load_holdout(Path(path))


@st.cache_data
def _full_queue(path: str) -> pd.DataFrame:
    return build_queue(_bundle(path), _accounts(path))


def main() -> None:
    artifacts = str(_artifacts_dir())
    st.set_page_config(page_title="Credit work queue", layout="wide")
    metrics = load_metrics(Path(artifacts))
    bundle = _bundle(artifacts)
    queue = _full_queue(artifacts)
    econ = bundle.config.economics

    st.title("Credit work queue")
    st.caption(
        f"Model: {metrics['champion_label']} trained on {metrics['data']['label']}. "
        "Accounts are the run's holdout set, ordered by model score."
    )

    with st.sidebar:
        st.header("Queue settings")
        capacity = st.slider(
            "Capacity (accounts worked this cycle)",
            min_value=10,
            max_value=len(queue),
            value=min(bundle.config.ranking.capacity, len(queue)),
            step=10,
        )
        st.markdown(
            f"Illustrative economics: contact cost {econ.cost_per_contact:,.0f}, "
            f"loss given default {econ.loss_given_default:,.0f}, "
            f"cure rate {econ.cure_rate_if_worked:.0%}."
        )

    worked = queue.head(capacity)
    ev = expected_value_from_probabilities(worked["probability"].to_numpy(), econ)
    total_expected = queue["probability"].sum()
    cols = st.columns(4)
    cols[0].metric("Accounts in queue", f"{capacity:,} of {len(queue):,}")
    cols[1].metric("Expected defaults reached", f"{worked['probability'].sum():,.0f}")
    cols[2].metric(
        "Share of expected defaults", f"{worked['probability'].sum() / total_expected:.0%}"
    )
    cols[3].metric("Expected value of this queue", f"{ev.sum():,.0f}")

    st.subheader("Prioritized queue")
    st.dataframe(
        worked[["rank", "id", "probability", "decile", "reason_codes", "top_reasons"]],
        hide_index=True,
        width="stretch",
        column_config={
            "probability": st.column_config.ProgressColumn(
                "P(default)", min_value=0.0, max_value=1.0, format="%.3f"
            ),
            "top_reasons": st.column_config.TextColumn("Top reasons", width="large"),
        },
    )

    st.subheader("Account detail")
    account_id = st.selectbox("Account", worked["id"].tolist())
    detail = explain_account(bundle, _accounts(artifacts), int(account_id))
    left, right = st.columns([1, 2])
    with left:
        st.metric("P(default)", f"{detail.probability:.3f}")
        st.metric("Risk decile", detail.decile)
        st.markdown("**Reason codes**")
        for reason in detail.reasons:
            st.markdown(f"- `{reason.code}` {reason.text}")
        if not detail.reasons:
            st.markdown("No risk-increasing reasons.")
    with right:
        st.pyplot(waterfall(detail.contributions, detail.feature_values, detail.base_value))


main()
