from pathlib import Path

import pytest

from credit_ranking.pipeline import TrainResult

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_app_renders_queue_and_account_detail(
    trained_run: TrainResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("CREDIT_RANK_ARTIFACTS", str(trained_run.out_dir))
    at = testing.AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception
    assert at.title[0].value == "Credit work queue"
    assert len(at.dataframe) == 1
    at.slider[0].set_value(50).run()
    assert not at.exception
    assert at.metric[0].value.startswith("50 of")
