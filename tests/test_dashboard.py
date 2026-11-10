"""Smoke tests for the Streamlit dashboard.

Streamlit's AppTest runs the page script in process, so these tests catch an
exception or a bad widget call without a browser. They need a dataset on disk,
which is what the dashboard itself reads, so they are skipped when there is none.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stableguard.config import DEFAULT_DATASET_PATH

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"

pytestmark = pytest.mark.skipif(
    not Path(DEFAULT_DATASET_PATH).exists(),
    reason="dashboard tests need data/transactions.json, run scripts/generate_data.py",
)


def _run() -> AppTest:
    """Run the dashboard script and return the rendered app."""
    app = AppTest.from_file(str(APP_PATH), default_timeout=120)
    app.run()
    return app


def test_dashboard_renders_without_exceptions() -> None:
    """The page loads and shows its title."""
    app = _run()
    assert not app.exception
    assert any("StableGuard" in str(title.value) for title in app.title)


def test_review_mode_runs_an_investigation() -> None:
    """Clicking Investigate drives a full episode and shows the outcome."""
    app = _run()
    assert not app.exception
    # The Investigate button is the primary action in review mode.
    investigate = [b for b in app.button if b.label == "Investigate"]
    assert investigate, "the Investigate button is missing"
    investigate[0].click().run()
    assert not app.exception
    rendered = " ".join(str(m.value) for m in app.markdown)
    assert "Investigation" in rendered and "Outcome" in rendered


def test_leaderboard_mode_renders() -> None:
    """Switching to the leaderboard renders the table or the empty state."""
    app = _run()
    app.sidebar.radio[0].set_value("Leaderboard").run()
    assert not app.exception
    rendered = " ".join(str(m.value) for m in app.markdown) + " ".join(
        str(c.value) for c in app.caption
    )
    assert "Leaderboard" in " ".join(str(s.value) for s in app.subheader)
    assert app.dataframe or app.info


def test_agent_selector_offers_every_agent() -> None:
    """The sidebar lists all registered agents."""
    app = _run()
    options = app.sidebar.selectbox[0].options
    assert {"rule_based", "claude", "random"} <= set(options)


def test_recall_frame_shape() -> None:
    """The recall table has one row per fraud type."""
    import sys

    sys.path.insert(0, str(APP_PATH.parent.parent))
    from stableguard.metrics import leaderboard

    results = leaderboard()
    if not results:
        pytest.skip("no saved eval results to chart")

    from dashboard.app import recall_frame  # noqa: PLC0415 - imported lazily on purpose

    frame = recall_frame(results)
    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 6
