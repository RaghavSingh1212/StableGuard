"""Tests for the eval harness, the metrics and the leaderboard."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from stableguard.agent import RandomAgent, RuleBasedAgent
from stableguard.evaluator import Evaluator
from stableguard.metrics import (
    compute_metrics,
    confusion_counts,
    estimate_cost_usd,
    format_leaderboard,
    format_summary,
    leaderboard,
    load_all_results,
    recall_by_fraud_type,
    save_report,
)
from stableguard.models import Decision, EvalResult


def _decision(txn_id: str, verdict: str, is_fraud: bool, fraud_type: str = "none") -> Decision:
    """Build a scored decision for metric tests."""
    return Decision(
        transaction_id=txn_id,
        verdict=verdict,  # type: ignore[arg-type]
        is_fraud=is_fraud,
        fraud_type=fraud_type,
        correct=(verdict == "flag") == is_fraud,
        reward=1.0 if (verdict == "flag") == is_fraud else -1.0,
        latency_seconds=0.01,
        tokens_used=100,
    )


@pytest.fixture(scope="module")
def evaluator(dataset: dict[str, Any]) -> Evaluator:
    """An evaluator over the session dataset."""
    return Evaluator(dataset=dataset, dataset_path="in-memory")


def test_confusion_counts() -> None:
    """Confusion counts follow the flag as positive convention."""
    decisions = [
        _decision("a", "flag", True, "mule"),
        _decision("b", "flag", False),
        _decision("c", "approve", True, "geo"),
        _decision("d", "approve", False),
    ]
    assert confusion_counts(decisions) == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}


def test_metrics_on_a_known_confusion_matrix() -> None:
    """Precision, recall, F1 and the error rates are computed correctly."""
    decisions = (
        [_decision(f"tp{i}", "flag", True, "mule") for i in range(6)]
        + [_decision(f"fn{i}", "approve", True, "geo") for i in range(4)]
        + [_decision(f"fp{i}", "flag", False) for i in range(2)]
        + [_decision(f"tn{i}", "approve", False) for i in range(8)]
    )
    result = compute_metrics(decisions, agent_name="unit")
    assert result.n_transactions == 20
    assert result.accuracy == pytest.approx(14 / 20)
    assert result.precision == pytest.approx(6 / 8)
    assert result.recall == pytest.approx(6 / 10)
    assert result.f1 == pytest.approx(2 * 0.75 * 0.6 / (0.75 + 0.6), abs=1e-4)
    assert result.false_positive_rate == pytest.approx(2 / 10)
    assert result.false_negative_rate == pytest.approx(4 / 10)


def test_recall_by_fraud_type() -> None:
    """Per fraud type recall splits the catch rate by pattern."""
    decisions = [
        _decision("a", "flag", True, "mule"),
        _decision("b", "approve", True, "mule"),
        _decision("c", "flag", True, "geo"),
        _decision("d", "approve", False),
    ]
    assert recall_by_fraud_type(decisions) == {"geo": 1.0, "mule": 0.5}


def test_metrics_reject_empty_input() -> None:
    """Computing metrics over nothing is an error, not a zero."""
    with pytest.raises(ValueError):
        compute_metrics([], agent_name="unit")


def test_cost_estimate() -> None:
    """Cost scales with tokens at the configured per million rates."""
    assert estimate_cost_usd(0, 0) == 0.0
    assert estimate_cost_usd(1_000_000, 0) > 0
    assert estimate_cost_usd(1_000_000, 1_000_000) > estimate_cost_usd(1_000_000, 0)


def test_rule_based_agent_runs_end_to_end(evaluator: Evaluator) -> None:
    """The baseline reviews a slice and produces usable metrics."""
    result, decisions = evaluator.run(RuleBasedAgent(), n=120, save=False)
    assert result.n_transactions == 120
    assert len(decisions) == 120
    assert 0.0 <= result.accuracy <= 1.0
    assert result.avg_tool_calls <= 6
    assert all(d.verdict in ("approve", "flag") for d in decisions)


def test_rule_based_agent_beats_chance(evaluator: Evaluator) -> None:
    """The baseline is a real baseline: clearly better than a coin flip."""
    rule_result, _ = evaluator.run(RuleBasedAgent(), n=150, save=False)
    random_result, _ = evaluator.run(RandomAgent(seed=1), n=150, save=False)
    assert rule_result.f1 > random_result.f1
    assert rule_result.accuracy > 0.8


def test_rule_based_catches_the_patterns_it_targets(evaluator: Evaluator) -> None:
    """The baseline catches the four patterns its checklist covers."""
    result, _ = evaluator.run(RuleBasedAgent(), n=None, save=False)
    for fraud_type in ("sanctions", "velocity", "geo", "amount_spike"):
        assert result.recall_by_fraud_type.get(fraud_type, 0.0) > 0.9, fraud_type


def test_decisions_record_the_investigation(evaluator: Evaluator) -> None:
    """Each decision carries its tool calls, latency and ground truth."""
    _, decisions = evaluator.run(RuleBasedAgent(), n=30, save=False)
    with_tools = [d for d in decisions if d.tool_calls]
    assert with_tools
    call = with_tools[0].tool_calls[0]
    assert call.name and isinstance(call.result, dict)
    assert all(d.latency_seconds >= 0 for d in decisions)
    assert all(d.is_fraud is not None for d in decisions)


def test_review_one(evaluator: Evaluator) -> None:
    """A single payment can be reviewed on its own."""
    txn = evaluator.transactions[3]
    decision = evaluator.review_one(RuleBasedAgent(), txn.id)
    assert decision.transaction_id == txn.id
    assert decision.correct == ((decision.verdict == "flag") == txn.is_fraud)


def test_agents_see_the_same_slice(evaluator: Evaluator) -> None:
    """Two agents evaluated with the same n review the same payments."""
    _, first = evaluator.run(RuleBasedAgent(), n=25, save=False)
    _, second = evaluator.run(RandomAgent(seed=0), n=25, save=False)
    assert [d.transaction_id for d in first] == [d.transaction_id for d in second]


def test_budget_exhaustion_counts_as_an_approval(dataset: dict[str, Any]) -> None:
    """An agent that never decides is scored, not skipped."""

    class StallingAgent(RuleBasedAgent):
        """Calls one tool forever and never returns a verdict."""

        name = "stalling"

        def act(self, observation: dict[str, Any]) -> dict[str, Any]:
            return {
                "type": "tool",
                "name": "check_sanctions",
                "args": {"wallet_address": observation["transaction"]["sender_wallet"]},
            }

    evaluator = Evaluator(dataset=dataset, dataset_path="in-memory")
    result, decisions = evaluator.run(StallingAgent(), n=10, save=False)
    assert result.n_transactions == 10
    assert all(d.verdict == "approve" for d in decisions)


def test_report_round_trip(tmp_path: Path, evaluator: Evaluator) -> None:
    """A report can be written and read back as an EvalResult."""
    result, decisions = evaluator.run(RuleBasedAgent(), n=20, save=False)
    path = save_report(result, decisions, results_dir=tmp_path)
    assert path.exists()

    loaded = load_all_results(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].agent_name == result.agent_name
    assert loaded[0].f1 == result.f1


def test_leaderboard_ranks_by_f1(tmp_path: Path) -> None:
    """The leaderboard keeps the newest run per agent and ranks by F1."""
    now = datetime.now(timezone.utc)

    def make(agent: str, f1: float, when: datetime) -> EvalResult:
        return EvalResult(
            agent_name=agent,
            n_transactions=10,
            accuracy=f1,
            precision=f1,
            recall=f1,
            f1=f1,
            false_positive_rate=0.0,
            false_negative_rate=0.0,
            avg_tool_calls=1.0,
            avg_latency_seconds=0.1,
            timestamp=when,
        )

    results = [
        make("rule_based", 0.5, now - timedelta(hours=2)),
        make("rule_based", 0.7, now),
        make("claude", 0.9, now),
    ]
    board = leaderboard(results, results_dir=tmp_path)
    assert [r.agent_name for r in board] == ["claude", "rule_based"]
    assert board[1].f1 == 0.7
    assert "claude" in format_leaderboard(board)


def test_empty_leaderboard_message(tmp_path: Path) -> None:
    """An empty results directory produces a helpful message, not a crash."""
    assert "No eval results yet" in format_leaderboard(leaderboard(results_dir=tmp_path))


def test_format_summary_contains_key_metrics(evaluator: Evaluator) -> None:
    """The printed summary mentions the headline metrics."""
    result, _ = evaluator.run(RuleBasedAgent(), n=20, save=False)
    text = format_summary(result)
    for label in ("Accuracy", "Precision", "Recall", "F1", "Recall by fraud type"):
        assert label in text
