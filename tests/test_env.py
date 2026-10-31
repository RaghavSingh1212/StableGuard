"""Tests for the Gymnasium environment, including a random agent smoke test."""

from __future__ import annotations

from typing import Any

import pytest

from stableguard.env import StableGuardEnv
from stableguard.models import Transaction


@pytest.fixture()
def env(dataset: dict[str, Any]) -> StableGuardEnv:
    """A fresh environment over the session dataset."""
    return StableGuardEnv(dataset, seed=0)


def test_reset_returns_observation(env: StableGuardEnv) -> None:
    """Reset gives a transaction, an empty tool history and a full budget."""
    observation, info = env.reset()
    assert env.observation_space.contains(observation)
    assert observation["tool_results"] == []
    assert observation["steps_remaining"] == env.tool_budget
    assert info["transaction_id"] == observation["transaction"]["id"]


def test_observation_never_leaks_ground_truth(env: StableGuardEnv) -> None:
    """The label is absent from the observation before and after a tool call."""
    observation, _ = env.reset()
    assert "is_fraud" not in observation["transaction"]
    observation, _, _, _, _ = env.step(
        {"type": "tool", "name": "check_sanctions",
         "args": {"wallet_address": observation["transaction"]["sender_wallet"]}}
    )
    assert "fraud_type" not in observation["transaction"]


def test_reset_by_transaction_id(env: StableGuardEnv, transactions: list[Transaction]) -> None:
    """A specific payment can be selected for review."""
    wanted = transactions[5].id
    observation, _ = env.reset(options={"transaction_id": wanted})
    assert observation["transaction"]["id"] == wanted


def test_reset_with_unknown_id_raises(env: StableGuardEnv) -> None:
    """An unknown transaction id is an error, not a silent fallback."""
    with pytest.raises(KeyError):
        env.reset(options={"transaction_id": "txn_nope"})


def test_tool_call_costs_a_small_penalty(env: StableGuardEnv) -> None:
    """Every tool call is charged, so investigation is not free."""
    observation, _ = env.reset()
    observation, reward, terminated, truncated, _ = env.step(
        {"type": "tool", "name": "check_wallet_history",
         "args": {"wallet_address": observation["transaction"]["sender_wallet"]}}
    )
    assert reward == pytest.approx(env.reward_tool_call)
    assert not terminated and not truncated
    assert len(observation["tool_results"]) == 1
    assert observation["steps_remaining"] == env.tool_budget - 1


def test_correct_verdict_is_rewarded(
    dataset: dict[str, Any], transactions: list[Transaction]
) -> None:
    """A correct verdict ends the episode with a positive reward."""
    fraud = next(t for t in transactions if t.is_fraud)
    env = StableGuardEnv(dataset)
    env.reset(options={"transaction_id": fraud.id})
    # Spend three tool calls so the efficiency bonus does not apply.
    for _ in range(3):
        env.step({"type": "tool", "name": "check_sanctions",
                  "args": {"wallet_address": fraud.sender_wallet}})
    _, reward, terminated, _, info = env.step({"type": "decision", "verdict": "flag"})
    assert terminated
    assert info["correct"] is True
    assert reward == pytest.approx(env.reward_correct)


def test_wrong_verdict_is_penalised(
    dataset: dict[str, Any], transactions: list[Transaction]
) -> None:
    """A wrong verdict ends the episode with a negative reward."""
    clean = next(t for t in transactions if not t.is_fraud)
    env = StableGuardEnv(dataset)
    env.reset(options={"transaction_id": clean.id})
    _, reward, terminated, _, info = env.step({"type": "decision", "verdict": "flag"})
    assert terminated
    assert info["correct"] is False
    assert reward == pytest.approx(env.reward_wrong)


def test_efficient_fraud_catch_gets_a_bonus(
    dataset: dict[str, Any], transactions: list[Transaction]
) -> None:
    """Catching fraud in two tool calls or fewer pays a bonus."""
    fraud = next(t for t in transactions if t.is_fraud)
    env = StableGuardEnv(dataset)
    env.reset(options={"transaction_id": fraud.id})
    env.step({"type": "tool", "name": "check_sanctions",
              "args": {"wallet_address": fraud.sender_wallet}})
    _, reward, _, _, info = env.step({"type": "decision", "verdict": "flag"})
    assert info["efficiency_bonus"] == pytest.approx(env.efficient_bonus)
    assert reward == pytest.approx(env.reward_correct + env.efficient_bonus)


def test_no_bonus_for_approving_clean_traffic(
    dataset: dict[str, Any], transactions: list[Transaction]
) -> None:
    """Approving without investigating earns no efficiency bonus."""
    clean = next(t for t in transactions if not t.is_fraud)
    env = StableGuardEnv(dataset)
    env.reset(options={"transaction_id": clean.id})
    _, reward, _, _, info = env.step({"type": "decision", "verdict": "approve"})
    assert info["efficiency_bonus"] == 0.0
    assert reward == pytest.approx(env.reward_correct)


def test_budget_exhaustion_ends_the_episode(dataset: dict[str, Any]) -> None:
    """Running out of tool calls without deciding scores like a wrong answer."""
    env = StableGuardEnv(dataset, tool_budget=2)
    observation, _ = env.reset()
    wallet = observation["transaction"]["sender_wallet"]
    call = {"type": "tool", "name": "check_sanctions", "args": {"wallet_address": wallet}}
    env.step(call)
    env.step(call)
    _, reward, terminated, truncated, info = env.step(call)
    assert truncated and not terminated
    assert reward == pytest.approx(env.reward_wrong)
    assert info["reason"] == "tool_budget_exhausted"


def test_step_after_episode_end_raises(dataset: dict[str, Any]) -> None:
    """The environment refuses to continue a finished episode."""
    env = StableGuardEnv(dataset)
    env.reset()
    env.step({"type": "decision", "verdict": "approve"})
    with pytest.raises(RuntimeError):
        env.step({"type": "decision", "verdict": "approve"})


def test_malformed_action_is_absorbed(dataset: dict[str, Any]) -> None:
    """A malformed action costs a step and reports an error, nothing more."""
    env = StableGuardEnv(dataset)
    env.reset()
    observation, reward, terminated, truncated, _ = env.step({"type": "nonsense"})
    assert not terminated and not truncated
    assert reward == pytest.approx(env.reward_tool_call)
    assert observation["tool_results"][-1]["result"]["error"] == "invalid_action"


def test_episodes_advance_through_the_dataset(dataset: dict[str, Any]) -> None:
    """Consecutive resets walk through the dataset rather than repeating one row."""
    env = StableGuardEnv(dataset)
    seen = []
    for _ in range(5):
        observation, _ = env.reset()
        seen.append(observation["transaction"]["id"])
        env.step({"type": "decision", "verdict": "approve"})
    assert len(set(seen)) == 5


def test_random_agent_completes_episodes(dataset: dict[str, Any]) -> None:
    """A random agent can run many episodes without ever breaking the env."""
    env = StableGuardEnv(dataset, seed=3)
    total_reward = 0.0
    for _ in range(40):
        env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
    assert isinstance(total_reward, float)


def test_action_space_contains(dataset: dict[str, Any]) -> None:
    """The action space accepts valid actions and rejects invalid ones."""
    env = StableGuardEnv(dataset)
    assert env.action_space.contains({"type": "decision", "verdict": "flag"})
    assert env.action_space.contains({"type": "tool", "name": "check_geo", "args": {}})
    assert not env.action_space.contains({"type": "decision", "verdict": "maybe"})
    assert not env.action_space.contains("flag")
