"""Tests for the agents.

The Claude agent is tested against a stub client, so the suite needs no API key
and makes no network calls. What is tested is the part StableGuard owns: the
prompt, the JSON parsing, the token accounting and the failure handling.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from stableguard.agent import (
    AGENT_REGISTRY,
    LLMAgent,
    RandomAgent,
    RuleBasedAgent,
    _extract_json,
    build_agent,
)
from stableguard.env import StableGuardEnv
from stableguard.models import Transaction


class _StubUsage:
    """Minimal stand in for the SDK usage object."""

    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _StubTextBlock:
    """Minimal stand in for a text content block."""

    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _StubResponse:
    """Minimal stand in for a Message response."""

    def __init__(self, text: str) -> None:
        self.content = [_StubTextBlock(text)]
        self.usage = _StubUsage(120, 30)


class _StubMessages:
    """Records the requests made and replays a scripted list of replies."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _StubResponse:
        self.requests.append(kwargs)
        reply = self.replies.pop(0) if self.replies else '{"reasoning": "done", "action": {"type": "decision", "verdict": "approve"}}'
        return _StubResponse(reply)


class _StubClient:
    """A stand in for anthropic.Anthropic."""

    def __init__(self, replies: list[str] | None = None) -> None:
        self.messages = _StubMessages(replies or [])


class _FailingClient:
    """A client whose every request raises, to exercise the fallback path."""

    class _Messages:
        def create(self, **kwargs: Any) -> Any:
            raise RuntimeError("simulated API outage")

    def __init__(self) -> None:
        self.messages = self._Messages()


def _observation(env: StableGuardEnv, txn_id: str) -> dict[str, Any]:
    observation, _ = env.reset(options={"transaction_id": txn_id})
    return observation


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------


def test_registry_names() -> None:
    """The registry exposes the three agents the project ships with."""
    assert set(AGENT_REGISTRY) == {"rule_based", "claude", "random"}


def test_build_unknown_agent() -> None:
    """An unknown agent name is rejected with a helpful error."""
    with pytest.raises(KeyError):
        build_agent("gradient_boosted_guesswork")


def test_random_agent_decides_immediately() -> None:
    """The random agent spends no tools."""
    action = RandomAgent(seed=0).act({"transaction": {}, "tool_results": [], "steps_remaining": 6})
    assert action["type"] == "decision"


# ----------------------------------------------------------------------
# Rule based baseline
# ----------------------------------------------------------------------


def test_rule_based_flags_velocity_without_tools(
    dataset: dict[str, Any], transactions: list[Transaction]
) -> None:
    """A velocity burst is visible on the payment, so no tool call is needed."""
    txn = next(t for t in transactions if t.fraud_type == "velocity")
    env = StableGuardEnv(dataset)
    observation = _observation(env, txn.id)

    agent = RuleBasedAgent()
    agent.begin_episode(observation)
    action = agent.act(observation)
    assert action == {"type": "decision", "verdict": "flag"}
    assert "Velocity" in agent.reasoning


def test_rule_based_flags_sanctions_hit(
    dataset: dict[str, Any], transactions: list[Transaction]
) -> None:
    """A sanctioned counterparty is caught within the checklist's budget."""
    txn = next(t for t in transactions if t.fraud_type == "sanctions")
    env = StableGuardEnv(dataset)
    observation = _observation(env, txn.id)

    agent = RuleBasedAgent()
    agent.begin_episode(observation)
    terminated = truncated = False
    info: dict[str, Any] = {}
    while not (terminated or truncated):
        observation, _, terminated, truncated, info = env.step(agent.act(observation))
    assert info["verdict"] == "flag"
    assert info["correct"] is True


def test_rule_based_approves_clean_traffic(
    dataset: dict[str, Any], transactions: list[Transaction]
) -> None:
    """A clean payment passes the whole checklist and is approved."""
    txn = next(t for t in transactions if not t.is_fraud)
    env = StableGuardEnv(dataset)
    observation = _observation(env, txn.id)

    agent = RuleBasedAgent()
    agent.begin_episode(observation)
    terminated = truncated = False
    info: dict[str, Any] = {}
    while not (terminated or truncated):
        observation, _, terminated, truncated, info = env.step(agent.act(observation))
    assert info["verdict"] == "approve"
    assert info["correct"] is True


def test_rule_based_stays_inside_the_budget(
    dataset: dict[str, Any], transactions: list[Transaction]
) -> None:
    """The checklist never needs more than four tool calls."""
    env = StableGuardEnv(dataset)
    agent = RuleBasedAgent()
    for txn in transactions[:40]:
        observation = _observation(env, txn.id)
        agent.begin_episode(observation)
        terminated = truncated = False
        info: dict[str, Any] = {}
        while not (terminated or truncated):
            observation, _, terminated, truncated, info = env.step(agent.act(observation))
        assert info["n_tool_calls"] <= 4


# ----------------------------------------------------------------------
# JSON extraction
# ----------------------------------------------------------------------


def test_extract_plain_json() -> None:
    """Clean JSON parses."""
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_fenced_json() -> None:
    """A fenced code block is unwrapped."""
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_with_prose_around_it() -> None:
    """A JSON object embedded in prose is recovered."""
    assert _extract_json('Here you go: {"a": {"b": 2}} hope that helps') == {"a": {"b": 2}}


def test_extract_json_failure() -> None:
    """Unparseable text returns None rather than raising."""
    assert _extract_json("no json here") is None


# ----------------------------------------------------------------------
# Claude agent against a stub client
# ----------------------------------------------------------------------


def test_llm_agent_parses_a_tool_call(
    dataset: dict[str, Any], transactions: list[Transaction]
) -> None:
    """A tool action from the model becomes a tool action for the env."""
    reply = json.dumps(
        {
            "reasoning": "Screening both wallets first.",
            "action": {
                "name": "check_sanctions",
                "type": "tool",
                "args": {"wallet_address": transactions[0].sender_wallet},
            },
        }
    )
    env = StableGuardEnv(dataset)
    observation = _observation(env, transactions[0].id)

    agent = LLMAgent(client=_StubClient([reply]))
    agent.begin_episode(observation)
    action = agent.act(observation)
    assert action["type"] == "tool"
    assert action["name"] == "check_sanctions"
    assert agent.reasoning.startswith("Screening")
    assert env.action_space.contains(action)


def test_llm_agent_parses_a_verdict(
    dataset: dict[str, Any], transactions: list[Transaction]
) -> None:
    """A decision action is normalised to approve or flag."""
    reply = json.dumps({"reasoning": "Clear mule pattern.", "action": {"type": "decision", "verdict": "FLAG"}})
    env = StableGuardEnv(dataset)
    observation = _observation(env, transactions[0].id)

    agent = LLMAgent(client=_StubClient([reply]))
    agent.begin_episode(observation)
    assert agent.act(observation) == {"type": "decision", "verdict": "flag"}


def test_llm_agent_counts_tokens(
    dataset: dict[str, Any], transactions: list[Transaction]
) -> None:
    """Tokens are accumulated per episode and per run."""
    env = StableGuardEnv(dataset)
    observation = _observation(env, transactions[0].id)

    agent = LLMAgent(client=_StubClient([]))
    agent.begin_episode(observation)
    agent.act(observation)
    agent.act(observation)
    assert agent.tokens_used == 2 * 150
    assert agent.total_input_tokens == 240
    assert agent.total_output_tokens == 60

    # A new episode resets the per episode counter but not the run total.
    agent.begin_episode(observation)
    assert agent.tokens_used == 0
    assert agent.total_input_tokens == 240


def test_llm_agent_prompt_contains_the_payment_and_results(
    dataset: dict[str, Any], transactions: list[Transaction]
) -> None:
    """The prompt carries the payment, the tool results and the budget."""
    env = StableGuardEnv(dataset)
    observation = _observation(env, transactions[0].id)
    observation, _, _, _, _ = env.step(
        {
            "type": "tool",
            "name": "check_sanctions",
            "args": {"wallet_address": transactions[0].sender_wallet},
        }
    )

    client = _StubClient([])
    agent = LLMAgent(client=client)
    agent.begin_episode(observation)
    agent.act(observation)

    request = client.messages.requests[-1]
    prompt = request["messages"][0]["content"]
    assert transactions[0].id in prompt
    assert "check_sanctions" in prompt
    assert "Tool calls remaining" in prompt
    # Ground truth must never reach the model.
    assert "is_fraud" not in prompt
    assert "fraud_type" not in prompt


def test_llm_agent_system_prompt_lists_the_tools(
    dataset: dict[str, Any], transactions: list[Transaction]
) -> None:
    """The system prompt documents every tool and is marked cacheable."""
    env = StableGuardEnv(dataset)
    observation = _observation(env, transactions[0].id)

    client = _StubClient([])
    agent = LLMAgent(client=client)
    agent.begin_episode(observation)
    agent.act(observation)

    system = client.messages.requests[-1]["system"][0]
    assert system["cache_control"] == {"type": "ephemeral"}
    for tool in ("check_wallet_history", "check_velocity", "check_merchant"):
        assert tool in system["text"]


def test_llm_agent_handles_unparseable_output(
    dataset: dict[str, Any], transactions: list[Transaction]
) -> None:
    """Garbage from the model ends the episode instead of hanging it."""
    env = StableGuardEnv(dataset)
    observation = _observation(env, transactions[0].id)

    agent = LLMAgent(client=_StubClient(["I am not JSON"]))
    agent.begin_episode(observation)
    action = agent.act(observation)
    assert action == {"type": "decision", "verdict": "approve"}
    assert "parseable" in agent.reasoning


def test_llm_agent_falls_back_on_api_error(
    dataset: dict[str, Any], transactions: list[Transaction]
) -> None:
    """An API outage falls back to the checklist so a run still finishes."""
    txn = next(t for t in transactions if t.fraud_type == "velocity")
    env = StableGuardEnv(dataset)
    observation = _observation(env, txn.id)

    agent = LLMAgent(client=_FailingClient())
    agent.begin_episode(observation)
    action = agent.act(observation)
    assert action == {"type": "decision", "verdict": "flag"}
    assert agent.api_errors == 1
    assert "fell back" in agent.reasoning


def test_llm_agent_can_raise_instead_of_falling_back(
    dataset: dict[str, Any], transactions: list[Transaction]
) -> None:
    """With the fallback disabled, an API error propagates."""
    env = StableGuardEnv(dataset)
    observation = _observation(env, transactions[0].id)

    agent = LLMAgent(client=_FailingClient(), fallback_on_error=False)
    agent.begin_episode(observation)
    with pytest.raises(RuntimeError):
        agent.act(observation)


def test_llm_agent_requires_a_key_without_an_injected_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a key and without a client, construction fails with advice."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        LLMAgent()
