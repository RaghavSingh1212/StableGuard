"""Agents that review payments.

Every agent implements the same tiny interface so the evaluator can run any of
them against the environment without knowing which one it has:

    agent.begin_episode(observation)      # called once per payment
    action = agent.act(observation)       # called until the episode ends
    agent.reasoning                       # a short explanation of the verdict
    agent.tokens_used                     # tokens spent on this episode

Three implementations ship with StableGuard: a random agent used as a floor in
tests, a rule based baseline that needs no API key, and a Claude agent.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from abc import ABC, abstractmethod
from typing import Any

from .config import anthropic_api_key, claude_model, load_config
from .tools import TOOL_SCHEMAS

logger = logging.getLogger(__name__)

Action = dict[str, Any]
Observation = dict[str, Any]


class Agent(ABC):
    """Base class for every reviewing agent."""

    #: Short name used in reports, on the leaderboard and in the API.
    name: str = "agent"

    def __init__(self) -> None:
        self.reasoning: str = ""
        self.tokens_used: int = 0

    def begin_episode(self, observation: Observation) -> None:
        """Reset per episode state before the first action."""
        self.reasoning = ""
        self.tokens_used = 0

    @abstractmethod
    def act(self, observation: Observation) -> Action:
        """Return the next action: a tool call or a final verdict."""

    def describe(self) -> dict[str, Any]:
        """Return a small dict describing the agent, used by the API."""
        return {"name": self.name, "type": type(self).__name__}


class RandomAgent(Agent):
    """Flips a coin. Only useful as a floor to compare real agents against."""

    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self._rng = random.Random(seed)

    def act(self, observation: Observation) -> Action:
        """Decide immediately, at random."""
        verdict = self._rng.choice(["approve", "flag"])
        self.reasoning = "Random baseline."
        return {"type": "decision", "verdict": verdict}


class RuleBasedAgent(Agent):
    """The baseline analyst: a short checklist, no model, no API key.

    It applies the rules in the architecture, in increasing order of cost:

    1. velocity above the threshold (free, the value is on the payment itself),
    2. either wallet on a sanctions list,
    3. amount far above the wallet's usual size,
    4. payment sent from outside the wallet's home country.

    The checklist stops at the first hit, so an obvious case is caught with one
    tool call or none, which is also what the environment's efficiency bonus
    rewards. Everything else is approved.
    """

    name = "rule_based"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config = config or load_config()
        thresholds = self.config["thresholds"]
        self.velocity_flag: int = int(thresholds["velocity_flag"])
        self.amount_ratio_flag: float = float(thresholds["amount_ratio_flag"])
        self._plan: list[Action] = []

    def begin_episode(self, observation: Observation) -> None:
        """Build the tool plan for this payment."""
        super().begin_episode(observation)
        txn = observation["transaction"]
        sender = txn["sender_wallet"]
        self._plan = [
            {"type": "tool", "name": "check_sanctions", "args": {"wallet_address": sender}},
            {
                "type": "tool",
                "name": "check_sanctions",
                "args": {"wallet_address": txn["receiver_wallet"]},
            },
            {
                "type": "tool",
                "name": "check_amount_profile",
                "args": {"wallet_address": sender, "amount_usd": txn["amount_usd"]},
            },
            {
                "type": "tool",
                "name": "check_geo",
                "args": {"sender_country": txn["sender_country"], "wallet_address": sender},
            },
        ]

    def act(self, observation: Observation) -> Action:
        """Run the checklist, stopping at the first rule that fires."""
        txn = observation["transaction"]
        results = observation["tool_results"]

        # Rule 1: a velocity burst is visible on the payment itself.
        if txn["velocity_1h"] > self.velocity_flag:
            self.reasoning = (
                f"Velocity of {txn['velocity_1h']} payments in the last hour is above the "
                f"threshold of {self.velocity_flag}."
            )
            return {"type": "decision", "verdict": "flag"}

        # Rules 2 to 4: inspect whatever the tools have returned so far.
        for entry in results:
            result = entry["result"]
            if entry["name"] == "check_sanctions" and result.get("sanctioned"):
                self.reasoning = (
                    f"Wallet {result['wallet_address']} appears on {result['list_name']}."
                )
                return {"type": "decision", "verdict": "flag"}
            if entry["name"] == "check_amount_profile" and result.get("ratio", 0) > self.amount_ratio_flag:
                self.reasoning = (
                    f"Amount is {result['ratio']}x the wallet's typical "
                    f"{result['typical_amount_usd']} USD payment."
                )
                return {"type": "decision", "verdict": "flag"}
            if entry["name"] == "check_geo" and result.get("mismatch"):
                self.reasoning = (
                    f"Payment sent from {result['sender_country']} but the wallet's home "
                    f"country is {result['wallet_home_country']}."
                )
                return {"type": "decision", "verdict": "flag"}

        # Nothing fired yet. Run the next check if there is budget for it.
        next_index = len(results)
        if next_index < len(self._plan) and observation["steps_remaining"] > 0:
            return self._plan[next_index]

        self.reasoning = "No sanctions hit, normal velocity, normal amount, no geo mismatch."
        return {"type": "decision", "verdict": "approve"}


# ----------------------------------------------------------------------
# Claude agent
# ----------------------------------------------------------------------

SYSTEM_PROMPT = """You are a payments fraud analyst reviewing stablecoin transactions for a \
payment network that settles USDC and USDT. Your job is to decide whether each payment should \
be approved or flagged for manual review.

You investigate by calling tools. You have a budget of at most {tool_budget} tool calls per \
payment, and every call is charged, so call only what you need. These tools are available:

{tool_docs}

Fraud patterns worth looking for:
- velocity: a burst of payments from the same wallet inside an hour (the payment already shows velocity_1h).
- geo: the payment was sent from a country that is not the wallet's home country.
- sanctions: either wallet appears on a sanctions list.
- mule: a wallet opened in the last few days moving a large amount straight back out.
- amount_spike: an amount many times larger than the wallet usually sends.
- merchant_mismatch: a high risk merchant category (crypto exchange, gambling) that this wallet \
has no history with. A high risk category on its own is not fraud if the wallet uses it regularly.

Most payments are legitimate, so do not flag on weak evidence alone. Flagging clean traffic has \
a real cost for the business.

Reply with one JSON object and nothing else, in exactly this shape:

{{"reasoning": "one or two sentences", "action": {{"type": "tool", "name": "<tool name>", "args": {{...}}}}}}

or, when you are ready to decide:

{{"reasoning": "one or two sentences", "action": {{"type": "decision", "verdict": "approve"}}}}

The verdict must be either "approve" or "flag"."""


def _tool_documentation() -> str:
    """Render the tool schemas as readable lines for the system prompt."""
    lines: list[str] = []
    for schema in TOOL_SCHEMAS:
        params = ", ".join(schema["input_schema"]["properties"])
        lines.append(f"- {schema['name']}({params}): {schema['description']}")
    return "\n".join(lines)


def _extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of a model response.

    Models usually return clean JSON here, but a stray code fence or a sentence
    before the object should not cost a whole episode.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    # Fall back to the first balanced object in the text.
    start = text.find("{")
    while start != -1:
        depth = 0
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : index + 1])
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


class LLMAgent(Agent):
    """A fraud analyst backed by the Anthropic Claude API.

    The agent is prompted fresh on every step with the payment and every tool
    result so far, and answers with one JSON object holding its reasoning and
    its next action. That keeps the system prompt stable (so it caches well),
    makes each step easy to log, and means a malformed answer costs one step
    rather than the whole episode.

    The API key is read from the ANTHROPIC_API_KEY environment variable and is
    never stored in the repository.
    """

    name = "claude"

    def __init__(
        self,
        model: str | None = None,
        config: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        client: Any | None = None,
        fallback_on_error: bool = True,
    ) -> None:
        super().__init__()
        self.config = config or load_config()
        llm_cfg = self.config["llm"]

        self.model: str = model or claude_model()
        self.max_tokens: int = int(max_tokens or llm_cfg["max_tokens"])
        self.effort: str = str(llm_cfg.get("effort", "low"))
        self.tool_budget: int = int(self.config["env"]["tool_budget"])

        # Token counters: per episode (reset by begin_episode) and per run.
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.api_errors: int = 0

        # On an API failure the agent can fall back to the rule based checklist
        # so a long eval run finishes instead of dying on one bad request.
        self.fallback_on_error = fallback_on_error
        self._fallback = RuleBasedAgent(config=self.config)

        self._system_prompt = SYSTEM_PROMPT.format(
            tool_budget=self.tool_budget, tool_docs=_tool_documentation()
        )
        self._client = client if client is not None else self._build_client()

    @staticmethod
    def _build_client() -> Any:
        """Create an Anthropic client, failing with a clear message if unkeyed."""
        if anthropic_api_key() is None and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key, "
                "or run the rule_based agent, which needs no key."
            )
        import anthropic  # imported lazily so the baseline runs without the SDK

        return anthropic.Anthropic()

    def begin_episode(self, observation: Observation) -> None:
        """Reset per episode token counters and the fallback checklist."""
        super().begin_episode(observation)
        self.input_tokens = 0
        self.output_tokens = 0
        self._fallback.begin_episode(observation)

    def describe(self) -> dict[str, Any]:
        """Return the agent description, including the model in use."""
        return {"name": self.name, "type": type(self).__name__, "model": self.model}

    # ------------------------------------------------------------------
    # Prompting
    # ------------------------------------------------------------------

    def _user_prompt(self, observation: Observation) -> str:
        """Render the payment and the investigation so far as a prompt."""
        parts = [
            "Payment under review:",
            json.dumps(observation["transaction"], indent=2),
        ]
        if observation["tool_results"]:
            parts.append("\nTool results so far:")
            for entry in observation["tool_results"]:
                parts.append(
                    f"- {entry['name']}({json.dumps(entry['args'])}) -> "
                    f"{json.dumps(entry['result'])}"
                )
        else:
            parts.append("\nNo tools called yet.")
        parts.append(
            f"\nTool calls remaining: {observation['steps_remaining']}. "
            "Reply with one JSON object holding your reasoning and your next action."
        )
        if observation["steps_remaining"] <= 0:
            parts.append("Your tool budget is spent. You must return a verdict now.")
        return "\n".join(parts)

    def act(self, observation: Observation) -> Action:
        """Ask Claude for the next action and return it."""
        try:
            text = self._call_model(self._user_prompt(observation))
        except Exception as exc:  # noqa: BLE001 - any API failure is handled the same way
            self.api_errors += 1
            logger.warning("Claude API call failed (%s), falling back to the checklist", exc)
            if not self.fallback_on_error:
                raise
            action = self._fallback.act(observation)
            self.reasoning = f"API error, fell back to the rule checklist: {self._fallback.reasoning}"
            return action

        payload = _extract_json(text)
        if payload is None:
            self.reasoning = "Model did not return parseable JSON."
            return {"type": "decision", "verdict": "approve"}

        self.reasoning = str(payload.get("reasoning", ""))[:1000]
        action = payload.get("action")
        if not isinstance(action, dict):
            return {"type": "decision", "verdict": "approve"}

        if action.get("type") == "decision":
            verdict = str(action.get("verdict", "approve")).lower().strip()
            return {"type": "decision", "verdict": "flag" if verdict == "flag" else "approve"}

        return {
            "type": "tool",
            "name": str(action.get("name", "")),
            "args": action.get("args") or {},
        }

    def _call_model(self, prompt: str) -> str:
        """Send one request to Claude and return the text of the reply.

        Sampling parameters are deliberately not sent: the current Claude models
        reject temperature and top_p, and a low effort setting plus a fully
        specified prompt is what keeps this reproducible.
        """
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": self._system_prompt,
                    # The system prompt is identical on every call in a run, so
                    # caching it makes a 200 payment eval noticeably cheaper.
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": prompt}],
        )

        usage = getattr(response, "usage", None)
        if usage is not None:
            self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
            self.total_input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            self.total_output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
            self.tokens_used = self.input_tokens + self.output_tokens

        for block in response.content:
            if getattr(block, "type", None) == "text":
                return str(block.text)
        return ""


#: Agents that can be built by name from the CLI, the API and the dashboard.
AGENT_REGISTRY: dict[str, type[Agent]] = {
    "rule_based": RuleBasedAgent,
    "claude": LLMAgent,
    "random": RandomAgent,
}


def build_agent(name: str, **kwargs: Any) -> Agent:
    """Construct an agent by name.

    Raises:
        KeyError: if the name is not one of ``rule_based``, ``claude`` or ``random``.
    """
    try:
        agent_class = AGENT_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown agent {name!r}, choose one of {sorted(AGENT_REGISTRY)}"
        ) from exc
    return agent_class(**kwargs)
