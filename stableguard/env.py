"""The StableGuard Gymnasium environment.

One episode is one payment review. The agent sees the payment with the ground
truth removed, may call investigation tools, and must end the episode with a
verdict of approve or flag.

Reward shaping (kept deliberately simple, values live in ``config.yaml``):

* ``+1.0``  for a correct verdict
* ``-1.0``  for a wrong verdict
* ``-0.05`` for every tool call, so investigation is not free
* ``+0.2``  bonus when fraud is caught with two tool calls or fewer
* ``-1.0``  if the tool budget runs out before the agent decides

The bonus is paid only for catching fraud, not for approving clean traffic,
because approving without looking is exactly the behaviour we do not want to
reward.
"""

from __future__ import annotations

from typing import Any, Iterable

import gymnasium as gym
import numpy as np

from .config import load_config
from .models import Transaction
from .tools import TOOL_NAMES, InvestigationTools

VERDICTS: tuple[str, ...] = ("approve", "flag")


class StructuredActionSpace(gym.Space):
    """The action space: either a tool call or a final verdict.

    Actions are dictionaries rather than integers because the agents here are
    language models that emit JSON. The space still supports ``sample()`` so a
    random agent can be used as a sanity check, and ``contains()`` so malformed
    actions can be rejected early.
    """

    def __init__(self, tool_names: Iterable[str] = TOOL_NAMES, seed: int | None = None) -> None:
        super().__init__(shape=None, dtype=None, seed=seed)
        self.tool_names: tuple[str, ...] = tuple(tool_names)

    def sample(self, mask: Any = None) -> dict[str, Any]:  # noqa: D102 (documented above)
        rng: np.random.Generator = self.np_random
        if rng.random() < 0.5:
            name = self.tool_names[int(rng.integers(len(self.tool_names)))]
            return {"type": "tool", "name": name, "args": {}}
        verdict = VERDICTS[int(rng.integers(len(VERDICTS)))]
        return {"type": "decision", "verdict": verdict}

    def contains(self, x: Any) -> bool:  # noqa: D102 (documented above)
        if not isinstance(x, dict):
            return False
        if x.get("type") == "tool":
            return isinstance(x.get("name"), str)
        if x.get("type") == "decision":
            return x.get("verdict") in VERDICTS
        return False

    def __repr__(self) -> str:
        return f"StructuredActionSpace(tools={len(self.tool_names)})"


class ObservationSpace(gym.Space):
    """The observation space: a transaction, tool results so far, steps left."""

    REQUIRED_KEYS: tuple[str, ...] = ("transaction", "tool_results", "steps_remaining")

    def __init__(self, seed: int | None = None) -> None:
        super().__init__(shape=None, dtype=None, seed=seed)

    def sample(self, mask: Any = None) -> dict[str, Any]:  # noqa: D102
        return {"transaction": {}, "tool_results": [], "steps_remaining": 0}

    def contains(self, x: Any) -> bool:  # noqa: D102
        return isinstance(x, dict) and all(key in x for key in self.REQUIRED_KEYS)

    def __repr__(self) -> str:
        return "ObservationSpace(transaction, tool_results, steps_remaining)"


class StableGuardEnv(gym.Env):
    """Gymnasium environment for reviewing stablecoin payments."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        dataset: dict[str, Any],
        transactions: list[Transaction] | None = None,
        tool_budget: int | None = None,
        shuffle: bool = False,
        seed: int | None = None,
        config: dict[str, Any] | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Build an environment over a dataset produced by the generator.

        Args:
            dataset: the dict written by ``generate_dataset``.
            transactions: an optional subset to review, in order. Defaults to
                every transaction in the dataset.
            tool_budget: maximum tool calls per episode. Defaults to config.
            shuffle: shuffle the episode order.
            seed: seed for shuffling and for action sampling.
            config: an already loaded config dict, mostly for tests.
            render_mode: "human" prints a short summary on render().
        """
        super().__init__()
        self.config = config or load_config()
        env_cfg = self.config["env"]

        self.tool_budget: int = int(env_cfg["tool_budget"] if tool_budget is None else tool_budget)
        self.reward_correct: float = float(env_cfg["reward_correct"])
        self.reward_wrong: float = float(env_cfg["reward_wrong"])
        self.reward_tool_call: float = float(env_cfg["reward_tool_call"])
        self.efficient_bonus: float = float(env_cfg["reward_efficient_catch_bonus"])
        self.efficient_max_tools: int = int(env_cfg["efficient_catch_max_tools"])

        self.dataset = dataset
        self.transactions: list[Transaction] = (
            transactions
            if transactions is not None
            else [Transaction(**row) for row in dataset["transactions"]]
        )
        if not self.transactions:
            raise ValueError("the environment needs at least one transaction")

        self.tools = InvestigationTools(dataset, config=self.config)
        self.render_mode = render_mode

        self.action_space = StructuredActionSpace(TOOL_NAMES, seed=seed)
        self.observation_space = ObservationSpace(seed=seed)

        self._order: list[int] = list(range(len(self.transactions)))
        self._shuffle = shuffle
        self._cursor: int = 0

        # Episode state.
        self.current: Transaction | None = None
        self.tool_results: list[dict[str, Any]] = []
        self.steps_used: int = 0
        self.done: bool = True

        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(self._order)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start a new review episode and return the first observation.

        ``options`` may carry ``{"transaction_id": "txn_000007"}`` or
        ``{"index": 7}`` to review one specific payment, which is what the API
        and the dashboard use.
        """
        super().reset(seed=seed)
        options = options or {}

        index: int | None = None
        if "transaction_id" in options:
            wanted = options["transaction_id"]
            index = next(
                (i for i, t in enumerate(self.transactions) if t.id == wanted),
                None,
            )
            if index is None:
                raise KeyError(f"transaction {wanted!r} is not in this dataset")
        elif "index" in options:
            index = int(options["index"]) % len(self.transactions)
        else:
            index = self._order[self._cursor % len(self._order)]
            self._cursor += 1

        self.current = self.transactions[index]
        self.tools.set_context(self.current)
        self.tool_results = []
        self.steps_used = 0
        self.done = False

        info: dict[str, Any] = {
            "transaction_id": self.current.id,
            "tool_budget": self.tool_budget,
        }
        return self._observation(), info

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply one action: either a tool call or a final verdict."""
        if self.done or self.current is None:
            raise RuntimeError("step() called on a finished episode, call reset() first")

        if not self.action_space.contains(action):
            # A malformed action is treated as a tool call that returns an
            # error. It costs the agent a step but does not end the run.
            return self._apply_tool_call(
                {"type": "tool", "name": "invalid_action", "args": {}},
                forced_error={"error": "invalid_action", "action": repr(action)[:200]},
            )

        if action["type"] == "decision":
            return self._apply_decision(str(action["verdict"]))
        return self._apply_tool_call(action)

    def render(self) -> None:
        """Print a short human readable summary of the episode so far."""
        if self.render_mode != "human" or self.current is None:
            return
        txn = self.current
        print(f"[{txn.id}] {txn.amount_usd:,.2f} {txn.stablecoin} on {txn.chain}")
        print(f"  from {txn.sender_country} to {txn.receiver_country}, {txn.merchant_category}")
        print(f"  tool calls used: {self.steps_used}/{self.tool_budget}")
        for result in self.tool_results:
            print(f"  - {result['name']}({result['args']}) -> {result['result']}")

    def close(self) -> None:
        """Release the tool context."""
        self.tools.set_context(None)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _observation(self) -> dict[str, Any]:
        """Build the observation: payment, tool results so far, steps left."""
        assert self.current is not None
        return {
            "transaction": self.current.observation(),
            "tool_results": list(self.tool_results),
            "steps_remaining": self.tool_budget - self.steps_used,
        }

    def _apply_tool_call(
        self,
        action: dict[str, Any],
        forced_error: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Run one tool, or end the episode if the budget is exhausted."""
        assert self.current is not None

        if self.steps_used >= self.tool_budget:
            # The agent spent its whole budget without deciding. Out of budget
            # is scored like a wrong answer, so stalling is never a good policy.
            self.done = True
            reward = self.reward_wrong
            info = self._terminal_info(verdict=None, correct=False, reward=reward)
            info["reason"] = "tool_budget_exhausted"
            return self._observation(), reward, False, True, info

        name = str(action.get("name", ""))
        args = dict(action.get("args") or {})
        result = forced_error if forced_error is not None else self.tools.call(name, args)

        self.steps_used += 1
        self.tool_results.append({"name": name, "args": args, "result": result})

        info = {
            "transaction_id": self.current.id,
            "tool": name,
            "steps_used": self.steps_used,
            "steps_remaining": self.tool_budget - self.steps_used,
        }
        return self._observation(), self.reward_tool_call, False, False, info

    def _apply_decision(
        self, verdict: str
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Score a verdict and end the episode."""
        assert self.current is not None
        correct = (verdict == "flag") == self.current.is_fraud

        reward = self.reward_correct if correct else self.reward_wrong
        caught_fraud_efficiently = (
            correct
            and self.current.is_fraud
            and self.steps_used <= self.efficient_max_tools
        )
        if caught_fraud_efficiently:
            reward += self.efficient_bonus

        self.done = True
        info = self._terminal_info(verdict=verdict, correct=correct, reward=reward)
        info["efficiency_bonus"] = self.efficient_bonus if caught_fraud_efficiently else 0.0
        return self._observation(), reward, True, False, info

    def _terminal_info(
        self, verdict: str | None, correct: bool, reward: float
    ) -> dict[str, Any]:
        """Build the info dict returned when an episode ends.

        This is the only place the ground truth leaves the environment, and it
        goes to the evaluator, never into an observation.
        """
        assert self.current is not None
        return {
            "transaction_id": self.current.id,
            "verdict": verdict,
            "is_fraud": self.current.is_fraud,
            "fraud_type": self.current.fraud_type,
            "correct": correct,
            "reward": reward,
            "n_tool_calls": self.steps_used,
            "tool_calls": list(self.tool_results),
        }
