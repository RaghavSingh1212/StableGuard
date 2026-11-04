"""The eval harness: run an agent over a dataset and score it.

The evaluator owns the loop. For every payment it resets the environment, lets
the agent act until the episode ends, and records what happened: the verdict,
the ground truth, whether the verdict was right, which tools were called, how
long it took and how many tokens it cost.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Iterable

from .agent import Agent, build_agent
from .config import DEFAULT_DATASET_PATH, load_config
from .env import StableGuardEnv
from .generator import load_dataset
from .metrics import compute_metrics, save_report
from .models import Decision, EvalResult, ToolCall, Transaction

#: Called after each payment with (index, total, decision). Used by the
#: dashboard to draw a progress bar.
ProgressCallback = Callable[[int, int, Decision], None]


class Evaluator:
    """Runs an agent over a dataset and produces an :class:`EvalResult`."""

    def __init__(
        self,
        dataset: dict[str, Any] | None = None,
        dataset_path: str | Path = DEFAULT_DATASET_PATH,
        tool_budget: int | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Load a dataset either from memory or from disk."""
        self.config = config or load_config()
        self.dataset_path = str(dataset_path)
        self.dataset = dataset if dataset is not None else load_dataset(dataset_path)
        self.transactions: list[Transaction] = [
            Transaction(**row) for row in self.dataset["transactions"]
        ]
        self.tool_budget = tool_budget

    # ------------------------------------------------------------------
    # Single payment
    # ------------------------------------------------------------------

    def review_one(
        self,
        agent: Agent,
        transaction_id: str,
        env: StableGuardEnv | None = None,
    ) -> Decision:
        """Let the agent review one payment and return the scored decision."""
        env = env or self._make_env(self.transactions)
        observation, _ = env.reset(options={"transaction_id": transaction_id})
        return self._run_episode(agent, env, observation)

    def _make_env(self, transactions: list[Transaction]) -> StableGuardEnv:
        """Build an environment over a list of transactions."""
        return StableGuardEnv(
            self.dataset,
            transactions=transactions,
            tool_budget=self.tool_budget,
            config=self.config,
        )

    def _run_episode(
        self, agent: Agent, env: StableGuardEnv, observation: dict[str, Any]
    ) -> Decision:
        """Drive one episode to its end and build the Decision record."""
        agent.begin_episode(observation)
        started = time.perf_counter()

        terminated = truncated = False
        info: dict[str, Any] = {}
        while not (terminated or truncated):
            action = agent.act(observation)
            observation, _, terminated, truncated, info = env.step(action)

        latency = time.perf_counter() - started

        # An episode that ran out of budget has no verdict of its own. Treat it
        # as an approval, which is the conservative reading: the payment went
        # through because the review never concluded.
        verdict = info.get("verdict") or "approve"
        tool_calls = [
            ToolCall(name=entry["name"], args=entry["args"], result=entry["result"])
            for entry in info.get("tool_calls", [])
        ]
        correct = (verdict == "flag") == bool(info.get("is_fraud"))

        return Decision(
            transaction_id=info["transaction_id"],
            verdict=verdict,  # type: ignore[arg-type]
            reasoning=agent.reasoning or "",
            tool_calls=tool_calls,
            latency_seconds=round(latency, 4),
            tokens_used=int(getattr(agent, "tokens_used", 0) or 0),
            is_fraud=bool(info.get("is_fraud")),
            fraud_type=info.get("fraud_type"),
            correct=correct,
            reward=float(info.get("reward", 0.0)),
        )

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------

    def select(self, n: int | None = None, offset: int = 0) -> list[Transaction]:
        """Return the slice of the dataset to evaluate on.

        The slice is taken from the front of the dataset, which keeps runs
        comparable between agents: everyone reviews the same payments.
        """
        rows = self.transactions[offset:]
        return rows if n is None else rows[:n]

    def run(
        self,
        agent: Agent | str,
        n: int | None = None,
        offset: int = 0,
        progress: ProgressCallback | None = None,
        save: bool = True,
        results_dir: str | Path | None = None,
    ) -> tuple[EvalResult, list[Decision]]:
        """Run an agent over ``n`` payments and return its metrics and decisions."""
        if isinstance(agent, str):
            agent = build_agent(agent)

        transactions = self.select(n=n, offset=offset)
        if not transactions:
            raise ValueError("the selected dataset slice is empty")

        env = self._make_env(transactions)
        decisions: list[Decision] = []

        for index, txn in enumerate(transactions):
            observation, _ = env.reset(options={"transaction_id": txn.id})
            decision = self._run_episode(agent, env, observation)
            decisions.append(decision)
            if progress is not None:
                progress(index + 1, len(transactions), decision)

        env.close()

        result = compute_metrics(
            decisions,
            agent_name=agent.name,
            dataset=self.dataset_path,
            model=getattr(agent, "model", None),
            input_tokens=int(getattr(agent, "total_input_tokens", 0) or 0),
            output_tokens=int(getattr(agent, "total_output_tokens", 0) or 0),
            config=self.config,
        )

        if save:
            kwargs = {"results_dir": results_dir} if results_dir is not None else {}
            save_report(result, decisions, **kwargs)  # type: ignore[arg-type]

        return result, decisions


def run_eval(
    agent: Agent | str,
    n: int | None = 200,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    offset: int = 0,
    save: bool = True,
    progress: ProgressCallback | None = None,
) -> tuple[EvalResult, list[Decision]]:
    """Convenience wrapper: build an evaluator, run one agent, return results."""
    evaluator = Evaluator(dataset_path=dataset_path)
    return evaluator.run(agent, n=n, offset=offset, save=save, progress=progress)


def compare(
    agents: Iterable[Agent | str],
    n: int | None = 200,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    save: bool = True,
) -> list[EvalResult]:
    """Run several agents over the same payments and return their results."""
    evaluator = Evaluator(dataset_path=dataset_path)
    return [evaluator.run(agent, n=n, save=save)[0] for agent in agents]
