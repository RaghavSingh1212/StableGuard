#!/usr/bin/env python3
"""CLI to run an eval and print the summary table.

Examples:
    python scripts/run_eval.py --agent rule_based --n 200
    python scripts/run_eval.py --agent claude --n 200 --dataset data/transactions.json
    python scripts/run_eval.py --leaderboard
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stableguard.agent import AGENT_REGISTRY, build_agent  # noqa: E402
from stableguard.config import DEFAULT_DATASET_PATH  # noqa: E402
from stableguard.evaluator import Evaluator  # noqa: E402
from stableguard.metrics import (  # noqa: E402
    format_leaderboard,
    format_summary,
    leaderboard,
    save_report,
)
from stableguard.models import Decision  # noqa: E402


def _progress(index: int, total: int, decision: Decision) -> None:
    """Print progress: a self overwriting line on a terminal, else a digest.

    Redirecting the output to a file would otherwise produce one very long line
    of carriage returns, so a piped run prints every twenty fifth payment.
    """
    mark = "ok " if decision.correct else "MISS"
    if sys.stdout.isatty():
        print(
            f"\r[{index:>4}/{total}] {decision.transaction_id} "
            f"{decision.verdict:<7} {mark}",
            end="",
            flush=True,
        )
    elif index % 25 == 0 or index == total:
        print(f"[{index:>4}/{total}] reviewed", flush=True)


def main() -> int:
    """Parse arguments, run the eval and print the report."""
    parser = argparse.ArgumentParser(description="Run a StableGuard eval.")
    parser.add_argument(
        "--agent",
        type=str,
        default="rule_based",
        choices=sorted(AGENT_REGISTRY),
        help="which agent to evaluate",
    )
    parser.add_argument("--n", type=int, default=200, help="number of transactions")
    parser.add_argument("--offset", type=int, default=0, help="skip this many transactions")
    parser.add_argument(
        "--dataset", type=str, default=str(DEFAULT_DATASET_PATH), help="dataset JSON file"
    )
    parser.add_argument("--no-save", action="store_true", help="do not write a report file")
    parser.add_argument(
        "--leaderboard",
        action="store_true",
        help="print the leaderboard from saved reports and exit",
    )
    args = parser.parse_args()

    if args.leaderboard:
        print(format_leaderboard(leaderboard()))
        return 0

    if not Path(args.dataset).exists():
        print(
            f"Dataset {args.dataset} not found. Generate one first:\n"
            "  python scripts/generate_data.py --n 1000 --fraud-rate 0.15 --seed 42"
        )
        return 1

    try:
        agent = build_agent(args.agent)
    except RuntimeError as exc:
        # Raised when the Claude agent is asked for without an API key.
        print(str(exc))
        return 1

    evaluator = Evaluator(dataset_path=args.dataset)
    print(f"Running {agent.name} over {args.n} transactions from {args.dataset}")
    result, decisions = evaluator.run(
        agent, n=args.n, offset=args.offset, save=False, progress=_progress
    )
    print()
    print(format_summary(result))

    if not args.no_save:
        path = save_report(result, decisions)
        print(f"\nReport written to {path}")

    print()
    print("Leaderboard")
    print(format_leaderboard(leaderboard()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
