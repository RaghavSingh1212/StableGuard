"""Metric computation, report formatting and the leaderboard.

Everything here works on a list of :class:`~stableguard.models.Decision`
objects that the evaluator has already scored, so metrics can be recomputed
from a saved report without rerunning an agent.

Conventions: "positive" means the agent flagged the payment, so a true positive
is fraud that was caught and a false positive is clean traffic that was flagged.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import RESULTS_DIR, load_config
from .models import FRAUD_TYPES, Decision, EvalResult


def _safe_divide(numerator: float, denominator: float) -> float:
    """Divide, returning 0.0 instead of raising when the denominator is zero."""
    return numerator / denominator if denominator else 0.0


def confusion_counts(decisions: Iterable[Decision]) -> dict[str, int]:
    """Count true and false positives and negatives over scored decisions."""
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for decision in decisions:
        flagged = decision.verdict == "flag"
        is_fraud = bool(decision.is_fraud)
        if flagged and is_fraud:
            counts["tp"] += 1
        elif flagged and not is_fraud:
            counts["fp"] += 1
        elif not flagged and is_fraud:
            counts["fn"] += 1
        else:
            counts["tn"] += 1
    return counts


def recall_by_fraud_type(decisions: Iterable[Decision]) -> dict[str, float]:
    """Return recall for each fraud type that appears in the decisions.

    This is the most useful view in the whole harness: an agent can look strong
    overall while missing one pattern completely.
    """
    caught: dict[str, int] = {}
    total: dict[str, int] = {}
    for decision in decisions:
        if not decision.is_fraud:
            continue
        fraud_type = decision.fraud_type or "unknown"
        total[fraud_type] = total.get(fraud_type, 0) + 1
        if decision.verdict == "flag":
            caught[fraud_type] = caught.get(fraud_type, 0) + 1

    ordered = [t for t in FRAUD_TYPES if t in total] + [
        t for t in sorted(total) if t not in FRAUD_TYPES
    ]
    return {t: round(_safe_divide(caught.get(t, 0), total[t]), 4) for t in ordered}


def estimate_cost_usd(
    input_tokens: int, output_tokens: int, config: dict[str, Any] | None = None
) -> float:
    """Estimate the API cost of a run from its token counts.

    Rates come from ``config.yaml`` and are per million tokens. This is an
    estimate: it ignores the discount on cached input tokens, so the real bill
    is usually a little lower.
    """
    llm_cfg = (config or load_config())["llm"]
    input_cost = input_tokens / 1_000_000 * float(llm_cfg["input_cost_per_mtok"])
    output_cost = output_tokens / 1_000_000 * float(llm_cfg["output_cost_per_mtok"])
    return round(input_cost + output_cost, 6)


def compute_metrics(
    decisions: list[Decision],
    agent_name: str,
    dataset: str = "",
    model: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    config: dict[str, Any] | None = None,
) -> EvalResult:
    """Turn scored decisions into an :class:`EvalResult`."""
    if not decisions:
        raise ValueError("cannot compute metrics over an empty list of decisions")

    counts = confusion_counts(decisions)
    tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
    n = len(decisions)

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    total_tokens = sum(d.tokens_used for d in decisions)
    if not input_tokens and not output_tokens:
        # No split available (for example when replaying a saved report), so
        # attribute everything to input for a conservative cost estimate.
        input_tokens = total_tokens

    return EvalResult(
        agent_name=agent_name,
        n_transactions=n,
        accuracy=round(_safe_divide(tp + tn, n), 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        # Share of clean payments that were wrongly flagged.
        false_positive_rate=round(_safe_divide(fp, fp + tn), 4),
        # Share of fraud that was missed.
        false_negative_rate=round(_safe_divide(fn, fn + tp), 4),
        avg_tool_calls=round(_safe_divide(sum(d.n_tool_calls for d in decisions), n), 3),
        avg_latency_seconds=round(_safe_divide(sum(d.latency_seconds for d in decisions), n), 4),
        recall_by_fraud_type=recall_by_fraud_type(decisions),
        timestamp=datetime.now(timezone.utc),
        n_fraud=tp + fn,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        avg_reward=round(
            _safe_divide(sum(d.reward or 0.0 for d in decisions), n), 4
        ),
        total_tokens=total_tokens or (input_tokens + output_tokens),
        estimated_cost_usd=estimate_cost_usd(input_tokens, output_tokens, config=config),
        dataset=dataset,
        model=model,
    )


def format_summary(result: EvalResult) -> str:
    """Render one eval result as a printable summary table."""
    width = 62
    lines = [
        "=" * width,
        f"StableGuard eval: {result.agent_name}" + (f" ({result.model})" if result.model else ""),
        "=" * width,
        f"{'Transactions':<26}{result.n_transactions}",
        f"{'Fraud in sample':<26}{result.n_fraud}",
        "-" * width,
        f"{'Accuracy':<26}{result.accuracy:.3f}",
        f"{'Precision':<26}{result.precision:.3f}",
        f"{'Recall':<26}{result.recall:.3f}",
        f"{'F1':<26}{result.f1:.3f}",
        f"{'False positive rate':<26}{result.false_positive_rate:.3f}",
        f"{'False negative rate':<26}{result.false_negative_rate:.3f}",
        "-" * width,
        f"{'Caught fraud':<26}{result.true_positives}",
        f"{'Missed fraud':<26}{result.false_negatives}",
        f"{'Wrongly flagged':<26}{result.false_positives}",
        f"{'Correctly approved':<26}{result.true_negatives}",
        "-" * width,
        f"{'Avg tool calls':<26}{result.avg_tool_calls:.2f}",
        f"{'Avg latency (s)':<26}{result.avg_latency_seconds:.3f}",
        f"{'Avg reward':<26}{result.avg_reward:.3f}",
        f"{'Tokens used':<26}{result.total_tokens:,}",
        f"{'Estimated cost (USD)':<26}{result.estimated_cost_usd:.4f}",
        "-" * width,
        "Recall by fraud type",
    ]
    for fraud_type, value in result.recall_by_fraud_type.items():
        bar = "#" * int(round(value * 20))
        lines.append(f"  {fraud_type:<20}{value:>6.2f}  {bar}")
    lines.append("=" * width)
    return "\n".join(lines)


def save_report(
    result: EvalResult,
    decisions: list[Decision] | None = None,
    results_dir: str | Path = RESULTS_DIR,
) -> Path:
    """Write a JSON report to ``results/<agent>_<timestamp>.json``."""
    directory = Path(results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
    path = directory / f"{result.agent_name}_{stamp}.json"

    payload: dict[str, Any] = {"result": result.model_dump(mode="json")}
    if decisions is not None:
        payload["decisions"] = [d.model_dump(mode="json") for d in decisions]

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def load_report(path: str | Path) -> dict[str, Any]:
    """Read one saved report."""
    with open(path, "r", encoding="utf-8") as handle:
        report: dict[str, Any] = json.load(handle)
    return report


def load_all_results(results_dir: str | Path = RESULTS_DIR) -> list[EvalResult]:
    """Load every saved eval result, newest first."""
    directory = Path(results_dir)
    if not directory.exists():
        return []

    results: list[EvalResult] = []
    for path in sorted(directory.glob("*.json")):
        try:
            report = load_report(path)
            results.append(EvalResult(**report["result"]))
        except (KeyError, ValueError, OSError):
            # A half written or hand edited report should not break the board.
            continue
    results.sort(key=lambda r: r.timestamp, reverse=True)
    return results


def leaderboard(
    results: list[EvalResult] | None = None,
    results_dir: str | Path = RESULTS_DIR,
    latest_per_agent: bool = True,
) -> list[EvalResult]:
    """Return eval results ranked by F1, best first.

    By default only the most recent run per agent is kept, which is what makes
    the board readable after a dozen experiments.
    """
    results = load_all_results(results_dir) if results is None else list(results)

    if latest_per_agent:
        newest: dict[str, EvalResult] = {}
        for result in results:
            current = newest.get(result.agent_name)
            if current is None or result.timestamp > current.timestamp:
                newest[result.agent_name] = result
        results = list(newest.values())

    results.sort(key=lambda r: (r.f1, r.recall, -r.avg_latency_seconds), reverse=True)
    return results


def format_leaderboard(results: list[EvalResult]) -> str:
    """Render the leaderboard as a printable table."""
    if not results:
        return "No eval results yet. Run scripts/run_eval.py to create one."

    header = (
        f"{'#':<3}{'agent':<14}{'n':>6}{'acc':>8}{'prec':>8}{'rec':>8}"
        f"{'f1':>8}{'fpr':>8}{'tools':>8}{'sec':>9}"
    )
    lines = [header, "-" * len(header)]
    for rank, result in enumerate(results, start=1):
        lines.append(
            f"{rank:<3}{result.agent_name:<14}{result.n_transactions:>6}"
            f"{result.accuracy:>8.3f}{result.precision:>8.3f}{result.recall:>8.3f}"
            f"{result.f1:>8.3f}{result.false_positive_rate:>8.3f}"
            f"{result.avg_tool_calls:>8.2f}{result.avg_latency_seconds:>9.3f}"
        )
    return "\n".join(lines)
