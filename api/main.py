"""The StableGuard HTTP API.

The app is deliberately thin: it holds a dataset in memory, builds agents on
demand and delegates every decision to the evaluator and the environment. All
of the fraud logic lives in the ``stableguard`` package.

Run it with:
    uvicorn api.main:app --reload
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

# Allow "uvicorn api.main:app" from a checkout that is not installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stableguard import __version__
from stableguard.agent import AGENT_REGISTRY, build_agent
from stableguard.config import DEFAULT_DATASET_PATH, anthropic_api_key, load_config
from stableguard.evaluator import Evaluator
from stableguard.metrics import compute_metrics, format_summary, leaderboard, load_all_results
from stableguard.models import Decision, EvalResult

app = FastAPI(
    title="StableGuard",
    version=__version__,
    description=(
        "An RL environment and eval harness for AI agents that review stablecoin "
        "payments for fraud and compliance."
    ),
)

# ----------------------------------------------------------------------
# Request and response models
# ----------------------------------------------------------------------


class ReviewRequest(BaseModel):
    """Ask one agent to review one payment."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: str
    agent: str = "rule_based"


class ReviewResponse(BaseModel):
    """What the agent did, what it concluded and whether it was right."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: str
    agent: str
    verdict: Literal["approve", "flag"]
    reasoning: str
    tool_calls: list[dict[str, Any]]
    n_tool_calls: int
    latency_seconds: float
    tokens_used: int
    is_fraud: bool
    fraud_type: str | None
    correct: bool
    reward: float


class EvalRequest(BaseModel):
    """Run a full eval over the first ``n`` payments of the dataset."""

    model_config = ConfigDict(extra="forbid")

    agent: str = "rule_based"
    n: int = Field(default=50, ge=1, le=5000)
    save: bool = True


class EvalResponse(BaseModel):
    """An eval result plus its printable summary."""

    model_config = ConfigDict(extra="forbid")

    result: EvalResult
    summary: str


# ----------------------------------------------------------------------
# Lazily loaded state
# ----------------------------------------------------------------------

_evaluator: Evaluator | None = None


def get_evaluator() -> Evaluator:
    """Return the shared evaluator, loading the dataset on first use."""
    global _evaluator
    if _evaluator is None:
        if not Path(DEFAULT_DATASET_PATH).exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    "No dataset found. Generate one with: python scripts/generate_data.py "
                    "--n 1000 --fraud-rate 0.15 --seed 42"
                ),
            )
        _evaluator = Evaluator(dataset_path=DEFAULT_DATASET_PATH)
    return _evaluator


def _build_agent_or_400(name: str) -> Any:
    """Build an agent by name, turning configuration problems into HTTP errors."""
    if name not in AGENT_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"unknown agent {name!r}, choose one of {sorted(AGENT_REGISTRY)}",
        )
    try:
        return build_agent(name)
    except RuntimeError as exc:
        # Raised when the Claude agent is requested without an API key.
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _to_review_response(decision: Decision, agent_name: str) -> ReviewResponse:
    """Convert a scored decision into the API response shape."""
    return ReviewResponse(
        transaction_id=decision.transaction_id,
        agent=agent_name,
        verdict=decision.verdict,
        reasoning=decision.reasoning,
        tool_calls=[call.model_dump(mode="json") for call in decision.tool_calls],
        n_tool_calls=decision.n_tool_calls,
        latency_seconds=decision.latency_seconds,
        tokens_used=decision.tokens_used,
        is_fraud=bool(decision.is_fraud),
        fraud_type=decision.fraud_type,
        correct=bool(decision.correct),
        reward=float(decision.reward or 0.0),
    )


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------


@app.get("/", summary="Service description")
def root() -> dict[str, Any]:
    """Return a short description of the service and its endpoints."""
    return {
        "name": "StableGuard",
        "version": __version__,
        "agents": sorted(AGENT_REGISTRY),
        "claude_agent_available": anthropic_api_key() is not None,
        "endpoints": [
            "GET /transactions/{id}",
            "GET /transactions?limit=&fraud_only=",
            "POST /review",
            "GET /leaderboard",
            "GET /metrics",
            "POST /eval/run",
        ],
    }


@app.get("/transactions", summary="List transactions")
def list_transactions(limit: int = 50, offset: int = 0) -> dict[str, Any]:
    """List payments without their ground truth labels."""
    evaluator = get_evaluator()
    rows = evaluator.transactions[offset : offset + max(1, min(limit, 500))]
    return {
        "total": len(evaluator.transactions),
        "offset": offset,
        "transactions": [t.observation() for t in rows],
    }


@app.get("/transactions/{transaction_id}", summary="Fetch one transaction")
def get_transaction(transaction_id: str) -> dict[str, Any]:
    """Return one payment with the ground truth label removed."""
    evaluator = get_evaluator()
    for txn in evaluator.transactions:
        if txn.id == transaction_id:
            return txn.observation()
    raise HTTPException(status_code=404, detail=f"transaction {transaction_id!r} not found")


@app.post("/review", response_model=ReviewResponse, summary="Review one transaction")
def review(request: ReviewRequest) -> ReviewResponse:
    """Run one agent over one payment and return the full investigation."""
    evaluator = get_evaluator()
    agent = _build_agent_or_400(request.agent)
    try:
        decision = evaluator.review_one(agent, request.transaction_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _to_review_response(decision, agent.name)


@app.get("/leaderboard", summary="Compare agents")
def get_leaderboard(all_runs: bool = False) -> dict[str, Any]:
    """Return saved eval results ranked by F1, best first."""
    board = leaderboard(latest_per_agent=not all_runs)
    return {
        "n_agents": len(board),
        "leaderboard": [result.model_dump(mode="json") for result in board],
    }


@app.get("/metrics", summary="Aggregate metrics")
def get_metrics() -> dict[str, Any]:
    """Return aggregate metrics across every saved eval run."""
    results = load_all_results()
    if not results:
        return {
            "n_runs": 0,
            "message": "No eval runs yet. Run POST /eval/run or scripts/run_eval.py.",
        }

    best = max(results, key=lambda r: r.f1)
    return {
        "n_runs": len(results),
        "n_agents": len({r.agent_name for r in results}),
        "transactions_reviewed": sum(r.n_transactions for r in results),
        "best_agent": best.agent_name,
        "best_f1": best.f1,
        "total_estimated_cost_usd": round(sum(r.estimated_cost_usd for r in results), 4),
        "latest_per_agent": {
            result.agent_name: result.model_dump(mode="json")
            for result in leaderboard(results, latest_per_agent=True)
        },
    }


@app.post("/eval/run", response_model=EvalResponse, summary="Run a full eval")
def run_eval_endpoint(request: EvalRequest) -> EvalResponse:
    """Run an agent over the first ``n`` payments and return the report.

    This call is synchronous. A Claude eval over a few hundred payments takes
    minutes and costs money, so keep ``n`` small when testing.
    """
    evaluator = get_evaluator()
    agent = _build_agent_or_400(request.agent)
    result, _ = evaluator.run(agent, n=request.n, save=request.save)
    return EvalResponse(result=result, summary=format_summary(result))


@app.get("/config", summary="Effective configuration")
def get_config() -> dict[str, Any]:
    """Return the non secret configuration the service is running with."""
    config = load_config()
    return {
        "env": config["env"],
        "thresholds": config["thresholds"],
        "dataset": config["dataset"],
        "llm_model": config["llm"]["model"],
    }
