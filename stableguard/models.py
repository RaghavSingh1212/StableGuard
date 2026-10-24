"""Pydantic v2 data models shared by every part of StableGuard."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Literal types keep the vocabulary of the dataset in one place.
Stablecoin = Literal["USDC", "USDT"]
Chain = Literal["Solana", "Ethereum", "Base"]
Verdict = Literal["approve", "flag"]
FraudType = Literal[
    "none",
    "velocity",
    "geo",
    "sanctions",
    "mule",
    "amount_spike",
    "merchant_mismatch",
]

#: Every fraud type except the "no fraud" sentinel. Useful for per-type recall.
FRAUD_TYPES: tuple[str, ...] = (
    "velocity",
    "geo",
    "sanctions",
    "mule",
    "amount_spike",
    "merchant_mismatch",
)

#: Fields an agent must never see. Stripped by Transaction.observation().
HIDDEN_FIELDS: tuple[str, ...] = ("is_fraud", "fraud_type")


class Transaction(BaseModel):
    """One synthetic stablecoin payment under review.

    ``is_fraud`` and ``fraud_type`` are ground truth. They are used by the
    environment to score a verdict and are never exposed to an agent.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    timestamp: datetime
    sender_wallet: str
    receiver_wallet: str
    amount_usd: float
    stablecoin: Stablecoin
    chain: Chain
    sender_country: str
    receiver_country: str
    merchant_id: str | None = None
    merchant_category: str
    sender_wallet_age_days: int
    sender_txn_count_30d: int
    sender_total_volume_30d_usd: float
    velocity_1h: int
    is_fraud: bool = False
    fraud_type: FraudType = "none"

    def observation(self) -> dict[str, Any]:
        """Return the transaction as a dict with the ground truth removed.

        This is what the agent and the API see. Timestamps are serialised to
        ISO 8601 strings so the result is JSON serialisable as is.
        """
        payload = self.model_dump(mode="json")
        for field in HIDDEN_FIELDS:
            payload.pop(field, None)
        return payload


class ToolCall(BaseModel):
    """A single investigation tool call made by an agent, with its result."""

    model_config = ConfigDict(extra="forbid")

    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)


class Decision(BaseModel):
    """The outcome of one review episode: what the agent did and concluded."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: str
    verdict: Verdict
    reasoning: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    latency_seconds: float = 0.0
    tokens_used: int = 0

    # Filled in by the evaluator, which is the only component that may look at
    # the ground truth label.
    is_fraud: bool | None = None
    fraud_type: str | None = None
    correct: bool | None = None
    reward: float | None = None

    @property
    def n_tool_calls(self) -> int:
        """Number of investigation tools the agent used before deciding."""
        return len(self.tool_calls)


class EvalResult(BaseModel):
    """Aggregate metrics for one agent over one dataset slice."""

    model_config = ConfigDict(extra="forbid")

    agent_name: str
    n_transactions: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    false_negative_rate: float
    avg_tool_calls: float
    avg_latency_seconds: float
    recall_by_fraud_type: dict[str, float] = Field(default_factory=dict)
    timestamp: datetime

    # Extras that the architecture asks metrics.py to compute and report.
    n_fraud: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    avg_reward: float = 0.0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    dataset: str = ""
    model: str | None = None
