"""The investigation tools an agent calls while reviewing a payment.

Each tool is a read only lookup against the world the generator built: wallet
profiles, the merchant directory and the sanctions list. Every tool returns a
plain JSON serialisable dict, and a bad argument produces an ``error`` key
rather than an exception, so an LLM agent that invents an address cannot crash
an eval run.

Some answers depend on the payment under review (velocity in the last hour, for
example, is a property of the transaction, not of the wallet in general). The
registry therefore holds an optional "current transaction" context that the
environment sets on every reset.
"""

from __future__ import annotations

from typing import Any, Callable

from .config import load_config, load_sanctions
from .generator import Merchant, WalletProfile
from .models import Transaction

#: Names of every tool, in the order they are advertised to an agent.
TOOL_NAMES: tuple[str, ...] = (
    "check_wallet_history",
    "check_sanctions",
    "check_velocity",
    "check_amount_profile",
    "check_merchant",
    "check_geo",
)

#: Tool schemas in Anthropic tool-use format. The LLM agent pastes these into
#: its system prompt, and they double as the human readable documentation.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "check_wallet_history",
        "description": (
            "Look up what is known about a wallet: how old it is, how much it has "
            "transacted in the last 30 days, its average payment size, its home "
            "country and the merchant categories it has a history with."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "wallet_address": {"type": "string", "description": "The wallet to look up."}
            },
            "required": ["wallet_address"],
        },
    },
    {
        "name": "check_sanctions",
        "description": "Check whether a wallet appears on a sanctions list.",
        "input_schema": {
            "type": "object",
            "properties": {
                "wallet_address": {"type": "string", "description": "The wallet to screen."}
            },
            "required": ["wallet_address"],
        },
    },
    {
        "name": "check_velocity",
        "description": (
            "Count how many payments this wallet sent in a recent time window and "
            "how much they were worth in total."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "wallet_address": {"type": "string"},
                "window_hours": {"type": "integer", "description": "Window size, default 1."},
            },
            "required": ["wallet_address"],
        },
    },
    {
        "name": "check_amount_profile",
        "description": (
            "Compare a payment amount with what this wallet normally sends. Returns "
            "the typical amount, the ratio, and whether the ratio is unusual."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "wallet_address": {"type": "string"},
                "amount_usd": {"type": "number"},
            },
            "required": ["wallet_address", "amount_usd"],
        },
    },
    {
        "name": "check_merchant",
        "description": "Look up a merchant's name, category and risk score from 0 to 1.",
        "input_schema": {
            "type": "object",
            "properties": {"merchant_id": {"type": "string"}},
            "required": ["merchant_id"],
        },
    },
    {
        "name": "check_geo",
        "description": (
            "Compare the country the payment was sent from with the wallet's home "
            "country. Pass wallet_home_country if you already know it, or pass "
            "wallet_address and the tool will resolve it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sender_country": {"type": "string"},
                "wallet_home_country": {"type": "string"},
                "wallet_address": {"type": "string"},
            },
            "required": ["sender_country"],
        },
    },
]


class InvestigationTools:
    """A registry of the six investigation tools, bound to one dataset."""

    def __init__(self, dataset: dict[str, Any], config: dict[str, Any] | None = None) -> None:
        self.config = config or load_config()
        self.thresholds = self.config["thresholds"]

        self.wallets: dict[str, WalletProfile] = {
            row["address"]: WalletProfile(**row) for row in dataset["wallets"]
        }
        self.merchants: dict[str, Merchant] = {
            row["merchant_id"]: Merchant(**row) for row in dataset["merchants"]
        }
        self.sanctions: dict[str, str] = load_sanctions()

        # The payment currently under review, set by the environment.
        self._current: Transaction | None = None

        self._dispatch: dict[str, Callable[..., dict[str, Any]]] = {
            "check_wallet_history": self.check_wallet_history,
            "check_sanctions": self.check_sanctions,
            "check_velocity": self.check_velocity,
            "check_amount_profile": self.check_amount_profile,
            "check_merchant": self.check_merchant,
            "check_geo": self.check_geo,
        }

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------

    def set_context(self, transaction: Transaction | None) -> None:
        """Tell the tools which payment is under review."""
        self._current = transaction

    def _is_current_sender(self, wallet_address: str) -> bool:
        """True if the address is the sender of the payment under review."""
        return self._current is not None and wallet_address == self._current.sender_wallet

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def check_wallet_history(self, wallet_address: str) -> dict[str, Any]:
        """Return the 30 day history and standing profile of a wallet."""
        profile = self.wallets.get(wallet_address)
        if profile is None:
            return {
                "error": "unknown_wallet",
                "wallet_address": wallet_address,
                "note": "No history on file. An unknown counterparty is itself a weak signal.",
            }
        return {
            "wallet_address": wallet_address,
            "wallet_age_days": profile.wallet_age_days,
            "txn_count_30d": profile.txn_count_30d,
            "total_volume_30d_usd": round(profile.total_volume_30d_usd, 2),
            "avg_amount_usd": round(profile.typical_amount_usd, 2),
            "home_country": profile.home_country,
            "known_merchant_categories": list(profile.categories),
        }

    def check_sanctions(self, wallet_address: str) -> dict[str, Any]:
        """Screen a wallet against the sanctions list."""
        list_name = self.sanctions.get(wallet_address)
        return {
            "wallet_address": wallet_address,
            "sanctioned": list_name is not None,
            "list_name": list_name,
        }

    def check_velocity(self, wallet_address: str, window_hours: int = 1) -> dict[str, Any]:
        """Count recent payments from a wallet inside a time window.

        For the sender of the payment under review this reports the observed
        burst. For any other wallet it reports the wallet's usual rate.
        """
        try:
            window = max(1, int(window_hours))
        except (TypeError, ValueError):
            window = 1

        profile = self.wallets.get(wallet_address)
        if profile is None:
            return {"error": "unknown_wallet", "wallet_address": wallet_address}

        if self._is_current_sender(wallet_address) and self._current is not None:
            per_hour = self._current.velocity_1h
        else:
            per_hour = profile.baseline_velocity_1h

        # Later hours are assumed quieter than the hour under review, so the
        # window is not a flat multiple.
        txn_count = per_hour if window == 1 else per_hour + int(
            profile.baseline_velocity_1h * (window - 1)
        )
        total_amount = round(txn_count * profile.typical_amount_usd, 2)
        return {
            "wallet_address": wallet_address,
            "window_hours": window,
            "txn_count": txn_count,
            "total_amount_usd": total_amount,
        }

    def check_amount_profile(self, wallet_address: str, amount_usd: float) -> dict[str, Any]:
        """Compare a payment amount with the wallet's typical amount."""
        profile = self.wallets.get(wallet_address)
        if profile is None:
            return {"error": "unknown_wallet", "wallet_address": wallet_address}

        try:
            amount = float(amount_usd)
        except (TypeError, ValueError):
            return {"error": "invalid_amount", "amount_usd": amount_usd}

        typical = max(1.0, profile.typical_amount_usd)
        ratio = amount / typical
        return {
            "wallet_address": wallet_address,
            "amount_usd": round(amount, 2),
            "typical_amount_usd": round(typical, 2),
            "ratio": round(ratio, 2),
            "unusual": bool(ratio > float(self.thresholds["amount_ratio_flag"])),
        }

    def check_merchant(self, merchant_id: str | None) -> dict[str, Any]:
        """Look up a merchant's category and risk score."""
        if merchant_id is None:
            return {
                "merchant_id": None,
                "name": None,
                "category": None,
                "risk_score": None,
                "note": "This payment has no merchant, it is a wallet to wallet transfer.",
            }
        merchant = self.merchants.get(merchant_id)
        if merchant is None:
            return {"error": "unknown_merchant", "merchant_id": merchant_id}
        return {
            "merchant_id": merchant.merchant_id,
            "name": merchant.name,
            "category": merchant.category,
            "risk_score": merchant.risk_score,
            "high_risk": merchant.risk_score >= float(self.thresholds["merchant_risk_flag"]),
        }

    def check_geo(
        self,
        sender_country: str,
        wallet_home_country: str | None = None,
        wallet_address: str | None = None,
    ) -> dict[str, Any]:
        """Compare where the payment came from with where the wallet lives."""
        home = wallet_home_country
        if home is None:
            address = wallet_address
            if address is None and self._current is not None:
                address = self._current.sender_wallet
            profile = self.wallets.get(address) if address else None
            if profile is None:
                return {
                    "error": "unknown_home_country",
                    "note": "Pass wallet_home_country or a known wallet_address.",
                }
            home = profile.home_country

        return {
            "sender_country": sender_country,
            "wallet_home_country": home,
            "mismatch": sender_country != home,
        }

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def call(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run a tool by name with keyword arguments.

        Unknown tools and bad arguments return an ``error`` dict instead of
        raising, so one malformed model output cannot end an eval run.
        """
        handler = self._dispatch.get(name)
        if handler is None:
            return {"error": "unknown_tool", "tool": name, "available": list(TOOL_NAMES)}
        try:
            return handler(**(args or {}))
        except TypeError as exc:
            return {"error": "bad_arguments", "tool": name, "detail": str(exc)}
