"""Tests for the six investigation tools."""

from __future__ import annotations

from typing import Any

import pytest

from stableguard.models import Transaction
from stableguard.tools import TOOL_NAMES, TOOL_SCHEMAS, InvestigationTools


@pytest.fixture(scope="module")
def tools(dataset: dict[str, Any]) -> InvestigationTools:
    """A tool registry bound to the session dataset."""
    return InvestigationTools(dataset)


def _first(transactions: list[Transaction], fraud_type: str) -> Transaction:
    return next(t for t in transactions if t.fraud_type == fraud_type)


def test_schemas_cover_every_tool() -> None:
    """The advertised schemas and the dispatch table stay in step."""
    assert {schema["name"] for schema in TOOL_SCHEMAS} == set(TOOL_NAMES)


def test_wallet_history_returns_expected_keys(
    tools: InvestigationTools, transactions: list[Transaction]
) -> None:
    """A known wallet returns its history."""
    result = tools.check_wallet_history(transactions[0].sender_wallet)
    for key in ("wallet_age_days", "txn_count_30d", "total_volume_30d_usd", "avg_amount_usd"):
        assert key in result
    assert result["wallet_age_days"] >= 0


def test_wallet_history_unknown_wallet(tools: InvestigationTools) -> None:
    """An unknown wallet reports an error instead of raising."""
    assert tools.check_wallet_history("0xdeadbeef")["error"] == "unknown_wallet"


def test_sanctions_detects_listed_wallet(
    tools: InvestigationTools, transactions: list[Transaction]
) -> None:
    """A sanctions fraud transaction has a listed wallet on one side."""
    txn = _first(transactions, "sanctions")
    sender = tools.check_sanctions(txn.sender_wallet)
    receiver = tools.check_sanctions(txn.receiver_wallet)
    assert sender["sanctioned"] or receiver["sanctioned"]
    hit = sender if sender["sanctioned"] else receiver
    assert isinstance(hit["list_name"], str)


def test_sanctions_clean_wallet(
    tools: InvestigationTools, transactions: list[Transaction]
) -> None:
    """A clean transaction screens clean on both sides."""
    txn = next(t for t in transactions if not t.is_fraud)
    assert tools.check_sanctions(txn.sender_wallet) == {
        "wallet_address": txn.sender_wallet,
        "sanctioned": False,
        "list_name": None,
    }


def test_velocity_reflects_transaction_under_review(
    tools: InvestigationTools, transactions: list[Transaction]
) -> None:
    """With a context set, velocity reports the burst in the payment itself."""
    txn = _first(transactions, "velocity")
    tools.set_context(txn)
    try:
        result = tools.check_velocity(txn.sender_wallet)
        assert result["txn_count"] == txn.velocity_1h
        assert result["window_hours"] == 1
        assert result["total_amount_usd"] > 0
    finally:
        tools.set_context(None)


def test_velocity_window_scales(
    tools: InvestigationTools, transactions: list[Transaction]
) -> None:
    """A wider window never reports fewer transactions."""
    txn = _first(transactions, "velocity")
    tools.set_context(txn)
    try:
        one_hour = tools.check_velocity(txn.sender_wallet, window_hours=1)
        six_hours = tools.check_velocity(txn.sender_wallet, window_hours=6)
        assert six_hours["txn_count"] >= one_hour["txn_count"]
    finally:
        tools.set_context(None)


def test_amount_profile_flags_spike(
    tools: InvestigationTools, transactions: list[Transaction]
) -> None:
    """An amount spike reports a high ratio and unusual True."""
    txn = _first(transactions, "amount_spike")
    result = tools.check_amount_profile(txn.sender_wallet, txn.amount_usd)
    assert result["ratio"] >= 9.9
    assert result["unusual"] is True


def test_amount_profile_normal_payment(
    tools: InvestigationTools, transactions: list[Transaction]
) -> None:
    """A clean payment is not flagged as unusual."""
    txn = next(t for t in transactions if not t.is_fraud)
    result = tools.check_amount_profile(txn.sender_wallet, txn.amount_usd)
    assert result["unusual"] is False


def test_amount_profile_rejects_bad_amount(
    tools: InvestigationTools, transactions: list[Transaction]
) -> None:
    """A non numeric amount returns an error."""
    result = tools.check_amount_profile(transactions[0].sender_wallet, "lots")
    assert result["error"] == "invalid_amount"


def test_merchant_lookup(tools: InvestigationTools, transactions: list[Transaction]) -> None:
    """A merchant id resolves to a name, a category and a risk score."""
    txn = next(t for t in transactions if t.merchant_id)
    result = tools.check_merchant(txn.merchant_id)
    assert result["category"] == txn.merchant_category
    assert 0.0 <= result["risk_score"] <= 1.0


def test_merchant_none_and_unknown(tools: InvestigationTools) -> None:
    """A missing merchant id is handled, an unknown one reports an error."""
    assert tools.check_merchant(None)["risk_score"] is None
    assert tools.check_merchant("mch_999999")["error"] == "unknown_merchant"


def test_geo_detects_mismatch(
    tools: InvestigationTools, transactions: list[Transaction]
) -> None:
    """Geo fraud reports a mismatch, with or without an explicit home country."""
    txn = _first(transactions, "geo")
    home = tools.check_wallet_history(txn.sender_wallet)["home_country"]
    assert tools.check_geo(txn.sender_country, home)["mismatch"] is True
    assert tools.check_geo(txn.sender_country, wallet_address=txn.sender_wallet)["mismatch"] is True


def test_geo_clean_transaction(
    tools: InvestigationTools, transactions: list[Transaction]
) -> None:
    """A clean payment comes from the wallet's home country."""
    txn = next(t for t in transactions if not t.is_fraud)
    result = tools.check_geo(txn.sender_country, wallet_address=txn.sender_wallet)
    assert result["mismatch"] is False


def test_merchant_mismatch_is_visible(
    tools: InvestigationTools, transactions: list[Transaction]
) -> None:
    """Merchant mismatch is detectable by combining two tools."""
    txn = _first(transactions, "merchant_mismatch")
    merchant = tools.check_merchant(txn.merchant_id)
    history = tools.check_wallet_history(txn.sender_wallet)
    assert merchant["high_risk"] is True
    assert merchant["category"] not in history["known_merchant_categories"]


def test_mule_is_visible(tools: InvestigationTools, transactions: list[Transaction]) -> None:
    """Mule fraud is visible as a very new wallet moving a large amount."""
    txn = _first(transactions, "mule")
    history = tools.check_wallet_history(txn.sender_wallet)
    assert history["wallet_age_days"] < 7
    assert txn.amount_usd >= 5000.0


def test_call_dispatch(tools: InvestigationTools, transactions: list[Transaction]) -> None:
    """The dispatcher runs tools by name."""
    result = tools.call("check_sanctions", {"wallet_address": transactions[0].sender_wallet})
    assert result["sanctioned"] is False


def test_call_unknown_tool(tools: InvestigationTools) -> None:
    """An unknown tool name is reported, not raised."""
    assert tools.call("check_vibes", {})["error"] == "unknown_tool"


def test_call_bad_arguments(tools: InvestigationTools) -> None:
    """Wrong keyword arguments are reported, not raised."""
    assert tools.call("check_sanctions", {"wallet": "0x1"})["error"] == "bad_arguments"


def test_results_are_json_serialisable(
    tools: InvestigationTools, transactions: list[Transaction]
) -> None:
    """Every tool result can be serialised for the API and the dashboard."""
    import json

    txn = transactions[0]
    tools.set_context(txn)
    try:
        payloads = [
            tools.call("check_wallet_history", {"wallet_address": txn.sender_wallet}),
            tools.call("check_sanctions", {"wallet_address": txn.receiver_wallet}),
            tools.call("check_velocity", {"wallet_address": txn.sender_wallet}),
            tools.call(
                "check_amount_profile",
                {"wallet_address": txn.sender_wallet, "amount_usd": txn.amount_usd},
            ),
            tools.call("check_merchant", {"merchant_id": txn.merchant_id}),
            tools.call("check_geo", {"sender_country": txn.sender_country}),
        ]
        for payload in payloads:
            json.dumps(payload)
    finally:
        tools.set_context(None)
