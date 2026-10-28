"""Tests for the synthetic transaction generator.

The point of these tests is that every fraud label is justified: a transaction
marked as a given fraud type must actually exhibit that pattern, and a clean
transaction must exhibit none of them.
"""

from __future__ import annotations

from typing import Any

import pytest

from stableguard.config import load_config, load_sanctions
from stableguard.generator import TransactionGenerator, generate_dataset
from stableguard.models import FRAUD_TYPES, Transaction

CONFIG = load_config()
THRESHOLDS = CONFIG["thresholds"]


def _wallets_by_address(dataset: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {w["address"]: w for w in dataset["wallets"]}


def test_dataset_shape(dataset: dict[str, Any]) -> None:
    """The dataset carries transactions plus the world they refer to."""
    assert set(dataset) == {"metadata", "wallets", "merchants", "transactions"}
    assert dataset["metadata"]["n_transactions"] == len(dataset["transactions"])
    assert dataset["wallets"] and dataset["merchants"]


def test_fraud_rate_is_respected(transactions: list[Transaction]) -> None:
    """The share of fraud matches the requested rate."""
    n_fraud = sum(1 for t in transactions if t.is_fraud)
    assert n_fraud == pytest.approx(len(transactions) * 0.2, abs=1)


def test_every_fraud_type_appears(transactions: list[Transaction]) -> None:
    """All six fraud patterns are represented in a dataset of this size."""
    present = {t.fraud_type for t in transactions if t.is_fraud}
    assert present == set(FRAUD_TYPES)


def test_labels_and_types_agree(transactions: list[Transaction]) -> None:
    """``is_fraud`` and ``fraud_type`` never disagree."""
    for txn in transactions:
        assert txn.is_fraud == (txn.fraud_type != "none")


def test_ids_are_unique(transactions: list[Transaction]) -> None:
    """Transaction ids are unique so results can be keyed by them."""
    ids = [t.id for t in transactions]
    assert len(ids) == len(set(ids))


def test_determinism() -> None:
    """The same seed produces exactly the same transactions."""
    first = generate_dataset(n=50, fraud_rate=0.2, seed=99, output_path=None)
    second = generate_dataset(n=50, fraud_rate=0.2, seed=99, output_path=None)
    assert first["transactions"] == second["transactions"]


def test_different_seeds_differ() -> None:
    """A different seed produces a different dataset."""
    first = generate_dataset(n=50, fraud_rate=0.2, seed=1, output_path=None)
    second = generate_dataset(n=50, fraud_rate=0.2, seed=2, output_path=None)
    assert first["transactions"] != second["transactions"]


def test_legit_transactions_are_clean(
    dataset: dict[str, Any], transactions: list[Transaction]
) -> None:
    """A clean transaction exhibits none of the six fraud signals."""
    wallets = _wallets_by_address(dataset)
    sanctions = load_sanctions()

    for txn in transactions:
        if txn.is_fraud:
            continue
        wallet = wallets[txn.sender_wallet]
        assert txn.sender_country == wallet["home_country"], txn.id
        assert txn.velocity_1h <= THRESHOLDS["velocity_flag"], txn.id
        assert txn.sender_wallet_age_days >= 30, txn.id
        assert txn.sender_wallet not in sanctions, txn.id
        assert txn.receiver_wallet not in sanctions, txn.id
        assert 10.0 <= txn.amount_usd <= 2000.0, txn.id
        ratio = txn.amount_usd / wallet["typical_amount_usd"]
        assert ratio <= THRESHOLDS["amount_ratio_flag"], txn.id
        assert txn.merchant_category in wallet["categories"], txn.id


def test_velocity_fraud_pattern(transactions: list[Transaction]) -> None:
    """Velocity fraud shows 15 to 40 transactions in the last hour."""
    rows = [t for t in transactions if t.fraud_type == "velocity"]
    assert rows
    for txn in rows:
        assert 15 <= txn.velocity_1h <= 40, txn.id


def test_geo_fraud_pattern(
    dataset: dict[str, Any], transactions: list[Transaction]
) -> None:
    """Geo fraud is sent from a country other than the wallet's home country."""
    wallets = _wallets_by_address(dataset)
    rows = [t for t in transactions if t.fraud_type == "geo"]
    assert rows
    for txn in rows:
        assert txn.sender_country != wallets[txn.sender_wallet]["home_country"], txn.id


def test_sanctions_fraud_pattern(transactions: list[Transaction]) -> None:
    """Sanctions fraud has a sanctioned wallet on one side of the payment."""
    sanctions = load_sanctions()
    rows = [t for t in transactions if t.fraud_type == "sanctions"]
    assert rows
    for txn in rows:
        assert txn.sender_wallet in sanctions or txn.receiver_wallet in sanctions, txn.id


def test_mule_fraud_pattern(transactions: list[Transaction]) -> None:
    """Mule fraud is a new wallet moving a large amount quickly."""
    rows = [t for t in transactions if t.fraud_type == "mule"]
    assert rows
    for txn in rows:
        assert txn.sender_wallet_age_days < THRESHOLDS["new_wallet_age_days"], txn.id
        assert txn.amount_usd >= THRESHOLDS["large_amount_usd"], txn.id
        assert txn.velocity_1h >= 5, txn.id


def test_amount_spike_fraud_pattern(
    dataset: dict[str, Any], transactions: list[Transaction]
) -> None:
    """Amount spike fraud is 10x to 50x the wallet's typical amount."""
    wallets = _wallets_by_address(dataset)
    rows = [t for t in transactions if t.fraud_type == "amount_spike"]
    assert rows
    for txn in rows:
        ratio = txn.amount_usd / wallets[txn.sender_wallet]["typical_amount_usd"]
        assert 9.9 <= ratio <= 50.1, txn.id


def test_merchant_mismatch_fraud_pattern(
    dataset: dict[str, Any], transactions: list[Transaction]
) -> None:
    """Merchant mismatch uses a high risk category the wallet has never used."""
    wallets = _wallets_by_address(dataset)
    high_risk = set(CONFIG["dataset"]["high_risk_categories"])
    rows = [t for t in transactions if t.fraud_type == "merchant_mismatch"]
    assert rows
    for txn in rows:
        assert txn.merchant_category in high_risk, txn.id
        assert txn.merchant_category not in wallets[txn.sender_wallet]["categories"], txn.id


def test_observation_hides_ground_truth(transactions: list[Transaction]) -> None:
    """An observation never leaks the label or the fraud type."""
    observation = transactions[0].observation()
    assert "is_fraud" not in observation
    assert "fraud_type" not in observation


def test_some_legit_wallets_use_high_risk_categories(dataset: dict[str, Any]) -> None:
    """Category risk alone must not separate fraud from clean traffic."""
    high_risk = set(CONFIG["dataset"]["high_risk_categories"])
    legit_high_risk = [
        t
        for t in dataset["transactions"]
        if not t["is_fraud"] and t["merchant_category"] in high_risk
    ]
    assert legit_high_risk, "no clean transactions use a high risk category"


def test_rejects_invalid_fraud_rate() -> None:
    """A fraud rate outside 0 to 1 is rejected."""
    with pytest.raises(ValueError):
        TransactionGenerator(seed=1, fraud_rate=1.5)
