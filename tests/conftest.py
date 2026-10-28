"""Shared pytest fixtures.

A single small dataset is generated once per test session and reused, which
keeps the suite fast while still exercising the real generator.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stableguard.generator import TransactionGenerator, generate_dataset  # noqa: E402
from stableguard.models import Transaction  # noqa: E402

TEST_SEED = 7
TEST_N = 300


@pytest.fixture(scope="session")
def dataset() -> dict[str, Any]:
    """A reproducible in-memory dataset, never written to disk."""
    return generate_dataset(n=TEST_N, fraud_rate=0.2, seed=TEST_SEED, output_path=None)


@pytest.fixture(scope="session")
def transactions(dataset: dict[str, Any]) -> list[Transaction]:
    """The dataset's transactions as Pydantic models."""
    return [Transaction(**row) for row in dataset["transactions"]]


@pytest.fixture(scope="session")
def generator() -> TransactionGenerator:
    """A generator instance for tests that need the wallet or merchant pools."""
    return TransactionGenerator(seed=TEST_SEED, fraud_rate=0.2)
