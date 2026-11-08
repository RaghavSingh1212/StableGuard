"""Tests for the FastAPI app, driven through Starlette's test client."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from stableguard.evaluator import Evaluator


@pytest.fixture()
def client(dataset: dict[str, Any], tmp_path: Path) -> Iterator[TestClient]:
    """A test client wired to the in-memory dataset, not to data/ on disk."""
    api_main._evaluator = Evaluator(dataset=dataset, dataset_path="in-memory")
    with TestClient(api_main.app) as test_client:
        yield test_client
    api_main._evaluator = None


def test_root(client: TestClient) -> None:
    """The root endpoint describes the service."""
    body = client.get("/").json()
    assert body["name"] == "StableGuard"
    assert "rule_based" in body["agents"]


def test_list_transactions(client: TestClient) -> None:
    """Listing returns observations without ground truth."""
    body = client.get("/transactions?limit=5").json()
    assert len(body["transactions"]) == 5
    assert all("is_fraud" not in t for t in body["transactions"])


def test_get_transaction(client: TestClient) -> None:
    """A single transaction can be fetched and hides its label."""
    first = client.get("/transactions?limit=1").json()["transactions"][0]
    body = client.get(f"/transactions/{first['id']}").json()
    assert body["id"] == first["id"]
    assert "is_fraud" not in body and "fraud_type" not in body


def test_get_unknown_transaction(client: TestClient) -> None:
    """An unknown id is a 404."""
    assert client.get("/transactions/txn_nope").status_code == 404


def test_review(client: TestClient) -> None:
    """A review returns the verdict, the reasoning and the tool calls."""
    first = client.get("/transactions?limit=1").json()["transactions"][0]
    response = client.post("/review", json={"transaction_id": first["id"], "agent": "rule_based"})
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] in ("approve", "flag")
    assert isinstance(body["tool_calls"], list)
    assert body["n_tool_calls"] == len(body["tool_calls"])
    assert isinstance(body["correct"], bool)


def test_review_unknown_agent(client: TestClient) -> None:
    """An unknown agent name is a 400."""
    first = client.get("/transactions?limit=1").json()["transactions"][0]
    response = client.post("/review", json={"transaction_id": first["id"], "agent": "psychic"})
    assert response.status_code == 400


def test_review_unknown_transaction(client: TestClient) -> None:
    """Reviewing a payment that is not in the dataset is a 404."""
    response = client.post("/review", json={"transaction_id": "txn_nope", "agent": "rule_based"})
    assert response.status_code == 404


def test_eval_run_and_leaderboard(client: TestClient, tmp_path: Path) -> None:
    """An eval can be triggered over HTTP and its result is well formed."""
    response = client.post("/eval/run", json={"agent": "rule_based", "n": 40, "save": False})
    assert response.status_code == 200
    body = response.json()
    assert body["result"]["n_transactions"] == 40
    assert 0.0 <= body["result"]["f1"] <= 1.0
    assert "Recall by fraud type" in body["summary"]


def test_leaderboard_endpoint(client: TestClient) -> None:
    """The leaderboard endpoint always answers, even with no runs saved."""
    body = client.get("/leaderboard").json()
    assert "leaderboard" in body
    assert body["n_agents"] == len(body["leaderboard"])


def test_metrics_endpoint(client: TestClient) -> None:
    """The metrics endpoint answers with either runs or a helpful message."""
    body = client.get("/metrics").json()
    assert "n_runs" in body


def test_config_endpoint(client: TestClient) -> None:
    """The config endpoint exposes thresholds but no secrets."""
    body = client.get("/config").json()
    assert body["env"]["tool_budget"] >= 1
    assert "api_key" not in str(body).lower()
