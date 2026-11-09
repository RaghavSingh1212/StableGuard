"""The StableGuard dashboard.

Two modes, both meant to be understood in about two minutes:

* Review a transaction: pick a payment, watch an agent investigate it one tool
  call at a time, then see the verdict against the ground truth.
* Leaderboard: compare every agent that has been evaluated, with recall broken
  down by fraud type, and start a new eval from the page.

Run it with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Allow "streamlit run dashboard/app.py" from a checkout that is not installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stableguard.agent import AGENT_REGISTRY, Agent, build_agent
from stableguard.config import DEFAULT_DATASET_PATH, anthropic_api_key, load_config
from stableguard.env import StableGuardEnv
from stableguard.evaluator import Evaluator
from stableguard.generator import load_dataset
from stableguard.metrics import leaderboard
from stableguard.models import FRAUD_TYPES, Decision, EvalResult, Transaction

st.set_page_config(page_title="StableGuard", page_icon="shield", layout="wide")

# A single accent colour is used for every chart so the page reads as one piece.
ACCENT = "#2f6feb"


# ----------------------------------------------------------------------
# Cached loading
# ----------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def get_dataset(path: str) -> dict[str, Any]:
    """Load the dataset once per session."""
    return load_dataset(path)


@st.cache_resource(show_spinner=False)
def get_evaluator(path: str) -> Evaluator:
    """Build the evaluator once per session."""
    return Evaluator(dataset_path=path)


def agent_or_warning(name: str) -> Agent | None:
    """Build an agent, showing a clear message if it cannot be configured."""
    try:
        return build_agent(name)
    except RuntimeError as exc:
        st.error(str(exc))
        return None


# ----------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------


def render_transaction_card(txn: Transaction) -> None:
    """Show the payment under review as a set of metrics."""
    top = st.columns(4)
    top[0].metric("Amount", f"${txn.amount_usd:,.2f}")
    top[1].metric("Asset", f"{txn.stablecoin} on {txn.chain}")
    top[2].metric("Route", f"{txn.sender_country} to {txn.receiver_country}")
    top[3].metric("Category", txn.merchant_category)

    bottom = st.columns(4)
    bottom[0].metric("Wallet age", f"{txn.sender_wallet_age_days} days")
    bottom[1].metric("Payments in last hour", txn.velocity_1h)
    bottom[2].metric("Payments in 30 days", txn.sender_txn_count_30d)
    bottom[3].metric("Volume in 30 days", f"${txn.sender_total_volume_30d_usd:,.0f}")

    with st.expander("Raw payment as the agent sees it"):
        st.json(txn.observation())


def render_verdict(decision_verdict: str, is_fraud: bool, correct: bool) -> None:
    """Show the verdict next to the ground truth."""
    columns = st.columns(3)
    columns[0].metric("Agent verdict", decision_verdict.upper())
    columns[1].metric("Ground truth", "FRAUD" if is_fraud else "LEGITIMATE")
    columns[2].metric("Outcome", "Correct" if correct else "Wrong")
    if correct:
        st.success("The agent got this one right.")
    else:
        missed = "missed fraud" if is_fraud else "flagged clean traffic"
        st.error(f"The agent got this one wrong: it {missed}.")


def recall_frame(results: list[EvalResult]) -> pd.DataFrame:
    """Build a fraud type by agent table of recall values."""
    rows: dict[str, dict[str, float]] = {}
    for result in results:
        for fraud_type in FRAUD_TYPES:
            rows.setdefault(fraud_type, {})[result.agent_name] = result.recall_by_fraud_type.get(
                fraud_type, 0.0
            )
    return pd.DataFrame(rows).T


def leaderboard_frame(results: list[EvalResult]) -> pd.DataFrame:
    """Build the leaderboard table in the column order the brief asks for."""
    return pd.DataFrame(
        [
            {
                "agent": r.agent_name,
                "n": r.n_transactions,
                "accuracy": r.accuracy,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
                "false positive rate": r.false_positive_rate,
                "avg tool calls": r.avg_tool_calls,
                "avg latency (s)": r.avg_latency_seconds,
                "est. cost (USD)": r.estimated_cost_usd,
            }
            for r in results
        ]
    )


# ----------------------------------------------------------------------
# Mode: review one transaction
# ----------------------------------------------------------------------


def review_mode(evaluator: Evaluator, dataset: dict[str, Any], agent_name: str) -> None:
    """Let the visitor pick a payment and watch an agent investigate it."""
    st.subheader("Review a transaction")
    st.caption(
        "Pick a payment, then let the agent investigate it. Every tool call is shown as it "
        "happens, and the ground truth label is revealed only after the verdict."
    )

    transactions = evaluator.transactions
    choice = st.radio(
        "Which payment",
        ["Random", "Random fraud", "Random legitimate", "Choose by id"],
        horizontal=True,
    )

    if choice == "Choose by id":
        txn_id = st.selectbox("Transaction id", [t.id for t in transactions[:500]])
    else:
        if st.button("Pick another payment") or "picked_id" not in st.session_state:
            pool = transactions
            if choice == "Random fraud":
                pool = [t for t in transactions if t.is_fraud]
            elif choice == "Random legitimate":
                pool = [t for t in transactions if not t.is_fraud]
            st.session_state.picked_id = (
                pd.Series([t.id for t in pool]).sample(1).iloc[0] if pool else transactions[0].id
            )
        txn_id = st.session_state.picked_id

    txn = next(t for t in transactions if t.id == txn_id)
    st.markdown(f"**{txn.id}** at {txn.timestamp:%Y-%m-%d %H:%M UTC}")
    render_transaction_card(txn)

    if not st.button("Investigate", type="primary"):
        return

    agent = agent_or_warning(agent_name)
    if agent is None:
        return

    env = StableGuardEnv(dataset, transactions=transactions, config=load_config())
    observation, _ = env.reset(options={"transaction_id": txn.id})
    agent.begin_episode(observation)

    st.markdown("#### Investigation")
    started = time.perf_counter()
    terminated = truncated = False
    info: dict[str, Any] = {}

    while not (terminated or truncated):
        with st.spinner(f"{agent.name} is thinking"):
            action = agent.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)

        if action["type"] == "tool":
            result = observation["tool_results"][-1]["result"]
            with st.expander(f"Tool call: {action['name']}", expanded=True):
                if agent.reasoning:
                    st.write(agent.reasoning)
                st.caption(f"arguments: {action.get('args', {})}")
                st.json(result)
        else:
            st.markdown(f"**Verdict:** {action['verdict']}")
            if agent.reasoning:
                st.write(agent.reasoning)

    latency = time.perf_counter() - started
    verdict = info.get("verdict") or "approve"
    if info.get("reason") == "tool_budget_exhausted":
        st.warning("The agent used its whole tool budget without deciding, so the payment passed.")

    st.markdown("#### Outcome")
    render_verdict(verdict, bool(info.get("is_fraud")), bool(info.get("correct")))

    stats = st.columns(4)
    stats[0].metric("Tool calls", info.get("n_tool_calls", 0))
    stats[1].metric("Latency", f"{latency:.2f}s")
    stats[2].metric("Tokens", f"{getattr(agent, 'tokens_used', 0):,}")
    stats[3].metric("Reward", f"{info.get('reward', 0.0):+.2f}")
    if info.get("is_fraud"):
        st.caption(f"Fraud type in the ground truth: {info.get('fraud_type')}")


# ----------------------------------------------------------------------
# Mode: leaderboard
# ----------------------------------------------------------------------


def leaderboard_mode(evaluator: Evaluator, agent_name: str) -> None:
    """Show saved eval results and offer to run a new eval."""
    st.subheader("Leaderboard")
    st.caption(
        "Every agent that has been evaluated, ranked by F1. Recall by fraud type is the view "
        "that matters most: an agent can look strong overall while missing one pattern entirely."
    )

    with st.form("run_eval"):
        columns = st.columns([2, 1, 1])
        n = columns[0].slider("Payments to review", min_value=20, max_value=1000, value=200, step=20)
        columns[1].markdown(f"**Agent**\n\n{agent_name}")
        submitted = columns[2].form_submit_button("Run eval", type="primary")

    if submitted:
        agent = agent_or_warning(agent_name)
        if agent is not None:
            if agent.name == "claude":
                st.info(
                    "This calls the Claude API once per step, so a few hundred payments takes "
                    "minutes and costs money."
                )
            progress = st.progress(0.0, text="Starting")
            running = {"correct": 0}

            def on_progress(index: int, total: int, decision: Decision) -> None:
                """Update the progress bar after each payment."""
                running["correct"] += int(bool(decision.correct))
                progress.progress(
                    index / total,
                    text=f"{index}/{total} reviewed, {running['correct']} correct so far",
                )

            result, _ = evaluator.run(agent, n=n, progress=on_progress)
            progress.empty()
            st.success(
                f"{result.agent_name}: F1 {result.f1:.3f}, recall {result.recall:.3f}, "
                f"false positive rate {result.false_positive_rate:.3f}"
            )

    results = leaderboard()
    if not results:
        st.info("No eval results yet. Run one above, or use scripts/run_eval.py.")
        return

    st.dataframe(leaderboard_frame(results), use_container_width=True, hide_index=True)

    st.markdown("#### Recall by fraud type")
    st.caption("Share of each fraud pattern that the agent caught. Higher is better.")
    st.bar_chart(recall_frame(results), height=320, color=None if len(results) > 1 else ACCENT)

    best = results[0]
    st.markdown("#### Where the leader stands")
    weakest = min(best.recall_by_fraud_type.items(), key=lambda kv: kv[1], default=None)
    columns = st.columns(3)
    columns[0].metric("Best agent", best.agent_name, f"F1 {best.f1:.3f}")
    columns[1].metric("Fraud caught", f"{best.true_positives}/{best.n_fraud}")
    columns[2].metric("Clean traffic flagged", best.false_positives)
    if weakest is not None:
        st.caption(f"Weakest pattern for {best.agent_name}: {weakest[0]} at recall {weakest[1]:.2f}")


# ----------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------


def main() -> None:
    """Render the dashboard."""
    st.title("StableGuard")
    st.caption(
        "An RL environment and eval harness for AI agents that review stablecoin payments "
        "for fraud and compliance."
    )

    dataset_path = str(DEFAULT_DATASET_PATH)
    if not Path(dataset_path).exists():
        st.error(
            "No dataset found. Generate one first:\n\n"
            "python scripts/generate_data.py --n 1000 --fraud-rate 0.15 --seed 42"
        )
        return

    dataset = get_dataset(dataset_path)
    evaluator = get_evaluator(dataset_path)

    with st.sidebar:
        st.header("Setup")
        mode = st.radio("Mode", ["Review a transaction", "Leaderboard"])
        agent_name = st.selectbox(
            "Agent",
            sorted(AGENT_REGISTRY),
            index=sorted(AGENT_REGISTRY).index("rule_based"),
        )
        if agent_name == "claude":
            if anthropic_api_key() is None:
                st.warning("ANTHROPIC_API_KEY is not set, so the Claude agent cannot run.")
            else:
                st.success(f"Claude agent ready: {load_config()['llm']['model']}")

        st.divider()
        metadata = dataset.get("metadata", {})
        n_fraud = sum(1 for t in dataset["transactions"] if t["is_fraud"])
        st.caption("Dataset")
        st.write(
            {
                "transactions": metadata.get("n_transactions", len(dataset["transactions"])),
                "fraud": n_fraud,
                "fraud rate": metadata.get("fraud_rate"),
                "seed": metadata.get("seed"),
                "wallets": len(dataset["wallets"]),
                "merchants": len(dataset["merchants"]),
            }
        )
        st.caption("Tool budget per payment: " + str(load_config()["env"]["tool_budget"]))

    if mode == "Review a transaction":
        review_mode(evaluator, dataset, agent_name)
    else:
        leaderboard_mode(evaluator, agent_name)


main()
