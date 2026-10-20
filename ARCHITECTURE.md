# StableGuard: Architecture

## What to make

**StableGuard** is an RL environment plus an eval harness for AI agents that review stablecoin payments for fraud and compliance.

It generates synthetic stablecoin transactions, lets an AI agent investigate each one by calling tools, and scores the agent on how accurately and efficiently it catches fraud. The agent sees a payment, calls tools like wallet history, sanctions check, and velocity check, then decides approve or flag. The harness runs the agent over hundreds of transactions and reports precision, recall, false positives, and speed, and compares agents on a leaderboard.

Visa is settling USDC and building stablecoin rails, and fraud and compliance is the biggest problem in that space, so this is the project that makes a payments hiring manager pay attention.

## Tech stack

- Python 3.11+
- FastAPI + Uvicorn (backend API)
- Streamlit (dashboard)
- Gymnasium (RL environment)
- Anthropic Claude API for the LLM agent (API key in `.env` as `ANTHROPIC_API_KEY`); a rule-based baseline agent needs no API key
- Pydantic v2 (data models)
- pytest (tests)
- Results stored as JSON files (no database needed for v1)

## Core data model

Transaction (Pydantic model, `stableguard/models.py`):

- `id`: str, e.g. `"txn_000123"`
- `timestamp`: datetime
- `sender_wallet`: str
- `receiver_wallet`: str
- `amount_usd`: float
- `stablecoin`: str, `"USDC"` or `"USDT"`
- `chain`: str, `"Solana"` or `"Ethereum"` or `"Base"`
- `sender_country`: str (ISO code, e.g. `"US"`)
- `receiver_country`: str
- `merchant_id`: str or None
- `merchant_category`: str (e.g. `"ecommerce"`, `"crypto_exchange"`, `"gambling"`, `"remittance"`)
- `sender_wallet_age_days`: int
- `sender_txn_count_30d`: int
- `sender_total_volume_30d_usd`: float
- `velocity_1h`: int (transactions from the sender in the last hour)
- `is_fraud`: bool (ground truth label, hidden from the agent)
- `fraud_type`: str, `"none"` or `"velocity"` or `"geo"` or `"sanctions"` or `"mule"` or `"amount_spike"` or `"merchant_mismatch"`

Decision (Pydantic model): `transaction_id`, `verdict` (`"approve"` or `"flag"`), `reasoning`, `tool_calls` (list of tool name + args + result), `latency_seconds`, `tokens_used`.

EvalResult (Pydantic model): `agent_name`, `n_transactions`, `accuracy`, `precision`, `recall`, `f1`, `false_positive_rate`, `false_negative_rate`, `avg_tool_calls`, `avg_latency_seconds`, `recall_by_fraud_type` (dict), `timestamp`.

## Components

### 1. Transaction generator (`stableguard/generator.py`)

Generates N synthetic transactions with a configurable fraud rate (default 15%). Deterministic with a seed so datasets are reproducible.

Legit transactions: pick a known wallet (age 30 to 900 days), normal amount ($10 to $2,000), sender country matches the wallet's home country, velocity 0 to 2 per hour, normal merchant category.

Fraud transactions: pick a fraud type and craft the transaction to exhibit it.

- `velocity`: velocity_1h is 15 to 40, moderate amount.
- `geo`: sender_country differs from the wallet's home country.
- `sanctions`: sender or receiver wallet is in the sanctions list (see `config/sanctions.json`).
- `mule`: wallet age under 7 days, large amount ($5,000+), high velocity (forwards quickly).
- `amount_spike`: amount is 10 to 50 times the wallet's typical amount.
- `merchant_mismatch`: merchant category is high risk (`crypto_exchange`, `gambling`) and the wallet has no history with that category.

Save to `data/transactions.json`. CLI: `python scripts/generate_data.py --n 1000 --fraud-rate 0.15 --seed 42`.

### 2. Investigation tools (`stableguard/tools.py`)

The agent calls these to investigate a transaction. Each tool reads from the generated dataset and a small in-memory state (wallet profiles, sanctions list). Every tool returns a JSON-serializable dict. The agent may call at most 6 tools per transaction (configurable).

- `check_wallet_history(wallet_address)` returns `{wallet_age_days, txn_count_30d, total_volume_30d_usd, avg_amount_usd}`.
- `check_sanctions(wallet_address)` returns `{sanctioned: bool, list_name: str or None}`.
- `check_velocity(wallet_address, window_hours=1)` returns `{txn_count, total_amount_usd}`.
- `check_amount_profile(wallet_address, amount_usd)` returns `{typical_amount_usd, ratio, unusual: bool}`.
- `check_merchant(merchant_id)` returns `{name, category, risk_score}` where risk_score is 0 to 1.
- `check_geo(sender_country, wallet_home_country)` returns `{mismatch: bool}`.

### 3. RL environment (`stableguard/env.py`)

A Gymnasium environment (`gymnasium.Env`).

- Observation: the transaction under review (as a dict) plus the list of tool results so far plus steps remaining. The ground truth label is never in the observation.
- Action: a structured dict, either `{"type": "tool", "name": ..., "args": {...}}` or `{"type": "decision", "verdict": "approve" or "flag"}`.
- `reset()`: picks the next transaction from the dataset and clears the tool history.
- `step(action)`: if the action is a tool call, run the tool, return the result, and give a small negative reward (-0.05). If the action is a decision, end the episode and give +1 for a correct verdict or -1 for a wrong one. If the agent exceeds the max steps without deciding, end the episode with reward -1.
- Reward shaping: +1 correct, -1 wrong, -0.05 per tool call, +0.2 bonus if fraud is caught using 2 or fewer tool calls. Keep the reward logic simple and document it in the code.

### 4. Agent (`stableguard/agent.py`)

Base class `Agent` with a method `act(observation) -> action`. Two implementations, same interface so the evaluator can run either.

- `RuleBasedAgent`: a baseline using simple rules. Flag if sanctioned, or velocity over 10, or amount ratio over 8, or geo mismatch. Otherwise approve. No API key needed. Build this first.
- `LLMAgent`: uses the Anthropic Claude API. System prompt: you are a payments fraud analyst, here are the tools, you must output JSON with `{"reasoning": "...", "action": {...}}`. It reads the transaction and the tool results so far, then picks the next tool or a verdict. Temperature 0 for determinism. Track tokens used per decision.

### 5. Eval harness (`stableguard/evaluator.py`, `stableguard/metrics.py`)

- `Evaluator` takes an agent and a dataset and runs the agent over each transaction through the env. For each transaction it records the verdict, ground truth, whether it was correct, tool calls used, latency, and tokens used.
- `metrics.py` computes: accuracy, precision, recall, F1, false positive rate, false negative rate, average tool calls, average latency, estimated cost, and recall broken down by fraud type.
- Output: a JSON report at `results/<agent_name>_<timestamp>.json` plus a printed summary table. A leaderboard function compares all agents run so far.
- CLI: `python scripts/run_eval.py --agent rule_based --n 200 --dataset data/transactions.json` and `python scripts/run_eval.py --agent claude --n 200 --dataset data/transactions.json`.

### 6. API (`api/main.py`)

FastAPI app. Keep it thin; it calls the evaluator and the env.

- `GET /transactions/{id}` returns a transaction without the ground truth label.
- `POST /review` with `{transaction_id, agent}` runs the agent on that transaction and returns the tool calls, reasoning, verdict, and whether it was correct.
- `GET /leaderboard` returns the eval results for all agents.
- `GET /metrics` returns aggregate metrics.
- `POST /eval/run` with `{agent, n}` runs a full eval and returns the report.

### 7. Dashboard (`dashboard/app.py`)

Streamlit app with two modes. Keep the UI clean and simple; the point is a demo a recruiter can click through in 2 minutes.

- "Review a transaction": shows a transaction, a button to let the agent investigate (shows each tool call and the reasoning live), then the verdict and whether it was right.
- "Leaderboard": shows the eval results table (agent, accuracy, precision, recall, F1, false positive rate, average tool calls, average latency) and a per-fraud-type recall chart.
- A "Run eval" button to trigger an eval from the UI.

## File structure

```
stableguard/
  README.md
  requirements.txt
  .env.example
  ARCHITECTURE.md            (this file)
  config/
    config.yaml              (chains, stablecoins, fraud rate, tool budget, thresholds)
    sanctions.json           (synthetic sanctioned wallets)
  stableguard/
    __init__.py
    models.py                (Transaction, Decision, EvalResult)
    generator.py             (synthetic transaction generator)
    tools.py                 (investigation tools)
    env.py                   (Gymnasium RL environment)
    agent.py                 (Agent base, RuleBasedAgent, LLMAgent)
    evaluator.py             (eval harness)
    metrics.py               (metric computation)
  api/
    main.py                  (FastAPI app)
  dashboard/
    app.py                   (Streamlit dashboard)
  scripts/
    generate_data.py         (CLI to generate a dataset)
    run_eval.py              (CLI to run an eval)
  tests/
    test_generator.py
    test_tools.py
    test_env.py
    test_evaluator.py
  data/
    transactions.json        (generated, gitignored)
  results/
    (eval reports, gitignored)
```

## Build milestones (in order)

1. Models + generator. Generate 1000 transactions and verify the fraud labels are correct.
2. Tools. Implement the 6 investigation tools, unit-tested.
3. Env. Gymnasium env with reset, step, and reward. Test it with a random agent.
4. Agents. Rule-based baseline first, then the Claude LLM agent.
5. Evaluator + metrics. Run both agents, compare them, save the report.
6. API. FastAPI with the 5 endpoints.
7. Dashboard. Streamlit with review mode and leaderboard.
8. README, demo instructions, tests, `.env.example`.

## How to run (v1)

```
pip install -r requirements.txt
cp .env.example .env        # add ANTHROPIC_API_KEY
python scripts/generate_data.py --n 1000 --fraud-rate 0.15 --seed 42
python scripts/run_eval.py --agent rule_based --n 200
python scripts/run_eval.py --agent claude --n 200
streamlit run dashboard/app.py
uvicorn api.main:app --reload     # API, in a separate terminal
```

## Definition of done

- 1000 synthetic transactions with correct labels.
- Both agents run end to end and produce a metrics report.
- The LLM agent beats the rule-based baseline on F1, or the report honestly shows where it does not.
- The dashboard lets someone review a transaction and see the leaderboard.
- README explains the project, how to run it, and why it matters for payments and stablecoins.
- A short demo video or screenshots is a bonus.

## Constraints for the build

- Keep the code clean, typed with type hints, and well commented.
- Do not use em dashes anywhere in user-facing text or the README.
- Do not hardcode API keys. Read them from environment variables.
- After each milestone, run the relevant tests and confirm it works before moving on.
- Ask before adding a new dependency.
