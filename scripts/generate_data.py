#!/usr/bin/env python3
"""CLI to generate a synthetic StableGuard dataset.

Example:
    python scripts/generate_data.py --n 1000 --fraud-rate 0.15 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Allow running the script directly from a checkout without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stableguard.config import DEFAULT_DATASET_PATH  # noqa: E402
from stableguard.generator import generate_dataset  # noqa: E402


def main() -> int:
    """Parse arguments, generate the dataset and print a short summary."""
    parser = argparse.ArgumentParser(description="Generate synthetic stablecoin transactions.")
    parser.add_argument("--n", type=int, default=1000, help="number of transactions")
    parser.add_argument("--fraud-rate", type=float, default=0.15, help="share of fraud, 0 to 1")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_DATASET_PATH),
        help="output path for the dataset JSON file",
    )
    args = parser.parse_args()

    dataset = generate_dataset(
        n=args.n, fraud_rate=args.fraud_rate, seed=args.seed, output_path=args.out
    )

    transactions = dataset["transactions"]
    fraud_counts = Counter(t["fraud_type"] for t in transactions if t["is_fraud"])
    n_fraud = sum(fraud_counts.values())

    print(f"Wrote {len(transactions)} transactions to {args.out}")
    print(f"Wallets: {len(dataset['wallets'])}   Merchants: {len(dataset['merchants'])}")
    print(f"Fraud: {n_fraud} ({n_fraud / len(transactions):.1%})")
    for fraud_type, count in sorted(fraud_counts.items()):
        print(f"  {fraud_type:<18} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
