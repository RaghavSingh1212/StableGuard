"""Synthetic stablecoin transaction generator.

The generator builds three things that travel together in one dataset file:

* wallet profiles, which give the investigation tools something to look up,
* merchants, with a category and a risk score,
* transactions, each labelled with ground truth.

Everything is driven by a single seeded ``random.Random`` instance, so the same
seed always produces the same dataset. Fraud transactions are crafted so that
each one exhibits the signal of its own fraud type and, as far as possible, not
the signals of the others. That keeps per fraud type recall meaningful.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from random import Random
from typing import Any

from .config import DEFAULT_DATASET_PATH, load_config, load_sanctions
from .models import FRAUD_TYPES, Transaction

# Countries used for wallet home countries and counterparties. The "high risk"
# pool is only used to make geo mismatches look plausible.
HOME_COUNTRIES: tuple[str, ...] = ("US", "GB", "DE", "SG", "BR", "MX", "NG", "PH", "IN", "FR")
FOREIGN_COUNTRIES: tuple[str, ...] = ("RU", "IR", "KP", "VE", "BY", "MM", "AF", "SY", "PA", "CY")

# Baseline risk score per merchant category, 0 to 1.
CATEGORY_RISK: dict[str, float] = {
    "payroll": 0.08,
    "ecommerce": 0.18,
    "remittance": 0.34,
    "marketplace": 0.42,
    "crypto_exchange": 0.74,
    "gambling": 0.86,
}

MERCHANT_NAME_PREFIXES: tuple[str, ...] = (
    "Northwind", "Bluepeak", "Cedar", "Orbital", "Lumen", "Harbor", "Vertex",
    "Juniper", "Solace", "Pioneer", "Tidal", "Granite", "Everlyn", "Kestrel",
)
MERCHANT_NAME_SUFFIXES: dict[str, tuple[str, ...]] = {
    "ecommerce": ("Retail", "Goods", "Store", "Shop"),
    "crypto_exchange": ("Exchange", "Digital Assets", "Trading"),
    "gambling": ("Gaming", "Casino", "Betting"),
    "remittance": ("Remit", "Transfer", "Money"),
    "payroll": ("Payroll", "Payments", "HR Services"),
    "marketplace": ("Marketplace", "Market", "Bazaar"),
}


@dataclass
class WalletProfile:
    """What the investigation tools know about a wallet.

    This is the "bank side" view: history that an analyst could pull up, not
    anything about the specific transaction under review.
    """

    address: str
    chain: str
    home_country: str
    wallet_age_days: int
    typical_amount_usd: float
    txn_count_30d: int
    total_volume_30d_usd: float
    categories: list[str] = field(default_factory=list)
    baseline_velocity_1h: int = 1
    is_sanctioned: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON serialisable copy of the profile."""
        return asdict(self)


@dataclass
class Merchant:
    """A merchant the payment is made to."""

    merchant_id: str
    name: str
    category: str
    risk_score: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON serialisable copy of the merchant."""
        return asdict(self)


def _eth_address(rng: Random) -> str:
    """Return a synthetic Ethereum or Base style address."""
    return "0x" + "".join(rng.choice("0123456789abcdef") for _ in range(40))


def _sol_address(rng: Random) -> str:
    """Return a synthetic Solana style base58 address."""
    alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
    return "".join(rng.choice(alphabet) for _ in range(44))


def _address_for_chain(rng: Random, chain: str) -> str:
    """Return an address whose shape matches the chain."""
    return _sol_address(rng) if chain == "Solana" else _eth_address(rng)


class TransactionGenerator:
    """Builds a reproducible synthetic dataset of stablecoin payments."""

    def __init__(
        self,
        seed: int | None = None,
        fraud_rate: float | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.config = config or load_config()
        dataset_cfg = self.config["dataset"]

        self.seed: int = int(dataset_cfg["default_seed"] if seed is None else seed)
        self.fraud_rate: float = float(
            dataset_cfg["default_fraud_rate"] if fraud_rate is None else fraud_rate
        )
        if not 0.0 <= self.fraud_rate <= 1.0:
            raise ValueError("fraud_rate must be between 0 and 1")

        self.rng = Random(self.seed)
        self.chains: list[str] = list(dataset_cfg["chains"])
        self.stablecoins: list[str] = list(dataset_cfg["stablecoins"])
        self.categories: list[str] = list(dataset_cfg["merchant_categories"])
        self.high_risk_categories: list[str] = list(dataset_cfg["high_risk_categories"])
        self.normal_categories: list[str] = [
            c for c in self.categories if c not in self.high_risk_categories
        ]

        thresholds = self.config["thresholds"]
        self.new_wallet_age_days: int = int(thresholds["new_wallet_age_days"])
        self.large_amount_usd: float = float(thresholds["large_amount_usd"])

        self.sanctions: dict[str, str] = load_sanctions()
        self.sanctioned_addresses: list[str] = sorted(self.sanctions)

        self.wallets: dict[str, WalletProfile] = {}
        self.merchants: dict[str, Merchant] = {}
        self._build_wallets(int(dataset_cfg["n_wallets"]))
        self._build_merchants(int(dataset_cfg["n_merchants"]))

    # ------------------------------------------------------------------
    # Static world: wallets and merchants
    # ------------------------------------------------------------------

    def _build_wallets(self, n_wallets: int) -> None:
        """Create the pool of established wallets plus profiles for sanctioned ones."""
        for _ in range(n_wallets):
            profile = self._new_established_wallet()
            self.wallets[profile.address] = profile

        # Sanctioned wallets get a profile too, otherwise a lookup on one would
        # fail when a sanctioned address appears as the sender.
        for address in self.sanctioned_addresses:
            chain = "Solana" if not address.startswith("0x") else self.rng.choice(["Ethereum", "Base"])
            profile = WalletProfile(
                address=address,
                chain=chain,
                home_country=self.rng.choice(FOREIGN_COUNTRIES),
                wallet_age_days=self.rng.randint(20, 600),
                typical_amount_usd=round(self.rng.uniform(400.0, 4000.0), 2),
                txn_count_30d=self.rng.randint(5, 90),
                total_volume_30d_usd=0.0,
                categories=[self.rng.choice(self.categories)],
                baseline_velocity_1h=self.rng.randint(0, 3),
                is_sanctioned=True,
            )
            profile.total_volume_30d_usd = round(
                profile.typical_amount_usd * profile.txn_count_30d, 2
            )
            self.wallets[address] = profile

    def _new_established_wallet(self) -> WalletProfile:
        """Create one ordinary wallet with 30 to 900 days of history."""
        chain = self.rng.choice(self.chains)
        address = _address_for_chain(self.rng, chain)
        typical = round(self.rng.uniform(40.0, 900.0), 2)
        txn_count = self.rng.randint(4, 120)

        # Most wallets transact in one or two categories. A minority of them
        # legitimately use a high risk category, which is what stops a naive
        # "high risk category means fraud" rule from scoring well.
        n_categories = self.rng.choice([1, 1, 2, 2, 3])
        pool = list(self.normal_categories)
        if self.rng.random() < 0.18:
            pool.append(self.rng.choice(self.high_risk_categories))
        categories = self.rng.sample(pool, k=min(n_categories, len(pool)))

        return WalletProfile(
            address=address,
            chain=chain,
            home_country=self.rng.choice(HOME_COUNTRIES),
            wallet_age_days=self.rng.randint(30, 900),
            typical_amount_usd=typical,
            txn_count_30d=txn_count,
            total_volume_30d_usd=round(typical * txn_count * self.rng.uniform(0.8, 1.2), 2),
            categories=categories,
            baseline_velocity_1h=self.rng.randint(0, 2),
            is_sanctioned=False,
        )

    def _new_young_wallet(self) -> WalletProfile:
        """Create a wallet that was opened in the last few days (mule pattern)."""
        chain = self.rng.choice(self.chains)
        address = _address_for_chain(self.rng, chain)
        typical = round(self.rng.uniform(80.0, 600.0), 2)
        profile = WalletProfile(
            address=address,
            chain=chain,
            home_country=self.rng.choice(HOME_COUNTRIES + FOREIGN_COUNTRIES),
            wallet_age_days=self.rng.randint(0, self.new_wallet_age_days - 1),
            typical_amount_usd=typical,
            txn_count_30d=self.rng.randint(1, 6),
            total_volume_30d_usd=0.0,
            categories=[self.rng.choice(self.categories)],
            baseline_velocity_1h=self.rng.randint(5, 15),
            is_sanctioned=False,
        )
        profile.total_volume_30d_usd = round(typical * profile.txn_count_30d, 2)
        self.wallets[address] = profile
        return profile

    def _build_merchants(self, n_merchants: int) -> None:
        """Create the merchant directory, spread across every category."""
        for index in range(n_merchants):
            category = self.categories[index % len(self.categories)]
            base_risk = CATEGORY_RISK[category]
            risk = min(1.0, max(0.0, base_risk + self.rng.uniform(-0.06, 0.06)))
            name = "{} {}".format(
                self.rng.choice(MERCHANT_NAME_PREFIXES),
                self.rng.choice(MERCHANT_NAME_SUFFIXES[category]),
            )
            merchant_id = f"mch_{index:04d}"
            self.merchants[merchant_id] = Merchant(
                merchant_id=merchant_id,
                name=name,
                category=category,
                risk_score=round(risk, 3),
            )

    def _merchant_in_category(self, category: str) -> Merchant:
        """Return a random merchant from the given category."""
        candidates = [m for m in self.merchants.values() if m.category == category]
        return self.rng.choice(candidates)

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    def generate(self, n: int) -> list[Transaction]:
        """Generate ``n`` transactions at the configured fraud rate."""
        if n <= 0:
            raise ValueError("n must be a positive integer")

        n_fraud = int(round(n * self.fraud_rate))
        labels = [True] * n_fraud + [False] * (n - n_fraud)
        self.rng.shuffle(labels)

        # Spread fraud types as evenly as the fraud count allows.
        fraud_types: list[str] = [
            FRAUD_TYPES[i % len(FRAUD_TYPES)] for i in range(n_fraud)
        ]
        self.rng.shuffle(fraud_types)
        fraud_iter = iter(fraud_types)

        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        transactions: list[Transaction] = []
        for index, is_fraud in enumerate(labels):
            timestamp = start + timedelta(minutes=self.rng.randint(0, 60 * 24 * 60))
            txn_id = f"txn_{index:06d}"
            if is_fraud:
                txn = self._make_fraud(txn_id, timestamp, next(fraud_iter))
            else:
                txn = self._make_legit(txn_id, timestamp)
            transactions.append(txn)

        transactions.sort(key=lambda t: t.timestamp)
        return transactions

    def _base_fields(
        self,
        txn_id: str,
        timestamp: datetime,
        wallet: WalletProfile,
        amount: float,
        category: str,
        merchant: Merchant | None,
    ) -> dict[str, Any]:
        """Assemble the fields that every transaction shares."""
        return {
            "id": txn_id,
            "timestamp": timestamp,
            "sender_wallet": wallet.address,
            "receiver_wallet": _address_for_chain(self.rng, wallet.chain),
            "amount_usd": round(amount, 2),
            "stablecoin": self.rng.choice(self.stablecoins),
            "chain": wallet.chain,
            "sender_country": wallet.home_country,
            "receiver_country": wallet.home_country,
            "merchant_id": merchant.merchant_id if merchant else None,
            "merchant_category": category,
            "sender_wallet_age_days": wallet.wallet_age_days,
            "sender_txn_count_30d": wallet.txn_count_30d,
            "sender_total_volume_30d_usd": wallet.total_volume_30d_usd,
            "velocity_1h": wallet.baseline_velocity_1h,
            "is_fraud": False,
            "fraud_type": "none",
        }

    def _pick_established_wallet(self) -> WalletProfile:
        """Return a random wallet that is neither new nor sanctioned."""
        while True:
            profile = self.rng.choice(list(self.wallets.values()))
            if not profile.is_sanctioned and profile.wallet_age_days >= 30:
                return profile

    def _make_legit(self, txn_id: str, timestamp: datetime) -> Transaction:
        """Build a clean transaction that exhibits none of the fraud signals."""
        wallet = self._pick_established_wallet()
        category = self.rng.choice(wallet.categories)
        merchant = self._merchant_in_category(category)

        # Normal amount: within a small multiple of what this wallet usually
        # sends, and inside the $10 to $2,000 band the architecture specifies.
        amount = min(2000.0, max(10.0, wallet.typical_amount_usd * self.rng.uniform(0.3, 2.5)))

        fields = self._base_fields(txn_id, timestamp, wallet, amount, category, merchant)
        fields["velocity_1h"] = min(2, wallet.baseline_velocity_1h)
        fields["receiver_country"] = (
            wallet.home_country if self.rng.random() < 0.75 else self.rng.choice(HOME_COUNTRIES)
        )
        return Transaction(**fields)

    def _make_fraud(self, txn_id: str, timestamp: datetime, fraud_type: str) -> Transaction:
        """Build a transaction that exhibits exactly one fraud pattern."""
        builder = {
            "velocity": self._fraud_velocity,
            "geo": self._fraud_geo,
            "sanctions": self._fraud_sanctions,
            "mule": self._fraud_mule,
            "amount_spike": self._fraud_amount_spike,
            "merchant_mismatch": self._fraud_merchant_mismatch,
        }[fraud_type]
        txn = builder(txn_id, timestamp)
        txn.is_fraud = True
        txn.fraud_type = fraud_type  # type: ignore[assignment]
        return txn

    def _fraud_velocity(self, txn_id: str, timestamp: datetime) -> Transaction:
        """Burst of transactions from one wallet inside an hour."""
        wallet = self._pick_established_wallet()
        category = self.rng.choice(wallet.categories)
        merchant = self._merchant_in_category(category)
        amount = wallet.typical_amount_usd * self.rng.uniform(0.8, 2.0)
        fields = self._base_fields(txn_id, timestamp, wallet, amount, category, merchant)
        fields["velocity_1h"] = self.rng.randint(15, 40)
        return Transaction(**fields)

    def _fraud_geo(self, txn_id: str, timestamp: datetime) -> Transaction:
        """Payment sent from a country the wallet has never transacted from."""
        wallet = self._pick_established_wallet()
        category = self.rng.choice(wallet.categories)
        merchant = self._merchant_in_category(category)
        amount = wallet.typical_amount_usd * self.rng.uniform(0.8, 2.5)
        fields = self._base_fields(txn_id, timestamp, wallet, amount, category, merchant)
        foreign = self.rng.choice(FOREIGN_COUNTRIES)
        fields["sender_country"] = foreign
        fields["receiver_country"] = self.rng.choice(FOREIGN_COUNTRIES + HOME_COUNTRIES)
        fields["velocity_1h"] = min(2, wallet.baseline_velocity_1h)
        return Transaction(**fields)

    def _fraud_sanctions(self, txn_id: str, timestamp: datetime) -> Transaction:
        """Counterparty appears on a sanctions list."""
        sanctioned = self.rng.choice(self.sanctioned_addresses)
        sender_is_sanctioned = self.rng.random() < 0.4

        if sender_is_sanctioned:
            wallet = self.wallets[sanctioned]
        else:
            wallet = self._pick_established_wallet()

        category = self.rng.choice(wallet.categories)
        merchant = self._merchant_in_category(category)
        amount = wallet.typical_amount_usd * self.rng.uniform(0.8, 3.0)
        fields = self._base_fields(txn_id, timestamp, wallet, amount, category, merchant)
        if not sender_is_sanctioned:
            fields["receiver_wallet"] = sanctioned
        fields["velocity_1h"] = min(2, wallet.baseline_velocity_1h)
        return Transaction(**fields)

    def _fraud_mule(self, txn_id: str, timestamp: datetime) -> Transaction:
        """Freshly opened wallet moving a large sum straight back out."""
        wallet = self._new_young_wallet()
        category = self.rng.choice(wallet.categories)
        merchant = self._merchant_in_category(category)
        amount = self.rng.uniform(self.large_amount_usd, self.large_amount_usd * 8)
        fields = self._base_fields(txn_id, timestamp, wallet, amount, category, merchant)
        fields["velocity_1h"] = self.rng.randint(6, 14)
        return Transaction(**fields)

    def _fraud_amount_spike(self, txn_id: str, timestamp: datetime) -> Transaction:
        """Amount far above anything this wallet has sent before."""
        wallet = self._pick_established_wallet()
        category = self.rng.choice(wallet.categories)
        merchant = self._merchant_in_category(category)
        amount = wallet.typical_amount_usd * self.rng.uniform(10.0, 50.0)
        fields = self._base_fields(txn_id, timestamp, wallet, amount, category, merchant)
        fields["velocity_1h"] = min(2, wallet.baseline_velocity_1h)
        return Transaction(**fields)

    def _fraud_merchant_mismatch(self, txn_id: str, timestamp: datetime) -> Transaction:
        """High risk merchant category the wallet has no history with."""
        while True:
            wallet = self._pick_established_wallet()
            unseen = [c for c in self.high_risk_categories if c not in wallet.categories]
            if unseen:
                break
        category = self.rng.choice(unseen)
        merchant = self._merchant_in_category(category)
        amount = wallet.typical_amount_usd * self.rng.uniform(1.0, 4.0)
        fields = self._base_fields(txn_id, timestamp, wallet, amount, category, merchant)
        fields["velocity_1h"] = min(2, wallet.baseline_velocity_1h)
        return Transaction(**fields)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def build_dataset(self, n: int) -> dict[str, Any]:
        """Generate transactions and wrap them with the world they refer to."""
        transactions = self.generate(n)
        return {
            "metadata": {
                "n_transactions": len(transactions),
                "fraud_rate": self.fraud_rate,
                "seed": self.seed,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            "wallets": [w.to_dict() for w in self.wallets.values()],
            "merchants": [m.to_dict() for m in self.merchants.values()],
            "transactions": [t.model_dump(mode="json") for t in transactions],
        }


def generate_dataset(
    n: int,
    fraud_rate: float | None = None,
    seed: int | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate a dataset and, unless ``output_path`` is None, write it to disk."""
    generator = TransactionGenerator(seed=seed, fraud_rate=fraud_rate)
    dataset = generator.build_dataset(n)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(dataset, handle, indent=2)
    return dataset


def load_dataset(path: str | Path = DEFAULT_DATASET_PATH) -> dict[str, Any]:
    """Read a dataset file written by :func:`generate_dataset`."""
    with open(path, "r", encoding="utf-8") as handle:
        dataset: dict[str, Any] = json.load(handle)
    return dataset


def load_transactions(path: str | Path = DEFAULT_DATASET_PATH) -> list[Transaction]:
    """Read a dataset file and return its transactions as Pydantic models."""
    dataset = load_dataset(path)
    return [Transaction(**row) for row in dataset["transactions"]]
