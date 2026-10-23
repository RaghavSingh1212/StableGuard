"""Loading of project configuration and static reference data.

Everything that the generator, the tools and the agents must agree on lives in
``config/config.yaml`` and ``config/sanctions.json``. This module is the single
place that reads them, so nothing else in the codebase needs to know the paths.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Any

import yaml

# Repository root: this file is <root>/stableguard/config.py
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
CONFIG_DIR: Path = ROOT_DIR / "config"
DATA_DIR: Path = ROOT_DIR / "data"
RESULTS_DIR: Path = ROOT_DIR / "results"

DEFAULT_CONFIG_PATH: Path = CONFIG_DIR / "config.yaml"
DEFAULT_SANCTIONS_PATH: Path = CONFIG_DIR / "sanctions.json"
DEFAULT_DATASET_PATH: Path = DATA_DIR / "transactions.json"


@functools.lru_cache(maxsize=4)
def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Read ``config.yaml`` and return it as a plain dictionary.

    The result is cached because the config is read on nearly every call path
    (tools, env, agents) and never changes during a run.
    """
    with open(path, "r", encoding="utf-8") as handle:
        config: dict[str, Any] = yaml.safe_load(handle)
    return config


@functools.lru_cache(maxsize=4)
def load_sanctions(path: str | Path = DEFAULT_SANCTIONS_PATH) -> dict[str, str]:
    """Return a mapping of sanctioned wallet address to the list that names it.

    The addresses are synthetic and match no real entity.
    """
    with open(path, "r", encoding="utf-8") as handle:
        document: dict[str, Any] = json.load(handle)
    return {entry["wallet_address"]: entry["list_name"] for entry in document["entries"]}


def claude_model() -> str:
    """Return the Claude model id to use, honouring an environment override."""
    config = load_config()
    return os.environ.get("STABLEGUARD_CLAUDE_MODEL") or str(config["llm"]["model"])


def anthropic_api_key() -> str | None:
    """Return the Anthropic API key from the environment, or None if unset.

    The key is never read from a file in the repository and never written to one.
    """
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    return key or None
