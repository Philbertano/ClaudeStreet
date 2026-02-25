"""Configuration loader — cascading config from SSM, env, and YAML.

Resolution order (last wins):
  1. config/settings.yaml (defaults baked into deploy)
  2. SSM Parameter Store (/claudestreet/{env}/config)
  3. Environment variables (CLAUDESTREET_*)
  4. Secrets Manager (for sensitive values only)

For local development, set CLAUDESTREET_LOCAL=true to skip AWS calls
and load only from YAML + env vars.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "env": "dev",
    "region": "us-east-1",
    "event_bus_name": "claudestreet",
    "trades_table": "claudestreet-trades",
    "strategies_table": "claudestreet-strategies",
    "snapshots_table": "claudestreet-snapshots",
    "events_table": "claudestreet-events",
    "session_bucket": "claudestreet-sessions",
    "secrets_name": "claudestreet/broker",
    "watchlist": ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "TSLA", "META", "JPM", "V", "SPY"],
    "max_positions": 10,
    "max_position_pct": 0.15,
    "max_portfolio_risk_pct": 0.02,
    "initial_capital": 100000.0,
    "paper_trading": True,
    "stop_loss_default_pct": 0.05,
    "take_profit_default_pct": 0.10,
    "max_daily_loss_pct": 0.03,
    "max_trades_per_day": 20,
    "min_time_between_trades_sec": 60,
    "restricted_hours": ["09:30-09:45", "15:45-16:00"],
    "population_size": 20,
    "survivors_per_generation": 5,
    "mutation_rate": 0.15,
    "crossover_rate": 0.6,
    "min_trades_for_fitness": 10,
}


def _load_yaml(path: str = "config/settings.yaml") -> dict:
    """Load config from YAML file."""
    p = Path(path)
    if not p.exists():
        # Also check relative to the Lambda task root
        alt = Path(os.environ.get("LAMBDA_TASK_ROOT", "")) / path
        if alt.exists():
            p = alt
        else:
            return {}
    try:
        data = yaml.safe_load(p.read_text()) or {}
        # Flatten nested structure for simpler access
        flat = {}
        for section in data.values():
            if isinstance(section, dict):
                flat.update(section)
        return flat
    except Exception:
        logger.exception("Failed to load YAML config from %s", path)
        return {}


def _load_ssm(prefix: str) -> dict:
    """Load config from SSM Parameter Store."""
    try:
        import boto3

        ssm = boto3.client("ssm")
        response = ssm.get_parameters_by_path(
            Path=prefix,
            Recursive=True,
            WithDecryption=True,
        )
        config = {}
        for param in response.get("Parameters", []):
            key = param["Name"].split("/")[-1]
            value = param["Value"]
            # Try to parse JSON values
            try:
                config[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                config[key] = value
        return config
    except Exception:
        logger.debug("SSM parameter load skipped (not available)")
        return {}


def _load_secrets(secret_name: str) -> dict:
    """Load secrets from AWS Secrets Manager."""
    try:
        import boto3

        client = boto3.client("secretsmanager")
        response = client.get_secret_value(SecretId=secret_name)
        return json.loads(response["SecretString"])
    except Exception:
        logger.debug("Secrets Manager load skipped (not available)")
        return {}


def _load_env() -> dict:
    """Load config from environment variables."""
    prefix = "CLAUDESTREET_"
    config = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            config_key = key[len(prefix):].lower()
            try:
                config[config_key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                config[config_key] = value
    return config


@lru_cache(maxsize=1)
def load_config() -> dict:
    """Load configuration with cascading precedence.

    Returns a merged dict. Safe to call multiple times (cached).
    """
    config = dict(_DEFAULTS)

    # 1. YAML defaults
    config.update(_load_yaml())

    is_local = os.environ.get("CLAUDESTREET_LOCAL", "").lower() in ("true", "1", "yes")

    if not is_local:
        # 2. SSM parameters
        env = os.environ.get("CLAUDESTREET_ENV", "dev")
        config.update(_load_ssm(f"/claudestreet/{env}/"))

        # 3. Secrets (broker keys etc.)
        secrets_name = config.get("secrets_name", "claudestreet/broker")
        config.update(_load_secrets(secrets_name))

    # 4. Environment variables (always, highest precedence)
    config.update(_load_env())

    logger.info(
        "Config loaded: env=%s, local=%s, watchlist=%d symbols",
        config.get("env", "dev"),
        is_local,
        len(config.get("watchlist", [])),
    )
    return config
