"""Sandbox — safe execution of LLM-generated strategy code.

Loads strategy Python source via restricted exec() with only
whitelisted modules available. Catches all exceptions and
enforces a timeout to prevent infinite loops.
"""

from __future__ import annotations

import logging
import math
import signal
import traceback
from typing import Any, Type

import numpy as np
import pandas as pd

from claudestreet.evolve_engine.strategy_template import CustomStrategy

logger = logging.getLogger(__name__)

# Modules that generated strategy code is allowed to import
_ALLOWED_MODULES = {
    "math", "numpy", "np", "pandas", "pd",
    "claudestreet.evolve_engine.strategy_template",
    "claudestreet.evolve_engine",
    "claudestreet",
}

_real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Import function that only allows whitelisted modules."""
    if name in _ALLOWED_MODULES or any(name.startswith(m + ".") for m in _ALLOWED_MODULES):
        return _real_import(name, globals, locals, fromlist, level)
    raise ImportError(f"Import of '{name}' is not allowed in strategy code")


def _make_safe_builtins() -> dict:
    """Create safe builtins: real builtins minus dangerous functions."""
    import builtins
    safe = dict(vars(builtins))
    # Replace import with restricted version
    safe["__import__"] = _restricted_import
    # Remove dangerous builtins
    for dangerous in ("open", "exec", "eval", "compile", "breakpoint",
                      "exit", "quit", "input", "memoryview"):
        safe.pop(dangerous, None)
    return safe


# Only these are available to generated code
SAFE_GLOBALS = {
    "__builtins__": _make_safe_builtins(),
    "np": np,
    "numpy": np,
    "pd": pd,
    "pandas": pd,
    "math": math,
    "CustomStrategy": CustomStrategy,
}

EXECUTION_TIMEOUT_SECONDS = 30


class SandboxError(Exception):
    """Raised when sandbox execution fails."""
    pass


class TimeoutError(SandboxError):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError(f"Strategy execution timed out after {EXECUTION_TIMEOUT_SECONDS}s")


def load_strategy_from_source(
    source_code: str,
    strategy_name: str | None = None,
) -> Type[CustomStrategy] | None:
    """Load a CustomStrategy subclass from Python source code.

    Executes the code in a restricted namespace and extracts
    the first CustomStrategy subclass found.

    Args:
        source_code: Python source defining a CustomStrategy subclass.
        strategy_name: Optional specific class name to look for.

    Returns:
        The strategy class, or None if loading failed.
    """
    namespace: dict[str, Any] = dict(SAFE_GLOBALS)

    try:
        # Set timeout to prevent infinite loops
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(EXECUTION_TIMEOUT_SECONDS)

        try:
            exec(compile(source_code, "<strategy>", "exec"), namespace)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    except TimeoutError:
        logger.error("Strategy code timed out during load")
        return None
    except SyntaxError as e:
        logger.error("Syntax error in strategy code: %s (line %d)", e.msg, e.lineno or 0)
        return None
    except Exception as e:
        logger.error("Failed to load strategy: %s\n%s", e, traceback.format_exc())
        return None

    # Find the CustomStrategy subclass
    for name, obj in namespace.items():
        if name.startswith("_"):
            continue
        if (
            isinstance(obj, type)
            and issubclass(obj, CustomStrategy)
            and obj is not CustomStrategy
        ):
            if strategy_name and name != strategy_name:
                continue
            logger.info("Loaded strategy class: %s", name)
            return obj

    logger.warning("No CustomStrategy subclass found in source")
    return None


def validate_strategy(strategy_cls: Type[CustomStrategy]) -> list[str]:
    """Run basic validation on a loaded strategy class.

    Returns list of error messages (empty = valid).
    """
    errors: list[str] = []

    # Must have name
    if not getattr(strategy_cls, "name", None) or strategy_cls.name == "unnamed":
        errors.append("Strategy must have a non-default 'name'")

    # Must implement evaluate
    try:
        instance = strategy_cls()
        # Create minimal test data
        test_df = pd.DataFrame({
            "Open": np.random.uniform(100, 200, 50),
            "High": np.random.uniform(100, 200, 50),
            "Low": np.random.uniform(100, 200, 50),
            "Close": np.random.uniform(100, 200, 50),
            "Volume": np.random.randint(1000, 100000, 50),
        })

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(10)
        try:
            result = instance.evaluate(test_df, instance.get_default_params())
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        if not isinstance(result, tuple) or len(result) != 2:
            errors.append("evaluate() must return (recommendation, confidence) tuple")
        else:
            rec, conf = result
            valid_recs = {"strong_buy", "buy", "hold", "sell", "strong_sell"}
            if rec not in valid_recs:
                errors.append(f"Invalid recommendation '{rec}', must be one of {valid_recs}")
            if not isinstance(conf, (int, float)) or not 0.0 <= conf <= 1.0:
                errors.append(f"Confidence must be float 0.0-1.0, got {conf}")

    except TimeoutError:
        errors.append("evaluate() timed out during validation")
    except Exception as e:
        errors.append(f"evaluate() raised exception: {e}")

    return errors
