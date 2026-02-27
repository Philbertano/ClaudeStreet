"""Fargate Evolution Engine — Claude Agent SDK autonomous loop.

Runs as an ECS Fargate task. Uses the Anthropic Python SDK with
tool_use to create an autonomous agent that:

  1. Reviews current strategy performance (DynamoDB)
  2. Analyzes market data (yfinance)
  3. Writes NEW Python strategy code (Claude generates it)
  4. Backtests candidates against historical data
  5. Deploys winners, retires losers

This is the "brain that builds better brains" — a meta-agent
that evolves the swarm through actual code generation.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

import anthropic
import boto3

from claudestreet.connectors.market_data import MarketDataConnector
from claudestreet.core.config import load_config
from claudestreet.core.event_bus import EventBridgeClient
from claudestreet.core.memory import DynamoMemory
from claudestreet.evolve_engine.backtest import BacktestEngine, BacktestResult
from claudestreet.evolve_engine.sandbox import load_strategy_from_source, validate_strategy
from claudestreet.evolve_engine.tools import TOOLS
from claudestreet.models.events import Event, EventType

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the Evolution Engine for ClaudeStreet, an autonomous stock trading system.

Your mission: analyze existing trading strategy performance and evolve the strategy
population to become more profitable over time. You do this by writing real Python code.

## Your Capabilities
You have tools to:
- List active strategies and their fitness metrics
- Read strategy source code and performance data
- Read historical market data for any stock
- Write new strategy code (Python)
- Backtest strategies against historical data
- Deploy successful strategies and retire underperformers

## Strategy Code Requirements
Strategies must inherit from CustomStrategy and implement evaluate():

```python
import math
import numpy as np
import pandas as pd
from claudestreet.evolve_engine.strategy_template import CustomStrategy

class MyStrategy(CustomStrategy):
    name = "my-strategy"
    description = "Description of the approach"
    version = "1.0"

    def evaluate(self, df: pd.DataFrame, params: dict[str, float]) -> tuple[str, float]:
        close = df["Close"].values
        high = df["High"].values
        low = df["Low"].values
        volume = df["Volume"].values

        # Your analysis logic here...

        return "hold", 0.0  # (recommendation, confidence)

    def get_default_params(self) -> dict[str, float]:
        return {"param1": 14.0, "param2": 0.05}
```

Available imports: numpy, pandas, math (nothing else).
Recommendations: "strong_buy", "buy", "hold", "sell", "strong_sell"
Confidence: 0.0 to 1.0

## Your Workflow
1. **Assess**: List strategies, read their performance data
2. **Analyze**: Identify what's working and what's failing. Look at market data.
3. **Create**: Write 2-3 new strategy variations. Be creative — try different approaches:
   - Technical indicators (RSI, MACD, Bollinger, ATR, etc.)
   - Pattern recognition (breakouts, reversals, channels)
   - Multi-timeframe analysis
   - Volatility-adaptive logic
   - Mean reversion vs momentum
4. **Test**: Backtest each new strategy across multiple symbols
5. **Deploy**: Deploy strategies with composite fitness > 0.55
6. **Retire**: Retire strategies with composite fitness < 0.35

## Market Regime Awareness
The system detects market regimes: TRENDING_BULL, TRENDING_BEAR, MEAN_REVERTING, HIGH_VOLATILITY.
When the current regime is provided, write regime-specific strategies:
- TRENDING_BULL: Write momentum strategies — trend-following, breakout, moving average systems
- TRENDING_BEAR: Write short-biased momentum or defensive strategies
- MEAN_REVERTING: Write mean-reversion strategies — Bollinger band bounce, RSI extremes, z-score
- HIGH_VOLATILITY: Write volatility-adaptive strategies — wider stops, smaller positions, VIX-aware

Tag each strategy with its target regime using `regime_preference` in deploy metadata.

## Principles
- Risk-adjusted returns matter more than raw returns (Sharpe ratio)
- Low drawdown is critical — capital preservation first
- Diversify strategies — don't make them all the same approach
- A strategy that returns 0% but never loses is better than one swinging wildly
- Test on at least 3 different symbols before deploying
- Write strategies that match the current market regime
"""

MAX_AGENT_TURNS = 25


class EvolutionEngine:
    """Claude Agent SDK-powered evolution engine."""

    def __init__(self) -> None:
        self.config = load_config()
        self.memory = DynamoMemory()
        self.eb = EventBridgeClient()
        self.market = MarketDataConnector()
        self.backtester = BacktestEngine(
            initial_capital=self.config.get("initial_capital", 100000.0),
        )
        self._s3 = boto3.client("s3")
        self._bucket = os.environ.get("SESSION_BUCKET", "claudestreet-sessions")

        # Candidate strategies written during this session
        self._candidates: dict[str, dict] = {}

        # Agent SDK client
        self._client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )

    def _detect_current_regime(self) -> str:
        """Detect the current market regime from SPY data."""
        try:
            from claudestreet.skills.regime_detection import RegimeDetector
            df = self.market.get_historical("SPY", period="6mo")
            if df is not None and not df.empty:
                detector = RegimeDetector()
                regime = detector.detect(df)
                return regime.value
        except Exception:
            logger.exception("Failed to detect market regime")
        return "mean_reverting"

    def run(self) -> None:
        """Run the autonomous evolution agent loop."""
        logger.info("=== Evolution Engine Started (Agent SDK) ===")
        start = datetime.now(timezone.utc)

        # Detect current regime for context
        current_regime = self._detect_current_regime()
        logger.info("Current market regime: %s", current_regime)

        messages: list[dict] = [
            {
                "role": "user",
                "content": (
                    "Run an evolution cycle. The current watchlist is: "
                    f"{self.config.get('watchlist', [])}. "
                    f"The current market regime is: {current_regime.upper()}. "
                    f"Write strategies optimized for this regime. "
                    "Start by listing active strategies and their performance, "
                    "then analyze, create new strategies, backtest them, and "
                    "deploy the best ones."
                ),
            }
        ]

        turns = 0
        strategies_written = 0
        strategies_deployed = 0
        strategies_retired = 0

        while turns < MAX_AGENT_TURNS:
            turns += 1
            logger.info("--- Agent turn %d / %d ---", turns, MAX_AGENT_TURNS)

            try:
                response = self._client.messages.create(
                    model="claude-sonnet-4-6-20250514",
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=messages,
                )
            except anthropic.APIError as e:
                logger.error("Anthropic API error: %s", e)
                break

            # Collect assistant response
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            # Log any text output
            for block in assistant_content:
                if block.type == "text" and block.text.strip():
                    logger.info("[Claude] %s", block.text[:500])

            # If no tool use, the agent is done
            if response.stop_reason == "end_turn":
                logger.info("Agent completed (end_turn)")
                break

            # Execute tool calls
            tool_results = []
            for block in assistant_content:
                if block.type != "tool_use":
                    continue

                tool_name = block.name
                tool_input = block.input
                logger.info("Tool call: %s(%s)", tool_name, json.dumps(tool_input)[:200])

                try:
                    result = self._execute_tool(tool_name, tool_input)

                    # Track metrics
                    if tool_name == "write_strategy_code":
                        strategies_written += 1
                    elif tool_name == "deploy_strategy":
                        strategies_deployed += 1
                    elif tool_name == "retire_strategy":
                        strategies_retired += 1

                except Exception as e:
                    result = f"Error: {e}"
                    logger.exception("Tool execution failed: %s", tool_name)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result) if not isinstance(result, str) else result,
                })

            messages.append({"role": "user", "content": tool_results})

        # Publish summary event
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        self.eb.put_event(Event(
            type=EventType.STRATEGY_EVOLVED,
            source="evolve_engine",
            payload={
                "agent_turns": turns,
                "strategies_written": strategies_written,
                "strategies_deployed": strategies_deployed,
                "strategies_retired": strategies_retired,
                "duration_seconds": round(elapsed, 1),
            },
        ))

        # Archive conversation to S3
        self._archive_conversation(messages, turns)

        logger.info(
            "=== Evolution Complete: %d turns | %d written | %d deployed | "
            "%d retired | %.1fs ===",
            turns, strategies_written, strategies_deployed,
            strategies_retired, elapsed,
        )

    def _execute_tool(self, name: str, input: dict) -> str:
        """Route a tool call to its implementation."""
        match name:
            case "list_strategies":
                return self._tool_list_strategies()
            case "read_strategy_code":
                return self._tool_read_strategy_code(input["strategy_id"])
            case "read_performance_data":
                return self._tool_read_performance(input["strategy_id"])
            case "read_market_data":
                return self._tool_read_market_data(
                    input["symbol"], input.get("period", "6mo")
                )
            case "write_strategy_code":
                return self._tool_write_strategy(input)
            case "run_backtest":
                return self._tool_run_backtest(
                    input["strategy_id"],
                    input["symbol"],
                    input.get("period", "1y"),
                )
            case "deploy_strategy":
                return self._tool_deploy(
                    input["strategy_id"],
                    input.get("backtest_composite", 0.0),
                )
            case "retire_strategy":
                return self._tool_retire(
                    input["strategy_id"], input.get("reason", "")
                )
            case _:
                return f"Unknown tool: {name}"

    # ── Tool implementations ──

    def _tool_list_strategies(self) -> str:
        strategies = self.memory.get_active_strategies()
        if not strategies:
            return "No active strategies found. The population is empty — you should create new ones."

        lines = []
        for s in strategies:
            fitness = s.get("fitness", {})
            lines.append(
                f"- {s['id']} | {s.get('name', 'unnamed')} | "
                f"type={s.get('strategy_type', '?')} | "
                f"gen={s.get('generation', 0)} | "
                f"trades={s.get('total_trades', 0)} | "
                f"pnl=${s.get('total_pnl', 0):.2f} | "
                f"composite={fitness.get('composite', 0):.4f} | "
                f"sharpe={fitness.get('sharpe_ratio', 0):.2f} | "
                f"dd={fitness.get('max_drawdown', 0):.1%} | "
                f"wr={fitness.get('win_rate', 0):.0%}"
            )
        return f"Active strategies ({len(strategies)}):\n" + "\n".join(lines)

    def _tool_read_strategy_code(self, strategy_id: str) -> str:
        # Check candidates first (written this session)
        if strategy_id in self._candidates:
            return self._candidates[strategy_id].get("source_code", "No source code found")

        # Try S3
        try:
            key = f"strategies/{strategy_id}/strategy.py"
            response = self._s3.get_object(Bucket=self._bucket, Key=key)
            return response["Body"].read().decode("utf-8")
        except Exception:
            return f"No source code found for strategy {strategy_id}. It may use genome-based parameters instead of custom code."

    def _tool_read_performance(self, strategy_id: str) -> str:
        trades = self.memory.get_strategy_trades(strategy_id)
        if not trades:
            return f"No trades found for strategy {strategy_id}"

        closed = [t for t in trades if t.get("status") == "closed"]
        open_trades = [t for t in trades if t.get("status") == "open"]

        lines = [f"Strategy {strategy_id}: {len(closed)} closed, {len(open_trades)} open trades"]
        total_pnl = sum(t.get("pnl", 0) for t in closed)
        lines.append(f"Total P&L: ${total_pnl:,.2f}")

        if closed:
            wins = [t for t in closed if t.get("pnl", 0) > 0]
            losses = [t for t in closed if t.get("pnl", 0) < 0]
            lines.append(f"Win rate: {len(wins)}/{len(closed)} ({len(wins)/len(closed):.0%})")
            lines.append(f"\nRecent trades (last 15):")
            for t in closed[-15:]:
                lines.append(
                    f"  {t.get('symbol')} {t.get('side')} "
                    f"entry={t.get('entry_price', 0):.2f} "
                    f"exit={t.get('exit_price', 0):.2f} "
                    f"pnl=${t.get('pnl', 0):+.2f}"
                )

        return "\n".join(lines)

    def _tool_read_market_data(self, symbol: str, period: str) -> str:
        df = self.market.get_historical(symbol, period=period)
        if df is None or df.empty:
            return f"No data available for {symbol}"

        close = df["Close"]
        volume = df["Volume"]

        # Basic stats
        return (
            f"{symbol} ({period}, {len(df)} bars):\n"
            f"  Latest: ${close.iloc[-1]:.2f}\n"
            f"  Range: ${close.min():.2f} — ${close.max():.2f}\n"
            f"  Mean: ${close.mean():.2f}, Std: ${close.std():.2f}\n"
            f"  Return: {((close.iloc[-1] / close.iloc[0]) - 1):.1%}\n"
            f"  Avg Volume: {volume.mean():,.0f}\n"
            f"  Recent 5 closes: {[round(float(x), 2) for x in close.tail(5).values]}"
        )

    def _tool_write_strategy(self, input: dict) -> str:
        source_code = input["source_code"]
        strategy_name = input["strategy_name"]

        # Validate by loading in sandbox
        strategy_cls = load_strategy_from_source(source_code)
        if strategy_cls is None:
            return "FAILED: Could not load strategy — syntax error or missing CustomStrategy subclass."

        errors = validate_strategy(strategy_cls)
        if errors:
            return f"FAILED validation:\n" + "\n".join(f"  - {e}" for e in errors)

        # Assign ID and store
        strategy_id = f"custom-{uuid.uuid4().hex[:8]}"
        self._candidates[strategy_id] = {
            "id": strategy_id,
            "name": strategy_name,
            "strategy_type": input.get("strategy_type", "custom"),
            "description": input.get("description", ""),
            "source_code": source_code,
            "class_name": strategy_cls.__name__,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Upload to S3
        try:
            key = f"strategies/{strategy_id}/strategy.py"
            self._s3.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=source_code.encode("utf-8"),
                ContentType="text/x-python",
            )
        except Exception as e:
            logger.warning("Failed to upload to S3: %s", e)

        return (
            f"Strategy written and validated successfully.\n"
            f"  ID: {strategy_id}\n"
            f"  Name: {strategy_name}\n"
            f"  Class: {strategy_cls.__name__}\n"
            f"Now run backtests to evaluate it before deploying."
        )

    def _tool_run_backtest(
        self, strategy_id: str, symbol: str, period: str
    ) -> str:
        # Load strategy source
        candidate = self._candidates.get(strategy_id)
        if not candidate:
            return f"Strategy {strategy_id} not found. Write it first with write_strategy_code."

        source_code = candidate["source_code"]
        strategy_cls = load_strategy_from_source(source_code)
        if strategy_cls is None:
            return "Failed to load strategy for backtesting."

        # Fetch market data
        df = self.market.get_historical(symbol, period=period)
        if df is None or df.empty:
            return f"No market data for {symbol} ({period})"

        # Run backtest
        result = self.backtester.run(strategy_cls, df, symbol=symbol)

        if result.error:
            return f"Backtest FAILED: {result.error}"

        # Store result in candidate
        if "backtest_results" not in candidate:
            candidate["backtest_results"] = []
        candidate["backtest_results"].append({
            "symbol": symbol,
            "period": period,
            "fitness": result.fitness.model_dump(),
            "num_trades": len(result.trades),
        })

        f = result.fitness
        return (
            f"Backtest Results — {strategy_cls.name} on {symbol} ({period}, {result.total_bars} bars):\n"
            f"  Total Return: ${f.total_return:,.2f}\n"
            f"  Sharpe Ratio: {f.sharpe_ratio:.4f}\n"
            f"  Max Drawdown: {f.max_drawdown:.1%}\n"
            f"  Win Rate: {f.win_rate:.0%}\n"
            f"  Profit Factor: {f.profit_factor:.2f}\n"
            f"  Total Trades: {f.total_trades}\n"
            f"  Composite Score: {f.composite:.4f}\n"
            f"\n{'GOOD — consider deploying.' if f.composite > 0.55 else 'BELOW threshold (0.55) — needs improvement.'}"
        )

    def _tool_deploy(self, strategy_id: str, backtest_composite: float) -> str:
        candidate = self._candidates.get(strategy_id)
        if not candidate:
            return f"Strategy {strategy_id} not found in this session's candidates."

        backtests = candidate.get("backtest_results", [])
        if not backtests:
            return "Cannot deploy without backtesting first. Run run_backtest first."

        avg_composite = sum(b["fitness"]["composite"] for b in backtests) / len(backtests)
        if avg_composite < 0.45:
            return f"Average composite {avg_composite:.4f} is too low. Improve the strategy first."

        # Save to DynamoDB
        self.memory.save_strategy({
            "id": strategy_id,
            "name": candidate["name"],
            "strategy_type": candidate["strategy_type"],
            "genome": {},  # custom strategies use code, not genomes
            "fitness": backtests[-1]["fitness"],
            "generation": 0,
            "is_active": True,
            "created_at": candidate["created_at"],
            "total_trades": 0,
            "total_pnl": 0.0,
        })

        logger.info("Deployed strategy %s (composite=%.4f)", strategy_id, avg_composite)
        return (
            f"Strategy {strategy_id} ({candidate['name']}) deployed to production.\n"
            f"  Avg backtest composite: {avg_composite:.4f}\n"
            f"  Backtested on {len(backtests)} symbol(s)\n"
            f"  It will be used by the trading agents starting next cycle."
        )

    def _tool_retire(self, strategy_id: str, reason: str) -> str:
        self.memory.retire_strategy(strategy_id)
        logger.info("Retired strategy %s: %s", strategy_id, reason)
        return f"Strategy {strategy_id} retired. Reason: {reason}"

    # ── Archival ──

    def _archive_conversation(self, messages: list[dict], turns: int) -> None:
        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            key = f"evolution/conversations/{timestamp}-{turns}turns.json"

            # Serialize, handling content blocks
            serializable = []
            for msg in messages:
                entry = {"role": msg["role"]}
                content = msg.get("content")
                if isinstance(content, str):
                    entry["content"] = content
                elif isinstance(content, list):
                    entry["content"] = [
                        block if isinstance(block, dict) else
                        {"type": block.type, "text": getattr(block, "text", "")}
                        if hasattr(block, "type") else str(block)
                        for block in content
                    ]
                else:
                    entry["content"] = str(content)
                serializable.append(entry)

            self._s3.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=json.dumps(serializable, indent=2, default=str),
                ContentType="application/json",
            )
            logger.info("Archived conversation to s3://%s/%s", self._bucket, key)
        except Exception:
            logger.exception("Failed to archive conversation")


def main() -> None:
    """Entry point for Fargate task."""
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)-8s] %(name)-30s %(message)s",
    )
    engine = EvolutionEngine()
    engine.run()


if __name__ == "__main__":
    main()
