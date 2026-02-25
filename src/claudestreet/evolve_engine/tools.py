"""Tool definitions for the Claude Agent SDK evolution loop.

Each tool is a function Claude can call to interact with the trading
system: read strategies, write new code, run backtests, and deploy.
The tools define what the autonomous agent CAN do — the system prompt
defines what it SHOULD do.
"""

from __future__ import annotations

TOOLS = [
    {
        "name": "list_strategies",
        "description": (
            "List all active trading strategies with their performance metrics. "
            "Returns strategy ID, name, type, generation, fitness scores, "
            "total trades, and total P&L for each."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "read_strategy_code",
        "description": (
            "Read the Python source code of a specific strategy from S3. "
            "Returns the full source code that implements the CustomStrategy interface."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy_id": {
                    "type": "string",
                    "description": "The ID of the strategy to read",
                },
            },
            "required": ["strategy_id"],
        },
    },
    {
        "name": "read_performance_data",
        "description": (
            "Get detailed performance data for a strategy including its "
            "closed trade history with entry/exit prices and P&L for each trade."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy_id": {
                    "type": "string",
                    "description": "The ID of the strategy to analyze",
                },
            },
            "required": ["strategy_id"],
        },
    },
    {
        "name": "read_market_data",
        "description": (
            "Fetch historical OHLCV (Open, High, Low, Close, Volume) data "
            "for a stock symbol. Returns a summary with latest values, "
            "basic statistics, and recent price action."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g., 'AAPL', 'MSFT')",
                },
                "period": {
                    "type": "string",
                    "description": "Data period: '1mo', '3mo', '6mo', '1y', '2y'",
                    "default": "6mo",
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "write_strategy_code",
        "description": (
            "Write a new trading strategy as Python code. The code MUST define "
            "a class that inherits from CustomStrategy and implements evaluate(). "
            "Available imports: numpy (np), pandas (pd), math. "
            "evaluate() receives a DataFrame with [Open, High, Low, Close, Volume] "
            "and params dict, must return (recommendation, confidence). "
            "Recommendations: 'strong_buy', 'buy', 'hold', 'sell', 'strong_sell'. "
            "Confidence: 0.0 to 1.0."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy_name": {
                    "type": "string",
                    "description": "Human-readable name for the strategy",
                },
                "strategy_type": {
                    "type": "string",
                    "description": "Category: 'momentum', 'mean_reversion', 'trend', 'volatility', 'hybrid'",
                },
                "description": {
                    "type": "string",
                    "description": "Brief description of the strategy logic",
                },
                "source_code": {
                    "type": "string",
                    "description": (
                        "Complete Python source code. Must define a class inheriting "
                        "from CustomStrategy with an evaluate() method."
                    ),
                },
            },
            "required": ["strategy_name", "strategy_type", "description", "source_code"],
        },
    },
    {
        "name": "run_backtest",
        "description": (
            "Backtest a strategy against historical market data. "
            "Simulates trading with stop-loss and take-profit, then "
            "computes fitness metrics: total return, Sharpe ratio, "
            "max drawdown, win rate, profit factor, and composite score. "
            "Use this to validate strategies before deploying."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy_id": {
                    "type": "string",
                    "description": "ID of a previously written strategy to test",
                },
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker to backtest against",
                },
                "period": {
                    "type": "string",
                    "description": "Historical period: '3mo', '6mo', '1y', '2y'",
                    "default": "1y",
                },
            },
            "required": ["strategy_id", "symbol"],
        },
    },
    {
        "name": "deploy_strategy",
        "description": (
            "Deploy a backtested strategy to production. This activates "
            "the strategy in DynamoDB so the Lambda trading agents will "
            "use it for live/paper trading. Only deploy strategies that "
            "have been backtested with good results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy_id": {
                    "type": "string",
                    "description": "ID of the strategy to deploy",
                },
                "backtest_composite": {
                    "type": "number",
                    "description": "Composite fitness score from backtesting (for audit)",
                },
            },
            "required": ["strategy_id"],
        },
    },
    {
        "name": "retire_strategy",
        "description": (
            "Deactivate an underperforming strategy. It will no longer "
            "be used for trading but remains in the database for analysis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy_id": {
                    "type": "string",
                    "description": "ID of the strategy to retire",
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for retirement (logged for audit)",
                },
            },
            "required": ["strategy_id", "reason"],
        },
    },
]
