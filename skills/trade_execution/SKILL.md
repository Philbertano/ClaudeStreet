# Trade Execution Skill

## Purpose
Execute approved trades via broker API with proper order management,
fill tracking, and error handling.

## Execution Modes
- **Paper Trading** (default) — simulated fills at entry price, no real money
- **Live Trading** — submits limit orders via Alpaca API

## Order Flow
1. Receive RISK_APPROVED event with full trade specification
2. Submit limit order to broker (or simulate in paper mode)
3. Record trade in Memory (SQLite trades table)
4. Publish TRADE_EXECUTED event with fill details
5. On failure, publish TRADE_FAILED event

## Position Management
- Track all open positions in memory
- Monitor for stop-loss and take-profit triggers via Sentinel
- Close positions when SL/TP events arrive

## Safety
- Paper trading is the default — live trading requires explicit configuration
- All executions are logged with full audit trail
- Failed orders are retried once then dead-lettered
