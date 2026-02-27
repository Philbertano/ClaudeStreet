# ClaudeStreet

AWS-native serverless AgentSwarm for autonomous stock trading. Event-driven architecture using Lambda, DynamoDB, EventBridge, Kinesis, Step Functions, and ECS Fargate.

## Quick Reference

```bash
source .venv/bin/activate   # Always activate venv first
make test                   # Run all tests (must pass: 21+)
make synth                  # CDK synth (must produce 4 stacks, 0 errors)
make lint                   # Ruff check + format
make deploy                 # Deploy all stacks to AWS
```

## Project Structure

```
src/claudestreet/
  agents/          # Event-driven agents (Lambda handlers)
    sentinel.py    # Market data scanner (heartbeat polls yfinance)
    analyst.py     # Technical + sentiment analysis
    strategist.py  # Strategy selection (Thompson Sampling) + trade proposals (Kelly sizing)
    risk_guard.py  # Risk validation, circuit breakers, position reconciliation
    executor.py    # Order execution via Alpaca broker API
    chronicler.py  # Trade journaling, signal attribution, online fitness updates
    base.py        # BaseAgent ABC
  connectors/
    broker.py      # Alpaca broker API (with tenacity retry)
    market_data.py # yfinance + correlation matrix (with tenacity retry)
    websocket_feeder.py  # Fargate: Alpaca WebSocket -> Kinesis
  core/
    config.py      # SSM + Secrets Manager config loader
    event_bus.py   # EventBridge publish/parse
    memory.py      # DynamoDB state store (trades, strategies, snapshots, events)
  evolution/       # Genetic algorithm: DNA, mutation, crossover, fitness, population
  evolve_engine/   # Claude Agent SDK meta-agent (Fargate task)
    engine.py      # Autonomous loop: review -> analyze -> write code -> backtest -> deploy
    backtest.py    # Walk-forward backtesting with slippage/fees
    sandbox.py     # Strategy code loader + validator
    strategy_template.py  # CustomStrategy base class
    tools.py       # Tool definitions for Claude agent
  handlers/        # Lambda entry points (one per agent + stream_processor + dlq_replayer)
    base.py        # Handler factory: SQS unwrap -> agent.process/heartbeat -> EventBridge publish
  models/
    events.py      # Event, EventType, all payload models (Pydantic)
    strategy.py    # Strategy, StrategyGenome, StrategyFitness
    order_state.py # Order lifecycle state machine
    trade.py       # Trade model
  skills/
    technical_analysis.py  # RSI, StochRSI, MACD, Bollinger, MFI, A/D, ATR, relative strength
    sentiment_analysis.py  # Claude Haiku LLM scoring + keyword fallback + DynamoDB cache
    regime_detection.py    # Market regime classifier (trending/mean-reverting/high-vol)
    change_detection.py    # CUSUM change-point detection on PnL streams

cdk/               # AWS CDK infrastructure (Python)
  app.py           # CDK app entry point
  stacks/
    core_stack.py           # DynamoDB tables, EventBridge, S3, Secrets, KMS, Kinesis
    agents_stack.py         # Lambda functions, SQS queues, DLQ, Fargate services
    order_workflow_stack.py # Step Functions express workflow
    monitoring_stack.py     # CloudWatch dashboards + alarms

tests/unit/        # Pytest unit tests
```

## Architecture: Event Flow

```
Sentinel (heartbeat) -> MARKET_TICK
  -> Analyst -> ANALYSIS_COMPLETE
    -> Strategist (Thompson Sampling + Kelly) -> TRADE_PROPOSED
      -> RiskGuard (validates) -> RISK_APPROVED / RISK_REJECTED
        -> Executor (Alpaca API) -> TRADE_EXECUTED
          -> Chronicler (journal + attribution + fitness update)

Evolution Engine (Fargate, scheduled):
  Claude Agent SDK loop -> writes strategy code -> backtests -> deploys/retires
```

## Event Types (EventType enum in models/events.py)

MARKET_TICK, ANALYSIS_COMPLETE, TRADE_PROPOSED, RISK_APPROVED, RISK_REJECTED,
TRADE_EXECUTED, RISK_ALERT, STRATEGY_EVOLVED, REGIME_CHANGE, STRATEGY_RETIRED

## DynamoDB Tables

| Table | PK | SK | GSIs |
|-------|----|----|------|
| claudestreet-trades | trade_id | - | symbol-status-index, strategy-index |
| claudestreet-strategies | strategy_id | - | active-index (is_active, total_pnl) |
| claudestreet-snapshots | date | timestamp | - |
| claudestreet-events | event_id | timestamp | correlation-index |
| claudestreet-attributions | correlation_id | timestamp | strategy-index, signal-type-index |

## Key Design Decisions

- **All DynamoDB writes are idempotent** via ConditionExpression (prevents Lambda retry duplicates)
- **SQS between EventBridge and Lambda** for retry/DLQ (maxReceiveCount=3)
- **Order state machine** tracks lifecycle: PENDING -> SUBMITTED -> FILLED etc.
- **Thompson Sampling** for strategy selection (Beta distribution, exploration/exploitation)
- **Half-Kelly Criterion** for position sizing, scaled by signal confidence
- **US Eastern timezone** for market hours and restricted trading windows
- **Walk-forward backtesting** with slippage (5bps) and commission ($1/trade)
- **Regime-aware**: strategies tagged with regime preference, filtered before trading

## Verification Checklist

Every change MUST pass these before committing:

1. `make test` — all tests pass (currently 21, should only increase)
2. `make synth` — all 4 CDK stacks synthesize with 0 errors
3. `make lint` — no ruff errors (run `make format` to auto-fix)

## Autonomous Improvement Workflow

When asked to improve the system or fix bugs, follow this loop:

### 1. Discover Issues
- Run `make test` and `make synth` to find failures
- Use the code-review agent to scan for bugs, logic errors, security issues
- Check for missing test coverage (agents, handlers, connectors, skills have minimal tests)
- Look for TODO/FIXME/HACK comments in source
- Verify DynamoDB access patterns match actual queries (GSI key schema vs code)
- Check that all EventType variants have matching EventBridge rules in agents_stack.py

### 2. Fix Issues
- Fix the root cause, not symptoms
- Keep changes minimal and focused
- Maintain idempotency on all DynamoDB writes
- All times related to market hours must use US Eastern (ZoneInfo("America/New_York"))
- All broker/market API calls must use tenacity retry decorators
- All Pydantic models must stay backwards-compatible (add fields with defaults)

### 3. Add Tests
- Every bug fix should include a regression test
- Test files go in `tests/unit/test_<module>.py`
- Use `moto` for AWS service mocks (DynamoDB, EventBridge, S3, SQS, SSM, Secrets Manager)
- Test edge cases: empty data, API failures, duplicate events, Lambda retries
- Test the full event chain: signal -> analysis -> proposal -> risk check -> execution

### 4. Verify
- `make test` — all pass
- `make synth` — 4 stacks, 0 errors
- Review the diff before committing

### 5. Repeat
- After fixing, re-run discovery to find new issues exposed by the changes
- Continue until `make test` and `make synth` are clean AND code review finds no high-confidence bugs

## Known Gaps (prioritized)

These are areas that still need work, in rough priority order:

1. **Test coverage is thin** — only 21 tests for 48 source files. Critical gaps:
   - No tests for: handlers/base.py, memory.py, event_bus.py, strategist, executor, chronicler
   - No integration tests for the full event chain
   - No tests for DynamoDB conditional write failure paths
2. **CDK: EventBridge scheduled rules target default bus** — heartbeat rules in agents_stack.py should target the custom event bus, not the default bus
3. **CDK: Step Functions approval state** — order_workflow_stack.py may have a pass-through approval that doesn't actually validate
4. **No graceful degradation** — if yfinance/Alpaca APIs are down, agents crash instead of emitting health alerts
5. **Chronicler attribution** — needs to handle missing correlation_id gracefully
6. **Sector concentration check** — referenced in risk_guard plan but not implemented

## Patterns to Follow

- **Agents**: Inherit from BaseAgent, implement process() and optionally heartbeat()
- **Handlers**: Use `create_handler(AgentClass)` factory in handlers/base.py
- **Events**: Define payload as Pydantic model in models/events.py, add EventType enum value
- **DynamoDB writes**: Always use ConditionExpression for idempotency
- **External API calls**: Always wrap with @retry from tenacity (3 attempts, exponential backoff)
- **Config**: Use self.config.get("key", default) — loaded from SSM/Secrets Manager
- **Logging**: Use module-level `logger = logging.getLogger(__name__)`

## Do NOT

- Remove existing tests or lower test standards
- Change DynamoDB table schemas without updating both core_stack.py AND memory.py
- Add `async` to functions called from synchronous Lambda handlers
- Use UTC for market-hour-related time comparisons (use US Eastern)
- Skip ConditionExpression on DynamoDB writes (Lambda retries will cause duplicates)
- Commit without running `make test` and `make synth`
