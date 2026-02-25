# Risk Management Skill

## Purpose
Protect the portfolio from catastrophic losses through position-level
and portfolio-level risk controls.

## Risk Checks (Applied to Every Trade Proposal)
1. **Circuit breaker** — halt all trading after 3% daily loss or 10% drawdown
2. **Max trades per day** — prevent overtrading (default: 20)
3. **Cooldown timer** — minimum 60 seconds between trades
4. **Position size cap** — no single position > 15% of portfolio
5. **Cumulative exposure** — combined same-symbol exposure capped
6. **Max open positions** — default 10 concurrent positions
7. **Mandatory stop loss** — every trade must have a stop loss
8. **Risk/reward ratio** — minimum 1.5:1 reward-to-risk
9. **Time restrictions** — no trading in first/last 15min of session

## Portfolio-Level Monitoring (Heartbeat)
- Total exposure percentage
- Positions missing stop losses
- Daily P&L vs circuit breaker threshold
- Correlation risk across positions

## Decision Output
Each proposal receives APPROVED or REJECTED with detailed reasoning.
