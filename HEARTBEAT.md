# ClaudeStreet Heartbeat Checklist

## Every Tick (Sentinel - 60s)
- [ ] Fetch latest prices for watchlist
- [ ] Check for price anomalies (>2σ moves)
- [ ] Monitor volume spikes (>3x average)
- [ ] Scan open positions for stop-loss / take-profit hits

## Every 5 Minutes (Analyst)
- [ ] Update technical indicators for active signals
- [ ] Check RSI extremes (<30 or >70)
- [ ] Detect MACD crossovers
- [ ] Evaluate Bollinger Band breakouts

## Every 15 Minutes (Strategist)
- [ ] Review strategy fitness scores
- [ ] Trigger evolution cycle if enough data
- [ ] Promote top-performing strategies
- [ ] Retire underperforming strategies

## Every 2 Minutes (RiskGuard)
- [ ] Calculate current portfolio exposure
- [ ] Check daily P&L against circuit breakers
- [ ] Verify all positions have stop-losses
- [ ] Monitor correlation risk across positions

## Every 10 Minutes (Chronicler)
- [ ] Snapshot portfolio state
- [ ] Update cumulative performance metrics
- [ ] Log agent activity summary
- [ ] Archive completed event chains
