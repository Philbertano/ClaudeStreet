# Technical Analysis Skill

## Purpose
Compute standard technical indicators on OHLCV market data and generate
buy/hold/sell signals based on multi-indicator consensus.

## Indicators Computed
- **RSI** (Relative Strength Index) — momentum oscillator (oversold < 30, overbought > 70)
- **MACD** (Moving Average Convergence Divergence) — trend and momentum via histogram crossovers
- **Bollinger Bands** — volatility envelope using %B position
- **EMA Crossover** — fast/slow exponential moving average trend
- **Volume Analysis** — spike detection relative to 20-day average
- **ATR** (Average True Range) — volatility for position sizing
- **SMA 50/200** — long-term trend (golden cross / death cross)

## Signal Evaluation
Each indicator contributes a weighted score from -1 (bearish) to +1 (bullish).
The composite score maps to: strong_buy > 0.6, buy > 0.2, hold, sell < -0.2, strong_sell < -0.6.
High volume amplifies the signal by 1.2x.

## Parameters (Tunable by Evolution)
All indicator periods and thresholds are part of the strategy genome
and can be evolved by the genetic algorithm for optimal performance.
