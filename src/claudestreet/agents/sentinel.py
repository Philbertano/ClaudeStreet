"""Sentinel agent — the eyes and ears of the swarm.

Monitors market data, detects price anomalies, volume spikes,
and stop-loss/take-profit triggers on open positions.
"""

from __future__ import annotations

import logging

from claudestreet.agents.base import BaseAgent
from claudestreet.connectors.market_data import MarketDataConnector
from claudestreet.models.events import (
    Event, EventType, EventPriority,
    MarketTickPayload, SignalPayload,
)

logger = logging.getLogger(__name__)


class SentinelAgent(BaseAgent):
    agent_id = "sentinel"
    description = "Market data monitor — detects anomalies and triggers"

    def __init__(self, memory, config) -> None:
        super().__init__(memory, config)
        self._price_history: dict[str, list[float]] = {}
        self._volume_history: dict[str, list[int]] = {}
        self._lookback = 50

    def process(self, event: Event) -> list[Event]:
        if event.type == EventType.MARKET_TICK:
            return self._process_tick(event)
        return []

    def heartbeat(self) -> list[Event]:
        """Fetch latest data for watchlist, scan for signals, and check cross-asset."""
        connector = MarketDataConnector()
        events: list[Event] = []
        watchlist = self.config.get("watchlist", [])

        for symbol in watchlist:
            try:
                tick = connector.get_latest_tick(symbol)
                if tick:
                    tick_event = Event(
                        type=EventType.MARKET_TICK,
                        source=self.agent_id,
                        payload=tick.model_dump(mode="json"),
                    )
                    events.extend(self._process_tick(tick_event))
            except Exception:
                logger.exception("Failed to fetch data for %s", symbol)

        # Cross-asset scan
        events.extend(self._scan_cross_asset(connector))

        return events

    def _scan_cross_asset(self, connector: MarketDataConnector) -> list[Event]:
        """Scan cross-asset signals: VIX spike, yield signals, dollar strength."""
        events: list[Event] = []
        try:
            signals = connector.get_cross_asset_signals()
            if not signals:
                return events

            # VIX spike: emit alert if VIX > 25
            vix = signals.get("vix", 0)
            if vix > 25:
                events.append(self.emit(
                    EventType.RISK_ALERT,
                    payload={
                        "alert_type": "vix_spike",
                        "vix": vix,
                        "vix_change": signals.get("vix_change", 0),
                        "cross_asset_signals": signals,
                    },
                    priority=EventPriority.HIGH,
                ))

            # VIX term structure inversion (backwardation = fear)
            vix_term = signals.get("vix_term_structure", 0)
            if vix_term > 1.05:  # VIX > VIX3M by 5%
                events.append(self.emit(
                    EventType.RISK_ALERT,
                    payload={
                        "alert_type": "vix_backwardation",
                        "vix_term_structure": vix_term,
                        "message": "VIX in backwardation — elevated short-term fear",
                    },
                    priority=EventPriority.HIGH,
                ))

        except Exception:
            logger.debug("Cross-asset scan failed (non-critical)")

        return events

    def _process_tick(self, event: Event) -> list[Event]:
        tick = MarketTickPayload(**event.payload)
        symbol = tick.symbol
        events: list[Event] = []

        # Track history
        self._price_history.setdefault(symbol, []).append(tick.price)
        self._volume_history.setdefault(symbol, []).append(tick.volume)
        self._price_history[symbol] = self._price_history[symbol][-self._lookback:]
        self._volume_history[symbol] = self._volume_history[symbol][-self._lookback:]

        prices = self._price_history[symbol]
        volumes = self._volume_history[symbol]

        # Price anomaly (>2σ move)
        if len(prices) >= 20:
            window = prices[-20:]
            mean = sum(window) / len(window)
            variance = sum((p - mean) ** 2 for p in window) / len(window)
            std = variance ** 0.5
            if std > 0:
                z_score = (tick.price - mean) / std
                if abs(z_score) > 2.0:
                    events.append(self.emit(
                        EventType.PRICE_ANOMALY,
                        payload={
                            "symbol": symbol,
                            "price": tick.price,
                            "z_score": round(z_score, 2),
                            "direction": "up" if z_score > 0 else "down",
                        },
                        parent=event,
                    ))

        # Volume spike (>2.5x average)
        mult = self.config.get("volume_spike_mult", 2.5)
        if len(volumes) >= 10:
            avg_vol = sum(volumes[-10:]) / 10
            if avg_vol > 0 and tick.volume > avg_vol * mult:
                events.append(self.emit(
                    EventType.VOLUME_SPIKE,
                    payload={
                        "symbol": symbol,
                        "volume": tick.volume,
                        "avg_volume": int(avg_vol),
                        "spike_ratio": round(tick.volume / avg_vol, 2),
                    },
                    parent=event,
                ))

        # Check open positions for SL/TP
        open_trades = self.memory.get_open_trades(symbol)
        for trade in open_trades:
            stop_loss = trade.get("stop_loss", 0)
            take_profit = trade.get("take_profit", 0)
            side = trade.get("side", "buy")

            triggered = False
            trigger_type = ""
            if side == "buy":
                if stop_loss > 0 and tick.price <= stop_loss:
                    triggered, trigger_type = True, "stop_loss"
                elif take_profit > 0 and tick.price >= take_profit:
                    triggered, trigger_type = True, "take_profit"
            else:
                if stop_loss > 0 and tick.price >= stop_loss:
                    triggered, trigger_type = True, "stop_loss"
                elif take_profit > 0 and tick.price <= take_profit:
                    triggered, trigger_type = True, "take_profit"

            if triggered:
                events.append(Event(
                    type=EventType.SIGNAL_DETECTED,
                    source=self.agent_id,
                    priority=EventPriority.HIGH,
                    payload=SignalPayload(
                        symbol=symbol,
                        signal_type=trigger_type,
                        strength=1.0,
                        indicators={"trigger_price": tick.price},
                        timeframe="tick",
                    ).model_dump(),
                    correlation_id=event.correlation_id or event.id,
                    parent_id=event.id,
                ))

        return events
