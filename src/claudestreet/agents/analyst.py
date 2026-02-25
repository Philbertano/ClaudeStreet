"""Analyst agent — technical and fundamental analysis.

Receives signals from Sentinel, runs indicator-based analysis,
and produces buy/hold/sell recommendations for the Strategist.
"""

from __future__ import annotations

import logging

from claudestreet.agents.base import BaseAgent
from claudestreet.connectors.market_data import MarketDataConnector
from claudestreet.models.events import (
    Event, EventType,
    SignalPayload, AnalysisPayload,
)
from claudestreet.skills.technical_analysis import TechnicalAnalysisSkill

logger = logging.getLogger(__name__)


class AnalystAgent(BaseAgent):
    agent_id = "analyst"
    description = "Technical/fundamental analysis on detected signals"

    def process(self, event: Event) -> list[Event]:
        if event.type == EventType.SIGNAL_DETECTED:
            return self._analyze_signal(event)
        if event.type in (EventType.PRICE_ANOMALY, EventType.VOLUME_SPIKE):
            return self._analyze_market_event(event)
        return []

    def heartbeat(self) -> list[Event]:
        """Re-analyze all watchlist symbols."""
        events: list[Event] = []
        for symbol in self.config.get("watchlist", []):
            try:
                result = self._run_full_analysis(symbol, parent_event=None)
                events.extend(result)
            except Exception:
                logger.exception("Heartbeat analysis failed for %s", symbol)
        return events

    def _analyze_signal(self, event: Event) -> list[Event]:
        signal = SignalPayload(**event.payload)

        # SL/TP triggers → immediate close signal
        if signal.signal_type in ("stop_loss", "take_profit"):
            return [self.emit(
                EventType.ANALYSIS_COMPLETE,
                payload=AnalysisPayload(
                    symbol=signal.symbol,
                    technical=signal.indicators,
                    recommendation="sell",
                    confidence=1.0,
                    summary=f"{signal.signal_type} triggered",
                ).model_dump(),
                parent=event,
            )]

        return self._run_full_analysis(signal.symbol, event)

    def _analyze_market_event(self, event: Event) -> list[Event]:
        symbol = event.payload.get("symbol", "")
        return self._run_full_analysis(symbol, event) if symbol else []

    def _run_full_analysis(
        self, symbol: str, parent_event: Event | None
    ) -> list[Event]:
        connector = MarketDataConnector()
        ta_skill = TechnicalAnalysisSkill()

        try:
            hist = connector.get_historical(symbol, period="3mo")
            if hist is None or hist.empty:
                return []

            indicators = ta_skill.compute_indicators(hist)
            recommendation, confidence = ta_skill.evaluate(indicators)

            analysis = AnalysisPayload(
                symbol=symbol,
                technical=indicators,
                recommendation=recommendation,
                confidence=confidence,
                summary=ta_skill.summarize(symbol, indicators, recommendation),
            )

            if recommendation in ("strong_buy", "buy", "strong_sell", "sell"):
                return [self.emit(
                    EventType.ANALYSIS_COMPLETE,
                    payload=analysis.model_dump(),
                    parent=parent_event,
                )]
        except Exception:
            logger.exception("Analysis failed for %s", symbol)

        return []
