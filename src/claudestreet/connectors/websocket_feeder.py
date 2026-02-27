"""WebSocket feeder — real-time market data via Alpaca WebSocket.

Runs as a Fargate task. Connects to Alpaca's real-time data WebSocket
and writes bars/trades to Kinesis Data Streams for consumption by
the Sentinel agent.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone

import boto3

logger = logging.getLogger(__name__)

_KINESIS_STREAM = os.environ.get("KINESIS_STREAM", "claudestreet-market-data")
_REGION = os.environ.get("AWS_REGION", "us-east-1")


class WebSocketFeeder:
    """Connects to Alpaca WebSocket and forwards data to Kinesis."""

    def __init__(self) -> None:
        self._kinesis = boto3.client("kinesis", region_name=_REGION)
        self._running = True
        self._api_key = os.environ.get("ALPACA_API_KEY", "")
        self._secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
        self._data_url = os.environ.get(
            "ALPACA_DATA_URL", "wss://stream.data.alpaca.markets/v2/iex"
        )
        self._watchlist = json.loads(
            os.environ.get("WATCHLIST", '["AAPL","MSFT","GOOG","AMZN","NVDA"]')
        )

        # Graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame) -> None:
        logger.info("Received shutdown signal %d", signum)
        self._running = False

    def run(self) -> None:
        """Main loop: connect to WebSocket and forward to Kinesis."""
        logger.info("WebSocket feeder starting for symbols: %s", self._watchlist)

        try:
            import websocket

            ws = websocket.WebSocketApp(
                self._data_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )

            while self._running:
                try:
                    ws.run_forever(ping_interval=30, ping_timeout=10)
                except Exception:
                    logger.exception("WebSocket connection lost, reconnecting in 5s")
                    time.sleep(5)

        except ImportError:
            logger.error("websocket-client not installed, falling back to polling mode")
            self._polling_fallback()

    def _on_open(self, ws) -> None:
        """Authenticate on connection open. Subscription happens after auth confirmation."""
        auth_msg = {
            "action": "auth",
            "key": self._api_key,
            "secret": self._secret_key,
        }
        ws.send(json.dumps(auth_msg))
        logger.info("Auth message sent, waiting for confirmation")

    def _on_message(self, ws, message: str) -> None:
        """Handle messages: auth confirmation triggers subscribe, bars go to Kinesis."""
        try:
            data = json.loads(message)
            if not isinstance(data, list):
                data = [data]

            # Check for auth confirmation — subscribe after successful auth
            for item in data:
                if isinstance(item, dict) and item.get("T") == "success" and item.get("msg") == "authenticated":
                    sub_msg = {
                        "action": "subscribe",
                        "bars": self._watchlist,
                    }
                    ws.send(json.dumps(sub_msg))
                    logger.info("Authenticated — subscribed to bars for %d symbols", len(self._watchlist))
                    return

            records = []
            for item in data:
                msg_type = item.get("T", "")

                if msg_type == "b":  # bar data
                    record = {
                        "type": "bar",
                        "symbol": item.get("S", ""),
                        "open": item.get("o", 0),
                        "high": item.get("h", 0),
                        "low": item.get("l", 0),
                        "close": item.get("c", 0),
                        "volume": item.get("v", 0),
                        "timestamp": item.get("t", datetime.now(timezone.utc).isoformat()),
                        "received_at": datetime.now(timezone.utc).isoformat(),
                    }
                    records.append({
                        "Data": json.dumps(record).encode("utf-8"),
                        "PartitionKey": record["symbol"],
                    })

            if records:
                self._kinesis.put_records(
                    StreamName=_KINESIS_STREAM,
                    Records=records,
                )
                logger.debug("Forwarded %d records to Kinesis", len(records))

        except Exception:
            logger.exception("Failed to process WebSocket message")

    def _on_error(self, ws, error) -> None:
        logger.error("WebSocket error: %s", error)

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        logger.info("WebSocket closed: %s %s", close_status_code, close_msg)

    def _polling_fallback(self) -> None:
        """Fallback: poll yfinance and push to Kinesis."""
        from claudestreet.connectors.market_data import MarketDataConnector
        connector = MarketDataConnector()

        while self._running:
            try:
                ticks = connector.get_batch_ticks(self._watchlist)
                records = []
                for symbol, tick in ticks.items():
                    record = {
                        "type": "tick",
                        "symbol": symbol,
                        "close": tick.price,
                        "volume": tick.volume,
                        "change_pct": tick.change_pct,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    records.append({
                        "Data": json.dumps(record).encode("utf-8"),
                        "PartitionKey": symbol,
                    })

                if records:
                    self._kinesis.put_records(
                        StreamName=_KINESIS_STREAM,
                        Records=records,
                    )
                    logger.info("Polled %d ticks → Kinesis", len(records))

            except Exception:
                logger.exception("Polling fallback error")

            time.sleep(60)  # poll every minute


def main() -> None:
    """Entry point for Fargate task."""
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)-8s] %(name)-30s %(message)s",
    )
    feeder = WebSocketFeeder()
    feeder.run()


if __name__ == "__main__":
    main()
