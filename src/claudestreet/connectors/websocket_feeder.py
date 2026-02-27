"""Market data feeder — yfinance polling to Kinesis.

Runs as a Fargate task. Polls yfinance for market data at regular
intervals and writes ticks to Kinesis Data Streams for consumption
by the Sentinel agent. No broker credentials required.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from datetime import datetime, timezone

import boto3

logger = logging.getLogger(__name__)

_KINESIS_STREAM = os.environ.get("KINESIS_STREAM", "claudestreet-market-data")
_REGION = os.environ.get("AWS_REGION", "us-east-1")
_POLL_INTERVAL_SECS = int(os.environ.get("POLL_INTERVAL_SECS", "60"))


class MarketDataFeeder:
    """Polls yfinance for market data and forwards to Kinesis."""

    def __init__(self) -> None:
        self._kinesis = boto3.client("kinesis", region_name=_REGION)
        self._running = True
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
        """Main loop: poll yfinance and forward to Kinesis."""
        logger.info("Market data feeder starting for symbols: %s", self._watchlist)

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
                    logger.info("Polled %d ticks -> Kinesis", len(records))

            except Exception:
                logger.exception("Polling error")

            time.sleep(_POLL_INTERVAL_SECS)


# Keep old class name as alias for backwards compatibility
WebSocketFeeder = MarketDataFeeder


def main() -> None:
    """Entry point for Fargate task."""
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)-8s] %(name)-30s %(message)s",
    )
    feeder = MarketDataFeeder()
    feeder.run()


if __name__ == "__main__":
    main()
