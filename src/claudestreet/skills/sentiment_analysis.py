"""Sentiment analysis skill — news and market sentiment scoring.

Fetches RSS news feeds and scores sentiment using keyword analysis.
Can be extended with LLM-based sentiment analysis for deeper
understanding of market narratives.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Weighted keyword lists for financial sentiment
BULLISH_KEYWORDS = {
    "surge": 0.8, "rally": 0.8, "soar": 0.9, "jump": 0.6,
    "beat": 0.6, "exceed": 0.6, "upgrade": 0.7, "bullish": 0.9,
    "growth": 0.5, "profit": 0.5, "gain": 0.5, "record high": 0.8,
    "breakthrough": 0.7, "outperform": 0.7, "buy": 0.4,
    "strong": 0.4, "positive": 0.4, "optimistic": 0.6,
}

BEARISH_KEYWORDS = {
    "crash": 0.9, "plunge": 0.8, "tumble": 0.7, "drop": 0.6,
    "miss": 0.6, "downgrade": 0.7, "bearish": 0.9, "loss": 0.5,
    "decline": 0.5, "fall": 0.5, "record low": 0.8, "sell": 0.4,
    "weak": 0.4, "negative": 0.4, "pessimistic": 0.6,
    "recession": 0.8, "layoff": 0.6, "bankrupt": 0.9,
    "investigation": 0.5, "lawsuit": 0.5, "fraud": 0.8,
}


class SentimentAnalysisSkill:
    """Analyze market sentiment from news feeds."""

    async def fetch_and_score(
        self, symbol: str, feed_urls: list[str] | None = None
    ) -> dict[str, Any]:
        """Fetch news for a symbol and compute sentiment score.

        Returns:
            Dict with sentiment_score (-1 to 1), article_count,
            and top headlines.
        """
        import feedparser

        default_feeds = [
            f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}",
        ]
        urls = feed_urls or default_feeds

        articles: list[dict] = []
        for url in urls:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:20]:
                    articles.append({
                        "title": entry.get("title", ""),
                        "summary": entry.get("summary", ""),
                        "published": entry.get("published", ""),
                        "link": entry.get("link", ""),
                    })
            except Exception:
                logger.exception("Failed to fetch feed: %s", url)

        if not articles:
            return {
                "symbol": symbol,
                "sentiment_score": 0.0,
                "article_count": 0,
                "headlines": [],
            }

        # Score each article
        scores: list[float] = []
        headlines: list[dict] = []
        for article in articles:
            text = f"{article['title']} {article['summary']}".lower()
            score = self._keyword_score(text)
            scores.append(score)
            headlines.append({
                "title": article["title"],
                "score": round(score, 3),
            })

        avg_score = sum(scores) / len(scores)

        return {
            "symbol": symbol,
            "sentiment_score": round(avg_score, 4),
            "article_count": len(articles),
            "headlines": sorted(headlines, key=lambda x: abs(x["score"]), reverse=True)[:5],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _keyword_score(self, text: str) -> float:
        """Score text based on bullish/bearish keyword presence."""
        bull_score = 0.0
        bear_score = 0.0
        matches = 0

        for keyword, weight in BULLISH_KEYWORDS.items():
            count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text))
            if count > 0:
                bull_score += weight * count
                matches += count

        for keyword, weight in BEARISH_KEYWORDS.items():
            count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text))
            if count > 0:
                bear_score += weight * count
                matches += count

        if matches == 0:
            return 0.0

        # Normalize to [-1, 1]
        total = bull_score + bear_score
        if total == 0:
            return 0.0
        return (bull_score - bear_score) / total
