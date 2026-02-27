"""Sentiment analysis skill — LLM-powered news sentiment scoring.

Uses Claude Haiku for structured sentiment analysis of financial headlines.
Falls back to keyword-based scoring if the API is unavailable.
Results are cached in DynamoDB with 1-hour TTL.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Fallback weighted keyword lists for when LLM is unavailable
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

_SENTIMENT_CACHE_TTL = 3600  # 1 hour in seconds


class SentimentAnalysisSkill:
    """Analyze market sentiment from news feeds using Claude Haiku."""

    def __init__(self, memory=None) -> None:
        self._memory = memory  # DynamoMemory for caching
        self._client = None

    def _get_anthropic_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(
                    api_key=os.environ.get("ANTHROPIC_API_KEY"),
                )
            except Exception:
                logger.warning("Anthropic client unavailable, using keyword fallback")
        return self._client

    def _cache_key(self, symbol: str, headlines: list[str]) -> str:
        content = f"{symbol}:{':'.join(sorted(headlines[:10]))}"
        return f"sentiment-{hashlib.md5(content.encode()).hexdigest()[:12]}"

    def _get_cached(self, cache_key: str) -> dict | None:
        """Check DynamoDB cache for recent sentiment result."""
        if not self._memory:
            return None
        try:
            from boto3.dynamodb.conditions import Key
            response = self._memory._events.query(
                KeyConditionExpression=Key("event_id").eq(cache_key),
                Limit=1,
            )
            items = response.get("Items", [])
            if items:
                item = items[0]
                ttl = int(item.get("ttl", 0))
                if ttl > int(time.time()):
                    return json.loads(item.get("payload_json", "{}"))
        except Exception:
            pass
        return None

    def _set_cached(self, cache_key: str, result: dict) -> None:
        """Cache sentiment result in DynamoDB events table."""
        if not self._memory:
            return
        try:
            from claudestreet.core.memory import _to_decimal
            self._memory._events.put_item(
                Item=_to_decimal({
                    "event_id": cache_key,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": "sentiment_cache",
                    "source": "sentiment_analysis",
                    "payload_json": json.dumps(result),
                    "ttl": int(time.time()) + _SENTIMENT_CACHE_TTL,
                }),
            )
        except Exception:
            logger.debug("Failed to cache sentiment result")

    def fetch_and_score(
        self, symbol: str, feed_urls: list[str] | None = None
    ) -> dict[str, Any]:
        """Fetch news for a symbol and compute sentiment score.

        Uses Claude Haiku for LLM-powered sentiment analysis.
        Falls back to keyword scoring if unavailable.
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
                "confidence": 0.0,
                "article_count": 0,
                "headlines": [],
                "method": "none",
            }

        headlines = [a["title"] for a in articles if a["title"]]

        # Check cache
        cache_key = self._cache_key(symbol, headlines)
        cached = self._get_cached(cache_key)
        if cached:
            logger.debug("Sentiment cache hit for %s", symbol)
            return cached

        # Try LLM-powered analysis
        result = self._llm_score(symbol, articles)
        if result is None:
            # Fallback to keywords
            result = self._keyword_score_articles(symbol, articles)

        # Cache result
        self._set_cached(cache_key, result)
        return result

    def _llm_score(self, symbol: str, articles: list[dict]) -> dict | None:
        """Score sentiment using Claude Haiku API."""
        client = self._get_anthropic_client()
        if client is None:
            return None

        # Batch headlines (max 10)
        headlines = [a["title"] for a in articles[:10] if a["title"]]
        if not headlines:
            return None

        headlines_text = "\n".join(f"- {h}" for h in headlines)

        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Analyze the sentiment of these financial news headlines for {symbol}. "
                        f"Return ONLY valid JSON with this exact format:\n"
                        f'{{"sentiment": <float -1 to 1>, "confidence": <float 0 to 1>, '
                        f'"reasoning": "<brief explanation>"}}\n\n'
                        f"Headlines:\n{headlines_text}"
                    ),
                }],
            )

            text = response.content[0].text.strip()
            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                sentiment = float(parsed.get("sentiment", 0))
                confidence = float(parsed.get("confidence", 0))
                reasoning = parsed.get("reasoning", "")

                return {
                    "symbol": symbol,
                    "sentiment_score": round(max(-1, min(1, sentiment)), 4),
                    "confidence": round(max(0, min(1, confidence)), 4),
                    "reasoning": reasoning,
                    "article_count": len(articles),
                    "headlines": [{"title": h, "score": sentiment} for h in headlines[:5]],
                    "method": "llm",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
        except Exception:
            logger.exception("LLM sentiment analysis failed for %s", symbol)

        return None

    def _keyword_score_articles(
        self, symbol: str, articles: list[dict]
    ) -> dict:
        """Fallback: score articles using keyword matching."""
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

        avg_score = sum(scores) / len(scores) if scores else 0.0

        return {
            "symbol": symbol,
            "sentiment_score": round(avg_score, 4),
            "confidence": round(min(abs(avg_score) * 2, 1.0), 4),
            "article_count": len(articles),
            "headlines": sorted(headlines, key=lambda x: abs(x["score"]), reverse=True)[:5],
            "method": "keyword",
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

        total = bull_score + bear_score
        if total == 0:
            return 0.0
        return (bull_score - bear_score) / total
