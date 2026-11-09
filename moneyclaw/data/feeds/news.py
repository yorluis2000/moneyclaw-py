"""Financial news headlines from RSS feeds."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from time import mktime

import feedparser
import httpx
import structlog

log = structlog.get_logger()

DEFAULT_FEEDS = {
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cointelegraph": "https://cointelegraph.com/rss",
    "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
}


@dataclass
class NewsItem:
    title: str
    link: str
    source: str
    published: datetime | None = None
    summary: str = ""


class NewsFeed:
    """Aggregate headlines from a set of RSS feeds."""

    def __init__(self, feeds: dict[str, str] | None = None, timeout: float = 10.0) -> None:
        self._feeds = feeds or dict(DEFAULT_FEEDS)
        self._timeout = timeout

    async def _fetch(self, source: str, url: str) -> list[NewsItem]:
        try:
            async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()
            parsed = await asyncio.to_thread(feedparser.parse, resp.text)
        except Exception:
            log.warning("news_feed.fetch_failed", source=source, exc_info=True)
            return []

        items = []
        for e in parsed.entries:
            ts = e.get("published_parsed") or e.get("updated_parsed")
            items.append(
                NewsItem(
                    title=e.get("title", ""),
                    link=e.get("link", ""),
                    source=source,
                    published=datetime.fromtimestamp(mktime(ts), tz=UTC) if ts else None,
                    summary=e.get("summary", ""),
                )
            )
        return items

    async def get_headlines(self, limit: int = 20) -> list[NewsItem]:
        """Latest headlines across all feeds, newest first."""
        results = await asyncio.gather(*(self._fetch(s, u) for s, u in self._feeds.items()))
        items = [i for batch in results for i in batch]
        epoch = datetime.min.replace(tzinfo=UTC)
        items.sort(key=lambda i: i.published or epoch, reverse=True)
        return items[:limit]
