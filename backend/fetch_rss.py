"""
backend/fetch_rss.py — PulseIQ RSS Ingestion Module
====================================================
Secondary data source to prevent dependence on a single API.
Uses feedparser to get latest financial news.
"""

import logging
import feedparser
from datetime import datetime, timezone

from backend.database import init_db, insert_articles, article_count

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Fallback/Secondary source
RSS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY,AAPL,MSFT,TSLA",
    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664" # Finance
]

def fetch_rss_articles(feed_url: str) -> list[dict]:
    log.info(f"Fetching RSS feed: {feed_url}")
    feed = feedparser.parse(feed_url)
    articles = []

    for entry in feed.entries:
        title = entry.get("title", "").strip()
        link = entry.get("link", "").strip()
        description = entry.get("summary", "").strip()
        
        if not title or not link:
            continue
            
        published_at = entry.get("published")
        if not published_at:
            published_at = datetime.now(timezone.utc).isoformat()
            
        articles.append({
            "title": title,
            "description": description,
            "source": feed.feed.get("title", "RSS Feed"),
            "url": link,
            "published_at": published_at
        })
        
    return articles

def fetch_and_store_rss() -> int:
    init_db()
    
    all_articles = []
    for feed_url in RSS_FEEDS:
        all_articles.extend(fetch_rss_articles(feed_url))
        
    if not all_articles:
        log.warning("No articles fetched from RSS.")
        return 0
        
    inserted = insert_articles(all_articles)
    log.info(f"RSS Fetch complete — {len(all_articles)} fetched | {inserted} newly inserted | {article_count()} total in DB")
    return inserted

if __name__ == "__main__":
    fetch_and_store_rss()
