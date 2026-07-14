"""
backend/fetch_news.py — PulseIQ News Ingestion Module
======================================================
Fetches financial news from the NewsAPI /everything endpoint,
normalises the payload, and persists articles to SQLite via database.py.

Usage (standalone):
    python -m backend.fetch_news
"""

import os
import logging
from datetime import datetime, timedelta, timezone

import requests
from rapidfuzz import fuzz, process
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.database import init_db, insert_articles, article_count, fetch_all_articles
from config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

NEWSAPI_URL  = "https://newsapi.org/v2/everything"
DEFAULT_QUERY = (
    "stock market OR earnings OR Federal Reserve OR inflation "
    "OR IPO OR cryptocurrency OR GDP OR interest rates"
)
PAGE_SIZE    = 100
MAX_PAGES    = 3
FUZZY_THRESHOLD = 85  # Score out of 100

def _build_params(query: str, api_key: str, page: int, days_back: int) -> dict:
    since = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "q":        query,
        "from":     since,
        "sortBy":   "publishedAt",
        "language": "en",
        "pageSize": PAGE_SIZE,
        "page":     page,
        "apiKey":   api_key,
    }

def _normalise(raw: dict) -> dict | None:
    url   = (raw.get("url") or "").strip()
    title = (raw.get("title") or "").strip()
    if not url or not title or title == "[Removed]":
        return None

    return {
        "title":        title,
        "description":  (raw.get("description") or "").strip() or None,
        "source":       (raw.get("source", {}).get("name") or "Unknown").strip(),
        "url":          url,
        "published_at": raw.get("publishedAt"),
    }

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def _fetch_page(params: dict) -> dict:
    resp = requests.get(NEWSAPI_URL, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()

def filter_duplicates_fuzzy(new_articles: list[dict], existing_articles: list[dict]) -> list[dict]:
    if not new_articles:
        return []
    
    # Create corpus of existing titles + descriptions for fast fuzzy matching
    corpus = [f"{a.get('title', '')} {a.get('description', '')}" for a in existing_articles]
    
    unique_articles = []
    
    for na in new_articles:
        na_text = f"{na.get('title', '')} {na.get('description', '')}"
        
        # If DB is empty, nothing to compare against
        if not corpus:
            unique_articles.append(na)
            corpus.append(na_text)
            continue
            
        # extractOne returns (match, score, index)
        match = process.extractOne(na_text, corpus, scorer=fuzz.token_set_ratio)
        
        if match and match[1] >= FUZZY_THRESHOLD:
            log.debug(f"Skipping fuzzy duplicate: {na['title']} (Score: {match[1]})")
            continue
            
        unique_articles.append(na)
        corpus.append(na_text)
        
    return unique_articles


def fetch_articles(
    query: str    = DEFAULT_QUERY,
    api_key: str  = "",
    days_back: int = 7,
    max_pages: int = MAX_PAGES,
) -> list[dict]:
    key = api_key or settings.news_api_key
    if not key:
        log.warning("NewsAPI key not found, skipping fetch.")
        return []

    all_articles: list[dict] = []

    for page in range(1, max_pages + 1):
        params = _build_params(query, key, page, days_back)
        log.info("Fetching page %d …", page)

        try:
            data = _fetch_page(params)
        except Exception as exc:
            log.error("NewsAPI request failed after retries: %s", exc)
            break

        if data.get("status") != "ok":
            log.error("NewsAPI error: %s", data.get("message", "unknown"))
            break

        raw_articles = data.get("articles", [])
        if not raw_articles:
            log.info("No more articles on page %d — stopping.", page)
            break

        normalised = [n for a in raw_articles if (n := _normalise(a)) is not None]
        all_articles.extend(normalised)
        log.info("  Page %d → %d usable articles (total so far: %d)", page, len(normalised), len(all_articles))

        total_results = data.get("totalResults", 0)
        if page * PAGE_SIZE >= total_results:
            break

    return all_articles


def fetch_and_store(
    query: str    = DEFAULT_QUERY,
    api_key: str  = "",
    days_back: int = 7,
) -> int:
    init_db()

    log.info("Starting news fetch  (query: %r, days_back=%d)", query, days_back)
    articles = fetch_articles(query=query, api_key=api_key, days_back=days_back)

    if not articles:
        log.warning("No articles returned from NewsAPI.")
        return 0
        
    existing_articles = fetch_all_articles()
    
    log.info(f"Filtering {len(articles)} articles against {len(existing_articles)} existing articles for fuzzy duplicates.")
    unique_articles = filter_duplicates_fuzzy(articles, existing_articles)
    log.info(f"Fuzzy match retained {len(unique_articles)} unique articles.")
    
    if not unique_articles:
        return 0

    inserted = insert_articles(unique_articles)
    log.info(
        "Fetch complete — %d fetched | %d unique | %d newly inserted | %d total in DB",
        len(articles),
        len(unique_articles),
        inserted,
        article_count(),
    )
    return inserted


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch financial news into PulseIQ DB")
    parser.add_argument("--query",     default=DEFAULT_QUERY, help="Search query")
    parser.add_argument("--days-back", type=int, default=7,   help="Days of history to fetch")
    parser.add_argument("--api-key",   default="",            help="NewsAPI key")
    args = parser.parse_args()
    n = fetch_and_store(query=args.query, api_key=args.api_key, days_back=args.days_back)
    print(f"\n🗞️   {n} new articles stored.")
