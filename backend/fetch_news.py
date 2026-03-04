"""
backend/fetch_news.py — PulseIQ News Ingestion Module
======================================================
Fetches financial news from the NewsAPI /everything endpoint,
normalises the payload, and persists articles to SQLite via database.py.

Why NewsAPI?
  - Free tier supports 100 requests/day with up to 100 articles each.
  - The /everything endpoint lets us pass finance-specific keywords so
    we get relevant signal rather than generic headlines.

Usage (standalone):
    python -m backend.fetch_news

Environment variable:
    NEWS_API_KEY  — your NewsAPI key (https://newsapi.org/register)
                    Can also be passed to fetch_and_store() directly.
"""

import os
import logging
from datetime import datetime, timedelta, timezone

import requests

from backend.database import init_db, insert_articles, article_count

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
NEWSAPI_URL  = "https://newsapi.org/v2/everything"
DEFAULT_QUERY = (
    "stock market OR earnings OR Federal Reserve OR inflation "
    "OR IPO OR cryptocurrency OR GDP OR interest rates"
)
PAGE_SIZE    = 100   # maximum NewsAPI allows per request
MAX_PAGES    = 3     # cap at 300 articles per run to respect free-tier limits


# ── Core fetch logic ───────────────────────────────────────────────────────────

def _build_params(query: str, api_key: str, page: int, days_back: int) -> dict:
    """Construct the query-string parameters for one API call."""
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
    """
    Map a raw NewsAPI article dict to our DB schema.
    Returns None if the article lacks a URL or title (unusable for ML).
    """
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


def fetch_articles(
    query: str    = DEFAULT_QUERY,
    api_key: str  = "",
    days_back: int = 7,
    max_pages: int = MAX_PAGES,
) -> list[dict]:
    """
    Hit the NewsAPI /everything endpoint and return normalised article dicts.

    Args:
        query     — free-text search string
        api_key   — NewsAPI key (falls back to NEWS_API_KEY env var)
        days_back — how far back to search (free tier: max 30 days)
        max_pages — max pages to fetch (100 articles each)

    Returns:
        List of normalised article dicts ready for DB insertion.
    """
    key = api_key or os.getenv("NEWS_API_KEY", "")
    if not key:
        raise ValueError(
            "NewsAPI key not found. Set the NEWS_API_KEY environment variable "
            "or pass api_key= to fetch_articles()."
        )

    all_articles: list[dict] = []

    for page in range(1, max_pages + 1):
        params = _build_params(query, key, page, days_back)
        log.info("Fetching page %d …", page)

        try:
            resp = requests.get(NEWSAPI_URL, params=params, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as exc:
            log.error("NewsAPI request failed: %s", exc)
            break

        data = resp.json()

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

        # NewsAPI paginates by totalResults; stop early if we have everything
        total_results = data.get("totalResults", 0)
        if page * PAGE_SIZE >= total_results:
            break

    return all_articles


# ── Top-level convenience function ────────────────────────────────────────────

def fetch_and_store(
    query: str    = DEFAULT_QUERY,
    api_key: str  = "",
    days_back: int = 7,
) -> int:
    """
    Fetch articles from NewsAPI and persist them to SQLite.

    Returns:
        Number of newly inserted articles (duplicates are skipped).
    """
    init_db()

    log.info("Starting news fetch  (query: %r, days_back=%d)", query, days_back)
    articles = fetch_articles(query=query, api_key=api_key, days_back=days_back)

    if not articles:
        log.warning("No articles returned from NewsAPI.")
        return 0

    inserted = insert_articles(articles)
    log.info(
        "Fetch complete — %d fetched | %d newly inserted | %d total in DB",
        len(articles),
        inserted,
        article_count(),
    )
    return inserted


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch financial news into PulseIQ DB")
    parser.add_argument("--query",     default=DEFAULT_QUERY, help="Search query")
    parser.add_argument("--days-back", type=int, default=7,   help="Days of history to fetch")
    parser.add_argument("--api-key",   default="",            help="NewsAPI key (overrides env var)")
    args = parser.parse_args()

    n = fetch_and_store(query=args.query, api_key=args.api_key, days_back=args.days_back)
    print(f"\n🗞️   {n} new articles stored.")
