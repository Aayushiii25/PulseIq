"""
backend/pipeline.py — PulseIQ Full ML Pipeline Orchestrator
============================================================
Runs every stage in sequence:
  1. fetch_news & fetch_rss
  2. embed_articles
  3. cluster_articles (PCA → UMAP → HDBSCAN)
  4. sentiment_analysis (FinBERT)
  5. market_benchmark
"""

import argparse
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def run_pipeline(skip_fetch: bool = False, api_key: str = "") -> dict:
    start = time.perf_counter()
    banner("PulseIQ — Full ML Pipeline")

    results = {}

    if not skip_fetch:
        stage("1 / 5  Fetching financial news & RSS …")
        
        from backend.fetch_news import fetch_and_store
        n_news = fetch_and_store(api_key=api_key)
        
        from backend.fetch_rss import fetch_and_store_rss
        n_rss = fetch_and_store_rss()
        
        log.info("   → %d new articles stored from NewsAPI.", n_news)
        log.info("   → %d new articles stored from RSS.", n_rss)
        results['fetch'] = n_news + n_rss
    else:
        log.info("⏩  Skipping fetch stage.\n")

    stage("2 / 5  Generating embeddings …")
    from backend.embed_articles import embed_articles
    n_embed = embed_articles()
    results['embed'] = n_embed
    print()

    stage("3 / 5  Clustering articles …")
    from backend.cluster_articles import cluster_articles
    summary = cluster_articles()
    if summary:
        log.info("   → %d clusters discovered.\n", summary.get("n_clusters", 0))
    results['cluster'] = summary

    stage("4 / 5  Running FinBERT sentiment analysis …")
    from backend.sentiment_analysis import run_sentiment_analysis
    n_sent = run_sentiment_analysis()
    results['sentiment'] = n_sent

    stage("5 / 5  Market Benchmark Correlation …")
    from backend.market_benchmark import calculate_correlations
    calculate_correlations()
    results['benchmark'] = True

    elapsed = time.perf_counter() - start
    banner(f"Pipeline complete  ({elapsed:.1f}s)")
    
    results['elapsed_time'] = elapsed
    return results


def banner(msg: str) -> None:
    log.info("━" * 55)
    log.info("  %s", msg)
    log.info("━" * 55)


def stage(msg: str) -> None:
    log.info("\n── %s", msg)


if __name__ == "__main__":
    from config import settings
    parser = argparse.ArgumentParser(description="Run the PulseIQ ML pipeline end-to-end")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip the news-fetching stage")
    parser.add_argument("--api-key",   default="",           help="NewsAPI key")
    args = parser.parse_args()
    key = args.api_key or settings.news_api_key
    run_pipeline(skip_fetch=args.skip_fetch, api_key=key)
