"""
backend/pipeline.py — PulseIQ Full ML Pipeline Orchestrator
============================================================
Runs every stage in sequence:
  1. fetch_news        — pull articles from NewsAPI
  2. embed_articles    — generate sentence-transformer vectors
  3. cluster_articles  — PCA → UMAP → HDBSCAN
  4. sentiment_analysis — FinBERT per-article scoring

Usage:
    python -m backend.pipeline
    python -m backend.pipeline --skip-fetch   (use articles already in DB)
"""

import argparse
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def run_pipeline(skip_fetch: bool = False, api_key: str = "") -> None:
    start = time.perf_counter()
    banner("PulseIQ — Full ML Pipeline")

    # ── Stage 1: Fetch ────────────────────────────────────────────────────────
    if not skip_fetch:
        stage("1 / 4  Fetching financial news …")
        from backend.fetch_news import fetch_and_store
        n = fetch_and_store(api_key=api_key)
        log.info("   → %d new articles stored.\n", n)
    else:
        log.info("⏩  Skipping fetch stage.\n")

    # ── Stage 2: Embed ────────────────────────────────────────────────────────
    stage("2 / 4  Generating embeddings …")
    from backend.embed_articles import embed_articles
    embed_articles()
    print()

    # ── Stage 3: Cluster ──────────────────────────────────────────────────────
    stage("3 / 4  Clustering articles …")
    from backend.cluster_articles import cluster_articles
    summary = cluster_articles()
    if summary:
        log.info("   → %d clusters discovered.\n", summary.get("n_clusters", 0))

    # ── Stage 4: Sentiment ────────────────────────────────────────────────────
    stage("4 / 4  Running FinBERT sentiment analysis …")
    from backend.sentiment_analysis import run_sentiment_analysis
    run_sentiment_analysis()

    elapsed = time.perf_counter() - start
    banner(f"Pipeline complete  ({elapsed:.1f}s)")
    log.info("Launch the dashboard:  streamlit run app.py")


# ── Helpers ────────────────────────────────────────────────────────────────────

def banner(msg: str) -> None:
    log.info("━" * 55)
    log.info("  %s", msg)
    log.info("━" * 55)


def stage(msg: str) -> None:
    log.info("\n── %s", msg)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the PulseIQ ML pipeline end-to-end")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip the news-fetching stage")
    parser.add_argument("--api-key",   default="",           help="NewsAPI key")
    args = parser.parse_args()
    run_pipeline(skip_fetch=args.skip_fetch, api_key=args.api_key)
