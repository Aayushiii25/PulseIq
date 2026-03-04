"""
run_pipeline.py — Run the ML pipeline on already-seeded data.

Skips the NewsAPI fetch step and runs:
  embed → cluster → sentiment

Perfect for the first run after `python seed_db.py`.

Usage:
    python run_pipeline.py

To also fetch fresh news from NewsAPI:
    python run_pipeline.py --fetch
"""

import argparse
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true", help="Also fetch news from NewsAPI")
    args = parser.parse_args()

    t_start = time.perf_counter()

    if args.fetch:
        import os
        key = os.getenv("NEWS_API_KEY", "60bedc90538343c28014f9b2e34ce758")
        log.info("── Stage 1/4: Fetching news from NewsAPI …")
        from backend.fetch_news import fetch_and_store
        n = fetch_and_store(api_key=key, days_back=14)
        log.info("   %d new articles stored.\n", n)
    else:
        log.info("⏩  Skipping fetch (using seeded data). Pass --fetch to pull live news.\n")

    log.info("── Stage 2/4: Generating embeddings …")
    from backend.embed_articles import embed_articles
    embed_articles()
    print()

    log.info("── Stage 3/4: Clustering (PCA → UMAP → HDBSCAN) …")
    from backend.cluster_articles import cluster_articles
    summary = cluster_articles()
    if summary:
        log.info("   %d clusters found, %d noise articles.\n",
                 summary.get("n_clusters", 0), summary.get("noise_articles", 0))

    log.info("── Stage 4/4: FinBERT sentiment analysis …")
    from backend.sentiment_analysis import run_sentiment_analysis
    run_sentiment_analysis()

    elapsed = time.perf_counter() - t_start
    print(f"\n{'━'*55}")
    print(f"  ✅ Pipeline complete in {elapsed:.1f}s")
    print(f"{'━'*55}")
    print("\nNext step:")
    print("  Terminal 1:  uvicorn api.main:app --reload --port 8000")
    print("  Terminal 2:  streamlit run app.py")


if __name__ == "__main__":
    main()
