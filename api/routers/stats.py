"""
api/routers/stats.py
GET /api/stats  — returns a single platform-wide stats snapshot.

This is the first endpoint the frontend calls on load to
decide which panels to show (e.g. hide cluster map if no clusters yet).
"""

from fastapi import APIRouter
from backend.database import get_connection
from api.schemas import PlatformStats

router = APIRouter()


@router.get("/stats", response_model=PlatformStats)
def get_stats() -> PlatformStats:
    """Return platform-wide counts and pipeline readiness flags."""
    conn = get_connection()

    total_articles     = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    embedded_articles  = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    clustered_articles = conn.execute(
        "SELECT COUNT(*) FROM clusters WHERE cluster_label != -1"
    ).fetchone()[0]
    analysed_articles  = conn.execute("SELECT COUNT(*) FROM sentiment").fetchone()[0]

    cluster_labels = conn.execute(
        "SELECT DISTINCT cluster_label FROM clusters WHERE cluster_label != -1"
    ).fetchall()
    n_clusters = len(cluster_labels)

    noise_articles = conn.execute(
        "SELECT COUNT(*) FROM clusters WHERE cluster_label = -1"
    ).fetchone()[0]

    avg_row = conn.execute(
        "SELECT AVG(sentiment_score) FROM sentiment"
    ).fetchone()[0]
    avg_sentiment = round(avg_row or 0.0, 4)

    conn.close()

    return PlatformStats(
        total_articles=total_articles,
        embedded_articles=embedded_articles,
        clustered_articles=clustered_articles,
        analysed_articles=analysed_articles,
        n_clusters=n_clusters,
        noise_articles=noise_articles,
        avg_sentiment=avg_sentiment,
        pipeline_status={
            "fetched":   total_articles > 0,
            "embedded":  embedded_articles > 0,
            "clustered": clustered_articles > 0,
            "analysed":  analysed_articles > 0,
        },
    )
