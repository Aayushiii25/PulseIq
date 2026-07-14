"""
api/routers/stats.py
GET /api/stats  — returns a single platform-wide stats snapshot.
"""

from fastapi import APIRouter
from backend.database import SessionLocal, Article, Embedding, Cluster, Sentiment
from api.schemas import PlatformStats
from cachetools import cached, TTLCache
from sqlalchemy import select, func

router = APIRouter()
cache = TTLCache(maxsize=10, ttl=60)

@router.get("/stats", response_model=PlatformStats)
@cached(cache)
def get_stats() -> PlatformStats:
    with SessionLocal() as db:
        total_articles = db.scalar(select(func.count(Article.id))) or 0
        embedded_articles = db.scalar(select(func.count(Embedding.article_id))) or 0
        clustered_articles = db.scalar(select(func.count(Cluster.article_id)).where(Cluster.cluster_label != -1)) or 0
        analysed_articles = db.scalar(select(func.count(Sentiment.article_id))) or 0
        
        n_clusters = db.scalar(select(func.count(func.distinct(Cluster.cluster_label))).where(Cluster.cluster_label != -1)) or 0
        noise_articles = db.scalar(select(func.count(Cluster.article_id)).where(Cluster.cluster_label == -1)) or 0
        avg_sentiment = db.scalar(select(func.avg(Sentiment.sentiment_score))) or 0.0

    return PlatformStats(
        total_articles=total_articles,
        embedded_articles=embedded_articles,
        clustered_articles=clustered_articles,
        analysed_articles=analysed_articles,
        n_clusters=n_clusters,
        noise_articles=noise_articles,
        avg_sentiment=round(float(avg_sentiment), 4),
        pipeline_status={
            "fetched":   total_articles > 0,
            "embedded":  embedded_articles > 0,
            "clustered": clustered_articles > 0,
            "analysed":  analysed_articles > 0,
        },
    )
