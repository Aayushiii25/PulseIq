"""
api/routers/clusters.py
=======================
GET /api/clusters            — list all cluster summaries
GET /api/clusters/{label}    — one cluster with full article list
"""

from fastapi import APIRouter, HTTPException
from backend.database import fetch_enriched_articles, get_latest_cluster_history
from api.schemas import ClusterSummary, ClusterDetail, ArticleEnriched
from cachetools import cached, TTLCache

router = APIRouter()
cache = TTLCache(maxsize=10, ttl=60)

def _build_cluster_summary(label: int, articles: list[dict], name: str = None) -> ClusterSummary:
    scores   = [a["sentiment_score"] if a["sentiment_score"] is not None else 0.0 for a in articles]
    avg_sent = round(sum(scores) / len(scores), 4) if scores else 0.0
    return ClusterSummary(
        cluster_label=label,
        article_count=len(articles),
        avg_sentiment=avg_sent,
        positive_count=sum(1 for a in articles if a.get("sentiment_label") == "positive"),
        neutral_count =sum(1 for a in articles if a.get("sentiment_label") == "neutral"),
        negative_count=sum(1 for a in articles if a.get("sentiment_label") == "negative"),
        top_titles=[a["title"] for a in articles[:5]],
        cluster_name=name
    )


@router.get("/clusters", response_model=list[ClusterSummary])
@cached(cache)
def list_clusters():
    all_articles = fetch_enriched_articles()
    cluster_history = get_latest_cluster_history()
    
    names_map = {r["current_label"]: r["human_name"] for r in cluster_history}

    groups: dict[int, list[dict]] = {}
    for a in all_articles:
        label = a.get("cluster_label")
        if label is None or label == -1:
            continue
        groups.setdefault(label, []).append(a)

    summaries = [
        _build_cluster_summary(label, arts, names_map.get(label))
        for label, arts in groups.items()
    ]
    return sorted(summaries, key=lambda s: s.article_count, reverse=True)


@router.get("/clusters/{label}", response_model=ClusterDetail)
def get_cluster(label: int):
    all_articles = fetch_enriched_articles()
    cluster_history = get_latest_cluster_history()
    
    name = None
    for r in cluster_history:
        if r["current_label"] == label:
            name = r["human_name"]
            break
            
    cluster_arts = [a for a in all_articles if a.get("cluster_label") == label]

    if not cluster_arts:
        raise HTTPException(status_code=404, detail=f"Cluster {label} not found")

    summary = _build_cluster_summary(label, cluster_arts, name)
    return ClusterDetail(
        **summary.model_dump(),
        articles=[ArticleEnriched(**a) for a in cluster_arts],
    )
