"""
api/routers/clusters.py
=======================
GET /api/clusters            — list all cluster summaries
GET /api/clusters/{label}    — one cluster with full article list
"""

from fastapi import APIRouter, HTTPException
from backend.database import fetch_enriched_articles
from api.schemas import ClusterSummary, ClusterDetail, ArticleEnriched

router = APIRouter()


def _build_cluster_summary(label: int, articles: list[dict]) -> ClusterSummary:
    scores   = [a["sentiment_score"] or 0.0 for a in articles]
    avg_sent = round(sum(scores) / len(scores), 4) if scores else 0.0
    return ClusterSummary(
        cluster_label=label,
        article_count=len(articles),
        avg_sentiment=avg_sent,
        positive_count=sum(1 for a in articles if a.get("sentiment_label") == "positive"),
        neutral_count =sum(1 for a in articles if a.get("sentiment_label") == "neutral"),
        negative_count=sum(1 for a in articles if a.get("sentiment_label") == "negative"),
        top_titles=[a["title"] for a in articles[:5]],
    )


# ── GET /api/clusters ─────────────────────────────────────────────────────────

@router.get("/clusters", response_model=list[ClusterSummary])
def list_clusters():
    """
    Return a summary for every cluster (excluding noise label -1).
    Sorted by article count descending so the biggest themes come first.
    """
    all_articles = fetch_enriched_articles()

    # Group by cluster_label
    groups: dict[int, list[dict]] = {}
    for a in all_articles:
        label = a.get("cluster_label")
        if label is None or label == -1:
            continue
        groups.setdefault(label, []).append(a)

    summaries = [
        _build_cluster_summary(label, arts)
        for label, arts in groups.items()
    ]
    return sorted(summaries, key=lambda s: s.article_count, reverse=True)


# ── GET /api/clusters/{label} ─────────────────────────────────────────────────

@router.get("/clusters/{label}", response_model=ClusterDetail)
def get_cluster(label: int):
    """Return full detail for one cluster including all its articles."""
    all_articles = fetch_enriched_articles()
    cluster_arts = [a for a in all_articles if a.get("cluster_label") == label]

    if not cluster_arts:
        raise HTTPException(status_code=404, detail=f"Cluster {label} not found")

    summary = _build_cluster_summary(label, cluster_arts)
    return ClusterDetail(
        **summary.model_dump(),
        articles=[ArticleEnriched(**a) for a in cluster_arts],
    )
