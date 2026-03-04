"""
api/routers/sentiment.py
========================
GET /api/sentiment/overview   — aggregate counts + daily timeline
GET /api/sentiment/timeline   — daily breakdown only
"""

from collections import defaultdict
from fastapi import APIRouter
from backend.database import fetch_enriched_articles
from api.schemas import SentimentOverview, SentimentTimelinePoint

router = APIRouter()


def _build_overview(articles: list[dict]) -> SentimentOverview:
    """Compute sentiment overview from a list of enriched article dicts."""
    analysed = [a for a in articles if a.get("sentiment_label")]
    total    = len(analysed)

    pos = sum(1 for a in analysed if a["sentiment_label"] == "positive")
    neu = sum(1 for a in analysed if a["sentiment_label"] == "neutral")
    neg = sum(1 for a in analysed if a["sentiment_label"] == "negative")
    scores = [a["sentiment_score"] or 0.0 for a in analysed]
    avg    = round(sum(scores) / total, 4) if total else 0.0

    # ── Daily timeline ─────────────────────────────────────────────────────────
    daily: dict[str, dict] = defaultdict(lambda: {
        "count": 0, "score_sum": 0.0,
        "positive": 0, "neutral": 0, "negative": 0,
    })

    for a in analysed:
        pub = (a.get("published_at") or "")[:10]  # "YYYY-MM-DD"
        if not pub:
            continue
        d = daily[pub]
        d["count"]     += 1
        d["score_sum"] += a["sentiment_score"] or 0.0
        d[a["sentiment_label"]] += 1

    timeline = sorted(
        [
            SentimentTimelinePoint(
                date=date,
                avg_score=round(v["score_sum"] / v["count"], 4),
                count=v["count"],
                positive=v["positive"],
                neutral=v["neutral"],
                negative=v["negative"],
            )
            for date, v in daily.items()
        ],
        key=lambda p: p.date,
    )

    return SentimentOverview(
        total=total,
        positive_count=pos,
        neutral_count=neu,
        negative_count=neg,
        avg_score=avg,
        timeline=timeline,
    )


@router.get("/sentiment/overview", response_model=SentimentOverview)
def sentiment_overview():
    """Full sentiment overview with daily timeline."""
    return _build_overview(fetch_enriched_articles())


@router.get("/sentiment/timeline", response_model=list[SentimentTimelinePoint])
def sentiment_timeline():
    """Daily sentiment timeline only (lighter payload)."""
    return _build_overview(fetch_enriched_articles()).timeline
