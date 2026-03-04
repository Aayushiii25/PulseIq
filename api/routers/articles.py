"""
api/routers/articles.py
=======================
GET /api/articles            — paginated list with optional filters
GET /api/articles/{id}       — single article detail
GET /api/articles/umap-coords — 2-D scatter data for cluster map
"""

from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from api.schemas import ArticleEnriched, UMAPPoint
from backend.database import fetch_enriched_articles, get_connection

router = APIRouter()

MODELS_DIR      = Path("models")
UMAP_2D_PATH    = MODELS_DIR / "umap_2d_coords.npy"
ARTICLE_ID_PATH = MODELS_DIR / "clustered_ids.npy"


# ── GET /api/articles ──────────────────────────────────────────────────────────

@router.get("/articles", response_model=dict)
def list_articles(
    page:      int           = Query(1, ge=1),
    page_size: int           = Query(20, ge=1, le=100),
    cluster:   Optional[int] = Query(None, description="Filter by cluster label"),
    sentiment: Optional[str] = Query(None, description="positive | neutral | negative"),
    search:    Optional[str] = Query(None, description="Search in title / description"),
):
    """
    Return a paginated, filterable list of enriched articles.

    Query params:
      page       — page number (1-indexed)
      page_size  — items per page (max 100)
      cluster    — filter to one cluster (-1 = noise)
      sentiment  — filter by FinBERT label
      search     — full-text search on title + description
    """
    all_rows = fetch_enriched_articles()

    # ── Filters ────────────────────────────────────────────────────────────────
    if cluster is not None:
        all_rows = [r for r in all_rows if r.get("cluster_label") == cluster]

    if sentiment:
        all_rows = [r for r in all_rows if r.get("sentiment_label") == sentiment]

    if search:
        q = search.lower()
        all_rows = [
            r for r in all_rows
            if q in (r.get("title") or "").lower()
            or q in (r.get("description") or "").lower()
        ]

    total = len(all_rows)

    # ── Pagination ─────────────────────────────────────────────────────────────
    start  = (page - 1) * page_size
    end    = start + page_size
    paged  = all_rows[start:end]

    return {
        "total":     total,
        "page":      page,
        "page_size": page_size,
        "pages":     (total + page_size - 1) // page_size,
        "items":     paged,
    }


# ── GET /api/articles/umap-coords ─────────────────────────────────────────────

@router.get("/articles/umap-coords", response_model=list[UMAPPoint])
def get_umap_coords():
    """
    Return 2-D UMAP coordinates for every clustered article.
    Used by the frontend to render the scatter plot without pulling
    all article text (keeps the payload small).
    """
    if not UMAP_2D_PATH.exists() or not ARTICLE_ID_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="UMAP coordinates not found. Run the clustering stage first.",
        )

    coords   = np.load(UMAP_2D_PATH)
    ids      = np.load(ARTICLE_ID_PATH).tolist()

    # Build a lookup from the DB
    conn = get_connection()
    rows = conn.execute("""
        SELECT a.id, a.title, a.source, c.cluster_label,
               s.label AS sentiment_label, s.sentiment_score
        FROM   articles a
        LEFT JOIN clusters  c ON c.article_id = a.id
        LEFT JOIN sentiment s ON s.article_id = a.id
    """).fetchall()
    conn.close()
    lookup = {r["id"]: dict(r) for r in rows}

    points: list[UMAPPoint] = []
    for i, art_id in enumerate(ids):
        meta = lookup.get(art_id, {})
        points.append(UMAPPoint(
            id=art_id,
            x=float(coords[i, 0]),
            y=float(coords[i, 1]),
            cluster_label=meta.get("cluster_label", -1) or -1,
            title=meta.get("title", ""),
            source=meta.get("source"),
            sentiment_label=meta.get("sentiment_label"),
            sentiment_score=meta.get("sentiment_score"),
        ))
    return points


# ── GET /api/articles/{id} ─────────────────────────────────────────────────────

@router.get("/articles/{article_id}", response_model=ArticleEnriched)
def get_article(article_id: int):
    """Fetch a single article by its primary key."""
    conn = get_connection()
    row  = conn.execute("""
        SELECT a.id, a.title, a.description, a.source, a.url, a.published_at,
               c.cluster_label, s.label AS sentiment_label, s.sentiment_score
        FROM   articles a
        LEFT JOIN clusters  c ON c.article_id = a.id
        LEFT JOIN sentiment s ON s.article_id = a.id
        WHERE  a.id = ?
    """, (article_id,)).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"Article {article_id} not found")
    return dict(row)
