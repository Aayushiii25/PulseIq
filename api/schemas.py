"""
api/schemas.py — PulseIQ Pydantic Response Models
==================================================
All request/response shapes for the API are defined here.
Using Pydantic ensures:
  - Automatic JSON serialisation
  - Auto-generated OpenAPI documentation
  - Runtime type validation (catches bad DB data early)
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


# ── Article ────────────────────────────────────────────────────────────────────

class ArticleBase(BaseModel):
    id: int
    title: str
    description: Optional[str] = None
    source: Optional[str] = None
    url: str
    published_at: Optional[str] = None


class ArticleEnriched(ArticleBase):
    """Article joined with cluster + sentiment data."""
    cluster_label:   Optional[int]   = None
    sentiment_label: Optional[str]   = None
    sentiment_score: Optional[float] = None


# ── Cluster ────────────────────────────────────────────────────────────────────

class ClusterSummary(BaseModel):
    cluster_label:   int
    article_count:   int
    avg_sentiment:   float
    positive_count:  int
    neutral_count:   int
    negative_count:  int
    top_titles:      list[str] = Field(default_factory=list)


class ClusterDetail(ClusterSummary):
    """Full cluster with all its articles."""
    articles: list[ArticleEnriched] = Field(default_factory=list)


class UMAPPoint(BaseModel):
    """Single data point for the 2-D scatter plot."""
    id:              int
    x:               float
    y:               float
    cluster_label:   int
    title:           str
    source:          Optional[str]  = None
    sentiment_label: Optional[str]  = None
    sentiment_score: Optional[float] = None


# ── Sentiment ──────────────────────────────────────────────────────────────────

class SentimentTimelinePoint(BaseModel):
    date:        str
    avg_score:   float
    count:       int
    positive:    int
    neutral:     int
    negative:    int


class SentimentOverview(BaseModel):
    total:          int
    positive_count: int
    neutral_count:  int
    negative_count: int
    avg_score:      float
    timeline:       list[SentimentTimelinePoint] = Field(default_factory=list)


# ── Stats ──────────────────────────────────────────────────────────────────────

class PlatformStats(BaseModel):
    total_articles:    int
    embedded_articles: int
    clustered_articles: int
    analysed_articles: int
    n_clusters:        int
    noise_articles:    int
    avg_sentiment:     float
    pipeline_status: dict[str, bool]


# ── Pipeline ───────────────────────────────────────────────────────────────────

class PipelineRunRequest(BaseModel):
    api_key:       Optional[str] = Field(None, description="NewsAPI key")
    run_fetch:     bool = True
    run_embed:     bool = True
    run_cluster:   bool = True
    run_sentiment: bool = True
    days_back:     int  = Field(7, ge=1, le=30)


class PipelineStageResult(BaseModel):
    stage:   str
    success: bool
    message: str
    elapsed: float   # seconds


class PipelineRunResponse(BaseModel):
    success:       bool
    total_elapsed: float
    stages:        list[PipelineStageResult]
