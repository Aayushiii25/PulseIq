"""
api/routers/pipeline.py
=======================
POST /api/pipeline/run     — run selected pipeline stages
GET  /api/pipeline/status  — check if a pipeline is currently running

The pipeline runs synchronously per-request (fine for a single-user dev
tool). For production you'd push this to a Celery / RQ background worker
and return a job-id that the frontend polls.
"""

import time
import threading
from fastapi import APIRouter, HTTPException
from api.schemas import PipelineRunRequest, PipelineRunResponse, PipelineStageResult

router = APIRouter()

# Simple in-process lock so two pipeline runs can't overlap
_pipeline_lock = threading.Lock()
_running       = False


@router.get("/pipeline/status")
def pipeline_status():
    """Check whether a pipeline run is currently in progress."""
    return {"running": _running}


@router.post("/pipeline/run", response_model=PipelineRunResponse)
def run_pipeline(req: PipelineRunRequest):
    """
    Execute selected pipeline stages in sequence and return per-stage results.

    Stages (each optional via request body flags):
      1. fetch_news        — pull articles from NewsAPI
      2. embed_articles    — generate transformer embeddings
      3. cluster_articles  — PCA → UMAP → HDBSCAN
      4. sentiment_analysis — FinBERT scoring
    """
    global _running

    if _running:
        raise HTTPException(status_code=409, detail="A pipeline run is already in progress.")

    if req.run_fetch and not req.api_key:
        raise HTTPException(
            status_code=422,
            detail="api_key is required when run_fetch=true",
        )

    _running      = True
    stages_done:  list[PipelineStageResult] = []
    overall_start = time.perf_counter()

    try:
        # ── Stage 1: Fetch ─────────────────────────────────────────────────────
        if req.run_fetch:
            t0 = time.perf_counter()
            try:
                from backend.fetch_news import fetch_and_store
                n = fetch_and_store(api_key=req.api_key, days_back=req.days_back)
                stages_done.append(PipelineStageResult(
                    stage="fetch", success=True,
                    message=f"{n} new articles stored",
                    elapsed=round(time.perf_counter() - t0, 2),
                ))
            except Exception as exc:
                stages_done.append(PipelineStageResult(
                    stage="fetch", success=False,
                    message=str(exc),
                    elapsed=round(time.perf_counter() - t0, 2),
                ))

        # ── Stage 2: Embed ─────────────────────────────────────────────────────
        if req.run_embed:
            t0 = time.perf_counter()
            try:
                from backend.embed_articles import embed_articles
                n = embed_articles()
                stages_done.append(PipelineStageResult(
                    stage="embed", success=True,
                    message=f"{n} articles embedded",
                    elapsed=round(time.perf_counter() - t0, 2),
                ))
            except Exception as exc:
                stages_done.append(PipelineStageResult(
                    stage="embed", success=False,
                    message=str(exc),
                    elapsed=round(time.perf_counter() - t0, 2),
                ))

        # ── Stage 3: Cluster ───────────────────────────────────────────────────
        if req.run_cluster:
            t0 = time.perf_counter()
            try:
                from backend.cluster_articles import cluster_articles
                summary = cluster_articles()
                stages_done.append(PipelineStageResult(
                    stage="cluster", success=True,
                    message=f"{summary.get('n_clusters', 0)} clusters found, "
                            f"{summary.get('noise_articles', 0)} noise articles",
                    elapsed=round(time.perf_counter() - t0, 2),
                ))
            except Exception as exc:
                stages_done.append(PipelineStageResult(
                    stage="cluster", success=False,
                    message=str(exc),
                    elapsed=round(time.perf_counter() - t0, 2),
                ))

        # ── Stage 4: Sentiment ─────────────────────────────────────────────────
        if req.run_sentiment:
            t0 = time.perf_counter()
            try:
                from backend.sentiment_analysis import run_sentiment_analysis
                n = run_sentiment_analysis()
                stages_done.append(PipelineStageResult(
                    stage="sentiment", success=True,
                    message=f"{n} articles analysed",
                    elapsed=round(time.perf_counter() - t0, 2),
                ))
            except Exception as exc:
                stages_done.append(PipelineStageResult(
                    stage="sentiment", success=False,
                    message=str(exc),
                    elapsed=round(time.perf_counter() - t0, 2),
                ))

    finally:
        _running = False

    total_elapsed = round(time.perf_counter() - overall_start, 2)
    all_ok        = all(s.success for s in stages_done)

    return PipelineRunResponse(
        success=all_ok,
        total_elapsed=total_elapsed,
        stages=stages_done,
    )
