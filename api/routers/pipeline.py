"""
api/routers/pipeline.py
=======================
POST /api/pipeline/run     — run selected pipeline stages
GET  /api/pipeline/status  — check if a pipeline is currently running
"""

import time
import threading
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from api.schemas import PipelineRunRequest, PipelineRunResponse, PipelineStageResult
from api.dependencies import get_api_key

router = APIRouter()

_pipeline_lock = threading.Lock()
_running       = False
_last_result   = None


@router.get("/pipeline/status")
def pipeline_status():
    """Check whether a pipeline run is currently in progress."""
    return {"running": _running, "last_result": _last_result}

def execute_pipeline(req: PipelineRunRequest):
    global _running, _last_result
    
    _running      = True
    stages_done:  list[PipelineStageResult] = []
    overall_start = time.perf_counter()

    try:
        if req.run_fetch:
            t0 = time.perf_counter()
            try:
                from backend.fetch_news import fetch_and_store
                from backend.fetch_rss import fetch_and_store_rss
                n1 = fetch_and_store(api_key=req.api_key, days_back=req.days_back)
                n2 = fetch_and_store_rss()
                stages_done.append(PipelineStageResult(
                    stage="fetch", success=True,
                    message=f"{n1} API articles, {n2} RSS articles stored",
                    elapsed=round(time.perf_counter() - t0, 2),
                ))
            except Exception as exc:
                stages_done.append(PipelineStageResult(
                    stage="fetch", success=False,
                    message=str(exc),
                    elapsed=round(time.perf_counter() - t0, 2),
                ))

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

        if req.run_sentiment:
            t0 = time.perf_counter()
            try:
                from backend.sentiment_analysis import run_sentiment_analysis
                n = run_sentiment_analysis()
                
                # also run market correlation
                from backend.market_benchmark import calculate_correlations
                calculate_correlations()
                
                stages_done.append(PipelineStageResult(
                    stage="sentiment", success=True,
                    message=f"{n} articles analysed + correlated",
                    elapsed=round(time.perf_counter() - t0, 2),
                ))
            except Exception as exc:
                stages_done.append(PipelineStageResult(
                    stage="sentiment", success=False,
                    message=str(exc),
                    elapsed=round(time.perf_counter() - t0, 2),
                ))

    finally:
        total_elapsed = round(time.perf_counter() - overall_start, 2)
        all_ok        = all(s.success for s in stages_done)
        
        result = PipelineRunResponse(
            success=all_ok,
            total_elapsed=total_elapsed,
            stages=stages_done,
        )
        _last_result = result.model_dump()
        _running = False
        _pipeline_lock.release()

@router.post("/pipeline/run", dependencies=[Depends(get_api_key)])
def run_pipeline(req: PipelineRunRequest, background_tasks: BackgroundTasks):
    global _running

    if not _pipeline_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A pipeline run is already in progress.")

    if req.run_fetch and not req.api_key:
        from config import settings
        if not settings.news_api_key:
            _pipeline_lock.release()
            raise HTTPException(
                status_code=422,
                detail="api_key is required when run_fetch=true",
            )
        else:
            req.api_key = settings.news_api_key

    background_tasks.add_task(execute_pipeline, req)
    return {"message": "Pipeline execution started in the background."}
