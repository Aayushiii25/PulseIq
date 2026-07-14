"""
api/routers/realtime.py
=======================
SSE endpoint for streaming pipeline status.
"""
import asyncio
from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

router = APIRouter()

async def pipeline_status_generator(request: Request):
    from api.routers.pipeline import _running, _last_result
    
    while True:
        if await request.is_disconnected():
            break
            
        data = {
            "running": _running,
            "last_result": _last_result
        }
        
        yield {
            "event": "message",
            "retry": 15000,
            "data": data
        }
        
        await asyncio.sleep(2)

@router.get("/pipeline/stream")
async def stream_pipeline_status(request: Request):
    """Server-Sent Events endpoint for real-time pipeline status"""
    return EventSourceResponse(pipeline_status_generator(request))
