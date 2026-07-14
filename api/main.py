"""
api/main.py — PulseIQ FastAPI Application
==========================================
The HTTP API layer that sits between the ML pipeline and the frontend.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from cachetools import TTLCache
import asyncio

from api.routers import articles, clusters, pipeline, sentiment, stats, realtime
from backend.database import init_db
from backend.scheduler import start_scheduler, stop_scheduler
from config import settings
from api.dependencies import get_api_key, api_key_header, API_KEY_NAME

# ── Auth & Rate Limiting ────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── App Lifespan ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    start_scheduler()
    yield
    stop_scheduler()


# ── Create FastAPI app ─────────────────────────────────────────────────────────
app = FastAPI(
    title="PulseIQ API",
    description="Financial Narrative Intelligence Platform — REST API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount routers ──────────────────────────────────────────────────────────────
app.include_router(stats.router,     prefix="/api", tags=["Stats"])
app.include_router(articles.router,  prefix="/api", tags=["Articles"])
app.include_router(clusters.router,  prefix="/api", tags=["Clusters"])
app.include_router(sentiment.router, prefix="/api", tags=["Sentiment"])
app.include_router(pipeline.router,  prefix="/api", tags=["Pipeline"])
app.include_router(realtime.router,  prefix="/api", tags=["Realtime"])


from fastapi import Request

@app.get("/", tags=["Health"])
@limiter.limit("10/minute")
def root(request: Request):
    """Health-check endpoint."""
    return {"status": "ok", "service": "PulseIQ API", "version": "2.0.0"}


@app.get("/health", tags=["Health"])
@limiter.limit("60/minute")
def health(request: Request):
    return {"status": "healthy"}
