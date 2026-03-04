"""
api/main.py — PulseIQ FastAPI Application
==========================================
The HTTP API layer that sits between the ML pipeline and the frontend.

Architecture:
  frontend (Streamlit)
        │  HTTP / JSON
        ▼
  api/main.py  ←  registers all routers
        │
        ├── routers/articles.py   GET  /api/articles
        ├── routers/clusters.py   GET  /api/clusters
        ├── routers/sentiment.py  GET  /api/sentiment
        ├── routers/pipeline.py   POST /api/pipeline/run
        └── routers/stats.py      GET  /api/stats

Run the API server:
    uvicorn api.main:app --reload --port 8000

The Streamlit frontend connects to http://localhost:8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import articles, clusters, pipeline, sentiment, stats
from backend.database import init_db

# ── Initialise DB on startup ───────────────────────────────────────────────────
init_db()

# ── Create FastAPI app ─────────────────────────────────────────────────────────
app = FastAPI(
    title="PulseIQ API",
    description="Financial Narrative Intelligence Platform — REST API",
    version="1.0.0",
    docs_url="/docs",        # Swagger UI at http://localhost:8000/docs
    redoc_url="/redoc",      # ReDoc UI at http://localhost:8000/redoc
)

# ── CORS — allow Streamlit (port 8501) to call the API ────────────────────────
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


@app.get("/", tags=["Health"])
def root():
    """Health-check endpoint."""
    return {"status": "ok", "service": "PulseIQ API", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}
