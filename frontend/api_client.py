"""
frontend/api_client.py — PulseIQ HTTP API Client
=================================================
All communication between the Streamlit frontend and the FastAPI backend
goes through this module.

Design principle: the frontend NEVER imports from `backend/` directly.
It only speaks HTTP. This means:
  - Frontend and backend can run on separate machines.
  - The API contract is explicit and testable.
  - Swapping the backend (e.g. to PostgreSQL) requires zero frontend changes.

Usage:
    from frontend.api_client import client
    stats  = client.get_stats()
    points = client.get_umap_coords()
"""

from __future__ import annotations
import os
import requests
from typing import Optional

# Default: API runs on localhost:8000
API_BASE = os.getenv("PULSEIQ_API_URL", "http://localhost:8000")
TIMEOUT  = 120   # seconds — pipeline calls can be slow


class APIError(Exception):
    """Raised when the API returns a non-2xx status."""
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail      = detail
        super().__init__(f"API {status_code}: {detail}")


class PulseIQClient:
    """Thin HTTP wrapper around the PulseIQ FastAPI backend."""

    def __init__(self, base_url: str = API_BASE):
        self.base = base_url.rstrip("/")
        self.session = requests.Session()

    def _get(self, path: str, params: dict | None = None) -> dict | list:
        resp = self.session.get(f"{self.base}{path}", params=params, timeout=TIMEOUT)
        if not resp.ok:
            raise APIError(resp.status_code, resp.json().get("detail", resp.text))
        return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        resp = self.session.post(f"{self.base}{path}", json=body, timeout=TIMEOUT)
        if not resp.ok:
            raise APIError(resp.status_code, resp.json().get("detail", resp.text))
        return resp.json()

    # ── Health ─────────────────────────────────────────────────────────────────

    def is_reachable(self) -> bool:
        """Return True if the API server responds to /health."""
        try:
            resp = self.session.get(f"{self.base}/health", timeout=3)
            return resp.ok
        except requests.ConnectionError:
            return False

    # ── Stats ──────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """GET /api/stats"""
        return self._get("/api/stats")

    # ── Articles ───────────────────────────────────────────────────────────────

    def get_articles(
        self,
        page:      int           = 1,
        page_size: int           = 20,
        cluster:   Optional[int] = None,
        sentiment: Optional[str] = None,
        search:    Optional[str] = None,
    ) -> dict:
        """GET /api/articles  (paginated + filtered)"""
        params = {"page": page, "page_size": page_size}
        if cluster is not None:
            params["cluster"] = cluster
        if sentiment:
            params["sentiment"] = sentiment
        if search:
            params["search"] = search
        return self._get("/api/articles", params)

    def get_article(self, article_id: int) -> dict:
        """GET /api/articles/{id}"""
        return self._get(f"/api/articles/{article_id}")

    def get_umap_coords(self) -> list[dict]:
        """GET /api/articles/umap-coords"""
        return self._get("/api/articles/umap-coords")

    # ── Clusters ───────────────────────────────────────────────────────────────

    def get_clusters(self) -> list[dict]:
        """GET /api/clusters"""
        return self._get("/api/clusters")

    def get_cluster(self, label: int) -> dict:
        """GET /api/clusters/{label}"""
        return self._get(f"/api/clusters/{label}")

    # ── Sentiment ──────────────────────────────────────────────────────────────

    def get_sentiment_overview(self) -> dict:
        """GET /api/sentiment/overview"""
        return self._get("/api/sentiment/overview")

    def get_sentiment_timeline(self) -> list[dict]:
        """GET /api/sentiment/timeline"""
        return self._get("/api/sentiment/timeline")

    # ── Pipeline ───────────────────────────────────────────────────────────────

    def pipeline_status(self) -> dict:
        """GET /api/pipeline/status"""
        return self._get("/api/pipeline/status")

    def run_pipeline(
        self,
        api_key:       str  = "",
        run_fetch:     bool = True,
        run_embed:     bool = True,
        run_cluster:   bool = True,
        run_sentiment: bool = True,
        days_back:     int  = 7,
    ) -> dict:
        """POST /api/pipeline/run"""
        return self._post("/api/pipeline/run", {
            "api_key":       api_key,
            "run_fetch":     run_fetch,
            "run_embed":     run_embed,
            "run_cluster":   run_cluster,
            "run_sentiment": run_sentiment,
            "days_back":     days_back,
        })


# ── Singleton ──────────────────────────────────────────────────────────────────
# Import this in app.py: from frontend.api_client import client
client = PulseIQClient()
