"""
backend/embed_articles.py — PulseIQ Embedding Generation Module
================================================================
Converts raw article text into dense semantic vector representations
using the 'thenlper/gte-small' SentenceTransformer model.

Why thenlper/gte-small?
  - 384-dimensional embeddings: rich enough for clustering, small enough
    to run comfortably on a MacBook M2 with MPS acceleration.
  - Trained on large-scale text-pair data; strong on short financial sentences.
  - ~33 M parameters — fast inference without needing a GPU server.

Pipeline position:  articles  →  [THIS MODULE]  →  PCA / UMAP / HDBSCAN

Usage (standalone):
    python -m backend.embed_articles
"""

import logging

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from backend.database import fetch_all_articles, upsert_embedding

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL_ID   = "thenlper/gte-small"
BATCH_SIZE = 64     # safe default; bump to 128 on M2 for extra speed


# ── Device detection ───────────────────────────────────────────────────────────

def get_device() -> str:
    """
    Probe for the best available compute device.

    MPS (Metal Performance Shaders) is Apple's GPU compute framework.
    Both is_available() and is_built() must be True to use it safely —
    is_available() alone can return True on non-M-series Macs.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        log.info("🍎  Apple Silicon MPS detected.")
        return "mps"
    if torch.cuda.is_available():
        log.info("⚡  CUDA GPU detected: %s", torch.cuda.get_device_name(0))
        return "cuda"
    log.info("💻  No GPU — using CPU.")
    return "cpu"


# ── Text preparation ───────────────────────────────────────────────────────────

def build_text(article: dict) -> str:
    """
    Combine title + description into one string for the encoder.

    A period-space separator signals a sentence boundary, which helps the
    tokeniser produce better sub-word segmentation for short titles.
    """
    title = (article.get("title") or "").strip()
    desc  = (article.get("description") or "").strip()
    return f"{title}. {desc}" if desc else title


# ── Normalisation ──────────────────────────────────────────────────────────────

def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """
    Unit-normalise every row vector.

    After L2 normalisation, cosine similarity == dot product, which
    makes downstream HDBSCAN / UMAP distance calculations faster and
    numerically more stable.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0   # guard against all-zero vectors
    return matrix / norms


# ── Main embedding routine ─────────────────────────────────────────────────────

def embed_articles(batch_size: int = BATCH_SIZE) -> int:
    """
    Encode all articles in the DB and persist their vectors.

    Steps:
      1. Load articles from SQLite.
      2. Build combined text strings.
      3. Encode in batches using SentenceTransformer (MPS / CPU).
      4. L2-normalise the resulting matrix.
      5. Upsert each vector back to the `embeddings` table.

    Returns:
        Number of articles embedded.
    """
    articles = fetch_all_articles()
    if not articles:
        log.warning("No articles found in DB — run fetch_news.py first.")
        return 0

    log.info("📰  %d articles to embed.", len(articles))

    device = get_device()
    log.info("📦  Loading model '%s' …", MODEL_ID)
    model = SentenceTransformer(MODEL_ID, device=device)

    texts = [build_text(a) for a in articles]

    log.info("🔄  Encoding …")
    raw_embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,   # we normalise explicitly below
    ).astype(np.float32)

    embeddings = l2_normalise(raw_embeddings)
    log.info("✅  Embeddings shape: %s", embeddings.shape)

    log.info("💾  Persisting vectors to DB …")
    for article, vector in zip(articles, embeddings):
        upsert_embedding(article["id"], vector)

    log.info("✅  %d embeddings saved.", len(articles))
    return len(articles)


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    n = embed_articles()
    print(f"\n🧠  {n} articles embedded and stored.")
