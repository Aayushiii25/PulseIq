"""
backend/database.py — PulseIQ Database Layer
=============================================
Manages all SQLite interactions for the PulseIQ platform.

Schema overview
---------------
articles    — raw news articles fetched from NewsAPI
embeddings  — serialised sentence-transformer vectors (one row per article)
clusters    — HDBSCAN cluster label assigned to each article
sentiment   — FinBERT compound sentiment score for each article

Design notes
------------
* Vectors are stored as raw bytes (numpy .tobytes()) so no extra dependency
  like sqlite-vss is required. At retrieval time they are deserialised back
  into float32 numpy arrays.
* All foreign keys reference articles.id — deleting an article cascades
  cleanly if you rebuild a run from scratch.
"""

import sqlite3
import numpy as np
from pathlib import Path

# ── Path configuration ─────────────────────────────────────────────────────────
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "news.db"


# ── Connection helper ──────────────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    """Return a connection with foreign-key enforcement enabled."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row   # columns accessible by name
    return conn


# ── Schema initialisation ──────────────────────────────────────────────────────

def init_db() -> None:
    """
    Create all tables if they don't already exist.
    Safe to call on every startup — uses IF NOT EXISTS.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # ── articles ──────────────────────────────────────────────────────────────
    # Stores the raw payload returned by NewsAPI.
    # `url` is UNIQUE so re-running the fetcher never duplicates articles.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            title        TEXT    NOT NULL,
            description  TEXT,
            source       TEXT,
            url          TEXT    UNIQUE NOT NULL,
            published_at TEXT
        )
    """)

    # ── embeddings ────────────────────────────────────────────────────────────
    # Stores the sentence-transformer vector for each article as a binary blob.
    # The `dim` column records vector length so deserialisation never guesses.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            article_id  INTEGER PRIMARY KEY
                            REFERENCES articles(id) ON DELETE CASCADE,
            vector      BLOB    NOT NULL,
            dim         INTEGER NOT NULL
        )
    """)

    # ── clusters ─────────────────────────────────────────────────────────────
    # HDBSCAN assigns each article an integer label.
    # Label -1 means "noise" (article didn't fit any cluster).
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS clusters (
            article_id    INTEGER PRIMARY KEY
                              REFERENCES articles(id) ON DELETE CASCADE,
            cluster_label INTEGER NOT NULL
        )
    """)

    # ── sentiment ─────────────────────────────────────────────────────────────
    # FinBERT outputs a score in [-1, +1]:
    #   +1 → strongly positive   0 → neutral   -1 → strongly negative
    # `label` stores the raw class string: "positive" | "neutral" | "negative"
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiment (
            article_id      INTEGER PRIMARY KEY
                                REFERENCES articles(id) ON DELETE CASCADE,
            label           TEXT    NOT NULL,
            sentiment_score REAL    NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    print(f"✅  Database ready → {DB_PATH}")


# ── articles helpers ───────────────────────────────────────────────────────────

def insert_articles(articles: list[dict]) -> int:
    """
    Bulk-insert a list of article dicts.
    Silently skips duplicates (same URL).
    Returns the number of newly inserted rows.
    """
    conn  = get_connection()
    cursor = conn.cursor()
    inserted = 0
    for a in articles:
        try:
            cursor.execute(
                """INSERT INTO articles (title, description, source, url, published_at)
                   VALUES (:title, :description, :source, :url, :published_at)""",
                a,
            )
            inserted += 1
        except sqlite3.IntegrityError:
            pass   # duplicate URL — skip
    conn.commit()
    conn.close()
    return inserted


def fetch_all_articles() -> list[dict]:
    """Return every article as a plain dict."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, title, description, source, url, published_at FROM articles ORDER BY id"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def article_count() -> int:
    conn = get_connection()
    n = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    conn.close()
    return n


# ── embeddings helpers ─────────────────────────────────────────────────────────

def upsert_embedding(article_id: int, vector: np.ndarray) -> None:
    """Save or overwrite the embedding for one article."""
    blob = vector.astype(np.float32).tobytes()
    conn = get_connection()
    conn.execute(
        """INSERT INTO embeddings (article_id, vector, dim)
           VALUES (?, ?, ?)
           ON CONFLICT(article_id) DO UPDATE SET vector=excluded.vector, dim=excluded.dim""",
        (article_id, blob, len(vector)),
    )
    conn.commit()
    conn.close()


def fetch_all_embeddings() -> tuple[list[int], np.ndarray]:
    """
    Return (article_ids, embedding_matrix).
    embedding_matrix shape: (n_articles, embedding_dim)
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT article_id, vector, dim FROM embeddings ORDER BY article_id"
    ).fetchall()
    conn.close()

    if not rows:
        return [], np.array([])

    ids     = [r["article_id"] for r in rows]
    vectors = [np.frombuffer(r["vector"], dtype=np.float32).reshape(r["dim"]) for r in rows]
    return ids, np.vstack(vectors)


# ── clusters helpers ───────────────────────────────────────────────────────────

def upsert_clusters(article_ids: list[int], labels: list[int]) -> None:
    """Persist HDBSCAN cluster labels for a batch of articles."""
    conn = get_connection()
    conn.executemany(
        """INSERT INTO clusters (article_id, cluster_label)
           VALUES (?, ?)
           ON CONFLICT(article_id) DO UPDATE SET cluster_label=excluded.cluster_label""",
        zip(article_ids, labels),
    )
    conn.commit()
    conn.close()


def fetch_clusters() -> dict[int, int]:
    """Return {article_id: cluster_label} mapping."""
    conn = get_connection()
    rows = conn.execute("SELECT article_id, cluster_label FROM clusters").fetchall()
    conn.close()
    return {r["article_id"]: r["cluster_label"] for r in rows}


# ── sentiment helpers ──────────────────────────────────────────────────────────

def upsert_sentiment(article_id: int, label: str, score: float) -> None:
    conn = get_connection()
    conn.execute(
        """INSERT INTO sentiment (article_id, label, sentiment_score)
           VALUES (?, ?, ?)
           ON CONFLICT(article_id) DO UPDATE SET label=excluded.label,
                                                 sentiment_score=excluded.sentiment_score""",
        (article_id, label, score),
    )
    conn.commit()
    conn.close()


def fetch_sentiment() -> dict[int, dict]:
    """Return {article_id: {'label': str, 'score': float}}."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT article_id, label, sentiment_score FROM sentiment"
    ).fetchall()
    conn.close()
    return {r["article_id"]: {"label": r["label"], "score": r["sentiment_score"]} for r in rows}


# ── combined view ──────────────────────────────────────────────────────────────

def fetch_enriched_articles() -> list[dict]:
    """
    JOIN articles + clusters + sentiment into one flat record per article.
    Articles not yet clustered / analysed are included with NULL fields.
    """
    conn = get_connection()
    rows = conn.execute("""
        SELECT
            a.id,
            a.title,
            a.description,
            a.source,
            a.url,
            a.published_at,
            c.cluster_label,
            s.label        AS sentiment_label,
            s.sentiment_score
        FROM       articles  a
        LEFT JOIN  clusters  c ON c.article_id = a.id
        LEFT JOIN  sentiment s ON s.article_id = a.id
        ORDER BY   a.id
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
