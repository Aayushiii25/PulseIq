"""
backend/database.py — PulseIQ Database Layer (SQLAlchemy)
==========================================================
Manages all DB interactions for the PulseIQ platform using SQLAlchemy.
Supports both SQLite (local dev) and PostgreSQL (production).
"""

import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, LargeBinary, ForeignKey, DateTime, select
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.exc import IntegrityError
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from config import settings

engine = create_engine(settings.database_url, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ── Models ─────────────────────────────────────────────────────────────────────

class Article(Base):
    __tablename__ = "articles"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    title = Column(Text, nullable=False)
    description = Column(Text)
    source = Column(String)
    url = Column(String, unique=True, nullable=False, index=True)
    published_at = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    embedding = relationship("Embedding", back_populates="article", uselist=False, cascade="all, delete-orphan")
    cluster = relationship("Cluster", back_populates="article", uselist=False, cascade="all, delete-orphan")
    sentiment = relationship("Sentiment", back_populates="article", uselist=False, cascade="all, delete-orphan")


class Embedding(Base):
    __tablename__ = "embeddings"
    article_id = Column(Integer, ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True)
    vector = Column(LargeBinary, nullable=False)
    dim = Column(Integer, nullable=False)
    
    article = relationship("Article", back_populates="embedding")


class Cluster(Base):
    __tablename__ = "clusters"
    article_id = Column(Integer, ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True)
    cluster_label = Column(Integer, nullable=False)
    
    article = relationship("Article", back_populates="cluster")


class Sentiment(Base):
    __tablename__ = "sentiment"
    article_id = Column(Integer, ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True)
    label = Column(String, nullable=False)
    sentiment_score = Column(Float, nullable=False)
    
    article = relationship("Article", back_populates="sentiment")


class ClusterHistory(Base):
    __tablename__ = "cluster_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_timestamp = Column(DateTime, default=datetime.utcnow)
    current_label = Column(Integer, nullable=False)
    previous_label = Column(Integer)
    similarity = Column(Float)
    human_name = Column(String)


class MarketCorrelation(Base):
    __tablename__ = "market_correlation"
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String, unique=True, nullable=False)
    avg_sentiment = Column(Float, nullable=False)
    market_return = Column(Float, nullable=False)


# ── DB Initialisation ──────────────────────────────────────────────────────────

def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    print(f"✅ Database ready → {settings.database_url}")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _get_insert_stmt(table):
    if "sqlite" in settings.database_url:
        return sqlite_insert(table)
    return pg_insert(table)

# ── articles helpers ───────────────────────────────────────────────────────────

def insert_articles(articles: list[dict]) -> int:
    """Bulk-insert articles, ignoring duplicates by URL."""
    if not articles:
        return 0
    with SessionLocal() as db:
        stmt = _get_insert_stmt(Article).values(articles)
        stmt = stmt.on_conflict_do_nothing(index_elements=['url'])
        res = db.execute(stmt)
        db.commit()
        return res.rowcount


def fetch_all_articles() -> list[dict]:
    with SessionLocal() as db:
        rows = db.scalars(select(Article).order_by(Article.id)).all()
        return [
            {
                "id": r.id,
                "title": r.title,
                "description": r.description,
                "source": r.source,
                "url": r.url,
                "published_at": r.published_at,
            }
            for r in rows
        ]


def article_count() -> int:
    with SessionLocal() as db:
        return db.query(Article).count()


# ── embeddings helpers ─────────────────────────────────────────────────────────

def upsert_embedding(article_id: int, vector: np.ndarray) -> None:
    upsert_embeddings_batch([article_id], np.array([vector]))


def fetch_all_embeddings() -> tuple[list[int], np.ndarray]:
    with SessionLocal() as db:
        rows = db.scalars(select(Embedding).order_by(Embedding.article_id)).all()
        if not rows:
            return [], np.array([])
        ids = [r.article_id for r in rows]
        vectors = [np.frombuffer(r.vector, dtype=np.float32).reshape(r.dim) for r in rows]
        return ids, np.vstack(vectors)


def upsert_embeddings_batch(article_ids: list[int], vectors: np.ndarray) -> None:
    if not article_ids:
        return
    data = []
    for aid, vec in zip(article_ids, vectors):
        data.append({"article_id": aid, "vector": vec.astype(np.float32).tobytes(), "dim": len(vec)})
    
    with SessionLocal() as db:
        stmt = _get_insert_stmt(Embedding).values(data)
        stmt = stmt.on_conflict_do_update(
            index_elements=['article_id'],
            set_=dict(vector=stmt.excluded.vector, dim=stmt.excluded.dim)
        )
        db.execute(stmt)
        db.commit()


def get_embedded_article_ids() -> set[int]:
    with SessionLocal() as db:
        rows = db.execute(select(Embedding.article_id)).all()
        return {r[0] for r in rows}


# ── clusters helpers ───────────────────────────────────────────────────────────

def upsert_clusters(article_ids: list[int], labels: list[int]) -> None:
    if not article_ids:
        return
    data = [{"article_id": aid, "cluster_label": lbl} for aid, lbl in zip(article_ids, labels)]
    with SessionLocal() as db:
        stmt = _get_insert_stmt(Cluster).values(data)
        stmt = stmt.on_conflict_do_update(
            index_elements=['article_id'],
            set_=dict(cluster_label=stmt.excluded.cluster_label)
        )
        db.execute(stmt)
        db.commit()


def fetch_clusters() -> dict[int, int]:
    with SessionLocal() as db:
        rows = db.execute(select(Cluster.article_id, Cluster.cluster_label)).all()
        return {r[0]: r[1] for r in rows}


# ── sentiment helpers ──────────────────────────────────────────────────────────

def upsert_sentiment(article_id: int, label: str, score: float) -> None:
    upsert_sentiments_batch([(article_id, label, score)])


def fetch_sentiment() -> dict[int, dict]:
    with SessionLocal() as db:
        rows = db.execute(select(Sentiment.article_id, Sentiment.label, Sentiment.sentiment_score)).all()
        return {r[0]: {"label": r[1], "score": r[2]} for r in rows}


def upsert_sentiments_batch(records: list[tuple[int, str, float]]) -> None:
    if not records:
        return
    data = [{"article_id": r[0], "label": r[1], "sentiment_score": r[2]} for r in records]
    with SessionLocal() as db:
        stmt = _get_insert_stmt(Sentiment).values(data)
        stmt = stmt.on_conflict_do_update(
            index_elements=['article_id'],
            set_=dict(label=stmt.excluded.label, sentiment_score=stmt.excluded.sentiment_score)
        )
        db.execute(stmt)
        db.commit()


def get_analysed_article_ids() -> set[int]:
    with SessionLocal() as db:
        rows = db.execute(select(Sentiment.article_id)).all()
        return {r[0] for r in rows}


# ── combined view ──────────────────────────────────────────────────────────────

def fetch_enriched_articles() -> list[dict]:
    with SessionLocal() as db:
        query = (
            select(
                Article.id, Article.title, Article.description, Article.source,
                Article.url, Article.published_at,
                Cluster.cluster_label,
                Sentiment.label.label("sentiment_label"), Sentiment.sentiment_score
            )
            .outerjoin(Cluster, Article.id == Cluster.article_id)
            .outerjoin(Sentiment, Article.id == Sentiment.article_id)
            .order_by(Article.id)
        )
        rows = db.execute(query).all()
        
        return [
            {
                "id": r.id,
                "title": r.title,
                "description": r.description,
                "source": r.source,
                "url": r.url,
                "published_at": r.published_at,
                "cluster_label": r.cluster_label,
                "sentiment_label": r.sentiment_label,
                "sentiment_score": r.sentiment_score
            }
            for r in rows
        ]


# ── cluster tracking and history ──────────────────────────────────────────────────────────────

def insert_cluster_history(history_records: list[dict]) -> None:
    """Bulk insert cluster history records"""
    if not history_records:
        return
    with SessionLocal() as db:
        db.bulk_insert_mappings(ClusterHistory, history_records)
        db.commit()

def get_latest_cluster_history() -> list[dict]:
    with SessionLocal() as db:
        latest_run = db.scalar(select(ClusterHistory.run_timestamp).order_by(ClusterHistory.run_timestamp.desc()).limit(1))
        if not latest_run:
            return []
        rows = db.scalars(select(ClusterHistory).where(ClusterHistory.run_timestamp == latest_run)).all()
        return [
            {
                "current_label": r.current_label,
                "previous_label": r.previous_label,
                "similarity": r.similarity,
                "human_name": r.human_name
            } for r in rows
        ]

# ── market correlation ──────────────────────────────────────────────────────────────

def upsert_market_correlation(date: str, avg_sentiment: float, market_return: float) -> None:
    data = {"date": date, "avg_sentiment": avg_sentiment, "market_return": market_return}
    with SessionLocal() as db:
        stmt = _get_insert_stmt(MarketCorrelation).values([data])
        stmt = stmt.on_conflict_do_update(
            index_elements=['date'],
            set_=dict(avg_sentiment=stmt.excluded.avg_sentiment, market_return=stmt.excluded.market_return)
        )
        db.execute(stmt)
        db.commit()

def get_market_correlations() -> list[dict]:
    with SessionLocal() as db:
        rows = db.scalars(select(MarketCorrelation).order_by(MarketCorrelation.date)).all()
        return [
            {"date": r.date, "avg_sentiment": r.avg_sentiment, "market_return": r.market_return}
            for r in rows
        ]

if __name__ == "__main__":
    init_db()
