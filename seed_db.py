"""
seed_db.py — Load seed article data into the SQLite database.

This script populates the DB with the pre-generated dataset in
data/seed_articles.json so you can run the ML pipeline immediately
without waiting for a NewsAPI fetch.

Usage:
    python seed_db.py
"""

import json
import sqlite3
from pathlib import Path

DB_PATH   = Path("data/news.db")
SEED_FILE = Path("data/seed_articles.json")


def main():
    if not SEED_FILE.exists():
        print(f"❌  Seed file not found: {SEED_FILE}")
        return

    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")

    # Create tables if they don't exist
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS articles (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        title        TEXT    NOT NULL,
        description  TEXT,
        source       TEXT,
        url          TEXT    UNIQUE NOT NULL,
        published_at TEXT
    );
    CREATE TABLE IF NOT EXISTS embeddings (
        article_id  INTEGER PRIMARY KEY REFERENCES articles(id) ON DELETE CASCADE,
        vector      BLOB    NOT NULL,
        dim         INTEGER NOT NULL
    );
    CREATE TABLE IF NOT EXISTS clusters (
        article_id    INTEGER PRIMARY KEY REFERENCES articles(id) ON DELETE CASCADE,
        cluster_label INTEGER NOT NULL
    );
    CREATE TABLE IF NOT EXISTS sentiment (
        article_id      INTEGER PRIMARY KEY REFERENCES articles(id) ON DELETE CASCADE,
        label           TEXT    NOT NULL,
        sentiment_score REAL    NOT NULL
    );
    """)

    with open(SEED_FILE) as f:
        articles = json.load(f)

    inserted = skipped = 0
    for a in articles:
        try:
            conn.execute(
                "INSERT INTO articles (title, description, source, url, published_at) VALUES (?,?,?,?,?)",
                (a["title"], a.get("description"), a["source"], a["url"], a.get("published_at")),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            skipped += 1

    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    conn.close()

    print(f"✅  Inserted: {inserted}  |  Skipped (dupes): {skipped}  |  Total in DB: {total}")


if __name__ == "__main__":
    main()
