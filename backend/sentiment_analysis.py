"""
backend/sentiment_analysis.py — PulseIQ FinBERT Sentiment Module
=================================================================
Runs ProsusAI/finbert — a BERT model fine-tuned on financial text —
on every article and stores a signed sentiment score in SQLite.

Why FinBERT instead of VADER / TextBlob?
  General-purpose lexicon models misclassify finance phrases like
  "the Fed cut rates" (positive for equities, misread as negative by
  VADER because of "cut").  FinBERT was trained on financial news and
  analyst reports, making its labels far more reliable in this domain.

Output
------
  label : "positive" | "neutral" | "negative"
  score : float in [-1.0, +1.0]
    Computed as:  P(positive) - P(negative)
    Neutral articles hover near 0; strong positives approach +1.

Usage:
    python -m backend.sentiment_analysis
"""

import logging

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from backend.database import fetch_all_articles, upsert_sentiments_batch, get_analysed_article_ids

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_NAME  = "ProsusAI/finbert"
BATCH_SIZE  = 16     # FinBERT is ~110 M params; smaller batches are safer
MAX_TOKENS  = 512    # BERT hard limit; titles + descriptions usually fit
LABEL_ORDER = ["positive", "negative", "neutral"]  # FinBERT output order


# ── Device detection ───────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        log.info("🍎  Using Apple Silicon MPS.")
        return torch.device("mps")
    if torch.cuda.is_available():
        log.info("⚡  Using CUDA GPU.")
        return torch.device("cuda")
    log.info("💻  Using CPU.")
    return torch.device("cpu")


# ── Text preparation ───────────────────────────────────────────────────────────

def build_text(article: dict) -> str:
    """
    Concatenate title and description.
    FinBERT was trained on short financial sentences, so we truncate
    via the tokeniser's max_length rather than pre-slicing here.
    """
    title = (article.get("title") or "").strip()
    desc  = (article.get("description") or "").strip()
    return f"{title}. {desc}" if desc else title


# ── Signed score helper ────────────────────────────────────────────────────────

def compute_score(probs: list[float]) -> tuple[str, float]:
    """
    Convert FinBERT softmax probabilities to a single signed score.

    FinBERT returns three logits in order: [positive, negative, neutral].
    We define:
        score = P(positive) - P(negative)

    This gives an intuitive range:
        +1.0  →  article is overwhelmingly positive
         0.0  →  balanced or neutral
        -1.0  →  article is overwhelmingly negative

    Args:
        probs — list of 3 floats from softmax output [pos, neg, neu]

    Returns:
        (label, score)  where label is the argmax class
    """
    p_pos, p_neg, p_neu = probs
    score = round(float(p_pos - p_neg), 4)

    # Label is simply the highest-probability class
    max_idx = probs.index(max(probs))
    label   = LABEL_ORDER[max_idx]

    return label, score


# ── Batch inference ────────────────────────────────────────────────────────────

def analyse_batch(
    texts: list[str],
    tokeniser: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
) -> list[tuple[str, float]]:
    """
    Run FinBERT on a single batch of texts.

    Returns:
        List of (label, score) tuples, one per input text.
    """
    encoded = tokeniser(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_TOKENS,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**encoded).logits

    probs_batch = torch.softmax(logits, dim=-1).tolist()
    return [compute_score(p) for p in probs_batch]


# ── Main routine ───────────────────────────────────────────────────────────────

def run_sentiment_analysis(batch_size: int = BATCH_SIZE) -> int:
    """
    Analyse every article in the DB and persist sentiment to SQLite.

    Returns:
        Number of articles analysed.
    """
    all_articles = fetch_all_articles()
    if not all_articles:
        log.warning("No articles in DB — run fetch_news.py first.")
        return 0

    # Skip articles that already have sentiment (incremental mode)
    already_done = get_analysed_article_ids()
    articles = [a for a in all_articles if a["id"] not in already_done]

    if not articles:
        log.info("✅  All %d articles already analysed — nothing to do.", len(all_articles))
        return 0

    log.info("📰  Analysing sentiment for %d new articles (%d already done).", len(articles), len(already_done))

    device = get_device()
    log.info("📦  Loading FinBERT ('%s') …", MODEL_NAME)
    tokeniser = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    total = len(articles)
    processed = 0
    results: list[tuple[int, str, float]] = []

    for batch_start in range(0, total, batch_size):
        batch   = articles[batch_start : batch_start + batch_size]
        texts   = [build_text(a) for a in batch]
        batch_results = analyse_batch(texts, tokeniser, model, device)

        for article, (label, score) in zip(batch, batch_results):
            results.append((article["id"], label, score))

        processed += len(batch)
        log.info("  Progress: %d / %d", processed, total)

    # Batch write all results in one connection
    upsert_sentiments_batch(results)

    log.info("✅  Sentiment analysis complete.")
    return total


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    n = run_sentiment_analysis()
    print(f"\n💬  Sentiment analysed for {n} articles.")
