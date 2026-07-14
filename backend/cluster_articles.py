"""
backend/cluster_articles.py — PulseIQ Dimensionality Reduction & Clustering
=============================================================================
Transforms high-dimensional article embeddings into interpretable clusters
using the pipeline:   PCA  →  UMAP  →  HDBSCAN
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import hdbscan
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.database import fetch_all_embeddings, fetch_all_articles, upsert_clusters, get_latest_cluster_history, insert_cluster_history

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(exist_ok=True)

UMAP_2D_PATH   = MODELS_DIR / "umap_2d_coords.npy"
ARTICLE_ID_PATH = MODELS_DIR / "clustered_ids.npy"

PCA_COMPONENTS   = 50
UMAP_COMPONENTS  = 5
UMAP_NEIGHBORS   = 15
UMAP_MIN_DIST    = 0.0
HDBSCAN_MIN_CLUSTER = 3
HDBSCAN_MIN_SAMPLES = 2
RANDOM_STATE     = 42

def run_pca(embeddings: np.ndarray) -> np.ndarray:
    log.info("📉  PCA: %d → %d dimensions …", embeddings.shape[1], PCA_COMPONENTS)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)

    n_components = min(PCA_COMPONENTS, scaled.shape[0], scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    reduced = pca.fit_transform(scaled)

    with open(MODELS_DIR / "pca_model.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "pca": pca}, f)

    return reduced.astype(np.float32)

def run_umap(pca_embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_neighbors = min(UMAP_NEIGHBORS, len(pca_embeddings) - 1)
    
    log.info("🌀  UMAP 5-D reduction (n_neighbors=%d) …", n_neighbors)
    reducer_5d = umap.UMAP(
        n_components=UMAP_COMPONENTS,
        n_neighbors=n_neighbors,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=RANDOM_STATE,
        low_memory=True,
    )
    coords_5d = reducer_5d.fit_transform(pca_embeddings)

    log.info("🌀  UMAP 2-D reduction (for visualisation) …")
    reducer_2d = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=RANDOM_STATE,
        low_memory=True,
    )
    coords_2d = reducer_2d.fit_transform(pca_embeddings)

    with open(MODELS_DIR / "umap_reducer.pkl", "wb") as f:
        pickle.dump({"umap_5d": reducer_5d, "umap_2d": reducer_2d}, f)

    return coords_5d.astype(np.float32), coords_2d.astype(np.float32)


def run_hdbscan(umap_embeddings: np.ndarray) -> np.ndarray:
    min_cluster = min(HDBSCAN_MIN_CLUSTER, max(2, len(umap_embeddings) // 10))
    log.info("🔍  HDBSCAN (min_cluster_size=%d) …", min_cluster)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(umap_embeddings)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    log.info("   Found %d clusters | %d noise articles", n_clusters, n_noise)

    with open(MODELS_DIR / "hdbscan_model.pkl", "wb") as f:
        pickle.dump(clusterer, f)

    return labels


def extract_cluster_names(articles: list[dict], labels: list[int]) -> dict[int, str]:
    """Use TF-IDF to assign a human-readable name (top 3 words) to each cluster."""
    clusters_text = {}
    
    for a, lbl in zip(articles, labels):
        if lbl == -1:
            continue
        text = f"{a.get('title', '')} {a.get('description', '')}"
        clusters_text[lbl] = clusters_text.get(lbl, "") + " " + text
        
    cluster_names = {-1: "Noise"}
    if not clusters_text:
        return cluster_names
        
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=1, max_features=1000)
    
    for lbl, text in clusters_text.items():
        if not text.strip():
            cluster_names[lbl] = f"Cluster {lbl}"
            continue
            
        try:
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            if len(feature_names) == 0:
                cluster_names[lbl] = f"Cluster {lbl}"
                continue
                
            scores = tfidf_matrix.toarray()[0]
            top_indices = scores.argsort()[-3:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            cluster_names[lbl] = " ".join(top_words).title()
        except ValueError:
            cluster_names[lbl] = f"Cluster {lbl}"
            
    return cluster_names


def track_cluster_history(current_labels: list[int], article_ids: list[int], names: dict[int, str]):
    """Track Jaccard similarity of clusters over time."""
    prev_history = get_latest_cluster_history()
    
    current_clusters = {}
    for aid, lbl in zip(article_ids, current_labels):
        if lbl != -1:
            current_clusters.setdefault(lbl, set()).add(aid)
            
    # Load previous cluster assignments to calculate similarity (mocked from prev_history DB or from articles)
    # Actually, previous labels are on the articles themselves before we upsert!
    # Let's get the previous labels from the DB.
    from backend.database import fetch_clusters
    old_labels_map = fetch_clusters()
    
    old_clusters = {}
    for aid, old_lbl in old_labels_map.items():
        if old_lbl != -1:
            old_clusters.setdefault(old_lbl, set()).add(aid)
            
    history_records = []
    
    from datetime import datetime
    now = datetime.utcnow()
    
    for curr_lbl, curr_set in current_clusters.items():
        best_match_old_lbl = None
        best_sim = 0.0
        
        for old_lbl, old_set in old_clusters.items():
            intersection = len(curr_set.intersection(old_set))
            union = len(curr_set.union(old_set))
            jaccard = intersection / union if union > 0 else 0
            
            if jaccard > best_sim:
                best_sim = jaccard
                best_match_old_lbl = old_lbl
                
        history_records.append({
            "run_timestamp": now,
            "current_label": int(curr_lbl),
            "previous_label": int(best_match_old_lbl) if best_match_old_lbl is not None else None,
            "similarity": float(best_sim),
            "human_name": names.get(curr_lbl, f"Cluster {curr_lbl}")
        })
        
    if history_records:
        insert_cluster_history(history_records)


def cluster_articles() -> dict:
    log.info("━" * 55)
    log.info("  PulseIQ — Clustering Pipeline")
    log.info("━" * 55)

    ids, embeddings = fetch_all_embeddings()
    if len(ids) == 0:
        log.warning("No embeddings in DB — run embed_articles.py first.")
        return {}

    log.info("📐  Loaded %d embeddings (dim=%d)", len(ids), embeddings.shape[1])

    if len(ids) < 5:
        log.warning("Too few articles (%d) for meaningful clustering. Need ≥ 5.", len(ids))
        return {}

    pca_embeddings = run_pca(embeddings)
    umap_5d, umap_2d = run_umap(pca_embeddings)
    labels = run_hdbscan(umap_5d)
    
    # Generate names and track history
    all_articles = fetch_all_articles()
    articles_by_id = {a['id']: a for a in all_articles}
    sorted_articles = [articles_by_id[i] for i in ids]
    
    cluster_names = extract_cluster_names(sorted_articles, labels)
    track_cluster_history(labels, ids, cluster_names)

    upsert_clusters(ids, labels.tolist())
    np.save(UMAP_2D_PATH, umap_2d)
    np.save(ARTICLE_ID_PATH, np.array(ids))

    unique_labels = sorted(set(labels))
    summary = {
        "total_articles": len(ids),
        "n_clusters":     len([l for l in unique_labels if l != -1]),
        "noise_articles": int((labels == -1).sum()),
        "cluster_sizes":  {
            int(l): int((labels == l).sum())
            for l in unique_labels if l != -1
        },
        "cluster_names": cluster_names
    }

    log.info("━" * 55)
    log.info("✅  Clustering complete!")
    log.info("   Clusters : %d", summary["n_clusters"])
    log.info("   Noise    : %d", summary["noise_articles"])
    for l, name in cluster_names.items():
        if l != -1:
            log.info(f"   Cluster {l} -> {name}")
    log.info("━" * 55)

    return summary


if __name__ == "__main__":
    summary = cluster_articles()
    if summary:
        print(f"\n📊  {summary['n_clusters']} clusters discovered across {summary['total_articles']} articles.")
        for cluster_id, size in summary["cluster_sizes"].items():
            print(f"   Cluster {cluster_id:>3}: {size} articles ({summary['cluster_names'].get(cluster_id)})")
