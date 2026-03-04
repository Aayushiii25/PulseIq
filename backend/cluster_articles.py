"""
backend/cluster_articles.py — PulseIQ Dimensionality Reduction & Clustering
=============================================================================
Transforms high-dimensional article embeddings into interpretable clusters
using the pipeline:   PCA  →  UMAP  →  HDBSCAN

Why this three-stage pipeline?
───────────────────────────────
1. PCA (50 dims)
   Raw GTE-small vectors are 384-dimensional.  PCA is a linear transform
   that keeps the directions of greatest variance.  Reducing to 50 dims:
     • removes noise dimensions that would confuse UMAP,
     • speeds up UMAP significantly (O(n·d) complexity),
     • and preserves ~95 %+ of explained variance for typical news corpora.

2. UMAP (5 dims)
   UMAP is a non-linear manifold-learning algorithm.  It preserves local
   neighbourhood structure far better than t-SNE and is much faster.
   5 dimensions gives HDBSCAN enough geometric resolution to find clusters
   while being low enough to avoid the curse of dimensionality.
   (2 dims are also stored separately for the 2-D scatter-plot in the UI.)

3. HDBSCAN
   Hierarchical DBSCAN auto-detects the number of clusters and assigns
   noisy/outlier articles the special label -1.  It outperforms k-Means
   for news because:
     • financial themes are irregularly shaped blobs, not spheres,
     • the number of themes is unknown in advance,
     • it gracefully marks ambiguous articles as noise.

Outputs stored in SQLite:
  clusters table  — cluster_label per article  (-1 == noise)
  models/         — pickled PCA + UMAP models (for reuse / inspection)

Usage:
    python -m backend.cluster_articles
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import hdbscan
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from backend.database import fetch_all_embeddings, fetch_all_articles, upsert_clusters

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(exist_ok=True)

UMAP_2D_PATH   = MODELS_DIR / "umap_2d_coords.npy"   # for scatter plot
ARTICLE_ID_PATH = MODELS_DIR / "clustered_ids.npy"    # id order matching coords

# ── Hyper-parameters ───────────────────────────────────────────────────────────
PCA_COMPONENTS   = 50
UMAP_COMPONENTS  = 5
UMAP_NEIGHBORS   = 15   # larger → more global structure preserved
UMAP_MIN_DIST    = 0.0  # 0.0 encourages tight, well-separated clusters
HDBSCAN_MIN_CLUSTER = 3  # minimum articles to form a cluster
HDBSCAN_MIN_SAMPLES = 2  # controls density threshold; lower = more clusters
RANDOM_STATE     = 42


# ── Pipeline stages ────────────────────────────────────────────────────────────

def run_pca(embeddings: np.ndarray) -> np.ndarray:
    """
    Reduce 384-dim embeddings to PCA_COMPONENTS dims.
    StandardScaler is applied first so all dimensions contribute equally.
    """
    log.info("📉  PCA: %d → %d dimensions …", embeddings.shape[1], PCA_COMPONENTS)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)

    n_components = min(PCA_COMPONENTS, scaled.shape[0], scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    reduced = pca.fit_transform(scaled)

    explained = pca.explained_variance_ratio_.sum()
    log.info("   Explained variance: %.1f%%", explained * 100)

    # Persist model for later inspection / incremental updates
    with open(MODELS_DIR / "pca_model.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "pca": pca}, f)

    return reduced.astype(np.float32)


def run_umap(pca_embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply UMAP twice:
      • 5-D reduction   → fed into HDBSCAN
      • 2-D reduction   → saved for scatter-plot visualisation

    Returns:
        (umap_5d, umap_2d)
    """
    n_neighbors = min(UMAP_NEIGHBORS, len(pca_embeddings) - 1)

    log.info("🌀  UMAP 5-D reduction (n_neighbors=%d) …", n_neighbors)
    reducer_5d = umap.UMAP(
        n_components=UMAP_COMPONENTS,
        n_neighbors=n_neighbors,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=RANDOM_STATE,
        low_memory=True,   # important for M2 RAM budget
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
    """
    Cluster the 5-D UMAP representation with HDBSCAN.

    cluster_selection_epsilon=0.0 lets HDBSCAN self-tune the cut threshold.
    prediction_data=True allows soft-membership queries later.
    """
    min_cluster = min(HDBSCAN_MIN_CLUSTER, max(2, len(umap_embeddings) // 10))
    log.info("🔍  HDBSCAN (min_cluster_size=%d) …", min_cluster)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",   # excess-of-mass: stable clusters
        prediction_data=True,
    )
    labels = clusterer.fit_predict(umap_embeddings)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    log.info("   Found %d clusters | %d noise articles", n_clusters, n_noise)

    with open(MODELS_DIR / "hdbscan_model.pkl", "wb") as f:
        pickle.dump(clusterer, f)

    return labels


# ── Top-level pipeline ─────────────────────────────────────────────────────────

def cluster_articles() -> dict:
    """
    Run the full PCA → UMAP → HDBSCAN pipeline.

    Returns summary dict with cluster statistics.
    """
    log.info("━" * 55)
    log.info("  PulseIQ — Clustering Pipeline")
    log.info("━" * 55)

    # 1. Load embeddings
    ids, embeddings = fetch_all_embeddings()
    if len(ids) == 0:
        log.warning("No embeddings in DB — run embed_articles.py first.")
        return {}

    log.info("📐  Loaded %d embeddings (dim=%d)", len(ids), embeddings.shape[1])

    if len(ids) < 5:
        log.warning("Too few articles (%d) for meaningful clustering. Need ≥ 5.", len(ids))
        return {}

    # 2. PCA
    pca_embeddings = run_pca(embeddings)

    # 3. UMAP
    umap_5d, umap_2d = run_umap(pca_embeddings)

    # 4. HDBSCAN
    labels = run_hdbscan(umap_5d)

    # 5. Persist cluster labels
    upsert_clusters(ids, labels.tolist())

    # 6. Save 2-D coordinates + ID order for the UI scatter plot
    np.save(UMAP_2D_PATH, umap_2d)
    np.save(ARTICLE_ID_PATH, np.array(ids))

    # 7. Build summary
    unique_labels = sorted(set(labels))
    summary = {
        "total_articles": len(ids),
        "n_clusters":     len([l for l in unique_labels if l != -1]),
        "noise_articles": int((labels == -1).sum()),
        "cluster_sizes":  {
            int(l): int((labels == l).sum())
            for l in unique_labels if l != -1
        },
    }

    log.info("━" * 55)
    log.info("✅  Clustering complete!")
    log.info("   Clusters : %d", summary["n_clusters"])
    log.info("   Noise    : %d", summary["noise_articles"])
    log.info("━" * 55)

    return summary


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    summary = cluster_articles()
    if summary:
        print(f"\n📊  {summary['n_clusters']} clusters discovered across {summary['total_articles']} articles.")
        for cluster_id, size in summary["cluster_sizes"].items():
            print(f"   Cluster {cluster_id:>3}: {size} articles")
