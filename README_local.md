# 📈 PulseIQ — Financial Narrative Intelligence Platform

Automatically discover emerging financial themes from market news using
transformer embeddings, UMAP clustering, and FinBERT sentiment analysis.

---

## 📁 Exact Folder Structure (save it exactly like this in VS Code)

```
PulseIQ/                          ← Open THIS folder in VS Code
│
├── .env                          ← API keys (already filled in)
├── .vscode/
│   ├── settings.json             ← Python path, formatter config
│   ├── launch.json               ← Debug configs for API + Streamlit
│   └── extensions.json           ← Recommended VS Code extensions
│
├── api/                          ← REST API layer (FastAPI)
│   ├── __init__.py
│   ├── main.py                   ← FastAPI app, CORS, router registration
│   ├── schemas.py                ← Pydantic response models
│   └── routers/
│       ├── __init__.py
│       ├── stats.py              ← GET /api/stats
│       ├── articles.py           ← GET /api/articles  (paginated + search)
│       ├── clusters.py           ← GET /api/clusters
│       ├── sentiment.py          ← GET /api/sentiment/overview|timeline
│       └── pipeline.py           ← POST /api/pipeline/run
│
├── backend/                      ← ML pipeline
│   ├── __init__.py
│   ├── database.py               ← SQLite helpers (all 4 tables)
│   ├── fetch_news.py             ← NewsAPI ingestion
│   ├── embed_articles.py         ← GTE-small embeddings (MPS on M2)
│   ├── cluster_articles.py       ← PCA → UMAP → HDBSCAN
│   ├── sentiment_analysis.py     ← FinBERT scoring
│   └── pipeline.py               ← Orchestrator
│
├── frontend/                     ← Frontend helpers
│   ├── __init__.py
│   └── api_client.py             ← HTTP client (Streamlit uses this only)
│
├── data/                         ← Auto-created by the app
│   ├── news.db                   ← SQLite database  ✅ pre-seeded
│   └── seed_articles.json        ← 295 financial articles  ✅ ready
│
├── models/                       ← Saved ML models (created after pipeline runs)
│   ├── pca_model.pkl
│   ├── umap_reducer.pkl
│   ├── hdbscan_model.pkl
│   ├── umap_2d_coords.npy
│   └── clustered_ids.npy
│
├── app.py                        ← Streamlit dashboard entry point
├── seed_db.py                    ← One-click DB seeder
├── run_pipeline.py               ← One-click ML pipeline runner
└── requirements.txt
```

---

## ⚡ Setup in VS Code (Step by Step)

### Step 1 — Open the folder
```
File → Open Folder → select the PulseIQ folder
```

### Step 2 — Create virtual environment
Open the VS Code terminal (`Ctrl+\`` or `Cmd+\``) and run:
```bash
python3 -m venv .venv
source .venv/bin/activate          # Mac / Linux
# .venv\Scripts\activate           # Windows
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```
> ⏳ First install takes ~5 min (downloads PyTorch + transformers models)

### Step 4 — Select Python interpreter in VS Code
```
Cmd+Shift+P  →  "Python: Select Interpreter"  →  choose  .venv/bin/python
```

### Step 5 — Seed the database (already done, just verify)
```bash
python seed_db.py
# Should print: ✅ Inserted: 295 | Total in DB: 295
```

### Step 6 — Run the ML pipeline
```bash
python run_pipeline.py
```
This runs: embed → cluster → sentiment on the 295 seeded articles.
> ⏳ Takes ~3-5 min on M2 (downloads GTE-small + FinBERT on first run)

### Step 7 — Start the API server (Terminal 1)
```bash
uvicorn api.main:app --reload --port 8000
```
API docs available at → **http://localhost:8000/docs**

### Step 8 — Start the dashboard (Terminal 2, new tab)
```bash
streamlit run app.py
```
Dashboard at → **http://localhost:8501**

---

## 🗞️ Fetch Live News (Optional)

Your NewsAPI key is already in `.env`. To pull real live articles:
```bash
python run_pipeline.py --fetch
```
Or trigger it from the sidebar in the dashboard UI.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/api/stats` | Counts + pipeline status flags |
| GET  | `/api/articles?page=1&search=fed&sentiment=positive` | Paginated filtered list |
| GET  | `/api/articles/umap-coords` | 2-D scatter plot data |
| GET  | `/api/clusters` | All cluster summaries |
| GET  | `/api/clusters/{label}` | One cluster + articles |
| GET  | `/api/sentiment/overview` | Counts + daily timeline |
| POST | `/api/pipeline/run` | Trigger pipeline stages |

---

## 🧠 ML Pipeline

```
295 articles in DB
      ↓
  embed_articles.py    → thenlper/gte-small (384-dim, MPS on M2)
      ↓
  cluster_articles.py  → PCA(50) → UMAP(5) → HDBSCAN
      ↓                  + UMAP(2) saved for scatter plot
  sentiment_analysis.py → ProsusAI/finbert  score ∈ [-1, +1]
      ↓
  SQLite (embeddings + clusters + sentiment tables)
```

---

## 🐛 Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Activate venv: `source .venv/bin/activate` |
| `Cannot reach API` | Start uvicorn first in Terminal 1 |
| MPS not available | Update macOS to 12.3+ and PyTorch to 2.0+ |
| `No articles found` | Run `python seed_db.py` first |
| Port 8000 in use | `lsof -ti:8000 \| xargs kill` |
| Port 8501 in use | `lsof -ti:8501 \| xargs kill` |

---

*Built with FastAPI · Streamlit · SentenceTransformers · UMAP · HDBSCAN · FinBERT*
