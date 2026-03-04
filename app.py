"""
app.py — PulseIQ Streamlit Dashboard (v2 — API-connected)
==========================================================
Pure frontend. Never imports from backend/ directly.
All data arrives via HTTP from the FastAPI server at localhost:8000.

Run order:
  Terminal 1:  uvicorn api.main:app --reload --port 8000
  Terminal 2:  streamlit run app.py
"""

import time
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from frontend.api_client import client, APIError

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PulseIQ",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = px.colors.qualitative.Plotly + px.colors.qualitative.Set3

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp                      { background: #060d1a; }
section[data-testid="stSidebar"] { background: #0d1b2e; border-right: 1px solid #1e3a5f; }
.piq-header {
    background: linear-gradient(120deg, #0d1b2e 0%, #0f2d50 60%, #0a1628 100%);
    border: 1px solid #1e3a5f; border-radius: 14px;
    padding: 28px 36px; margin-bottom: 28px;
    display: flex; align-items: center; justify-content: space-between;
}
.piq-header h1 { font-size:2rem; font-weight:700; color:#e2e8f0; margin:0; letter-spacing:-1px; }
.piq-header h1 span { color:#38bdf8; }
.piq-header p  { color:#64748b; font-size:0.85rem; margin:5px 0 0; }
.api-badge {
    font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#38bdf8;
    background:rgba(56,189,248,0.08); border:1px solid rgba(56,189,248,0.25);
    border-radius:6px; padding:6px 14px;
}
.kpi-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:14px; margin-bottom:24px; }
.kpi {
    background:#0d1b2e; border:1px solid #1e3a5f; border-radius:10px;
    padding:18px 16px; text-align:center; transition:border-color .2s;
}
.kpi:hover { border-color:#38bdf8; }
.kpi .k-label { color:#475569; font-size:0.72rem; text-transform:uppercase; letter-spacing:1.2px; }
.kpi .k-value { color:#f1f5f9; font-size:1.9rem; font-weight:700; line-height:1.1; margin:6px 0 4px; }
.kpi .k-sub   { color:#64748b; font-size:0.75rem; }
.sec-title {
    font-size:0.85rem; font-weight:600; color:#475569;
    text-transform:uppercase; letter-spacing:1.8px;
    margin:28px 0 14px; padding-bottom:8px; border-bottom:1px solid #1e3a5f;
}
.art-card {
    background:#0d1b2e; border:1px solid #1e3a5f; border-left:4px solid;
    border-radius:8px; padding:14px 18px; margin-bottom:9px; transition:background .15s;
}
.art-card:hover { background:#112033; }
.art-title { font-weight:600; font-size:0.9rem; color:#e2e8f0; text-decoration:none; }
.art-meta  { color:#475569; font-size:0.76rem; margin-top:3px; }
.art-desc  { color:#64748b; font-size:0.82rem; margin-top:6px; line-height:1.4; }
.badge { display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.7rem; font-weight:600; }
.b-pos { background:#052e16; color:#86efac; border:1px solid #166534; }
.b-neu { background:#0c1a3e; color:#93c5fd; border:1px solid #1d4ed8; }
.b-neg { background:#2d0a0a; color:#fca5a5; border:1px solid #991b1b; }
.api-error {
    background:#2d0a0a; border:1px solid #991b1b; border-radius:8px;
    padding:16px; color:#fca5a5; font-size:0.85rem;
}
.mono { font-family:'IBM Plex Mono',monospace; font-size:0.78rem; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def badge_html(label: str, score: float | None = None) -> str:
    cls = {"positive":"b-pos","neutral":"b-neu","negative":"b-neg"}.get(label,"b-neu")
    ico = {"positive":"▲","neutral":"●","negative":"▼"}.get(label,"●")
    sc  = f" {score:+.2f}" if score is not None else ""
    return f'<span class="badge {cls}">{ico} {label}{sc}</span>'

def cluster_colour(label):
    return "#2a3a50" if label == -1 else PALETTE[label % len(PALETTE)]

@st.cache_data(ttl=30)
def api_stats():         return client.get_stats()
@st.cache_data(ttl=30)
def api_clusters():      return client.get_clusters()
@st.cache_data(ttl=30)
def api_umap():          return client.get_umap_coords()
@st.cache_data(ttl=30)
def api_sentiment():     return client.get_sentiment_overview()
@st.cache_data(ttl=30)
def api_articles(**kw):  return client.get_articles(**kw)

api_ok = client.is_reachable()

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📈 PulseIQ")
    status_colour = "#22c55e" if api_ok else "#ef4444"
    status_text   = "ONLINE" if api_ok else "OFFLINE"
    st.markdown(
        f'<div class="api-badge" style="color:{status_colour};border-color:{status_colour}40">'
        f'API localhost:8000 &nbsp; {status_text}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("#### ⚙️ Pipeline Control")

    api_key       = st.text_input("NewsAPI Key", type="password", help="newsapi.org/register")
    days_back     = st.slider("Days of history", 1, 14, 7)
    run_fetch     = st.checkbox("① Fetch news",          value=True)
    run_embed     = st.checkbox("② Generate embeddings", value=True)
    run_cluster   = st.checkbox("③ Cluster articles",    value=True)
    run_sentiment = st.checkbox("④ Sentiment analysis",  value=True)

    if st.button("🚀 Run Pipeline", use_container_width=True, type="primary", disabled=not api_ok):
        if run_fetch and not api_key:
            st.error("Paste your NewsAPI key above.")
        else:
            log_box = st.empty()
            with st.spinner("Pipeline running…"):
                try:
                    result = client.run_pipeline(
                        api_key=api_key, run_fetch=run_fetch, run_embed=run_embed,
                        run_cluster=run_cluster, run_sentiment=run_sentiment, days_back=days_back,
                    )
                    lines = []
                    for s in result["stages"]:
                        ico = "✅" if s["success"] else "❌"
                        lines.append(f"{ico} [{s['stage']:10}] {s['message']}  ({s['elapsed']}s)")
                    lines.append(f"\n⏱ Total: {result['total_elapsed']}s")
                    log_box.code("\n".join(lines))
                    st.cache_data.clear()
                    time.sleep(1.5)
                    st.rerun()
                except APIError as e:
                    st.error(f"Error {e.status_code}: {e.detail}")

    st.markdown("---")
    st.markdown("#### 🔎 Filters")
    sent_filter = st.multiselect(
        "Sentiment", ["positive","neutral","negative"],
        default=["positive","neutral","negative"],
    )

# ══════════════════════════════════════════════════════════════════════════════
# Guard: API offline
# ══════════════════════════════════════════════════════════════════════════════
if not api_ok:
    st.markdown("""
    <div class="api-error">
      <strong>⚠️ API server is not reachable.</strong><br><br>
      Start it in a separate terminal:<br><br>
      <code style="background:#1a0505;padding:4px 10px;border-radius:4px">
        uvicorn api.main:app --reload --port 8000
      </code><br><br>
      Then refresh this page.
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Load stats ─────────────────────────────────────────────────────────────────
try:
    stats = api_stats()
except APIError as e:
    st.error(f"Failed to load stats: {e.detail}")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="piq-header">
  <div>
    <h1>Pulse<span>IQ</span></h1>
    <p>Financial Narrative Intelligence Platform · Discover emerging market themes via ML clustering</p>
  </div>
  <div class="api-badge">FastAPI + Streamlit · GTE-small · UMAP · HDBSCAN · FinBERT</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ① KPI Row
# ══════════════════════════════════════════════════════════════════════════════
avg_s  = stats["avg_sentiment"]
s_col  = "#22c55e" if avg_s > 0.05 else ("#ef4444" if avg_s < -0.05 else "#94a3b8")

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi">
    <div class="k-label">Articles</div>
    <div class="k-value">{stats['total_articles']}</div>
    <div class="k-sub">{stats['embedded_articles']} embedded</div>
  </div>
  <div class="kpi">
    <div class="k-label">Clusters</div>
    <div class="k-value">{stats['n_clusters']}</div>
    <div class="k-sub">{stats['noise_articles']} noise articles</div>
  </div>
  <div class="kpi">
    <div class="k-label">Clustered</div>
    <div class="k-value">{stats['clustered_articles']}</div>
    <div class="k-sub">articles assigned</div>
  </div>
  <div class="kpi">
    <div class="k-label">Analysed</div>
    <div class="k-value">{stats['analysed_articles']}</div>
    <div class="k-sub">FinBERT scored</div>
  </div>
  <div class="kpi">
    <div class="k-label">Market Tone</div>
    <div class="k-value" style="color:{s_col}">{avg_s:+.2f}</div>
    <div class="k-sub">avg sentiment</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ② Cluster Map + Sentiment Donut
# ══════════════════════════════════════════════════════════════════════════════
col_map, col_donut = st.columns([3, 1])

with col_map:
    st.markdown('<div class="sec-title">🗺 Cluster Map — UMAP 2-D Projection</div>', unsafe_allow_html=True)
    if not stats["pipeline_status"].get("clustered"):
        st.info("Run the clustering stage to populate this map.")
    else:
        try:
            pts = api_umap()
            if pts:
                df_p = pd.DataFrame(pts)
                if sent_filter:
                    df_p = df_p[df_p["sentiment_label"].isin(sent_filter) | df_p["sentiment_label"].isna()]
                df_p["cluster_str"] = df_p["cluster_label"].apply(
                    lambda l: "Noise" if l == -1 else f"Cluster {l}"
                )
                df_p["hover"] = (
                    df_p["title"].str[:72] + "<br>"
                    + df_p["source"].fillna("") + "<br>"
                    + df_p["sentiment_label"].fillna("—")
                )
                cnames = sorted(df_p["cluster_str"].unique(), key=lambda s:(s=="Noise",s))
                cmap   = {s:("#2a3a50" if s=="Noise" else PALETTE[int(s.split()[-1])%len(PALETTE)]) for s in cnames}
                fig    = px.scatter(df_p, x="x", y="y", color="cluster_str",
                                    color_discrete_map=cmap, hover_name="hover",
                                    opacity=0.83, height=460)
                fig.update_traces(marker_size=7, marker_line_width=0)
                fig.update_layout(
                    plot_bgcolor="#060d1a", paper_bgcolor="#060d1a", font_color="#64748b",
                    legend_title_text="",
                    xaxis=dict(showgrid=False,zeroline=False,title=""),
                    yaxis=dict(showgrid=False,zeroline=False,title=""),
                    margin=dict(l=0,r=0,t=4,b=0),
                    legend=dict(bgcolor="#0d1b2e",bordercolor="#1e3a5f",borderwidth=1),
                )
                st.plotly_chart(fig, use_container_width=True)
        except APIError as e:
            st.warning(f"UMAP data unavailable: {e.detail}")

with col_donut:
    st.markdown('<div class="sec-title">💬 Sentiment</div>', unsafe_allow_html=True)
    if not stats["pipeline_status"].get("analysed"):
        st.info("Run sentiment analysis first.")
    else:
        try:
            sd  = api_sentiment()
            fig_d = go.Figure(go.Pie(
                labels=["Positive","Neutral","Negative"],
                values=[sd["positive_count"],sd["neutral_count"],sd["negative_count"]],
                hole=0.65,
                marker_colors=["#22c55e","#3b82f6","#ef4444"],
                textfont_size=11,
                hovertemplate="%{label}: %{value}<extra></extra>",
            ))
            fig_d.update_layout(
                plot_bgcolor="#060d1a", paper_bgcolor="#060d1a", font_color="#94a3b8",
                showlegend=True,
                legend=dict(bgcolor="#0d1b2e", orientation="h", yanchor="bottom", y=-0.25),
                height=270, margin=dict(l=0,r=0,t=4,b=0),
                annotations=[dict(text=f"{sd['avg_score']:+.2f}", x=0.5, y=0.5,
                                  font_size=20, font_color="#e2e8f0", showarrow=False)],
            )
            st.plotly_chart(fig_d, use_container_width=True)
            st.markdown(
                f'<div style="text-align:center;color:#475569;font-size:0.78rem">'
                f'{sd["positive_count"]} pos · {sd["neutral_count"]} neu · {sd["negative_count"]} neg'
                f'</div>', unsafe_allow_html=True)
        except APIError as e:
            st.warning(str(e.detail))

# ══════════════════════════════════════════════════════════════════════════════
# ③ Sentiment Timeline
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-title">📅 Sentiment Timeline</div>', unsafe_allow_html=True)
if stats["pipeline_status"].get("analysed"):
    try:
        tl = client.get_sentiment_timeline()
        if tl:
            df_t = pd.DataFrame(tl)
            fig_t = go.Figure()
            fig_t.add_trace(go.Bar(x=df_t["date"], y=df_t["positive"],  name="Positive",
                                   marker_color="rgba(34,197,94,0.45)"))
            fig_t.add_trace(go.Bar(x=df_t["date"], y=df_t["negative"],  name="Negative",
                                   marker_color="rgba(239,68,68,0.45)"))
            fig_t.add_trace(go.Scatter(x=df_t["date"], y=df_t["avg_score"], name="Avg Score",
                                       yaxis="y2", line=dict(color="#38bdf8",width=2),
                                       mode="lines+markers", marker_size=5))
            fig_t.add_hline(y=0, line_dash="dot", line_color="#1e3a5f", line_width=1, yref="y2")
            fig_t.update_layout(
                plot_bgcolor="#060d1a", paper_bgcolor="#060d1a", font_color="#64748b",
                barmode="group", height=250, margin=dict(l=0,r=0,t=4,b=0),
                legend=dict(bgcolor="#0d1b2e",orientation="h",yanchor="bottom",y=-0.35),
                xaxis=dict(gridcolor="#0d1b2e"),
                yaxis=dict(gridcolor="#0d1b2e",title="Article count"),
                yaxis2=dict(title="Avg score",overlaying="y",side="right",
                            range=[-1,1],showgrid=False),
            )
            st.plotly_chart(fig_t, use_container_width=True)
    except APIError as e:
        st.warning(str(e.detail))
else:
    st.info("Run sentiment analysis to see the timeline.")

# ══════════════════════════════════════════════════════════════════════════════
# ④ Cluster Explorer
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-title">🔬 Cluster Explorer</div>', unsafe_allow_html=True)
if not stats["pipeline_status"].get("clustered"):
    st.info("Run clustering to explore narrative themes.")
else:
    try:
        clusters = api_clusters()
        if clusters:
            opts = {f"Cluster {c['cluster_label']}  ·  {c['article_count']} articles  ·  avg {c['avg_sentiment']:+.2f}": c["cluster_label"]
                   for c in clusters}
            sel_name = st.selectbox("Theme cluster", list(opts.keys()), label_visibility="collapsed")
            sel_id   = opts[sel_name]
            accent   = PALETTE[sel_id % len(PALETTE)]

            detail   = client.get_cluster(sel_id)
            k1,k2,k3,k4,k5 = st.columns(5)
            k1.metric("Articles",       detail["article_count"])
            k2.metric("Avg Sentiment",  f"{detail['avg_sentiment']:+.2f}")
            k3.metric("🟢 Positive",    detail["positive_count"])
            k4.metric("🟡 Neutral",     detail["neutral_count"])
            k5.metric("🔴 Negative",    detail["negative_count"])

            arts = [a for a in detail["articles"] if a.get("sentiment_label","neutral") in sent_filter]
            arts = sorted(arts, key=lambda a: a.get("sentiment_score") or 0, reverse=True)
            st.markdown(f"**{len(arts)} articles** (sorted by sentiment score)")

            for a in arts[:25]:
                sl  = a.get("sentiment_label","neutral")
                ss  = a.get("sentiment_score")
                pub = (a.get("published_at") or "")[:10]
                desc= (a.get("description") or "")[:160]
                st.markdown(f"""
                <div class="art-card" style="border-left-color:{accent}">
                  <a href="{a['url']}" target="_blank" class="art-title">{a['title']}</a>
                  <div class="art-meta">{a.get('source','')} · {pub} · {badge_html(sl,ss)}</div>
                  {"<div class='art-desc'>"+desc+"…</div>" if desc else ""}
                </div>""", unsafe_allow_html=True)
    except APIError as e:
        st.error(f"Cluster data error: {e.detail}")

# ══════════════════════════════════════════════════════════════════════════════
# ⑤ Article Search
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-title">🔍 Article Search</div>', unsafe_allow_html=True)
col_q, col_s = st.columns([4,1])
with col_q:
    search_q = st.text_input("", placeholder="Search: Federal Reserve, inflation, earnings…", label_visibility="collapsed")
with col_s:
    browse = st.checkbox("Browse all")

if search_q or browse:
    try:
        pg     = st.number_input("Page", 1, value=1, step=1, label_visibility="collapsed")
        res    = api_articles(page=pg, page_size=15, search=search_q or None)
        st.caption(f"{res['total']} results · page {res['page']} of {res['pages']}")
        for a in res["items"]:
            sl   = a.get("sentiment_label","neutral")
            ss   = a.get("sentiment_score")
            cl   = a.get("cluster_label")
            pub  = (a.get("published_at") or "")[:10]
            desc = (a.get("description") or "")[:160]
            acc  = cluster_colour(cl if cl is not None else -1)
            cl_lbl = f'<span class="mono" style="color:{acc}">cluster {cl}</span>' if cl not in (None,-1) else ""
            st.markdown(f"""
            <div class="art-card" style="border-left-color:{acc}">
              <a href="{a['url']}" target="_blank" class="art-title">{a['title']}</a>
              <div class="art-meta">{a.get('source','')} · {pub} · {badge_html(sl,ss)} {cl_lbl}</div>
              {"<div class='art-desc'>"+desc+"…</div>" if desc else ""}
            </div>""", unsafe_allow_html=True)
    except APIError as e:
        st.error(f"Search error: {e.detail}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#1e3a5f;font-size:0.75rem;font-family:\'IBM Plex Mono\',monospace">'
    'PulseIQ v2 · FastAPI REST API + Streamlit Frontend · '
    'GTE-small · PCA · UMAP · HDBSCAN · FinBERT'
    '</div>', unsafe_allow_html=True)
