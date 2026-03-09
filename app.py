# ============================================================
#  Market Narrative Intelligence System
#  app.py  —  Complete Final Version
#  Includes: live text paste, PDF report with charts, all fixes
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx

st.set_page_config(
    page_title="Market Narrative Intelligence",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.hero {
    background: linear-gradient(135deg,#050d1a 0%,#0a1628 40%,#071220 100%);
    border: 1px solid #1e3a5f; border-radius: 14px;
    padding: 2.4rem 2rem 2rem 2rem; margin-bottom: 2rem;
    text-align: center; position: relative; overflow: hidden;
}
.hero::before {
    content:''; position:absolute; inset:0;
    background: radial-gradient(ellipse at 30% 50%, rgba(56,189,248,0.07) 0%, transparent 60%),
                radial-gradient(ellipse at 70% 50%, rgba(99,102,241,0.07) 0%, transparent 60%);
}
.hero h1 { font-family:'IBM Plex Mono',monospace; font-size:2rem; color:#e2eaf6; margin:0; }
.hero p  { color:#6b8cba; font-size:1rem; margin-top:.5rem; }
.sec-head {
    border-left: 3px solid #38bdf8; padding-left:.9rem;
    margin: 2rem 0 1rem 0; font-family:'IBM Plex Mono',monospace;
    font-size:1.05rem; color:#cbd5e1;
}
div[data-testid="stMetric"] {
    background:#0d1526; border:1px solid #1e3a5f; border-radius:10px; padding:.8rem 1rem;
}
div[data-testid="stMetric"] label { color:#6b8cba !important; font-size:.78rem; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color:#e2eaf6 !important; font-family:'IBM Plex Mono',monospace;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>📈 Market Narrative Intelligence</h1>
  <p>Detect emerging financial narratives · Cluster · Analyse sentiment · Predict volatility</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("**Clustering**")
    min_cluster_size = st.slider("Min cluster size",  2, 30, 5)
    min_samples      = st.slider("Min samples",       1, 10, 2)
    umap_neighbors   = st.slider("UMAP neighbours",   5, 50, 15)
    umap_dims        = st.slider("UMAP dimensions",   2, 10,  5)
    st.markdown("**Sentiment**")
    use_finbert = st.checkbox("Use FinBERT (domain-accurate, slower)", value=False)
    st.markdown("**Volatility Model**")
    vol_window     = st.slider("Rolling window (days)", 3, 30, 7)
    vol_model_type = st.selectbox("Model", ["Ridge Regression","Random Forest","Gradient Boosting"])
    st.markdown("**Graph**")
    graph_layout    = st.selectbox("Layout", ["spring","kamada_kawai","spectral"])
    max_graph_nodes = st.slider("Max nodes", 50, 400, 150)
    st.markdown("---")
    st.caption("BSc Final Year AI Project")

# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading sentence-transformer…")
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Loading FinBERT…")
def load_finbert():
    from transformers import pipeline
    return pipeline("text-classification",
                    model="ProsusAI/finbert", tokenizer="ProsusAI/finbert",
                    truncation=True, max_length=512)

@st.cache_data(show_spinner="Encoding texts…")
def compute_embeddings(texts_tuple):
    embedder = load_embedder()
    return embedder.encode(list(texts_tuple), show_progress_bar=False,
                           batch_size=64, convert_to_numpy=True)

@st.cache_data(show_spinner="Running UMAP…")
def compute_umap(emb_bytes, n_neighbors, n_components):
    import umap
    emb = np.frombuffer(emb_bytes, dtype=np.float32).reshape(-1, 384)
    return umap.UMAP(n_neighbors=n_neighbors, n_components=n_components,
                     metric="cosine", random_state=42, min_dist=0.0).fit_transform(emb)

@st.cache_data(show_spinner="Running HDBSCAN…")
def compute_hdbscan(red_bytes, shape, min_clust, min_samp):
    import hdbscan
    red = np.frombuffer(red_bytes, dtype=np.float64).reshape(shape).copy()
    cl  = hdbscan.HDBSCAN(min_cluster_size=min_clust, min_samples=min_samp,
                           metric="euclidean", cluster_selection_method="eom",
                           prediction_data=True)
    cl.fit(red)
    return cl.labels_, cl.probabilities_

# ── Sentiment helpers ─────────────────────────────────────────────────────────
def finbert_scores(texts, pipe, batch=32):
    out = []
    for i in range(0, len(texts), batch):
        res = pipe(texts[i:i+batch], batch_size=batch)
        for r in res:
            s = (r["score"] if r["label"]=="positive"
                 else (-r["score"] if r["label"]=="negative" else 0.0))
            out.append(s)
    return out

def textblob_scores(texts):
    from textblob import TextBlob
    return [TextBlob(t).sentiment.polarity for t in texts]

# ── Plot style helpers ────────────────────────────────────────────────────────
BG, PANEL, BORDER = "#060b18", "#0d1526", "#1e3a5f"
TEXT_C, SUB_C     = "#e2eaf6", "#6b8cba"

def dark_fig(w=11, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG); ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT_C, labelsize=8)
    for s in ax.spines.values(): s.set_color(BORDER)
    ax.xaxis.label.set_color(SUB_C); ax.yaxis.label.set_color(SUB_C)
    return fig, ax

def dark_fig2(w=14, h=5):
    fig, axes = plt.subplots(1, 2, figsize=(w, h))
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT_C, labelsize=8)
        for s in ax.spines.values(): s.set_color(BORDER)
        ax.xaxis.label.set_color(SUB_C); ax.yaxis.label.set_color(SUB_C)
    return fig, axes

def fig_to_buf(fig):
    """Save a matplotlib figure to a bytes buffer for PDF embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf

# ── Volatility helpers ────────────────────────────────────────────────────────
def build_vol_features(df, window):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    daily_cnt = (df[df["Narrative"]>=0]
                 .groupby(["date","Narrative"]).size().unstack(fill_value=0))
    daily_cnt.columns = [f"nar_{c}_cnt" for c in daily_cnt.columns]
    daily_sent = (df[df["Narrative"]>=0]
                  .groupby(["date","Narrative"])["sent_score"].mean().unstack(fill_value=0.0))
    daily_sent.columns = [f"nar_{c}_sent" for c in daily_sent.columns]
    def _ent(x):
        p = x.value_counts(normalize=True)
        return -(p * np.log(p+1e-9)).sum()
    agg = df.groupby("date").agg(
        mean_sent=("sent_score","mean"),
        std_sent=("sent_score","std"),
        n_docs=("text","count"),
        nar_entropy=("Narrative", _ent),
    ).fillna(0)
    feats = daily_cnt.join(daily_sent,how="outer").join(agg,how="outer").fillna(0)
    for col in daily_cnt.columns:
        feats[f"{col}_roll{window}"] = feats[col].rolling(window, min_periods=1).mean()
    feats["volatility"] = feats["n_docs"].rolling(window, min_periods=1).std().fillna(0)
    if "return" in df.columns:
        rv = df.groupby("date")["return"].std().rolling(window, min_periods=1).std()
        feats["volatility"] = rv.reindex(feats.index).fillna(feats["volatility"])
    return feats.reset_index()

def train_vol_model(feats, model_type):
    from sklearn.linear_model    import Ridge
    from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing   import StandardScaler
    from sklearn.pipeline        import Pipeline
    from sklearn.metrics         import mean_squared_error, r2_score
    drop = ["date","volatility"]
    X = feats.drop(columns=drop, errors="ignore").values
    y = feats["volatility"].values
    if len(X) < 10: return None, None, None, None
    mdl = {"Ridge Regression":  Ridge(alpha=1.0),
          "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
           "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42)}[model_type]
    pipe  = Pipeline([("sc", StandardScaler()), ("m", mdl)])
    tscv  = TimeSeriesSplit(n_splits=5)
    rmses, r2s, y_pred = [], [], np.zeros_like(y, dtype=float)
    for tr, te in tscv.split(X):
        pipe.fit(X[tr], y[tr]); p = pipe.predict(X[te])
        y_pred[te] = p
        rmses.append(np.sqrt(mean_squared_error(y[te], p)))
        r2s.append(r2_score(y[te], p))
    pipe.fit(X, y)
    names = feats.drop(columns=drop, errors="ignore").columns.tolist()
    imps  = (dict(zip(names, pipe["m"].feature_importances_))
             if hasattr(pipe["m"],"feature_importances_")
             else dict(zip(names, np.abs(pipe["m"].coef_)))
             if hasattr(pipe["m"],"coef_") else None)
    return pipe, {"rmse": rmses, "r2": r2s}, y_pred, imps

def build_graph(df, reduced_2d, max_nodes):
    from sklearn.metrics.pairwise import cosine_similarity
    idx = np.random.choice(len(df), size=min(max_nodes,len(df)), replace=False)
    sdf = df.iloc[idx].copy().reset_index(drop=True)
    sim = cosine_similarity(reduced_2d[idx])
    G   = nx.Graph()
    for i, row in sdf.iterrows():
        G.add_node(i, narrative=int(row["Narrative"]),
                   sentiment=float(row.get("sent_score",0)),
                   text=str(row["text"])[:80])
    for i in range(len(sdf)):
        for j in range(i+1, len(sdf)):
            if (sdf.iloc[i]["Narrative"]==sdf.iloc[j]["Narrative"]
                    and sdf.iloc[i]["Narrative"]>=0
                    and sim[i,j]>0.85):
                G.add_edge(i, j, weight=float(sim[i,j]))
    return G, sdf

# ── Image buffer store (for PDF) ──────────────────────────────────────────────
img_bufs = {}

# ══════════════════════════════════════════════════════════════════════════════
#  QUICK TEXT INPUT  (live demo paste box)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-head">✍️ Quick Text Input</div>', unsafe_allow_html=True)
st.markdown("Paste headlines or sentences directly — one per line. Great for live demos.")

pasted = st.text_area("Paste news text here (one sentence per line)", height=140,
                       placeholder="Fed raises interest rates amid inflation fears.\nTech stocks fall as Nasdaq drops 2%.\nOil prices surge after OPEC cuts production.")

if pasted and st.button("▶ Analyse pasted text", type="primary"):
    lines = [l.strip() for l in pasted.split("\n") if len(l.strip()) > 10]
    if lines:
        buf = io.StringIO()
        pd.DataFrame({"text": lines}).to_csv(buf, index=False)
        st.download_button("⬇️ Download as CSV to upload below",
                           buf.getvalue().encode(),
                           "pasted_text.csv", "text/csv")
        st.success(f"✅ {len(lines)} sentences ready — download CSV above then upload it below.")
    else:
        st.warning("No usable text found.")

# ══════════════════════════════════════════════════════════════════════════════
#  UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-head">📂 Data Input</div>', unsafe_allow_html=True)
st.markdown("Upload a CSV with a **`text`** column. Optional: `date`, `sector`, `return`, `sentiment`.")

col_up, col_demo = st.columns([3,1])
with col_up:
    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
with col_demo:
    if st.button("🎲 Generate demo CSV"):
        from data_generator import make_demo_csv
        demo_df  = make_demo_csv(None)
        csv_bytes = demo_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download demo", csv_bytes, "demo_narratives.csv", "text/csv")

if not uploaded:
    st.info("Upload your CSV to begin. **all-data.csv** (4,846 financial sentences) works perfectly.")
    st.stop()

# ── Load ──────────────────────────────────────────────────────────────────────
try:
    df = pd.read_csv(uploaded, encoding="utf-8")
except UnicodeDecodeError:
    uploaded.seek(0)
    df = pd.read_csv(uploaded, encoding="latin1")

if "text" not in df.columns and df.shape[1] == 2:
    df.columns = ["sentiment","text"]
    st.info("Detected headerless file — assigned columns: `sentiment`, `text`")

if "text" not in df.columns:
    st.error("CSV must contain a **text** column."); st.stop()

df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() > 10].reset_index(drop=True)

st.markdown('<div class="sec-head">📋 Data Preview</div>', unsafe_allow_html=True)
c1,c2,c3,c4 = st.columns(4)
c1.metric("Documents",     f"{len(df):,}")
c2.metric("Columns",       len(df.columns))
c3.metric("Has Date",      "✅" if "date" in df.columns else "❌")
c4.metric("Has Sentiment", "✅" if "sentiment" in df.columns else "❌")
st.dataframe(df.head(8), use_container_width=True)
texts = df["text"].tolist()

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1  Embeddings
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-head">🔎 Step 1 · Semantic Embeddings</div>', unsafe_allow_html=True)
embeddings = compute_embeddings(tuple(texts))
st.success(f"Embeddings: **{embeddings.shape[0]:,}** docs × **{embeddings.shape[1]}** dims")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2  UMAP
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-head">🗺️ Step 2 · UMAP Dimensionality Reduction</div>', unsafe_allow_html=True)
emb_bytes = embeddings.astype(np.float32).tobytes()
reduced   = compute_umap(emb_bytes, umap_neighbors, umap_dims)

if umap_dims > 2:
    import umap as umap_lib
    reduced_2d = umap_lib.UMAP(n_neighbors=umap_neighbors, n_components=2,
                                metric="cosine", random_state=42,
                                min_dist=0.0).fit_transform(embeddings)
else:
    reduced_2d = reduced

st.success(f"384-D → **{reduced.shape[1]}-D** (+ 2-D for visualisation)")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3  HDBSCAN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-head">🧠 Step 3 · Narrative Clustering (HDBSCAN)</div>', unsafe_allow_html=True)
red_bytes       = reduced.astype(np.float64).tobytes()
labels, probs   = compute_hdbscan(red_bytes, reduced.shape, min_cluster_size, min_samples)
df["Narrative"] = labels
df["conf"]      = probs

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
noise_n    = (labels==-1).sum()
mean_conf  = probs[probs>0].mean() if probs.any() else 0

c1,c2,c3,c4 = st.columns(4)
c1.metric("Narratives Detected", n_clusters)
c2.metric("Noise Points",        f"{noise_n} ({noise_n/len(labels)*100:.1f}%)")
c3.metric("Avg Confidence",      f"{mean_conf:.3f}")
c4.metric("Total Documents",     f"{len(df):,}")

palette       = cm.get_cmap("tab20", max(n_clusters+1, 2))
unique_labels = sorted(set(labels))
fig, ax = dark_fig(10, 6)
for lbl in unique_labels:
    mask = labels==lbl
    col  = "#2a3a50" if lbl==-1 else palette(unique_labels.index(lbl))
    ax.scatter(reduced_2d[mask,0], reduced_2d[mask,1], c=[col],
               s=12, alpha=0.7, label=("Noise" if lbl==-1 else f"Nar {lbl}"), rasterized=True)
ax.legend(loc="upper right", fontsize=7, framealpha=0.3,
          labelcolor=TEXT_C, facecolor=BG, edgecolor=BORDER)
ax.set_title("UMAP 2-D Projection — Narrative Clusters", color=TEXT_C, fontsize=12, pad=10)
ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
st.pyplot(fig)
img_bufs["umap"] = fig_to_buf(fig)
plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4  Sentiment
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-head">💬 Step 4 · Sentiment Analysis</div>', unsafe_allow_html=True)

if "sentiment" in df.columns and df["sentiment"].dtype == object:
    st.markdown("**Ground-truth label distribution (from your data)**")
    gt = df["sentiment"].value_counts()
    fig, ax = dark_fig(7, 3)
    bar_c = ["#22c55e" if i=="positive" else ("#ef4444" if i=="negative" else "#64748b")
             for i in gt.index]
    ax.bar(gt.index, gt.values, color=bar_c, edgecolor=BORDER, linewidth=0.8)
    ax.set_title("Ground-Truth Label Distribution", color=TEXT_C, fontsize=11)
    ax.set_ylabel("Count")
    for bar, val in zip(ax.patches, gt.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
                str(val), ha="center", va="bottom", fontsize=8, color=TEXT_C)
    st.pyplot(fig)
    img_bufs["gt_dist"] = fig_to_buf(fig)
    plt.close(fig)

if use_finbert:
    with st.spinner("Running FinBERT inference…"):
        scores = finbert_scores(texts, load_finbert())
    st.success("FinBERT sentiment complete.")
else:
    with st.spinner("Running TextBlob sentiment…"):
        scores = textblob_scores(texts)

df["sent_score"] = scores

sent_by_nar = (df[df["Narrative"]>=0]
               .groupby("Narrative")["sent_score"]
               .agg(["mean","std","count"])
               .rename(columns={"mean":"Avg Sentiment","std":"Std Dev","count":"Docs"}))

fig, ax = dark_fig(10, 4)
bar_colors = ["#22c55e" if v>=0 else "#ef4444" for v in sent_by_nar["Avg Sentiment"]]
ax.bar(sent_by_nar.index.astype(str), sent_by_nar["Avg Sentiment"],
       color=bar_colors, edgecolor=BORDER, linewidth=0.8)
ax.axhline(0, color=SUB_C, linewidth=1, linestyle="--")
ax.set_xlabel("Narrative ID"); ax.set_ylabel("Avg Sentiment Score")
ax.set_title("Sentiment per Narrative  (green=bullish · red=bearish)", color=TEXT_C, fontsize=11)
st.pyplot(fig)
img_bufs["sentiment"] = fig_to_buf(fig)
plt.close(fig)
st.dataframe(sent_by_nar, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5  TF-IDF
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-head">🧩 Step 5 · Narrative Themes (TF-IDF)</div>', unsafe_allow_html=True)
from sklearn.feature_extraction.text import TfidfVectorizer

vec   = TfidfVectorizer(stop_words="english", max_features=2000, ngram_range=(1,2))
X_tf  = vec.fit_transform(texts)
terms = vec.get_feature_names_out()

themes = {}
for nar in sorted(df["Narrative"].unique()):
    if nar == -1: continue
    mask = df["Narrative"]==nar
    sub  = vec.transform(df[mask]["text"])
    sc   = np.asarray(sub.mean(axis=0)).flatten()
    top  = sc.argsort()[-10:][::-1]
    themes[nar] = [terms[i] for i in top]

ncols = min(3, max(1, len(themes)))
cols  = st.columns(ncols)
for i, (nar, kws) in enumerate(themes.items()):
    with cols[i % ncols]:
        sv = sent_by_nar.loc[nar,"Avg Sentiment"] if nar in sent_by_nar.index else 0
        em = "🟢" if sv>0.05 else ("🔴" if sv<-0.05 else "🟡")
        st.markdown(f"**Narrative {nar}** {em}")
        st.markdown(" · ".join(f"`{k}`" for k in kws))
        docs = int(sent_by_nar.loc[nar,"Docs"]) if nar in sent_by_nar.index else "?"
        st.caption(f"Sentiment: {sv:+.3f}  |  Docs: {docs}")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6  Narrative Strength
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-head">📊 Step 6 · Narrative Strength</div>', unsafe_allow_html=True)
strength = (df[df["Narrative"]>=0]
            .groupby("Narrative")
            .agg(Strength=("text","count"),
                 Avg_Confidence=("conf","mean"),
                 Avg_Sentiment=("sent_score","mean"))
            .sort_values("Strength", ascending=False).reset_index())

fig, ax = dark_fig(10, 4)
bar_c = cm.plasma(np.linspace(0.2, 0.9, len(strength)))
ax.barh(strength["Narrative"].astype(str), strength["Strength"],
        color=bar_c, edgecolor=BORDER)
ax.set_xlabel("Document Count"); ax.invert_yaxis()
ax.set_title("Narrative Strength (document volume)", color=TEXT_C, fontsize=11)
st.pyplot(fig)
img_bufs["strength"] = fig_to_buf(fig)
plt.close(fig)
st.dataframe(strength, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7  Storyline Graph
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-head">🕸️ Step 7 · Storyline Graph</div>', unsafe_allow_html=True)
with st.spinner("Building graph…"):
    G, sdf = build_graph(df, reduced_2d, max_graph_nodes)

c1,c2,c3 = st.columns(3)
c1.metric("Nodes", G.number_of_nodes())
c2.metric("Edges", G.number_of_edges())
c3.metric("Components", nx.number_connected_components(G) if G.number_of_nodes()>0 else 0)

if G.number_of_nodes() > 0:
    layout_fn = {
        "spring":       lambda: nx.spring_layout(G, seed=42, weight="weight"),
        "kamada_kawai": lambda: (nx.kamada_kawai_layout(G, weight="weight")
                                  if G.number_of_edges()>0 else nx.spring_layout(G, seed=42)),
        "spectral":     lambda: (nx.spectral_layout(G)
                                  if G.number_of_edges()>2 else nx.spring_layout(G, seed=42)),
    }
    pos         = layout_fn[graph_layout]()
    u_nars      = sorted(set(nx.get_node_attributes(G,"narrative").values()))
    nar_palette = cm.get_cmap("tab20", max(len(u_nars),2))
    node_colors = ["#2a3a50" if G.nodes[n]["narrative"]==-1
                   else nar_palette(u_nars.index(G.nodes[n]["narrative"]))
                   for n in G.nodes()]
    node_sizes  = [max(10, 30+50*abs(G.nodes[n].get("sentiment",0))) for n in G.nodes()]
    e_weights   = [G[u][v].get("weight",0.5)*2 for u,v in G.edges()]

    fig, ax = plt.subplots(figsize=(12,8))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis("off")
    nx.draw_networkx_edges(G, pos, ax=ax, width=e_weights, alpha=0.35, edge_color="#334155")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.88)
    handles = [plt.Line2D([0],[0], marker="o", color="w", markersize=8,
                           markerfacecolor=nar_palette(u_nars.index(n)) if n>=0 else "#2a3a50",
                           label=f"Narrative {n}" if n>=0 else "Noise")
               for n in u_nars]
    ax.legend(handles=handles, loc="upper right", fontsize=7,
              framealpha=0.2, labelcolor=TEXT_C, facecolor=BG, edgecolor=BORDER)
    ax.set_title("Storyline Graph — cosine-similarity weighted edges",
                 color=TEXT_C, fontsize=12, pad=10)
    st.pyplot(fig)
    img_bufs["graph"] = fig_to_buf(fig)
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 8  Temporal Trend
# ══════════════════════════════════════════════════════════════════════════════
if "date" in df.columns:
    st.markdown('<div class="sec-head">📅 Step 8 · Narrative Trends Over Time</div>', unsafe_allow_html=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    dated = df.dropna(subset=["date"])
    if not dated.empty:
        trend = (dated[dated["Narrative"]>=0]
                 .groupby(["date","Narrative"]).size()
                 .reset_index(name="count"))
        pivot = trend.pivot(index="date", columns="Narrative", values="count").fillna(0)
        pivot.columns = [f"Narrative {c}" for c in pivot.columns]
        fig, ax = dark_fig(12, 4)
        pal = cm.get_cmap("tab10", len(pivot.columns))
        for i, col in enumerate(pivot.columns):
            ax.plot(pivot.index, pivot[col], label=col, lw=1.8, color=pal(i))
        ax.set_xlabel("Date"); ax.set_ylabel("Document Count")
        ax.set_title("Narrative Volume Over Time", color=TEXT_C, fontsize=11)
        ax.legend(fontsize=7, framealpha=0.2, labelcolor=TEXT_C,
                  facecolor=BG, edgecolor=BORDER)
        st.pyplot(fig)
        img_bufs["trend"] = fig_to_buf(fig)
        plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 9  Volatility Prediction
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-head">📉 Step 9 · Sector Volatility Prediction</div>', unsafe_allow_html=True)

pipe_v = None; cv = None; y_pred = None; imps = None

if "date" not in df.columns:
    st.warning("Add a **date** column to enable volatility modelling.")
else:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df_v = df.dropna(subset=["date"])
    if df_v["date"].nunique() < 10:
        st.warning("Need ≥ 10 unique dates for volatility modelling.")
    else:
        with st.spinner("Engineering features…"):
            feats = build_vol_features(df_v, vol_window)
        with st.spinner(f"Training {vol_model_type} (TimeSeriesSplit CV)…"):
            pipe_v, cv, y_pred, imps = train_vol_model(feats, vol_model_type)

        if pipe_v:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("CV RMSE",  f"{np.mean(cv['rmse']):.4f}", f"±{np.std(cv['rmse']):.4f}")
            c2.metric("CV R²",    f"{np.mean(cv['r2']):.4f}",   f"±{np.std(cv['r2']):.4f}")
            c3.metric("Model",    vol_model_type)
            c4.metric("CV Folds", "5 (TimeSeriesSplit)")

            y_true = feats["volatility"].values
            fig, axes = dark_fig2(14, 5)
            axes[0].plot(feats["date"], y_true, label="Actual",    color="#38bdf8", lw=1.6)
            axes[0].plot(feats["date"], y_pred, label="Predicted", color="#f59e0b", lw=1.6, ls="--")
            axes[0].set_title("Volatility — Actual vs Predicted", color=TEXT_C, fontsize=11)
            axes[0].set_xlabel("Date"); axes[0].set_ylabel("Volatility")
            axes[0].legend(labelcolor=TEXT_C, facecolor=BG, edgecolor=BORDER, framealpha=0.3)
            resid = y_true - y_pred
            axes[1].scatter(y_pred, resid, alpha=0.5, color="#a78bfa", s=18)
            axes[1].axhline(0, color=SUB_C, lw=1, ls="--")
            axes[1].set_title("Residual Plot", color=TEXT_C, fontsize=11)
            axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Residual")
            st.pyplot(fig)
            img_bufs["volatility"] = fig_to_buf(fig)
            plt.close(fig)

            if imps:
                top15 = sorted(imps.items(), key=lambda x: x[1], reverse=True)[:15]
                fn, fs = zip(*top15)
                fig, ax = dark_fig(10, 5)
                bar_c = cm.viridis(np.linspace(0.15, 0.9, len(fn)))
                ax.barh(fn[::-1], fs[::-1], color=bar_c[::-1])
                ax.set_title("Feature Importance — Volatility Drivers", color=TEXT_C, fontsize=11)
                ax.set_xlabel("Importance Score")
                st.pyplot(fig)
                img_bufs["feat_imp"] = fig_to_buf(fig)
                plt.close(fig)

                st.markdown("**Narrative → Volatility interpretation**")
                for feat, score in [(n,s) for n,s in top15 if "nar_" in n][:6]:
                    parts = feat.split("_")
                    nid   = parts[1] if len(parts)>1 else "?"
                    kind  = "volume" if "cnt" in feat else "sentiment"
                    st.markdown(f"- **Narrative {nid}** ({kind}) → importance **{score:.4f}**")
        else:
            st.warning("Not enough data for cross-validation (need ≥ 10 unique dates).")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 10  Export CSV
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-head">📂 Step 10 · Annotated Dataset Export</div>', unsafe_allow_html=True)
export = df.copy()
export["umap_x"] = reduced_2d[:,0]
export["umap_y"] = reduced_2d[:,1]
st.dataframe(export, use_container_width=True)
st.download_button("⬇️  Download annotated CSV",
                   export.to_csv(index=False).encode(),
                   "narrative_annotated.csv", "text/csv")

st.success("✅  Pipeline complete — narrative detection and volatility modelling done.")

# ══════════════════════════════════════════════════════════════════════════════
#  PDF REPORT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-head">📄 Download PDF Report</div>', unsafe_allow_html=True)

if st.button("📄 Generate PDF Report", type="primary"):
    try:
        from reportlab.lib.pagesizes  import A4
        from reportlab.lib.styles     import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units      import cm
        from reportlab.lib             import colors
        from reportlab.platypus        import (SimpleDocTemplate, Paragraph,
                                               Spacer, Table, TableStyle,
                                               HRFlowable, Image as RLImage)
        from reportlab.lib.enums       import TA_CENTER
        import datetime

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                leftMargin=2*cm, rightMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle("T", fontSize=20, fontName="Helvetica-Bold",
                                     textColor=colors.HexColor("#0ea5e9"),
                                     spaceAfter=6, alignment=TA_CENTER)
        sub_style   = ParagraphStyle("S", fontSize=10, fontName="Helvetica",
                                     textColor=colors.HexColor("#64748b"),
                                     spaceAfter=16, alignment=TA_CENTER)
        h1_style    = ParagraphStyle("H", fontSize=12, fontName="Helvetica-Bold",
                                     textColor=colors.HexColor("#0f172a"),
                                     spaceBefore=14, spaceAfter=6,
                                     backColor=colors.HexColor("#e0f2fe"),
                                     leftIndent=0)
        body_style  = ParagraphStyle("B", fontSize=9, fontName="Helvetica",
                                     textColor=colors.HexColor("#1e293b"),
                                     spaceAfter=4, leading=14)
        cap_style   = ParagraphStyle("C", fontSize=8, fontName="Helvetica-Oblique",
                                     textColor=colors.HexColor("#94a3b8"),
                                     spaceAfter=6)

        def tbl(data, widths=None):
            t = Table(data, colWidths=widths, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND",   (0,0),(-1,0),  colors.HexColor("#0ea5e9")),
                ("TEXTCOLOR",    (0,0),(-1,0),  colors.white),
                ("FONTNAME",     (0,0),(-1,0),  "Helvetica-Bold"),
                ("FONTSIZE",     (0,0),(-1,-1), 8),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),
                 [colors.HexColor("#f8fafc"), colors.HexColor("#e2e8f0")]),
                ("GRID",         (0,0),(-1,-1), 0.4, colors.HexColor("#cbd5e1")),
                ("LEFTPADDING",  (0,0),(-1,-1), 5),
                ("RIGHTPADDING", (0,0),(-1,-1), 5),
                ("TOPPADDING",   (0,0),(-1,-1), 3),
                ("BOTTOMPADDING",(0,0),(-1,-1), 3),
                ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
            ]))
            return t

        def add_img(key, w=15, h=8):
            if key in img_bufs:
                img_bufs[key].seek(0)
                return [RLImage(img_bufs[key], width=w*cm, height=h*cm),
                        Spacer(1, 0.3*cm)]
            return []

        now   = datetime.datetime.now().strftime("%d %B %Y, %H:%M")
        story = []

        # ── Cover ─────────────────────────────────────────────────────
        story += [Spacer(1, 1*cm),
                  Paragraph("Market Narrative Intelligence", title_style),
                  Paragraph("Automated Financial Narrative Detection & Volatility Prediction", sub_style),
                  Paragraph(f"Report generated: {now}", cap_style),
                  HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#0ea5e9"), spaceAfter=14)]

        # ── 1. Dataset ─────────────────────────────────────────────────
        story.append(Paragraph("1. Dataset Summary", h1_style))
        story.append(Paragraph(
            f"Documents: <b>{len(df):,}</b>  |  "
            f"Columns: <b>{len(df.columns)}</b>  |  "
            f"Has date: <b>{'Yes' if 'date' in df.columns else 'No'}</b>",
            body_style))
        if "sentiment" in df.columns and df["sentiment"].dtype == object:
            gt = df["sentiment"].value_counts()
            story.append(tbl(
                [["Label","Count","Percentage"]] +
                [[l.title(), str(c), f"{c/len(df)*100:.1f}%"] for l,c in gt.items()],
                widths=[5*cm, 4*cm, 4*cm]))
        if "gt_dist" in img_bufs:
            story += [Spacer(1,0.2*cm),
                      Paragraph("Ground-Truth Sentiment Distribution", cap_style)]
            story += add_img("gt_dist", w=10, h=5)

        # ── 2. Clustering ──────────────────────────────────────────────
        story.append(Paragraph("2. Narrative Clustering Results", h1_style))
        story.append(Paragraph(
            f"Narratives: <b>{n_clusters}</b>  |  "
            f"Noise: <b>{noise_n} ({noise_n/len(labels)*100:.1f}%)</b>  |  "
            f"Avg confidence: <b>{mean_conf:.3f}</b>",
            body_style))
        story += [Spacer(1,0.2*cm),
                  Paragraph("UMAP 2-D Projection", cap_style)]
        story += add_img("umap", w=15, h=9)

        if not strength.empty:
            story.append(tbl(
                [["Narrative","Docs","Avg Conf","Avg Sentiment"]] +
                [[f"Narrative {int(r['Narrative'])}", str(int(r['Strength'])),
                  f"{r['Avg_Confidence']:.3f}", f"{r['Avg_Sentiment']:+.3f}"]
                 for _,r in strength.head(10).iterrows()],
                widths=[4*cm, 3*cm, 4*cm, 4*cm]))
            story += [Spacer(1,0.2*cm),
                      Paragraph("Narrative Strength — Document Volume", cap_style)]
            story += add_img("strength", w=15, h=7)

        # ── 3. Themes ──────────────────────────────────────────────────
        story.append(Paragraph("3. Narrative Themes (TF-IDF Keywords)", h1_style))
        if themes:
            story.append(tbl(
                [["Narrative","Top Keywords","Sentiment"]] +
                [[f"Narrative {n}", ", ".join(k[:5]),
                  f"{'Bullish' if (sent_by_nar.loc[n,'Avg Sentiment'] if n in sent_by_nar.index else 0)>0.05 else ('Bearish' if (sent_by_nar.loc[n,'Avg Sentiment'] if n in sent_by_nar.index else 0)<-0.05 else 'Neutral')} ({(sent_by_nar.loc[n,'Avg Sentiment'] if n in sent_by_nar.index else 0):+.3f})"]
                 for n,k in list(themes.items())[:15]],
                widths=[3*cm, 9*cm, 4*cm]))

        # ── 4. Sentiment ───────────────────────────────────────────────
        story.append(Paragraph("4. Sentiment Analysis", h1_style))
        story += [Spacer(1,0.2*cm),
                  Paragraph("Sentiment Score per Narrative", cap_style)]
        story += add_img("sentiment", w=15, h=6)
        if not sent_by_nar.empty:
            story.append(tbl(
                [["Narrative","Avg Sentiment","Std Dev","Docs"]] +
                [[f"Narrative {n}", f"{r['Avg Sentiment']:+.4f}",
                  f"{r['Std Dev']:.4f}", str(int(r['Docs']))]
                 for n,r in sent_by_nar.iterrows()],
                widths=[4*cm, 4*cm, 3.5*cm, 3.5*cm]))

        # ── 5. Temporal trend ──────────────────────────────────────────
        if "trend" in img_bufs:
            story.append(Paragraph("5. Narrative Trends Over Time", h1_style))
            story += add_img("trend", w=15, h=6)

        # ── 6. Volatility ──────────────────────────────────────────────
        if pipe_v is not None and cv is not None:
            story.append(Paragraph("6. Volatility Prediction Results", h1_style))
            story.append(Paragraph(
                f"Model: <b>{vol_model_type}</b>  |  "
                f"CV RMSE: <b>{np.mean(cv['rmse']):.4f} ± {np.std(cv['rmse']):.4f}</b>  |  "
                f"CV R²: <b>{np.mean(cv['r2']):.4f}</b>  |  "
                f"Folds: <b>5 (TimeSeriesSplit)</b>",
                body_style))
            story += add_img("volatility", w=15, h=7)
            if imps:
                top10 = sorted(imps.items(), key=lambda x: x[1], reverse=True)[:10]
                story.append(tbl(
                    [["Feature","Importance","Type"]] +
                    [[f, f"{s:.6f}",
                      "Volume" if "cnt" in f else ("Sentiment" if "sent" in f else "Aggregate")]
                     for f,s in top10],
                    widths=[7*cm, 4*cm, 4*cm]))
                story += [Spacer(1,0.2*cm)]
                story += add_img("feat_imp", w=15, h=7)

        # ── 7. Graph ───────────────────────────────────────────────────
        story.append(Paragraph("7. Storyline Graph", h1_style))
        story.append(Paragraph(
            f"Nodes: <b>{G.number_of_nodes()}</b>  |  "
            f"Edges: <b>{G.number_of_edges()}</b>  |  "
            f"Components: <b>{nx.number_connected_components(G) if G.number_of_nodes()>0 else 0}</b>  |  "
            f"Edge threshold: <b>cosine similarity &gt; 0.85</b>",
            body_style))
        story += add_img("graph", w=15, h=10)

        # ── 8. Methodology ─────────────────────────────────────────────
        story.append(Paragraph("8. Methodology", h1_style))
        story.append(tbl(
            [["Stage","Method","Parameters"],
             ["Embeddings",  "all-MiniLM-L6-v2",  "384-D, cosine similarity"],
             ["Reduction",   "UMAP",               f"n_neighbors={umap_neighbors}, n_components={umap_dims}"],
             ["Clustering",  "HDBSCAN",            f"min_cluster={min_cluster_size}, min_samples={min_samples}"],
             ["Sentiment",   "FinBERT / TextBlob", "ProsusAI/finbert (domain-aware)"],
             ["Keywords",    "TF-IDF",             "bigrams, max_features=2000"],
             ["Volatility",  vol_model_type,       f"window={vol_window}, CV=TimeSeriesSplit(5)"]],
            widths=[4*cm, 5*cm, 6*cm]))

        # ── Footer ─────────────────────────────────────────────────────
        story += [Spacer(1, 1*cm),
                  HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#cbd5e1"), spaceAfter=6),
                  Paragraph(
                      f"Market Narrative Intelligence System · BSc Final Year AI Project · {now}",
                      cap_style)]

        doc.build(story)
        buf.seek(0)

        st.download_button(
            label="⬇️  Download PDF Report",
            data=buf,
            file_name=f"narrative_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf",
        )
        st.success("✅ PDF ready — click above to download.")

    except ImportError:
        st.error("reportlab not installed. Run: pip install reportlab")
    except Exception as e:
        st.error(f"PDF generation error: {e}")
