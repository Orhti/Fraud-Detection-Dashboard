# ============================================================
# Financial Fraud Detection — Streamlit Dashboard (v2)
# Run:  streamlit run dashboard/streamlit_app.py
#       (from inside Project_1_Fraud_Detection/ folder)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib, os, warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection | Data Science Project",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme colours ─────────────────────────────────────────────────────────────
C_BLUE   = "#1976D2"
C_RED    = "#D32F2F"
C_GREEN  = "#388E3C"
C_ORANGE = "#F57C00"
C_PURPLE = "#7B1FA2"
C_TEAL   = "#00796B"
C_LIGHT  = "#E3F2FD"

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* Sidebar accent */
  section[data-testid="stSidebar"] {{ background: #0D1B2A; }}
  section[data-testid="stSidebar"] * {{ color: #E8EAF6 !important; }}
  section[data-testid="stSidebar"] .stRadio label {{ font-size: 15px !important; }}

  /* KPI cards */
  .kpi-card {{
      background: #FFFFFF;
      border-radius: 12px;
      padding: 18px 20px;
      text-align: center;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
      border-top: 4px solid {C_BLUE};
  }}
  .kpi-card.red   {{ border-top-color: {C_RED}; }}
  .kpi-card.green {{ border-top-color: {C_GREEN}; }}
  .kpi-card.orange{{ border-top-color: {C_ORANGE}; }}
  .kpi-card.purple{{ border-top-color: {C_PURPLE}; }}
  .kpi-val  {{ font-size: 2rem; font-weight: 800; color: #0D1B2A; margin: 4px 0; }}
  .kpi-label{{ font-size: 0.82rem; color: #555; font-weight: 600; letter-spacing:.5px; text-transform:uppercase; }}
  .kpi-sub  {{ font-size: 0.78rem; color: #888; margin-top: 2px; }}

  /* Section headers */
  .section-hdr {{
      font-size: 1.15rem; font-weight: 700; color: #0D1B2A;
      border-left: 4px solid {C_BLUE}; padding-left: 10px;
      margin: 22px 0 10px 0;
  }}

  /* Insight box */
  .insight-box {{
      background: {C_LIGHT};
      border-left: 4px solid {C_BLUE};
      border-radius: 6px;
      padding: 12px 16px;
      font-size: 0.88rem;
      color: #1A237E;
      margin: 8px 0;
  }}

  /* Cover page */
  .cover-title {{ font-size: 2.8rem; font-weight: 900; color: #0D1B2A; }}
  .cover-sub   {{ font-size: 1.25rem; color: #455A64; margin-bottom: 24px; }}
  .badge {{
      display: inline-block;
      background: {C_BLUE}; color: white;
      border-radius: 20px; padding: 4px 14px;
      font-size: 0.82rem; font-weight: 600;
      margin: 3px;
  }}
  .badge.red    {{ background: {C_RED}; }}
  .badge.green  {{ background: {C_GREEN}; }}
  .badge.orange {{ background: {C_ORANGE}; }}
  .badge.purple {{ background: {C_PURPLE}; }}
</style>
""", unsafe_allow_html=True)


# ── Helper: KPI card HTML ──────────────────────────────────────────────────────
def kpi(label, value, sub="", colour=""):
    cls = f"kpi-card {colour}"
    return f"""
    <div class="{cls}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-val">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>"""

def insight(text):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)

def section(text):
    st.markdown(f'<div class="section-hdr">{text}</div>', unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    return pd.read_csv("data/synthetic_fraud_dataset1.csv")

df = load_data()


# ── Train / load model (cached — runs only once per session) ──────────────────
@st.cache_resource(show_spinner="Preparing prediction model…")
def get_model():
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from imblearn.over_sampling import SMOTE

    # Try loading saved model first
    try:
        if os.path.exists("models/best_fraud_model.pkl"):
            m   = joblib.load("models/best_fraud_model.pkl")
            sc  = joblib.load("models/scaler.pkl")
            enc = joblib.load("models/label_encoders.pkl")
            feat= joblib.load("models/feature_columns.pkl")
            # Compatibility test
            m.predict(np.zeros((1, len(feat))))
            return {"model": m, "scaler": sc, "encoders": enc, "features": feat, "src": "saved"}
    except Exception:
        pass

    # Retrain a clean Random Forest in this Python environment
    raw = df.copy().drop(columns=["Transaction_ID", "User_ID", "Date"])
    cats = ["Transaction_Type","Device_Type","Location","Merchant_Category","Card_Type"]
    enc = {}
    for c in cats:
        le = LabelEncoder(); raw[c] = le.fit_transform(raw[c]); enc[c] = le

    X, y = raw.drop(columns=["Fraud_Label"]), raw["Fraud_Label"]
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    Xtr_s, ytr_s = SMOTE(random_state=42).fit_resample(Xtr, ytr)
    sc = StandardScaler()
    Xtr_sc = sc.fit_transform(Xtr_s)

    rf = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(Xtr_sc, ytr_s)
    return {"model": rf, "scaler": sc, "encoders": enc,
            "features": X.columns.tolist(), "src": "retrained"}

mdl = get_model()


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🛡️ Fraud Detection")
st.sidebar.markdown("*Data Science Project — Rohit*")
st.sidebar.markdown("---")

page = st.sidebar.radio("", [
    "🏠  Project Overview",
    "📊  EDA & Patterns",
    "🤖  Model Results",
    "🚨  Live Predictor",
])
st.sidebar.markdown("---")
fraud_pct = df["Fraud_Label"].mean() * 100
st.sidebar.markdown(f"**Dataset**  \n`{len(df):,}` transactions  \n`{fraud_pct:.1f}%` fraud rate")
st.sidebar.markdown(f"**Model source:** `{mdl['src']}`")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PROJECT OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown('<div class="cover-title">🛡️ Financial Fraud Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="cover-sub">End-to-end Machine Learning system to detect fraudulent transactions</div>', unsafe_allow_html=True)

    st.markdown("""
    <span class="badge">Random Forest</span>
    <span class="badge red">XGBoost</span>
    <span class="badge green">Logistic Regression</span>
    <span class="badge orange">Isolation Forest</span>
    <span class="badge purple">Voting Ensemble</span>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── KPI row ───────────────────────────────────────────────────────────────
    total        = len(df)
    fraud_n      = int(df["Fraud_Label"].sum())
    legit_n      = total - fraud_n
    avg_fraud_amt= df[df["Fraud_Label"]==1]["Transaction_Amount"].mean()
    avg_legit_amt= df[df["Fraud_Label"]==0]["Transaction_Amount"].mean()

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.markdown(kpi("Total Transactions", f"{total:,}","synthetic_fraud_dataset1.csv"), unsafe_allow_html=True)
    c2.markdown(kpi("Fraud Cases",  f"{fraud_n:,}", f"{fraud_pct:.1f}% of total","red"),    unsafe_allow_html=True)
    c3.markdown(kpi("Legitimate",   f"{legit_n:,}", f"{100-fraud_pct:.1f}% of total","green"), unsafe_allow_html=True)
    c4.markdown(kpi("Avg Fraud Amt",f"${avg_fraud_amt:,.0f}","per fraudulent txn","orange"),unsafe_allow_html=True)
    c5.markdown(kpi("Avg Legit Amt",f"${avg_legit_amt:,.0f}","per legitimate txn","purple"),unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns([1.1, 0.9])

    # Class balance chart
    with col_a:
        section("Class Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
        counts = df["Fraud_Label"].value_counts().sort_index()
        axes[0].bar(["Legitimate","Fraud"], counts.values,
                    color=[C_BLUE, C_RED], edgecolor="white", linewidth=1.5, width=0.5)
        for i,v in enumerate(counts.values):
            axes[0].text(i, v+300, f"{v:,}", ha="center", fontweight="bold", fontsize=11)
        axes[0].set_title("Transaction Count", fontweight="bold")
        axes[0].set_ylabel("Count"); axes[0].set_ylim(0, max(counts)*1.15)
        axes[0].spines[["top","right"]].set_visible(False)

        axes[1].pie(counts.values, labels=["Legitimate","Fraud"],
                    colors=[C_BLUE, C_RED], autopct="%1.1f%%",
                    startangle=90, textprops={"fontsize":11},
                    wedgeprops={"edgecolor":"white","linewidth":2})
        axes[1].set_title("Class Split (%)", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        insight(f"Dataset is <b>heavily imbalanced</b> — only {fraud_pct:.1f}% fraud. SMOTE was used during training to balance it.")

    # Project pipeline
    with col_b:
        section("Project Pipeline")
        steps = [
            ("1", "Data Loading", "50,000 transactions", C_BLUE),
            ("2", "EDA",          "Patterns & distributions", C_TEAL),
            ("3", "Preprocessing","Encode → SMOTE → Scale", C_ORANGE),
            ("4", "Model Training","6 ML algorithms", C_PURPLE),
            ("5", "Evaluation",   "F1, AUC-ROC, Confusion Matrix", C_GREEN),
            ("6", "Dashboard",    "Streamlit interactive app", C_RED),
        ]
        for num, title, desc, color in steps:
            st.markdown(f"""
            <div style="display:flex;align-items:center;margin:8px 0;">
              <div style="background:{color};color:white;border-radius:50%;
                          width:32px;height:32px;display:flex;align-items:center;
                          justify-content:center;font-weight:800;flex-shrink:0">{num}</div>
              <div style="margin-left:12px;">
                <b style="color:#0D1B2A">{title}</b>
                <span style="color:#666;font-size:0.83rem;margin-left:6px">{desc}</span>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    section("Models Trained & Compared")
    model_info = pd.DataFrame({
        "Model"      : ["Logistic Regression","Decision Tree","Random Forest",
                        "XGBoost","Isolation Forest","Voting Ensemble"],
        "Type"       : ["Supervised","Supervised","Supervised","Supervised","Unsupervised","Ensemble"],
        "Key Idea"   : [
            "Linear decision boundary between fraud and legit",
            "Tree of if/else rules — simple and interpretable",
            "100 decision trees vote on each transaction",
            "Sequential trees, each correcting the last",
            "Anomaly detection — fraud is easy to isolate",
            "Combines LR + RF + XGB for best overall result",
        ],
    })
    st.dataframe(model_info, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ════════════════════════════════════════════════════════════════════════════
elif "EDA" in page:
    st.title("📊 Exploratory Data Analysis")
    st.caption("Understand patterns that distinguish fraudulent from legitimate transactions.")

    # ── Sidebar filters ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🔽 Filters")
        sel_types = st.multiselect("Transaction Type",
                                   df["Transaction_Type"].unique().tolist(),
                                   default=df["Transaction_Type"].unique().tolist())
        sel_devices = st.multiselect("Device Type",
                                     df["Device_Type"].unique().tolist(),
                                     default=df["Device_Type"].unique().tolist())
        amt_range = st.slider("Transaction Amount ($)",
                              float(df["Transaction_Amount"].min()),
                              float(df["Transaction_Amount"].max()),
                              (float(df["Transaction_Amount"].min()),
                               float(df["Transaction_Amount"].max())))

    dff = df[
        df["Transaction_Type"].isin(sel_types) &
        df["Device_Type"].isin(sel_devices) &
        df["Transaction_Amount"].between(*amt_range)
    ]

    st.info(f"Showing **{len(dff):,}** transactions | Fraud rate: **{dff['Fraud_Label'].mean()*100:.2f}%**")
    st.markdown("---")

    # Row 1
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        section("Transaction Amount — Fraud vs Legitimate")
        fig, ax = plt.subplots(figsize=(7,3.8))
        dff[dff["Fraud_Label"]==0]["Transaction_Amount"].hist(
            bins=40, alpha=0.65, color=C_BLUE, label="Legitimate", ax=ax)
        dff[dff["Fraud_Label"]==1]["Transaction_Amount"].hist(
            bins=40, alpha=0.65, color=C_RED, label="Fraud", ax=ax)
        ax.set_xlabel("Amount ($)"); ax.set_ylabel("Count")
        ax.legend(); ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        m0 = dff[dff["Fraud_Label"]==0]["Transaction_Amount"].mean()
        m1 = dff[dff["Fraud_Label"]==1]["Transaction_Amount"].mean()
        insight(f"Avg legitimate: <b>${m0:,.0f}</b> &nbsp;|&nbsp; Avg fraud: <b>${m1:,.0f}</b>")

    with r1c2:
        section("Fraud Rate by Transaction Type")
        ft = (dff.groupby("Transaction_Type")["Fraud_Label"].mean()*100).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7,3.8))
        colors = [C_RED if v > ft.mean() else "#90CAF9" for v in ft.values]
        bars = ax.bar(ft.index, ft.values, color=colors, edgecolor="white", linewidth=1.2)
        ax.axhline(ft.mean(), color="gray", linestyle="--", linewidth=1, alpha=0.7, label=f"Avg {ft.mean():.1f}%")
        ax.set_ylabel("Fraud Rate (%)"); ax.set_title("Red = above average", fontsize=9, color="#666")
        ax.legend(fontsize=9)
        for b,v in zip(bars,ft.values):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, f"{v:.1f}%",
                    ha="center", fontsize=9, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        insight(f"Highest-risk type: <b>{ft.index[0]}</b> at <b>{ft.values[0]:.1f}%</b> fraud rate.")

    # Row 2
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        section("Fraud Rate by Merchant Category")
        mc = (dff.groupby("Merchant_Category")["Fraud_Label"].mean()*100).sort_values()
        fig, ax = plt.subplots(figsize=(7,4))
        colors = [C_RED if v > mc.mean() else C_TEAL for v in mc.values]
        ax.barh(mc.index, mc.values, color=colors, edgecolor="white")
        ax.axvline(mc.mean(), color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Fraud Rate (%)")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        insight(f"Riskiest category: <b>{mc.index[-1]}</b> ({mc.values[-1]:.1f}%)")

    with r2c2:
        section("Top 10 Locations by Fraud Rate")
        loc = (dff.groupby("Location")["Fraud_Label"].mean()*100).sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(7,4))
        ax.barh(loc.index[::-1], loc.values[::-1],
                color=[C_RED if v > loc.mean() else "#EF9A9A" for v in loc.values[::-1]],
                edgecolor="white")
        ax.set_xlabel("Fraud Rate (%)")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        insight(f"Highest-risk location: <b>{loc.index[0]}</b> ({loc.values[0]:.1f}%)")

    # Row 3 — Correlation heatmap
    section("Feature Correlation Heatmap")
    insight("Shows how strongly each feature is related to others. Values near +1 or -1 indicate a strong relationship.")
    num_cols = ["Transaction_Amount","Account_Balance","Daily_Transaction_Count",
                "Card_Age","Previous_Fraudulent_Activity","Fraud_Label"]
    available_num = [c for c in num_cols if c in dff.columns]
    corr = dff[available_num].corr()
    fig, ax = plt.subplots(figsize=(8, 4))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
                center=0, ax=ax, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Matrix (lower triangle)", fontweight="bold")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Row 4 — Account balance & card age boxplots
    r4c1, r4c2 = st.columns(2)
    with r4c1:
        section("Account Balance — Fraud vs Legitimate")
        fig, ax = plt.subplots(figsize=(7,3.5))
        data_plot = [dff[dff["Fraud_Label"]==0]["Account_Balance"].dropna(),
                     dff[dff["Fraud_Label"]==1]["Account_Balance"].dropna()]
        bp = ax.boxplot(data_plot, labels=["Legitimate","Fraud"],
                        patch_artist=True, notch=False,
                        medianprops={"color":"white","linewidth":2})
        bp["boxes"][0].set_facecolor(C_BLUE)
        bp["boxes"][1].set_facecolor(C_RED)
        ax.set_ylabel("Account Balance ($)")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with r4c2:
        section("Daily Transaction Count Distribution")
        fig, ax = plt.subplots(figsize=(7,3.5))
        dff[dff["Fraud_Label"]==0]["Daily_Transaction_Count"].hist(
            bins=25, alpha=0.65, color=C_BLUE, label="Legitimate", ax=ax)
        dff[dff["Fraud_Label"]==1]["Daily_Transaction_Count"].hist(
            bins=25, alpha=0.65, color=C_RED, label="Fraud", ax=ax)
        ax.set_xlabel("Daily Transaction Count"); ax.legend()
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL RESULTS
# ════════════════════════════════════════════════════════════════════════════
elif "Model" in page:
    st.title("🤖 Model Performance Results")
    st.caption("All 6 models trained on the same dataset and evaluated on a held-out 20% test set.")

    results_df = None
    if os.path.exists("model_comparison.csv"):
        results_df = pd.read_csv("model_comparison.csv")

    if results_df is None:
        st.warning("⚠️ `model_comparison.csv` not found. Run the notebook first to generate model metrics.")
        st.stop()

    # Clean up columns
    disp_cols = [c for c in ["Rank","Model","Accuracy","Precision","Recall","F1 Score","AUC-ROC"]
                 if c in results_df.columns]
    best = results_df.iloc[0]

    # Winner banner
    st.markdown(f"""
    <div style="background:linear-gradient(90deg,{C_GREEN},{C_TEAL});
                color:white;border-radius:12px;padding:16px 24px;margin-bottom:16px;">
      <span style="font-size:1.5rem;font-weight:800">🏆 Best Model: {best["Model"]}</span><br>
      <span style="font-size:0.95rem;opacity:0.9">
        F1 Score: <b>{best["F1 Score"]}%</b> &nbsp;|&nbsp;
        AUC-ROC: <b>{best["AUC-ROC"]}%</b> &nbsp;|&nbsp;
        Recall: <b>{best["Recall"]}%</b>
      </span>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # KPI row from best model
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(kpi("Accuracy",  f"{best['Accuracy']}%",  "overall correct","green"),  unsafe_allow_html=True)
    c2.markdown(kpi("Precision", f"{best['Precision']}%", "of flagged = real fraud"),   unsafe_allow_html=True)
    c3.markdown(kpi("Recall",    f"{best['Recall']}%",    "of frauds caught","orange"), unsafe_allow_html=True)
    c4.markdown(kpi("AUC-ROC",   f"{best['AUC-ROC']}%",  "separation ability","purple"),unsafe_allow_html=True)
    st.markdown("")

    # Full table
    section("All Models — Ranked by F1 Score")
    insight("F1 Score is our primary metric because the dataset is imbalanced — Accuracy alone is misleading for fraud detection.")

    def style_table(df):
        styled = df[disp_cols].style\
            .highlight_max(subset=[c for c in ["F1 Score","AUC-ROC","Recall"] if c in df.columns],
                           color="#C8E6C9")\
            .highlight_min(subset=[c for c in ["F1 Score","AUC-ROC"] if c in df.columns],
                           color="#FFCDD2")\
            .format({c: "{:.2f}%" for c in ["Accuracy","Precision","Recall","F1 Score","AUC-ROC"]
                     if c in df.columns})
        return styled

    st.dataframe(style_table(results_df), use_container_width=True, hide_index=True)

    st.markdown("---")
    r1, r2 = st.columns(2)

    with r1:
        section("F1 Score Comparison")
        fig, ax = plt.subplots(figsize=(7,4))
        colors = [C_GREEN if i==0 else "#90CAF9" for i in range(len(results_df))]
        bars = ax.bar(results_df["Model"], results_df["F1 Score"],
                      color=colors, edgecolor="white", linewidth=1.2)
        ax.set_ylabel("F1 Score (%)"); ax.set_ylim(0,115)
        ax.axhline(90, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.text(len(results_df)-0.5, 91, "90% line", color="gray", fontsize=8)
        for b,v in zip(bars, results_df["F1 Score"]):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                    f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
        plt.xticks(rotation=22, ha="right", fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with r2:
        section("Radar Chart — Top 3 Models")
        metrics = ["Accuracy","Precision","Recall","F1 Score","AUC-ROC"]
        avail_m = [m for m in metrics if m in results_df.columns]
        top3 = results_df.head(3)
        angles = np.linspace(0, 2*np.pi, len(avail_m), endpoint=False).tolist()
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
        radar_colors = [C_GREEN, C_BLUE, C_ORANGE]
        for (_, row), color in zip(top3.iterrows(), radar_colors):
            vals = [row[m] for m in avail_m] + [row[avail_m[0]]]
            ax.plot(angles, vals, color=color, linewidth=2)
            ax.fill(angles, vals, color=color, alpha=0.12)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(avail_m, fontsize=9)
        ax.set_ylim(0, 105)
        legend_patches = [mpatches.Patch(color=c, label=row["Model"])
                          for c,(_, row) in zip(radar_colors, top3.iterrows())]
        ax.legend(handles=legend_patches, loc="upper right",
                  bbox_to_anchor=(1.35,1.1), fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Metric breakdown
    section("All Metrics — Side by Side")
    fig, axes = plt.subplots(1, len(avail_m), figsize=(14,3.5))
    pal = [C_BLUE, C_GREEN, C_ORANGE, C_RED, C_PURPLE]
    for ax, metric, color in zip(axes, avail_m, pal):
        ax.barh(results_df["Model"], results_df[metric], color=color, alpha=0.85, edgecolor="white")
        ax.set_title(metric, fontweight="bold", fontsize=10)
        ax.set_xlim(0,110)
        for i,v in enumerate(results_df[metric]):
            ax.text(v+0.5, i, f"{v:.0f}%", va="center", fontsize=8)
        ax.spines[["top","right","bottom"]].set_visible(False)
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Metrics explained
    with st.expander("📖 What do these metrics mean?"):
        st.markdown("""
| Metric | What it measures | Why it matters for fraud |
|--------|-----------------|--------------------------|
| **Accuracy** | % of all predictions correct | Misleading on imbalanced data — a model that always says "not fraud" gets 68% accuracy |
| **Precision** | Of all flagged fraud cases, how many were real | Avoids false alarms that annoy legitimate customers |
| **Recall** | Of all real fraud, how much did we catch | Missing fraud is expensive — high recall is critical |
| **F1 Score** | Harmonic mean of Precision and Recall | Best single metric when both matter |
| **AUC-ROC** | How well the model separates classes at all thresholds | 1.0 = perfect, 0.5 = random guessing |
        """)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — LIVE PREDICTOR
# ════════════════════════════════════════════════════════════════════════════
elif "Predictor" in page:
    st.title("🚨 Live Transaction Fraud Predictor")
    st.caption("Fill in a transaction's details and the model will predict whether it's fraudulent.")

    if mdl.get("src") == "retrained":
        st.info("ℹ️ Using freshly trained model (Python version compatibility mode — works perfectly).")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 💳 Transaction Details")
        amount     = st.number_input("Transaction Amount ($)", 0.0, 50000.0, 250.0, 10.0)
        tx_type    = st.selectbox("Transaction Type", sorted(df["Transaction_Type"].unique()))
        merchant   = st.selectbox("Merchant Category",sorted(df["Merchant_Category"].unique()))

    with col2:
        st.markdown("#### 📱 User & Device")
        balance    = st.number_input("Account Balance ($)", 0.0, 200000.0, 15000.0, 500.0)
        device     = st.selectbox("Device Type", sorted(df["Device_Type"].unique()))
        location   = st.selectbox("Location",    sorted(df["Location"].unique()))

    with col3:
        st.markdown("#### 🃏 Card & History")
        card_type  = st.selectbox("Card Type", sorted(df["Card_Type"].unique()))
        card_age   = st.slider("Card Age (months)", 0, 300, 60)
        daily_tx   = st.slider("Transactions Today", 1, 30, 5)
        prev_fraud = st.selectbox("Previous Fraud Activity?", [0, 1],
                                  format_func=lambda x: "Yes (1)" if x==1 else "No (0)")

    st.markdown("---")
    predict_btn = st.button("🔍 Analyse Transaction", type="primary", use_container_width=True)

    if predict_btn:
        with st.spinner("Running fraud analysis…"):
            try:
                row = {
                    "Transaction_Amount"          : amount,
                    "Transaction_Type"            : tx_type,
                    "Account_Balance"             : balance,
                    "Device_Type"                 : device,
                    "Location"                    : location,
                    "Merchant_Category"           : merchant,
                    "Previous_Fraudulent_Activity": prev_fraud,
                    "Daily_Transaction_Count"     : daily_tx,
                    "Card_Type"                   : card_type,
                    "Card_Age"                    : card_age,
                }
                inp = pd.DataFrame([row])

                for col, le in mdl["encoders"].items():
                    if col in inp.columns:
                        try:   inp[col] = le.transform(inp[col])
                        except ValueError: inp[col] = 0

                for col in mdl["features"]:
                    if col not in inp.columns:
                        inp[col] = 0
                inp = inp[mdl["features"]]
                inp_sc = mdl["scaler"].transform(inp)

                pred = mdl["model"].predict(inp_sc)[0]
                prob = None
                try:
                    prob = mdl["model"].predict_proba(inp_sc)[0][1]
                except Exception:
                    pass

                st.markdown("---")
                if pred == 1:
                    st.error(f"## 🚨 FRAUD DETECTED")
                    if prob:
                        st.error(f"**Fraud probability: {prob*100:.1f}%** — This transaction is highly likely to be fraudulent.")
                else:
                    st.success(f"## ✅ LEGITIMATE TRANSACTION")
                    if prob:
                        st.success(f"**Fraud probability: {prob*100:.1f}%** — This transaction appears safe.")

                if prob is not None:
                    st.markdown("**Risk Level**")
                    risk_pct = prob * 100
                    bar_color = C_RED if risk_pct > 50 else (C_ORANGE if risk_pct > 25 else C_GREEN)
                    st.markdown(f"""
                    <div style="background:#eee;border-radius:8px;height:20px;width:100%">
                      <div style="background:{bar_color};width:{risk_pct:.1f}%;height:20px;
                                  border-radius:8px;transition:width 0.5s">
                      </div>
                    </div>
                    <p style="color:{bar_color};font-weight:700;margin-top:4px">{risk_pct:.1f}% fraud probability</p>
                    """, unsafe_allow_html=True)

                # Summary table
                st.markdown("---")
                section("Transaction Summary")
                summary = pd.DataFrame({
                    "Field": list(row.keys()),
                    "Value": [str(v) for v in row.values()]
                })
                st.dataframe(summary, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#888;font-size:0.8rem'>"
    "🛡️ Financial Fraud Detection Dashboard &nbsp;|&nbsp; "
    "Data Science Project &nbsp;|&nbsp; Rohit &nbsp;|&nbsp; 2026"
    "</p>",
    unsafe_allow_html=True
)
