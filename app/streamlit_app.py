
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================
# CONFIGURATION AND MODEL LOADING
# ============================================================

import json
import joblib
import xgboost as xgb
from pathlib import Path

st.set_page_config(page_title="CyberSentryFusion", page_icon="🛡️", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
CLASSIFIER_PATH = MODEL_DIR / "xgboost_classifier.json"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"

preprocessor = None
classifier = None
threshold = 0.5
model_error = None

try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    classifier = xgb.XGBClassifier()
    classifier.load_model(str(CLASSIFIER_PATH))
    with open(THRESHOLD_PATH, "r", encoding="utf-8") as threshold_file:
        threshold = float(json.load(threshold_file)["threshold"])
except Exception as exc:
    model_error = str(exc)

# ============================================================
# THEME AND CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #020b0a;
        --panel: rgba(8, 30, 27, 0.78);
        --teal: #37f5ce;
        --teal-soft: #13b99a;
        --text: #e9fffa;
        --muted: #8daaa3;
        --border: rgba(55, 245, 206, 0.19);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 50% 0%, rgba(0, 145, 117, 0.16), transparent 48%),
            radial-gradient(ellipse at 100% 50%, rgba(0, 110, 100, 0.08), transparent 38%),
            var(--bg);
        color: var(--text);
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.8rem;
        padding-bottom: 4rem;
    }

    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.8rem 0 1.4rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 2.2rem;
    }

    .brand {
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: -0.6px;
        color: var(--text);
    }

    .brand span {
        color: var(--teal);
    }

    .status {
        color: var(--teal);
        font-size: 0.78rem;
        padding: 0.48rem 0.8rem;
        border: 1px solid var(--border);
        border-radius: 999px;
        background: rgba(19, 185, 154, 0.08);
    }

    .hero {
        position: relative;
        overflow: hidden;
        text-align: center;
        padding: 4rem 1.4rem 3.4rem;
        border: 1px solid var(--border);
        border-radius: 24px;
        background:
            radial-gradient(ellipse at 50% 100%, rgba(0, 230, 190, 0.16), transparent 45%),
            linear-gradient(145deg, rgba(5, 24, 21, 0.94), rgba(1, 10, 9, 0.97));
        box-shadow: 0 0 60px rgba(0, 230, 190, 0.045);
        margin-bottom: 2rem;
    }

    .hero:before {
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        opacity: 0.15;
        background-image: radial-gradient(#43ffe0 0.7px, transparent 0.7px);
        background-size: 20px 20px;
        mask-image: linear-gradient(to bottom, black, transparent 85%);
    }

    .eyebrow {
        display: inline-block;
        color: var(--teal);
        font-size: 0.76rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        border: 1px solid var(--border);
        background: rgba(55, 245, 206, 0.06);
        padding: 0.5rem 0.9rem;
        border-radius: 999px;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        position: relative;
        font-size: clamp(2.2rem, 5vw, 4.3rem);
        line-height: 1.06;
        letter-spacing: -2.5px;
        margin: 0 auto 1.1rem;
        max-width: 900px;
        color: #f0fffc;
        font-weight: 800;
    }

    .hero h1 span {
        color: var(--teal);
        text-shadow: 0 0 35px rgba(55, 245, 206, 0.24);
    }

    .hero p {
        color: var(--muted);
        font-size: 1rem;
        max-width: 650px;
        line-height: 1.8;
        margin: 0 auto;
    }

    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--text);
        margin: 2.1rem 0 0.25rem;
    }

    .section-subtitle {
        color: var(--muted);
        font-size: 0.85rem;
        margin-bottom: 1.1rem;
    }

    .metric-card {
        padding: 1.25rem 1.3rem;
        border: 1px solid var(--border);
        border-radius: 16px;
        min-height: 130px;
        background: linear-gradient(145deg, rgba(10, 38, 33, 0.85), rgba(4, 18, 16, 0.92));
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.025);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(55, 245, 206, 0.48);
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.8rem;
        margin-bottom: 0.8rem;
    }

    .metric-value {
        color: var(--teal);
        font-size: 1.9rem;
        font-weight: 800;
        letter-spacing: -0.8px;
    }

    .metric-caption {
        color: #a5c3bb;
        font-size: 0.76rem;
        margin-top: 0.4rem;
    }

    .panel {
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: 1.3rem;
        background: var(--panel);
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid rgba(55, 245, 206, 0.45);
        background: linear-gradient(110deg, #0b806e, #13b99a);
        color: #001c17;
        font-weight: 700;
        min-height: 2.8rem;
        transition: all 0.2s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: var(--teal);
        box-shadow: 0 0 24px rgba(55, 245, 206, 0.16);
        color: #001c17;
    }

    .stTextInput input,
    .stNumberInput input {
        background: #061613;
        color: var(--text);
        border-color: var(--border);
    }

    [data-testid="stFileUploader"] {
        background: rgba(5, 25, 21, 0.7);
        border: 1px dashed rgba(55, 245, 206, 0.3);
        border-radius: 14px;
        padding: 0.7rem;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    hr {
        border-color: var(--border);
    }

    .footer {
        text-align: center;
        color: #6e8e85;
        font-size: 0.78rem;
        padding-top: 2rem;
        margin-top: 3rem;
        border-top: 1px solid var(--border);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# LOAD MODEL
# ============================================================



# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
CLASSIFIER_PATH = MODEL_DIR / "xgboost_classifier.json"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"

# Initialize model variables
preprocessor = None
classifier = None
threshold = 0.5
model_loaded = False

try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    classifier = xgb.XGBClassifier()
    classifier.load_model(str(CLASSIFIER_PATH))

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
        threshold = float(json.load(file)["threshold"])

    model_loaded = True
    print("CyberSentryFusion models loaded successfully.")

except Exception as exc:
    print(f"Model loading failed: {exc}")



# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="topbar">
        <div class="brand">🛡 Cyber<span>SentryFusion</span></div>
        <div class="status">● AI ENGINE · READY TO ANALYZE</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">✦ Intelligent Network Defense</div>
        <h1>See the Threat.<br><span>Stop the Unknown.</span></h1>
        <p>
            Machine-learning-powered network intrusion analysis.
            Explore traffic patterns, investigate suspicious records,
            and identify potential threats with CyberSentryFusion.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# MODEL STATUS
# ============================================================

if model_error:
    st.error("The model could not be loaded. Check the model files in the project's models/ directory.")
    st.code(model_error)
else:
    st.success("CyberSentryFusion model artifacts loaded successfully.")

# ============================================================
# SIDEBAR / ANALYSIS CONTROLS
# ============================================================

with st.sidebar:
    st.markdown("## 🛡 CyberSentryFusion")
    st.caption("Network threat analysis")

    st.markdown("---")
    st.markdown("### Analysis settings")

    st.write(f"Decision threshold: **{threshold:.2f}**")
    st.caption("The saved threshold is used for classification.")

    st.markdown("---")
    st.caption("Analysis runs when you submit a compatible CSV.")
    st.caption("This application does not capture live network packets.")

# ============================================================
# OVERVIEW METRICS
# ============================================================

st.markdown('<div class="section-title">Threat Intelligence Overview</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Summary of the most recent uploaded-data analysis.</div>',
    unsafe_allow_html=True,
)

# Only show live analysis counts after a CSV has actually been processed.
if "prediction_results" not in st.session_state:
    st.session_state.prediction_results = None

if st.session_state.prediction_results is None:
    total_records = 0
    attacks_detected = 0
    normal_records = 0
    attack_rate = 0.0
else:
    current_results = st.session_state.prediction_results
    total_records = len(current_results)
    attacks_detected = int((current_results["prediction"] == 1).sum())
    normal_records = total_records - attacks_detected
    attack_rate = (
        attacks_detected / total_records * 100 if total_records else 0.0
    )

metric_cols = st.columns(4)

metrics = [
    ("RECORDS ANALYZED", f"{total_records:,}", "Uploaded traffic records"),
    ("ATTACKS DETECTED", f"{attacks_detected:,}", "Predicted attack records"),
    ("NORMAL TRAFFIC", f"{normal_records:,}", "Predicted normal records"),
    ("ATTACK RATE", f"{attack_rate:.1f}%", "Within analyzed records"),
]

for col, (label, value, caption) in zip(metric_cols, metrics):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-caption">{caption}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# CSV UPLOAD AND PREDICTION
# ============================================================

st.markdown('<div class="section-title">Network Traffic Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Upload a CSV containing the expected network-flow features.</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload network-flow CSV",
    type=["csv"],
    help="Provide the 42 input features used during training. Optional id, attack_cat, and label columns are ignored.",
)

if "prediction_results" not in st.session_state:
    st.session_state.prediction_results = None

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        st.write(f"Uploaded dataset: **{len(uploaded_df):,} records**")

        if model_error or preprocessor is None or classifier is None:
            st.warning("Predictions are unavailable because the model artifacts did not load.")
        elif uploaded_df.empty:
            st.error("The uploaded CSV contains no records.")
        else:
            X_upload = uploaded_df.drop(columns=["id", "attack_cat", "label"], errors="ignore")

            # The fitted preprocessor stores the raw input columns used during training.
            expected_features = list(getattr(preprocessor, "feature_names_in_", []))
            if not expected_features:
                st.error("Could not read the expected feature schema from preprocessor.joblib.")
            else:
                missing_features = sorted(set(expected_features) - set(X_upload.columns))
                extra_features = sorted(set(X_upload.columns) - set(expected_features))

                if missing_features:
                    st.error(
                        f"Missing {len(missing_features)} required features: "
                        + ", ".join(missing_features)
                    )
                else:
                    if extra_features:
                        st.info("Ignoring unexpected input columns: " + ", ".join(extra_features))
                    X_upload = X_upload[expected_features]

                    # Include a few original fields in the output for investigation.
                    detail_columns = [c for c in ["proto", "service", "state", "dur", "sbytes", "dbytes"] if c in X_upload.columns]

                    if st.button("⚡ Analyze Network Traffic", use_container_width=True):
                        with st.spinner("Analyzing records with CyberSentryFusion..."):
                            X_processed = preprocessor.transform(X_upload)
                            probabilities = classifier.predict_proba(X_processed)[:, 1]
                            predictions = (probabilities >= threshold).astype(int)

                            results = pd.DataFrame({
                                "prediction": predictions,
                                "predicted_label": np.where(predictions == 1, "Attack", "Normal"),
                                "attack_probability": probabilities,
                            })
                            for col_name in detail_columns:
                                results.insert(len(results.columns) - 3, col_name, X_upload[col_name].to_numpy())

                            st.session_state.prediction_results = results
                            st.session_state.uploaded_source_name = uploaded_file.name
                        st.success("Analysis complete.")
    except Exception as exc:
        st.error(f"Could not process this CSV: {exc}")
# ============================================================
# RESULTS
# ============================================================

if st.session_state.prediction_results is not None:
    results = st.session_state.prediction_results.copy()

    st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Predictions generated by your saved XGBoost pipeline.</div>',
        unsafe_allow_html=True,
    )

    chart_col, probability_col = st.columns(2)

    with chart_col:
        label_counts = (
            results["predicted_label"]
            .value_counts()
            .rename_axis("Prediction")
            .reset_index(name="Records")
        )

        fig_labels = px.pie(
            label_counts,
            names="Prediction",
            values="Records",
            hole=0.68,
            title="Predicted Traffic Distribution",
            color="Prediction",
            color_discrete_map={
                "Normal": "#168c78",
                "Attack": "#ff6b78",
            },
        )
        fig_labels.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#dffcf5",
            legend_title_text="",
            margin=dict(t=60, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_labels, use_container_width=True)

    with probability_col:
        fig_prob = px.histogram(
            results,
            x="attack_probability",
            nbins=35,
            title="Attack Probability Distribution",
            labels={"attack_probability": "Attack probability"},
        )
        fig_prob.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="#37f5ce",
            annotation_text=f"Threshold = {threshold:.2f}",
        )
        fig_prob.update_traces(marker_color="#13b99a")
        fig_prob.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#dffcf5",
            margin=dict(t=60, b=30, l=20, r=20),
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    st.markdown("### Detection Records")

    attack_only = st.checkbox("Show only predicted attacks", value=False)
    visible_results = results

    if attack_only:
        visible_results = results[results["prediction"] == 1]

    st.dataframe(
        visible_results,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇ Download Prediction Results",
        data=csv_data,
        file_name="cybersentryfusion_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Offline test-set evaluation. These are not live monitoring metrics.</div>',
    unsafe_allow_html=True,
)

performance = [
    ("ACCURACY", "87.28%"),
    ("PRECISION", "81.83%"),
    ("ATTACK RECALL", "98.86%"),
    ("F1-SCORE", "89.54%"),
    ("ROC-AUC", "98.44%"),
    ("FALSE POSITIVE RATE", "26.90%"),
]

performance_cols = st.columns(3)

for idx, (label, value) in enumerate(performance):
    with performance_cols[idx % 3]:
        st.markdown(
            f"""
            <div class="metric-card" style="margin-bottom: 1rem;">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.info(
    "These metrics describe the model's recorded evaluation on the UNSW-NB15 "
    "test dataset. Real-world performance may differ."
)

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        CYBERSENTRYFUSION · AI-POWERED NETWORK INTRUSION ANALYSIS<br>
        <span style="color:#37f5ce;">Intelligence. Visibility. Defense.</span><br><br>
        Dataset-based analysis only · No live packet capture is configured
    </div>
    """,
    unsafe_allow_html=True,
)

