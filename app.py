#Importing libraries
import json
import io
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ===========================
# Page Setup & Global Styles
# ===========================
st.set_page_config(
    page_title="Mental Health in Tech - Analysis & Predictions",
    page_icon="🧠",
    layout="wide"
)

# --- Custom CSS (subtle glassy cards, gradient headers, nicer buttons) ---
st.markdown("""
<style>
/* Global font sizes a bit bigger */
html, body, [class*="css"] { font-size: 16px; }

/* Gradient page title */
.gradient-title {
  background: linear-gradient(90deg, #4f46e5 0%, #06b6d4 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

/* Cards */
.card {
  border-radius: 18px;
  padding: 18px;
  background: rgba(255,255,255,0.6);
  border: 1px solid rgba(0,0,0,0.05);
  box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}

/* Metric-like chips */
.kchip {
  display: inline-block;
  border-radius: 14px;
  padding: 4px 10px;
  margin: 2px 8px 2px 0;
  background: #eef2ff;
  border: 1px solid #e0e7ff;
  font-size: 0.9rem;
}

/* Section headers */
.section {
  padding: 6px 10px;
  border-left: 5px solid #6366f1;
  background: #f8fafc;
  border-radius: 8px;
}

/* Footer */
.footer {
  color: #6b7280;
  font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


# ===========================
# Helpers: IO & UI
# ===========================
IMG_DIR = Path("Images")
MODELS_DIR = Path("Models & Datasets")

def exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False

def show_image_if_exists(path: Path, caption: str = "", use_cols: int = 0):
    if exists(path):
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"ℹ️ Image not found at `{path.as_posix()}`")

def show_image_grid(paths_and_caps: List[Tuple[Path, str]], cols=3):
    if not paths_and_caps:
        return
    for i in range(0, len(paths_and_caps), cols):
        row = paths_and_caps[i:i+cols]
        columns = st.columns(len(row))
        for c, (p, cap) in zip(columns, row):
            with c:
                show_image_if_exists(p, cap)

def info_chip(text: str, bg="#6C63FF", color="white"):
    st.markdown(
        f"""
        <span style="
            background-color:{bg};
            color:{color};
            padding:3px 8px;
            border-radius:12px;
            font-size:12px;
            font-weight:500;
        ">{text}</span>
        """,
        unsafe_allow_html=True
    )

# Footer
def footer():
    st.markdown("---")
    st.markdown("""
    <small>Built with ❤️ and dedication by **Saaransh** 😎| 
    [LinkedIn](https://www.linkedin.com/in/saaransh-saxena-298256330/) • 
    [X](https://x.com/itsSaaransh) • 
    [GitHub](https://github.com/saaransh0602) • 
    [Medium Blog](https://medium.com/@saaransh2006/mental-health-in-tech-analysis-and-predictions-d14e4a7982fa)</small>
    """, unsafe_allow_html=True)

# ===========================
# Model Loading & Prediction API
# ===========================
@st.cache_resource(show_spinner=True)
def load_models_and_schemas():
    # You can change the paths below if needed
    clf = joblib.load(MODELS_DIR / "classification_model.pkl")
    reg = joblib.load(MODELS_DIR / "regression_model.pkl")

    with open(MODELS_DIR / "classification_columns.json") as f:
        clf_cols = json.load(f)
    with open(MODELS_DIR / "regression_columns.json") as f:
        reg_cols = json.load(f)

    return clf, reg, clf_cols, reg_cols


def build_input_df(user_input: Dict, expected_columns: List[str]) -> pd.DataFrame:
    """
    Build a one-row DataFrame with exactly the columns the model expects.
    - Any missing columns are added with 0.
    - Extra keys in user_input are ignored by subsetting to expected_columns.
    """
    df = pd.DataFrame([user_input])
    # add missing
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    # order & subset
    df = df[expected_columns]
    return df


def predict_classification(clf, expected_cols, user_input: Dict):
    df = build_input_df(user_input, expected_cols)
    pred = clf.predict(df)[0]
    # try predict_proba if supported
    try:
        proba = clf.predict_proba(df)[0]
    except Exception:
        # fallback: fake a probability if not available
        proba = np.array([0.5, 0.5]) if pred in [0, 1] else np.array([1.0])
    return pred, proba


def predict_regression(reg, expected_cols, user_input: Dict):
    df = build_input_df(user_input, expected_cols)
    pred = reg.predict(df)[0]
    return float(pred)


# ===========================
# Support Score (Mean) Mapping
# ===========================
SUPPORT_COLS = [
    "benefits", "care_options", "wellness_program", "seek_help",
    "anonymity", "supervisor", "coworkers", "leave"
]

VALUE_MAP = {
    'Yes': 1.0,
    'No': 0.0,
    "Don't know": 2.0,
    'Not sure': 2.0,
    'Some of them': 0.5,
    'Difficult': 0.0,
    'Medium': 0.5,
    'Easy': 1.0
}

def compute_support_score_mean_from_raw(user_raw: Dict) -> float:
    vals = [VALUE_MAP.get(user_raw.get(c), 0) for c in SUPPORT_COLS]
    return float(np.mean(vals)) if len(vals) else 0.0


# ===========================
# Sidebar Navigation
# ===========================
st.sidebar.title("🧭 Navigation")
menu = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 Exploratory Data Analysis",
     "🧮 Prediction using Classification", "📈 Prediction using Regression",
     "📌 Clustering Personas"],
    key="nav_radio"
)

# Load models once
try:
    clf_model, reg_model, clf_cols, reg_cols = load_models_and_schemas()
except Exception as e:
    st.error(f"❌ Failed to load models/schemas: {e}")
    st.stop()


if menu == "🏠 Home":
    st.markdown('<h1 class="gradient-title">🧠 Mental Health in Tech — Analysis & Predictions</h1>', unsafe_allow_html=True)
    st.write("")

    st.markdown("""
    <div style="text-align: justify;">
    Mental health is a critical aspect of workplace wellbeing.  
    This project explores survey data from the **tech industry** to understand patterns, 
    predict treatment-seeking behavior, and cluster employees into distinct personas.  
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # 🔹 Dataset Source
    st.subheader("📂 Dataset Source")
    st.markdown("""
    Dataset: **[Mental Health in Tech Survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)**  
    Collected by **OSMI (Open Sourcing Mental Illness)**
    """)

    # 🔹 Features
    st.subheader("🧾 Features include")
    st.markdown("""
    - Demographic details (age, gender, country)  
    - Workplace environment (mental health benefits, leave policies)  
    - Personal experiences (mental illness, family history)  
    - Attitudes towards mental health  
    """)

    # 🔹 Problem Statement
    st.subheader("❓ Problem Statement")
    st.markdown(""" As a Machine Learning Engineer at NeuronInsights Analytics, we've been contracted by a coalition of 
    leading tech companies including CodeLab, QuantumEdge, and SynapseWorks. Alarmed by rising burnout, 
    disengagement, and attrition linked to mental health, the consortium seeks data-driven strategies to 
    proactively identify and support at-risk employees. Our role is to analyze survey data from over 1,500 tech 
    professionals, covering workplace policies, personal mental health history, openness to seeking help, and perceived employer support.""")

    st.markdown("---")

    # ✨ Why this matters
    st.subheader("✨ Why this matters")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.success("📈 Data-driven insight for HR & well-being.")
    with c2:
        st.info("👩‍💻 Tech workplace pressure is real; stigma persists.")
    with c3:
        st.warning("🧩 Policy & culture shape outcomes more than you think.")

    st.markdown("---")

    # 📌 Highlights
    st.subheader("📌 Highlights from the study")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Responses", "1,259")
        info_chip("OSMI dataset")
    with k2:
        st.metric("Best Classifier", "XGBoost")
        info_chip("ROC-AUC ≈ 0.81")
    with k3:
        st.metric("Best Regressor", "RF 🌳")
        info_chip("R² Score ≈ 0.07")
    with k4:
        st.metric("Clustering Silhouette", "≈0.54")
        info_chip("KMeans + UMAP")

    st.info("💬 *“Mental health requires as much intentional support as physical health.”*")

    # Notes & Caveats
    st.subheader("🧩 Notes & Caveats")
    st.markdown("""
    - Classification uses workplace support signals + demographics; best model: **XGBoost** (~0.81 ROC-AUC).  
    - Regression (age) is noisy; **Random Forest** topped the list with low R² (~0.07).  
    - Clustering performed best with **KMeans** after **UMAP**, silhouette around **0.54**.  
    - Interpret predictions **contextually**; combine with policy improvements & culture work.  
    """)

    footer()

# ===========================
# EDA
# ===========================
elif menu == "📊 Exploratory Data Analysis":
    st.header("📊 Exploratory Data Analysis")
    st.markdown(
        "Quick tour of dataset distributions, relationships, and multi-feature patterns I explored."
    )

    st.markdown("#### 🔹 Dataset Overview")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Records", "1,259")
    with c2:
        st.metric("Features (core)", "27")
    with c3:
        st.metric("Countries", "60+")
    st.caption("*Info from raw data.. not processed data")

    # Cleaning details
    with st.expander("🧹 Data Cleaning"):
        st.markdown("""
        - Removed unrealistic or noisy ages; standardized gender labels  
        - Removed useless features from the dataset 
        - Imputed missing workplace policy fields with mode  
        - Removed Non-Techies  
        - Engineered a new feature **support_score** for better training
        - One-Hot Encoding with `handle_unknown='ignore'`
        """)

    st.markdown("#### 🔹 Univariate Distributions")
    st.image("Images/univariate1.png",  caption="Gender & Age distributions", use_container_width=True)
    show_image_grid([
        (IMG_DIR / "univariate2.png", "Country & Company size"),
        (IMG_DIR / "univariate3.png", "Workplace support items"),
    ], cols=3)

    st.markdown("#### 🔹 Bivariate Patterns")
    show_image_if_exists(IMG_DIR / "bivariate.png", "Treatment vs key features")
   
    st.markdown("#### 🔹 Multivariate Insights")
    show_image_grid([
        (IMG_DIR / "multivariate1.png", "Feature interplay & stigma indicators"),
        (IMG_DIR / "multivariate2.png", "Correlation matrix (all features)"),
    ], cols=1)
    st.markdown(
        "Stigma signals (e.g., perceived consequences) are strongly tied to treatment behavior."
        "These patterns of the correlation matrix (down) also explains why **age regression** is weak: features don't linearly capture age."
    )

    footer()

# ===========================
# CLASSIFICATION
# ===========================
elif menu == "🧮 Prediction using Classification":
    st.header("🧮 Prediction using Classification")
    st.markdown(
        "Enter respondent details to predict the likelihood of seeking mental health treatment. "
        "Your model was trained with a cleaned schema; we’ll transform inputs to match it."
    )

    # --- Collect inputs (original survey style) ---
    user = {}

    # Age with validation
    age_val = st.number_input("Age", min_value=0, max_value=120, step=1, value=25, key="cls_age")
    if age_val < 18 or age_val > 75:
        st.error("⚠️ Invalid Age. Please enter a valid age (18-75).")
    else:
        user["Age"] = age_val

    user["Gender"] = st.selectbox("Gender", ["Male", "Female", "Other"], key="cls_gender")

    countries = ["United States", "United Kingdom", "Canada", "Germany", "Ireland", "India", "Other"]
    user["Country"] = st.selectbox("Country", countries, key="cls_country")

    user["self_employed"] = st.selectbox("Are you self-employed?", ["Yes", "No"], key="cls_selfemp")
    user["family_history"] = st.selectbox("Family history of mental illness?", ["Yes", "No"], key="cls_famhx")

    user["work_interfere"] = st.selectbox(
        "If you have a mental health condition, does it interfere with work?",
        ["Never", "Rarely", "Sometimes", "Often"], key="cls_interfere"
    )

    user["no_employees"] = st.selectbox(
        "Company size (number of employees)",
        ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"], key="cls_size"
    )

    user["remote_work"] = st.selectbox("Do you work remotely (≥50% of the time)?", ["Yes", "No"], key="cls_remote")

    # Support-related original answers (we still collect all 8)
    st.markdown("##### Workplace support & policies")
    cols1 = st.columns(4)
    with cols1[0]:
        user["benefits"] = st.selectbox("Mental health benefits?", ["Yes", "No", "Don't know"], key="cls_benefits")
    with cols1[1]:
        user["care_options"] = st.selectbox("Know employer care options?", ["Yes", "No", "Not sure"], key="cls_care")
    with cols1[2]:
        user["wellness_program"] = st.selectbox("Wellness program discussed?", ["Yes", "No", "Don't know"], key="cls_wellness")
    with cols1[3]:
        user["seek_help"] = st.selectbox("Resources to seek help?", ["Yes", "No", "Don't know"], key="cls_seek")

    cols2 = st.columns(4)
    with cols2[0]:
        user["anonymity"] = st.selectbox("Is anonymity protected?", ["Yes", "No", "Don't know"], key="cls_anon")
    with cols2[1]:
        user["leave"] = st.selectbox("Ease of medical leave", ["Difficult", "Medium", "Easy"], key="cls_leave")
    with cols2[2]:
        user["coworkers"] = st.selectbox("Discuss with coworkers?", ["Yes", "No", "Some of them"], key="cls_cow")
    with cols2[3]:
        user["supervisor"] = st.selectbox("Discuss with supervisor(s)?", ["Yes", "No", "Some of them"], key="cls_sup")

    user["mental_health_consequence"] = st.selectbox("Negative consequence if discussing MH?", ["Yes", "No", "Maybe"], key="cls_mhc")
    user["phys_health_consequence"] = st.selectbox("Negative consequence if discussing PH?", ["Yes", "No", "Maybe"], key="cls_phc")
    user["mental_health_interview"] = st.selectbox("Discuss MH in interview?", ["Yes", "No", "Maybe"], key="cls_mhi")
    user["phys_health_interview"] = st.selectbox("Discuss PH in interview?", ["Yes", "No", "Maybe"], key="cls_phi")
    user["mental_vs_physical"] = st.selectbox("MH as important as PH?", ["Yes", "No", "Don't know"], key="cls_mvsp")
    user["obs_consequence"] = st.selectbox("Observed negative consequences?", ["Yes", "No"], key="cls_obs")

    # --- Compute support_score (mean) but keep originals in `user`.
    support_score = compute_support_score_mean_from_raw(user)
    user["support_score"] = support_score  # harmless if model doesn't need it; builder will drop extras

    st.caption("We compute an internal support score (mean of mapped answers) to align with the training logic. "
               "Your model will only receive the features it expects.")

    # --- Predict ---
    left, right = st.columns([1, 2])
    with left:
        go = st.button("Predict Treatment", key="btn_pred_treatment")

    if go:
        with st.spinner("Scoring..."):
            pred, proba = predict_classification(clf_model, clf_cols, user)

        st.markdown("### Result")
        colA, colB = st.columns([1, 3])

        label = "Seek Treatment" if pred == 1 else "Not Seek Treatment"
        with colA:
            if pred == 1:
                st.success(f"✅ Prediction: **{label}**")
            else:
                st.error(f"❌ Prediction: **{label}**")

            # Confidence bar (if binary proba known)
            try:
                p_seek = float(proba[1])
                st.progress(int(round(p_seek * 100)))
                st.caption(f"Confidence (seek): {p_seek:.2f}")
            except Exception:
                pass

        with colB:
            with st.expander("📎 Model details & notes"):
                st.markdown("""
                - Trained on cleaned survey data with categorical encoding  
                - Includes workplace policy perceptions and demographics  
                - **XGBoost** performed best on validation (ROC-AUC around ~0.81)  
                - Predictions are probabilistic; interpret alongside HR context  
                """)
    
    st.divider()
    st.markdown("#### 🤖 Models & Results")
    st.markdown("""
    | Model                     | Accuracy | F1 Score | ROC-AUC |
    |---------------------------|----------|----------|---------|
    | XGBoost                   | 0.727    | 0.708    | 0.811   |
    | Support Vector Classifier | 0.712    | 0.697    | 0.804   |
    | Logistic Regression       | 0.707    | 0.697    | 0.794   |
    | Random Forest             | 0.727    | 0.711    | 0.790   |

    🏆 **Best Model**: XGBoost Classifier """)
    st.success("🏆 XGBoost outperforms Logistic Regression, SVM and Random Forest for treatment prediction.")
    st.divider()

    st.markdown("#### 📈 Comparison Graph")
    show_image_if_exists(IMG_DIR / "ROC_Curve.png", "ROC-AUC comparison (your experiment)")

    footer()


# ===========================
# REGRESSION
# ===========================
elif menu == "📈 Prediction using Regression":
    st.header("📈 Prediction using Regression")
    st.markdown(
        "Estimate a respondent’s age from workplace and personal attributes. "
        "Note: age is weakly captured by these features, so treat results as indicative only."
    )

    # Collect regression inputs (exact columns you provided earlier for regression)
    reg_user = {}
    # All categorical; 'Age' is *not* in regression inputs (target)
    reg_user["Gender"] = st.selectbox("Gender", ["Male", "Female", "Other"], key="reg_gender")

    top_countries = ["United States", "United Kingdom", "Canada", "Germany", "Ireland", "India", "Other"]
    reg_user["Country"] = st.selectbox("Country", top_countries, key="reg_country")

    reg_user["self_employed"] = st.selectbox("Are you self-employed?", ["Yes", "No"], key="reg_self")
    reg_user["family_history"] = st.selectbox("Family history of mental illness?", ["Yes", "No"], key="reg_famhx")
    reg_user["treatment"] = st.selectbox("Have you sought treatment?", ["Yes", "No"], key="reg_treat")

    reg_user["work_interfere"] = st.selectbox(
        "If you have a mental health condition, does it interfere with work?",
        ["Never", "Rarely", "Sometimes", "Often"], key="reg_interfere"
    )
    reg_user["no_employees"] = st.selectbox(
        "Company size (number of employees)",
        ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"], key="reg_size"
    )
    reg_user["remote_work"] = st.selectbox("Do you work remotely (≥50% of the time)?", ["Yes", "No"], key="reg_remote")

    colsR1 = st.columns(4)
    with colsR1[0]:
        reg_user["benefits"] = st.selectbox("Mental health benefits?", ["Yes", "No", "Don't know"], key="reg_benefits")
    with colsR1[1]:
        reg_user["care_options"] = st.selectbox("Know employer care options?", ["Yes", "No", "Not sure"], key="reg_care")
    with colsR1[2]:
        reg_user["wellness_program"] = st.selectbox("Wellness program discussed?", ["Yes", "No", "Don't know"], key="reg_wellness")
    with colsR1[3]:
        reg_user["seek_help"] = st.selectbox("Resources to seek help?", ["Yes", "No", "Don't know"], key="reg_seek")

    colsR2 = st.columns(4)
    with colsR2[0]:
        reg_user["anonymity"] = st.selectbox("Is anonymity protected?", ["Yes", "No", "Don't know"], key="reg_anon")
    with colsR2[1]:
        reg_user["leave"] = st.selectbox("Ease of medical leave", ["Difficult", "Medium", "Easy"], key="reg_leave")
    with colsR2[2]:
        reg_user["mental_health_consequence"] = st.selectbox("Neg. consequence discussing MH?", ["Yes", "No", "Maybe"], key="reg_mhc")
    with colsR2[3]:
        reg_user["phys_health_consequence"] = st.selectbox("Neg. consequence discussing PH?", ["Yes", "No", "Maybe"], key="reg_phc")

    colsR3 = st.columns(4)
    with colsR3[0]:
        reg_user["coworkers"] = st.selectbox("Discuss with coworkers?", ["Yes", "No", "Some of them"], key="reg_cow")
    with colsR3[1]:
        reg_user["supervisor"] = st.selectbox("Discuss with supervisor(s)?", ["Yes", "No", "Some of them"], key="reg_sup")
    with colsR3[2]:
        reg_user["mental_health_interview"] = st.selectbox("Discuss MH in interview?", ["Yes", "No", "Maybe"], key="reg_mhi")
    with colsR3[3]:
        reg_user["phys_health_interview"] = st.selectbox("Discuss PH in interview?", ["Yes", "No", "Maybe"], key="reg_phi")

    reg_user["mental_vs_physical"] = st.selectbox("MH as important as PH?", ["Yes", "No", "Don't know"], key="reg_mvsp")
    reg_user["obs_consequence"] = st.selectbox("Observed negative consequences?", ["Yes", "No"], key="reg_obs")

    # Predict
    go_reg = st.button("Predict Age", key="btn_pred_age")
    if go_reg:
        with st.spinner("Scoring..."):
            yhat = predict_regression(reg_model, reg_cols, reg_user)

        st.markdown("### Result")
        cA, cB = st.columns([1, 2])
        with cA:
            st.success(f"🎯 Predicted Age: **{yhat:.1f} years**")
        with cB:
            with st.expander("📎 Model details & notes"):
                st.markdown("""
                - Multiple regressors tested; **Random Forest** performed best on validation  
                - R² was low (~0.07), implying weak predictability of age from given features  
                - Treat predictions as hints, not exact ages  
                """)

    st.divider()
    st.markdown("#### 🤖 Models & Results")
    st.markdown("""
    | Model                          | R² Score |   RMSE   |   MAE    |
    |--------------------------------|----------|----------|----------|
    | Random Forest Regressor        | 0.069362 | 6.980238 | 5.515165 |
    | Random Forest (log-transformed)| 0.054473 | 7.035854 | 5.450534 |
    | XGBoost (log-transformed)      | 0.051649 | 7.046350 | 5.473678 |
    | XGBRegressor                   | 0.050332 | 7.051243 | 5.560343 |
    | Gradient Boosting Regressor    | 0.049264 | 7.055208 | 5.587464 |
    | Lasso Regression               | 0.025909 | 7.141338 | 5.655086 |
    | Ridge Regression               | 0.015816 | 7.178238 | 5.690324 |
    | Support Vector Regressor       | -0.006819| 7.260316 | 5.683155 |
    | Linear Regression              | -0.036312| 7.365888 | 5.890834 |
    
    🏆 **Best Model**: Random forest Regressor (by R² score)""")

    st.success("🏆 Random Forest gives the best R² score for age prediction, though all models have low explanatory power.")

    footer()


# ===========================
# CLUSTERING
# ===========================
elif menu == "📌 Clustering Personas":
    st.header("📌 Clustering Personas (KMeans + UMAP)")
    st.markdown(
        "Unsupervised clustering segments employees into groups with different support needs and engagement patterns. "
        "These personas can inform targeted HR programs and communications."
    )

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Best Algorithm", "KMeans")
    with m2:
        st.metric("Silhouette Score", "≈0.54")

    # Clustering techniques
    st.subheader("Techniques Used: ")
    st.write(" - Principal Component Analysis (PCA)\n - t-distributed Stochastic Neighbor Embedding (t-SNE)\n - Uniform Manifold Approximation and Projection (UMAP)")
    st.write("Here is the plot for all three techniques applied on this dataset.")
    st.image("Images/reduce.png", caption="From these clusters we can see that `UMAP` forms the best and most clear and seggregated clusters out of the three.", use_container_width=True)
    
    st.write("The most optimal number of clusters were found to be 7, ranked by silhouette score for each cluster number.")
    st.image("Images/k_values.png", use_container_width=False)

    st.markdown("### Here is the comparison (silhouette score) for all the models applied for clustering: ")
    st.write(" - **K-Means Clustering:** 0.5407\n - **Agglomerative Clustering:** 0.5291\n - **DBSCAN:** 0.4824 (6 DBSCAN Clusters, 3 Noise Points)")

    st.markdown("### ✅ From these scores, we can easily say that `K-Means` is clearly our winner.")
    st.image("Images/algos.png", caption="Clusters formed by the models", use_container_width=True)

    st.image("Images/comparison.png", caption="Feature contrasts across clusters", use_container_width=True)
    st.divider()

    st.markdown("#### 🔹 Different Personas the respondents can be classified into ⬇️")
    tabs = st.tabs([
        "Balanced Support Seekers", "Cautious Engagers", "Policy-Dependent Advocates",
        "High-Risk Silent Sufferers", "Under-Supported Reluctants",
        "Stigma-Affected Minimalists", "Empowered Open Advocates"
    ])

    with tabs[0]:
        st.markdown("""
        - Moderately positive across support indicators  
        - Engage with programs but don’t necessarily lead them  
        - Respond well to consistent, low-friction resources
        """)

    with tabs[1]:
        st.markdown("""
        - Low openness initially; engage only when **trust** is established  
        - Prefer private, confidential channels to seek help  
        - Manager training & anonymous options increase participation
        """)

    with tabs[2]:
        st.markdown("""
        - Engagement tightly tied to clarity of **benefits** and **leave policies**  
        - Communicate entitlements simply; reduce approval friction  
        - Policy transparency drives usage
        """)

    with tabs[3]:
        st.markdown("""
        - High treatment needs but low support score  
        - Avoid disclosure due to stigma; risk of quiet attrition  
        - Proactive outreach & leadership openness matter most
        """)

    with tabs[4]:
        st.markdown("""
        - Encounter negative consequences without adequate resources  
        - Need immediate access & advocacy; buddy programs help  
        - Monitor for burnout signals and follow-ups
        """)

    with tabs[5]:
        st.markdown("""
        - Aware of wellness issues but engage minimally due to stigma  
        - Normalize MH through stories & frequent internal comms  
        - Short, low-effort interventions work better
        """)

    with tabs[6]:
        st.markdown("""
        - High openness & advocacy for mental wellness  
        - Can act as champions/mentors; amplify programs  
        - Involve them in campaigns and peer support
        """)

    with st.expander("📎 How to use personas"):
        st.markdown("""
        - Tailor benefits messaging based on cluster style  
        - Train managers to **reduce stigma** and handle conversations  
        - Run small A/B experiments on communications & measure uplift  
        """)


    footer()
