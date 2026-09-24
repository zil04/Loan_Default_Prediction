import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import textwrap
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# --- Base Directory for Deployment Safety ---
BASE_DIR = Path(__file__).resolve().parent

# --- Page Configuration ---
st.set_page_config(
    page_title="Loan Default Prediction | ML Risk Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Dark Theme CSS Injection ---
def apply_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global Dark Theme Settings */
    :root {
        --bg-main: #0b0f19;
        --bg-surface: #111827;
        --bg-card: #141e33;
        --bg-card-hover: #1a2744;
        --border-subtle: rgba(255, 255, 255, 0.08);
        --border-accent: rgba(59, 130, 246, 0.35);
        --primary-blue: #3b82f6;
        --primary-hover: #2563eb;
        --primary-glow: rgba(59, 130, 246, 0.25);
        --accent-emerald: #10b981;
        --accent-rose: #f43f5e;
        --accent-amber: #f59e0b;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
    }

    /* Core Streamlit Layout Overrides */
    .stApp {
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    [data-testid="stSidebar"] {
        display: none !important;
    }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 4rem !important;
        max-width: 1300px !important;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        font-family: 'Inter', sans-serif !important;
        color: var(--text-primary);
    }

    /* Custom Navbar Component */
    .nav-wrapper {
        background: rgba(17, 24, 39, 0.85);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 0.85rem 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }

    .nav-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .brand-icon {
        width: 38px;
        height: 38px;
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        box-shadow: 0 4px 14px var(--primary-glow);
    }

    .brand-title {
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #ffffff;
        margin: 0;
        line-height: 1.2;
    }

    .brand-subtitle {
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--primary-blue);
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* Glassmorphism Cards */
    .dashboard-card {
        background-color: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 1.75rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .dashboard-card:hover {
        border-color: var(--border-accent);
    }

    .card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1.25rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--border-subtle);
    }

    /* Hero Section Styles */
    .hero-container {
        padding: 3rem 1rem 2rem 1rem;
        text-align: left;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0.9rem;
        background: rgba(59, 130, 246, 0.12);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 9999px;
        color: #60a5fa;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 3.25rem !important;
        font-weight: 800 !important;
        line-height: 1.15 !important;
        letter-spacing: -0.03em !important;
        color: #ffffff !important;
        margin-bottom: 1.25rem !important;
    }

    .hero-highlight {
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: 1.15rem !important;
        color: var(--text-secondary) !important;
        line-height: 1.7 !important;
        max-width: 680px;
        margin-bottom: 2rem !important;
    }

    /* Stat Badges */
    .stat-box {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }

    .stat-number {
        font-size: 1.85rem;
        font-weight: 800;
        color: var(--primary-blue);
        line-height: 1;
        margin-bottom: 0.35rem;
    }

    .stat-label {
        font-size: 0.8rem;
        font-weight: 500;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Step Section Cards */
    .step-card {
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 1.5rem;
        height: 100%;
        position: relative;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .step-card:hover {
        transform: translateY(-4px);
        border-color: var(--border-accent);
    }

    .step-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    .step-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }

    .step-desc {
        font-size: 0.88rem;
        color: var(--text-secondary);
        line-height: 1.6;
        margin: 0;
    }

    /* Tech Badges */
    .tech-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.5rem 1rem;
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        font-size: 0.88rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0.35rem;
    }

    /* Result Card Styles */
    .result-container-safe {
        background: linear-gradient(145deg, rgba(16, 185, 129, 0.08) 0%, rgba(17, 24, 39, 0.9) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.1);
        margin-top: 1.5rem;
    }

    .result-container-danger {
        background: linear-gradient(145deg, rgba(244, 63, 94, 0.08) 0%, rgba(17, 24, 39, 0.9) 100%);
        border: 1px solid rgba(244, 63, 94, 0.4);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(244, 63, 94, 0.1);
        margin-top: 1.5rem;
    }

    .result-badge-safe {
        display: inline-block;
        padding: 0.35rem 1rem;
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 0.75rem;
    }

    .result-badge-danger {
        display: inline-block;
        padding: 0.35rem 1rem;
        background: rgba(244, 63, 94, 0.2);
        color: #fb7185;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 0.75rem;
    }

    /* Streamlit Native Inputs Theming */
    div[data-baseweb="select"] > div {
        background-color: var(--bg-card) !important;
        border-color: var(--border-subtle) !important;
        color: var(--text-primary) !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="select"] * {
        color: var(--text-primary) !important;
    }

    div[data-baseweb="input"] {
        background-color: var(--bg-card) !important;
        border-color: var(--border-subtle) !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="input"] input {
        color: var(--text-primary) !important;
        background-color: transparent !important;
    }

    .stNumberInput, .stSelectbox, .stSlider {
        margin-bottom: 0.85rem;
    }

    /* Streamlit Buttons */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        padding: 0.6rem 1.4rem !important;
        transition: all 0.25s ease !important;
        border: 1px solid var(--border-subtle) !important;
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
    }

    .stButton > button:hover {
        border-color: var(--primary-blue) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 14px var(--primary-glow) !important;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 4px 18px var(--primary-glow) !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 24px rgba(59, 130, 246, 0.4) !important;
    }

    /* Disclaimer / Alert Boxes */
    .disclaimer-box {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid var(--border-subtle);
        border-left: 3px solid var(--primary-blue);
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-top: 2rem;
        font-size: 0.85rem;
        color: var(--text-secondary);
        line-height: 1.6;
    }

    /* Custom Table Styling */
    .metrics-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        margin: 1.5rem 0;
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        overflow: hidden;
    }

    .metrics-table th {
        background-color: var(--bg-card);
        color: #ffffff;
        font-weight: 700;
        font-size: 0.88rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        padding: 1rem 1.25rem;
        border-bottom: 1px solid var(--border-subtle);
        text-align: left;
    }

    .metrics-table td {
        padding: 0.95rem 1.25rem;
        background-color: var(--bg-surface);
        border-bottom: 1px solid var(--border-subtle);
        color: var(--text-primary);
        font-size: 0.92rem;
    }

    .metrics-table tr:last-child td {
        border-bottom: none;
    }

    .metrics-table tr:hover td {
        background-color: var(--bg-card);
    }

    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    .badge-best {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)


# --- Cached Resource Loaders ---
@st.cache_resource
def load_all_models():
    """Loads all trained machine learning pipelines using relative deployment-safe paths."""
    models = {}
    model_configs = [
        ("Logistic Regression", "model_logistic_regression.pkl"),
        ("Decision Tree", "model_decision_tree.pkl"),
        ("KNN", "model_knn.pkl"),
        ("Random Forest", "model_random_forest.pkl")
    ]
    for display_name, file_name in model_configs:
        file_path = BASE_DIR / file_name
        if file_path.exists():
            try:
                models[display_name] = joblib.load(file_path)
            except Exception as e:
                st.warning(f"Unable to load {display_name}: {e}")
                
    # Fallback to model.pkl if available and nothing else loaded
    if not models:
        fallback_path = BASE_DIR / "model.pkl"
        if fallback_path.exists():
            try:
                models["Primary Model"] = joblib.load(fallback_path)
            except Exception:
                pass
    return models


@st.cache_data
def load_evaluation_metrics():
    """Loads precomputed evaluation metrics and ROC curves from JSON."""
    metrics_path = BASE_DIR / "evaluation_metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


# --- Navigation Component ---
def render_navbar():
    """Renders a static, professional dark navigation bar that persists on every page."""
    nav_items = ["Home", "Prediction", "Model Summary", "Analytics / Charts", "About"]
    
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"

    st.markdown('<div class="nav-wrapper">', unsafe_allow_html=True)
    col_brand, col_nav = st.columns([1.8, 3.2])

    with col_brand:
        st.markdown("""
        <div class="nav-brand">
            <div class="brand-icon">🏦</div>
            <div>
                <div class="brand-title">LoanRisk AI</div>
                <div class="brand-subtitle">Credit Risk Assessment System</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_nav:
        btn_cols = st.columns(len(nav_items))
        for idx, item in enumerate(nav_items):
            with btn_cols[idx]:
                is_active = (st.session_state.current_page == item)
                btn_type = "primary" if is_active else "secondary"
                if st.button(item, key=f"nav_btn_{item}", type=btn_type, use_container_width=True):
                    if st.session_state.current_page != item:
                        st.session_state.current_page = item
                        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# --- Helper: Plotly Dark Layout Formatter ---
def apply_dark_plotly_layout(fig, title="", height=420):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Inter", size=16, color="#f8fafc")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17, 24, 39, 0.7)",
        font=dict(family="Inter", color="#94a3b8"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=height,
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.06)",
            zerolinecolor="rgba(255, 255, 255, 0.12)",
            tickfont=dict(color="#94a3b8")
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.06)",
            zerolinecolor="rgba(255, 255, 255, 0.12)",
            tickfont=dict(color="#94a3b8")
        ),
        legend=dict(
            font=dict(color="#f8fafc"),
            bgcolor="rgba(17, 24, 39, 0.6)",
            bordercolor="rgba(255, 255, 255, 0.08)",
            borderwidth=1
        )
    )
    return fig


# ==========================================
# PAGE 1: HOME / LANDING PAGE
# ==========================================
def render_home():
    """Renders the comprehensive, professional Home / Landing Page."""
    # Hero Section
    h_col1, h_col2 = st.columns([1.3, 0.9])
    
    with h_col1:
        st.markdown('<div class="hero-badge">⚡ Enterprise Machine Learning Pipeline</div>', unsafe_allow_html=True)
        st.markdown("""
        <h1 class="hero-title">
            Loan Default <br><span class="hero-highlight">Prediction Platform</span>
        </h1>
        """, unsafe_allow_html=True)
        st.markdown("""
        <p class="hero-subtitle">
            This application uses state-of-the-art machine learning algorithms to estimate whether a loan applicant 
            is likely to default based on their financial history, credit metrics, and loan terms.
        </p>
        """, unsafe_allow_html=True)
        
        c_btn, _ = st.columns([1.1, 1.9])
        with c_btn:
            if st.button("Start Prediction ➔", type="primary", use_container_width=True, key="home_cta_btn"):
                st.session_state.current_page = "Prediction"
                st.rerun()

    with h_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        s1, s2 = st.columns(2)
        with s1:
            st.markdown("""
            <div class="stat-box">
                <div class="stat-number">4</div>
                <div class="stat-label">ML Algorithms</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div class="stat-box">
                <div class="stat-number">16</div>
                <div class="stat-label">Risk Features</div>
            </div>
            """, unsafe_allow_html=True)
        with s2:
            st.markdown("""
            <div class="stat-box">
                <div class="stat-number">88.6%</div>
                <div class="stat-label">Top Test Accuracy</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            st.markdown("""
            <div class="stat-box">
                <div class="stat-number">Real-Time</div>
                <div class="stat-label">Inference Engine</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height: 35px;'></div>", unsafe_allow_html=True)

    # Project Overview Section
    st.markdown("""
    <div class="dashboard-card">
        <div class="card-title">📖 Project Overview</div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem;">
            <div>
                <h4 style="font-size: 1.02rem; font-weight: 700; color: #60a5fa; margin-bottom: 0.4rem;">What is Loan Default Prediction?</h4>
                <p style="font-size: 0.9rem; color: #94a3b8; line-height: 1.6;">
                    A loan default takes place when a borrower fails to meet the legal repayment obligations of a credit agreement. 
                    Default prediction applies predictive classification models to quantify the probability of default before capital is disbursed.
                </p>
            </div>
            <div>
                <h4 style="font-size: 1.02rem; font-weight: 700; color: #60a5fa; margin-bottom: 0.4rem;">Why is it Critical for Lenders?</h4>
                <p style="font-size: 0.9rem; color: #94a3b8; line-height: 1.6;">
                    Accurate risk scoring protects credit institutions from bad debt write-offs, preserves liquidity, accelerates automated 
                    underwriting times, and enables fair, risk-adjusted interest rates for creditworthy applicants.
                </p>
            </div>
            <div>
                <h4 style="font-size: 1.02rem; font-weight: 700; color: #60a5fa; margin-bottom: 0.4rem;">What Information is Utilized?</h4>
                <p style="font-size: 0.9rem; color: #94a3b8; line-height: 1.6;">
                    The system evaluates applicant demographics (age, education, employment), credit history (credit score, credit lines, DTI ratio), 
                    and specific loan parameters (amount, term, interest rate, purpose, and collateral guarantees).
                </p>
            </div>
            <div>
                <h4 style="font-size: 1.02rem; font-weight: 700; color: #60a5fa; margin-bottom: 0.4rem;">How Machine Learning Helps</h4>
                <p style="font-size: 0.9rem; color: #94a3b8; line-height: 1.6;">
                    Unlike rigid legacy rule engines, machine learning captures subtle non-linear interactions across multivariate variables, 
                    offering calibrated probabilities and objective decision boundaries.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Step-by-Step "How Prediction Works" Section
    st.markdown("<h3 style='font-size: 1.4rem; font-weight: 700; margin: 2rem 0 1rem 0;'>⚙️ How the Prediction Process Works</h3>", unsafe_allow_html=True)
    
    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns(5)
    
    steps = [
        ("1", "Applicant Input", "The user inputs applicant demographics, income, credit score, and requested loan parameters."),
        ("2", "Data Preprocessing", "Numerical features are standardized using StandardScaler; categorical variables are encoded via OneHotEncoder."),
        ("3", "ML Inference", "The chosen pre-trained classification model processes the structured feature vector."),
        ("4", "Risk Outcome", "The system displays the binary classification (Default vs. No Default) with associated probability."),
        ("5", "Model Insights", "Feature importances and comparative evaluation analytics assist in interpreting the decision.")
    ]
    
    step_cols = [col_s1, col_s2, col_s3, col_s4, col_s5]
    for i, (num, title, desc) in enumerate(steps):
        with step_cols[i]:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-number">0{num}</div>
                <div class="step-title">{title}</div>
                <p class="step-desc">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height: 35px;'></div>", unsafe_allow_html=True)

    # Technology Section
    st.markdown("""
    <div class="dashboard-card">
        <div class="card-title">💻 Technologies Used in This Project</div>
        <p style="font-size: 0.92rem; color: #94a3b8; margin-bottom: 1.25rem;">
            This project utilizes a modern Python data science stack with automated preprocessing pipelines and interactive visual components:
        </p>
        <div>
            <span class="tech-pill">🐍 Python 3.11+</span>
            <span class="tech-pill">⚡ Streamlit</span>
            <span class="tech-pill">📊 Pandas</span>
            <span class="tech-pill">🔢 NumPy</span>
            <span class="tech-pill">🤖 Scikit-Learn</span>
            <span class="tech-pill">💾 Joblib</span>
            <span class="tech-pill">📈 Plotly Interactive Charts</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# PAGE 2: PREDICTION PAGE
# ==========================================
def render_prediction():
    """Renders the dedicated interactive applicant profiling and prediction interface."""
    st.markdown("<h2 style='font-size: 1.8rem; font-weight: 800; margin-bottom: 0.5rem;'>🔮 Applicant Risk Profiling & Prediction</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 0.95rem; margin-bottom: 1.5rem;'>Enter applicant details and loan specifications below to calculate real-time default risk.</p>", unsafe_allow_html=True)

    models = load_all_models()
    if not models:
        st.error("⚠️ No trained models detected. Ensure model pickle files exist in the project directory.")
        return

    # Top Configuration: Model Selector
    c_m1, c_m2 = st.columns([1.5, 2.5])
    with c_m1:
        selected_model_name = st.selectbox(
            "Select Prediction Algorithm",
            options=list(models.keys()),
            index=list(models.keys()).index("Random Forest") if "Random Forest" in models else 0
        )
    with c_m2:
        st.markdown(f"""
        <div style="background: rgba(17, 24, 39, 0.6); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 0.85rem 1.25rem; margin-top: 1.6rem; font-size: 0.88rem; color: #94a3b8;">
            Active Classifier: <strong style="color: #60a5fa;">{selected_model_name}</strong> 
            &nbsp;|&nbsp; Status: <span style="color: #34d399; font-weight: 600;">Ready for Inference</span>
        </div>
        """, unsafe_allow_html=True)

    selected_model = models[selected_model_name]
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Form Cards
    col_left, col_right = st.columns(2)

    with col_left:
        # Card 1: Applicant Demographics & Employment
        st.markdown("""
        <div class="dashboard-card">
            <div class="card-title">👤 Demographics & Employment</div>
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Applicant Age (Years)", min_value=18, max_value=100, value=35, help="Age of borrower")
            education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"], index=1)
            employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"], index=0)
        with c2:
            months_employed = st.number_input("Months Employed", min_value=0, max_value=600, value=36, step=1)
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], index=1)
            has_dependents = st.selectbox("Has Dependents?", ["Yes", "No"], index=1)

        # Card 2: Collateral & Guarantees
        st.markdown("""
        <div class="dashboard-card" style="margin-top: 1rem;">
            <div class="card-title">🛡️ Collateral & Guarantees</div>
        </div>
        """, unsafe_allow_html=True)
        
        g1, g2 = st.columns(2)
        with g1:
            has_mortgage = st.selectbox("Has Existing Mortgage?", ["Yes", "No"], index=1)
        with g2:
            has_cosigner = st.selectbox("Has Approved Co-Signer?", ["Yes", "No"], index=1)

    with col_right:
        # Card 3: Financial & Credit Profile
        st.markdown("""
        <div class="dashboard-card">
            <div class="card-title">💳 Financial & Credit Profile</div>
        </div>
        """, unsafe_allow_html=True)
        
        f1, f2 = st.columns(2)
        with f1:
            income = st.number_input("Annual Income ($)", min_value=1000, max_value=10000000, value=65000, step=2500)
            credit_score = st.slider("Credit Score (FICO)", min_value=300, max_value=850, value=710, help="Standard credit rating")
            num_credit_lines = st.number_input("Open Credit Lines", min_value=0, max_value=50, value=3, step=1)
        with f2:
            dti_ratio = st.slider("Debt-to-Income (DTI) Ratio", min_value=0.00, max_value=1.00, value=0.28, step=0.01, help="Total monthly debt payments divided by gross monthly income")
            loan_amount = st.number_input("Requested Loan Amount ($)", min_value=500, max_value=5000000, value=25000, step=1000)
            loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60], index=2)

        # Card 4: Loan Specifics
        st.markdown("""
        <div class="dashboard-card" style="margin-top: 1rem;">
            <div class="card-title">📝 Loan Specifications</div>
        </div>
        """, unsafe_allow_html=True)
        
        l1, l2 = st.columns(2)
        with l1:
            interest_rate = st.slider("Interest Rate (%)", min_value=0.5, max_value=35.0, value=6.5, step=0.1)
        with l2:
            loan_purpose = st.selectbox("Loan Purpose", ["Auto", "Business", "Education", "Home", "Other"], index=3)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Predict Button
    btn_predict = st.button("Predict Loan Default Risk ➔", type="primary", use_container_width=True)

    if btn_predict:
        # Construct feature DataFrame with exact column names expected by pipeline
        input_data = pd.DataFrame({
            'age': [int(age)],
            'income': [float(income)],
            'loanamount': [float(loan_amount)],
            'creditscore': [int(credit_score)],
            'monthsemployed': [int(months_employed)],
            'numcreditlines': [int(num_credit_lines)],
            'interestrate': [float(interest_rate)],
            'loanterm': [int(loan_term)],
            'dtiratio': [float(dti_ratio)],
            'education': [str(education)],
            'employmenttype': [str(employment_type)],
            'maritalstatus': [str(marital_status)],
            'hasmortgage': [str(has_mortgage)],
            'hasdependents': [str(has_dependents)],
            'loanpurpose': [str(loan_purpose)],
            'hascosigner': [str(has_cosigner)]
        })

        with st.spinner(f"Running inference with {selected_model_name}..."):
            try:
                prediction = selected_model.predict(input_data)[0]
                proba_available = hasattr(selected_model, "predict_proba")
                probabilities = selected_model.predict_proba(input_data)[0] if proba_available else None
                
                # Render Prominent Result Card
                if prediction == 1:
                    default_prob = probabilities[1] * 100 if proba_available else None
                    non_default_prob = probabilities[0] * 100 if proba_available else None
                    
                    st.markdown(f"""
                    <div class="result-container-danger">
                        <span class="result-badge-danger">⚠️ HIGH RISK DETECTED</span>
                        <h2 style="font-size: 2rem; font-weight: 800; color: #ffffff; margin: 0.5rem 0;">
                            Prediction: <span style="color: #fb7185;">LIKELY TO DEFAULT (Class 1)</span>
                        </h2>
                        <p style="color: #cbd5e1; font-size: 0.98rem; line-height: 1.6; max-width: 800px;">
                            The {selected_model_name} model classifies this applicant profile as having an elevated risk of loan default. 
                            The evaluated combination of credit score, debt-to-income ratio, and requested loan terms matches historical default patterns.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    default_prob = probabilities[1] * 100 if proba_available else None
                    non_default_prob = probabilities[0] * 100 if proba_available else None
                    
                    st.markdown(f"""
                    <div class="result-container-safe">
                        <span class="result-badge-safe">✅ LOW RISK CONFIRMED</span>
                        <h2 style="font-size: 2rem; font-weight: 800; color: #ffffff; margin: 0.5rem 0;">
                            Prediction: <span style="color: #34d399;">UNLIKELY TO DEFAULT (Class 0)</span>
                        </h2>
                        <p style="color: #cbd5e1; font-size: 0.98rem; line-height: 1.6; max-width: 800px;">
                            The {selected_model_name} model predicts that this loan applicant is unlikely to default. 
                            The applicant profile demonstrates financial stability indicators consistent with timely debt service.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # Probabilities & Confidence Metrics Visual
                if proba_available and probabilities is not None:
                    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
                    m1, m2, m3 = st.columns([1, 1, 1.5])
                    with m1:
                        st.markdown(f"""
                        <div class="stat-box" style="border-color: rgba(16, 185, 129, 0.3);">
                            <div class="stat-number" style="color: #34d399;">{non_default_prob:.1f}%</div>
                            <div class="stat-label">Repayment Confidence</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"""
                        <div class="stat-box" style="border-color: rgba(244, 63, 94, 0.3);">
                            <div class="stat-number" style="color: #fb7185;">{default_prob:.1f}%</div>
                            <div class="stat-label">Default Risk Probability</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with m3:
                        # Progress Bar Visualization
                        st.markdown("<div style='font-size: 0.85rem; color: #94a3b8; margin-bottom: 6px;'>Calibrated Probability Breakdown</div>", unsafe_allow_html=True)
                        st.progress(float(default_prob / 100.0), text=f"Default Risk: {default_prob:.1f}%")

            except Exception as ex:
                st.error(f"Prediction could not be completed: {str(ex)}")

    # Compliance Disclaimer
    st.markdown("""
    <div class="disclaimer-box">
        <strong>Factual Model Notice:</strong> This prediction is generated programmatically by a statistical classification model 
        trained on historical data. It is intended solely for machine learning evaluation and educational demonstration. It does not 
        constitute formal financial advice, credit granting, or underwriting commitments.
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# PAGE 3: MODEL SUMMARY PAGE
# ==========================================
def render_model_summary():
    """Renders the Model Summary page detailing all algorithms, metrics, and comparisons."""
    st.markdown("<h2 style='font-size: 1.8rem; font-weight: 800; margin-bottom: 0.5rem;'>📊 Machine Learning Model Summary</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 0.95rem; margin-bottom: 1.5rem;'>Comparative analysis and performance benchmarks for all trained classifiers evaluated on the test dataset.</p>", unsafe_allow_html=True)

    metrics = load_evaluation_metrics()
    if not metrics:
        st.warning("⚠️ Evaluation metrics file (`evaluation_metrics.json`) not found. Run `python train_model.py` to regenerate.")
        return

    # 1. KPI Metric Summary Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            """<div class="stat-box" style="border-color: rgba(16, 185, 129, 0.4);">
                <div class="stat-number" style="color: #34d399; font-size: 1.5rem;">Random Forest</div>
                <div class="stat-label">Top Overall Classifier</div>
            </div>""",
            unsafe_allow_html=True
        )
    with k2:
        st.markdown(
            """<div class="stat-box" style="border-color: rgba(59, 130, 246, 0.4);">
                <div class="stat-number" style="color: #60a5fa; font-size: 1.5rem;">88.58%</div>
                <div class="stat-label">Peak Test Accuracy</div>
            </div>""",
            unsafe_allow_html=True
        )
    with k3:
        st.markdown(
            """<div class="stat-box" style="border-color: rgba(245, 158, 11, 0.4);">
                <div class="stat-number" style="color: #fbbf24; font-size: 1.5rem;">72.22%</div>
                <div class="stat-label">Peak Precision Rate</div>
            </div>""",
            unsafe_allow_html=True
        )
    with k4:
        st.markdown(
            """<div class="stat-box" style="border-color: rgba(168, 85, 247, 0.4);">
                <div class="stat-number" style="color: #c084fc; font-size: 1.5rem;">0.7522</div>
                <div class="stat-label">Highest ROC-AUC</div>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # 2. Model Benchmark & Validation Matrix Table
    st.markdown(
        """<div class="dashboard-card" style="margin-bottom: 0.75rem;">
            <div class="card-title">🏆 Model Benchmark & Validation Matrix</div>
        </div>""",
        unsafe_allow_html=True
    )

    summary_rows = []
    for name, m in metrics.items():
        train_acc = f"{m.get('Train Accuracy', 0)*100:.2f}%" if m.get('Train Accuracy') is not None else "N/A"
        test_acc = f"{m.get('Test Accuracy', 0)*100:.2f}%" if m.get('Test Accuracy') is not None else "N/A"
        precision = f"{m.get('Precision', 0)*100:.2f}%" if m.get('Precision') is not None else "N/A"
        recall = f"{m.get('Recall', 0)*100:.2f}%" if m.get('Recall') is not None else "N/A"
        f1 = f"{m.get('F1 Score', 0)*100:.2f}%" if m.get('F1 Score') is not None else "N/A"
        auc = f"{m.get('ROC AUC', 0):.4f}" if m.get('ROC AUC') is not None else "N/A"
        
        status_tag = "🏆 Top Test Accuracy & Precision" if name == "Random Forest" else (
            "⚖️ Generalization Stability Leader" if name == "Logistic Regression" else (
                "📍 Optimal K=9 Neighborhood" if name == "KNN" else "🌲 Pruned Depth=10 Tree"
            )
        )
        
        summary_rows.append({
            "Algorithm": name,
            "Train Accuracy": train_acc,
            "Test Accuracy": test_acc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC-AUC": auc,
            "Validation Notes": status_tag
        })

    df_summary = pd.DataFrame(summary_rows)

    st.dataframe(
        df_summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Algorithm": st.column_config.TextColumn("Algorithm", width="medium"),
            "Train Accuracy": st.column_config.TextColumn("Train Accuracy"),
            "Test Accuracy": st.column_config.TextColumn("Test Accuracy (Holdout)"),
            "Precision": st.column_config.TextColumn("Precision"),
            "Recall": st.column_config.TextColumn("Recall"),
            "F1 Score": st.column_config.TextColumn("F1 Score"),
            "ROC-AUC": st.column_config.TextColumn("ROC-AUC"),
            "Validation Notes": st.column_config.TextColumn("Validation Notes", width="large")
        }
    )

    # 2. Overfitting & Underfitting Analysis
    st.markdown("<h3 style='font-size: 1.3rem; font-weight: 700; margin: 1.75rem 0 1rem 0;'>⚖️ Overfitting & Underfitting Diagnosis</h3>", unsafe_allow_html=True)
    
    c_diag1, c_diag2 = st.columns(2)
    
    with c_diag1:
        st.markdown(textwrap.dedent("""
        <div class="dashboard-card" style="height: 100%;">
            <div class="card-title">🔍 Bias-Variance Assessment</div>
            <ul style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.8; padding-left: 1.25rem;">
                <li><strong style="color: #60a5fa;">Random Forest:</strong> Perfect 100% training accuracy vs. 88.58% test accuracy indicates high capacity / variance on the training set, yet it attains the highest overall test accuracy and precision (72.22%).</li>
                <li><strong style="color: #60a5fa;">Logistic Regression:</strong> 88.50% train vs. 88.55% test demonstrates near-zero generalization gap, showing exceptional parameter stability on this financial distribution.</li>
                <li><strong style="color: #60a5fa;">Decision Tree:</strong> 90.36% train vs. 87.37% test. Pruned depth of 10 controls variance while maintaining interpretability.</li>
                <li><strong style="color: #60a5fa;">K-Nearest Neighbors:</strong> 88.90% train vs. 88.31% test with optimal k=9 neighbors.</li>
            </ul>
        </div>
        """).strip(), unsafe_allow_html=True)

    with c_diag2:
        st.markdown(textwrap.dedent("""
        <div class="dashboard-card" style="height: 100%;">
            <div class="card-title">📐 Hyperparameter Tuning & Cross-Validation</div>
            <div style="font-size: 0.9rem; color: #94a3b8; line-height: 1.7;">
                <p>All models were tuned using <strong>RandomizedSearchCV with 3-Fold Stratified Cross-Validation</strong> to optimize decision thresholds and ensure stability:</p>
                <div style="background: rgba(17, 24, 39, 0.7); border-radius: 8px; padding: 0.75rem 1rem; border: 1px solid rgba(255,255,255,0.06); font-family: monospace; font-size: 0.85rem; color: #e2e8f0;">
                    • Random Forest: n_estimators=100, max_depth=None<br>
                    • Decision Tree: max_depth=10, min_samples_split=10<br>
                    • KNN: n_neighbors=9, n_jobs=-1<br>
                    • Logistic Regression: C=0.1, max_iter=1000
                </div>
            </div>
        </div>
        """).strip(), unsafe_allow_html=True)

    # 3. Metric Explanation Guide
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    st.markdown(textwrap.dedent("""
    <div class="dashboard-card">
        <div class="card-title">📚 Evaluation Metrics Reference in Credit Risk</div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1.25rem;">
            <div>
                <strong style="color: #38bdf8;">Accuracy:</strong>
                <p style="font-size: 0.86rem; color: #94a3b8; margin-top: 0.25rem;">Overall proportion of correct default and non-default classifications across the test cohort.</p>
            </div>
            <div>
                <strong style="color: #38bdf8;">Precision:</strong>
                <p style="font-size: 0.86rem; color: #94a3b8; margin-top: 0.25rem;">When the model predicts a default, how often is it actually a default? Critical for preventing false alarms.</p>
            </div>
            <div>
                <strong style="color: #38bdf8;">Recall:</strong>
                <p style="font-size: 0.86rem; color: #94a3b8; margin-top: 0.25rem;">Out of all true defaulters, what percentage did the model catch? Essential for minimizing lender losses.</p>
            </div>
            <div>
                <strong style="color: #38bdf8;">ROC-AUC:</strong>
                <p style="font-size: 0.86rem; color: #94a3b8; margin-top: 0.25rem;">Discriminative capability across all classification probability thresholds. 1.0 is perfect; 0.5 is random.</p>
            </div>
        </div>
    </div>
    """).strip(), unsafe_allow_html=True)


# ==========================================
# PAGE 4: ANALYTICS / CHARTS PAGE
# ==========================================
def render_analytics():
    """Renders the comprehensive, interactive dark-themed charts and analytics dashboard."""
    st.markdown("<h2 style='font-size: 1.8rem; font-weight: 800; margin-bottom: 0.5rem;'>📈 Visual Analytics & Diagnostic Charts</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 0.95rem; margin-bottom: 1.5rem;'>Interactive visualizations displaying algorithm discrimination, confusion matrices, and feature importance.</p>", unsafe_allow_html=True)

    metrics = load_evaluation_metrics()
    if not metrics:
        st.warning("⚠️ Metrics data not found. Please train models to view visualizations.")
        return

    # Row 1: Overfitting Chart & ROC Curves
    ch_col1, ch_col2 = st.columns(2)

    with ch_col1:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 Train vs. Test Accuracy (Generalization Stability)</div>', unsafe_allow_html=True)
        
        models_list = list(metrics.keys())
        train_accs = [metrics[m].get("Train Accuracy", 0) for m in models_list]
        test_accs = [metrics[m].get("Test Accuracy", 0) for m in models_list]

        fig_acc = go.Figure(data=[
            go.Bar(name='Train Accuracy', x=models_list, y=train_accs, marker_color='#3b82f6'),
            go.Bar(name='Test Accuracy', x=models_list, y=test_accs, marker_color='#10b981')
        ])
        fig_acc.update_layout(
            barmode='group',
            yaxis=dict(range=[0.75, 1.02], title="Accuracy Score"),
            xaxis_title="Algorithm"
        )
        apply_dark_plotly_layout(fig_acc, height=380)
        st.plotly_chart(fig_acc, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with ch_col2:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📉 Receiver Operating Characteristic (ROC Curves)</div>', unsafe_allow_html=True)

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='#64748b', width=1.5),
            name='Random Chance (AUC = 0.500)'
        ))

        colors = ['#38bdf8', '#fb7185', '#a78bfa', '#f59e0b']
        for idx, (m_name, m_data) in enumerate(metrics.items()):
            roc_dict = m_data.get("ROC Curve", {})
            fpr = roc_dict.get("fpr", [])
            tpr = roc_dict.get("tpr", [])
            auc_val = m_data.get("ROC AUC", 0)
            if fpr and tpr:
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    line=dict(color=colors[idx % len(colors)], width=2),
                    name=f'{m_name} (AUC = {auc_val:.3f})'
                ))

        fig_roc.update_layout(
            xaxis=dict(range=[-0.02, 1.02], title="False Positive Rate"),
            yaxis=dict(range=[-0.02, 1.05], title="True Positive Rate")
        )
        apply_dark_plotly_layout(fig_roc, height=380)
        st.plotly_chart(fig_roc, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 2: Confusion Matrix & Feature Importances
    ch_col3, ch_col4 = st.columns(2)

    with ch_col3:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🎯 Confusion Matrix Heatmap</div>', unsafe_allow_html=True)
        
        cm_model = st.selectbox("Select Model for Confusion Matrix", list(metrics.keys()), key="cm_selector")
        cm = metrics[cm_model].get("Confusion Matrix")
        
        if cm:
            cm_z = cm
            cm_text = [[f"{val:,}" for val in row] for row in cm]
            fig_cm = px.imshow(
                cm_z,
                text_auto=False,
                color_continuous_scale='Blues',
                labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                x=['Non-Default (0)', 'Default (1)'],
                y=['Non-Default (0)', 'Default (1)']
            )
            # Add text annotations with clean formatting
            for i in range(2):
                for j in range(2):
                    fig_cm.add_annotation(
                        x=j, y=i,
                        text=f"{cm[i][j]:,}",
                        showarrow=False,
                        font=dict(color="#000000" if cm[i][j] > 1000 else "#000000", size=15, family="Inter")
                    )
            fig_cm.update_coloraxes(showscale=False)
            apply_dark_plotly_layout(fig_cm, height=380)
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("Confusion matrix data unavailable.")
        st.markdown('</div>', unsafe_allow_html=True)

    with ch_col4:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔍 Feature Importance Rankings</div>', unsafe_allow_html=True)

        fi_available = [m for m, d in metrics.items() if d.get("Feature Importances")]
        if fi_available:
            fi_model = st.selectbox("Select Tree Model", fi_available, key="fi_selector")
            fi_data = metrics[fi_model].get("Feature Importances", {})
            
            if fi_data:
                # Top 12 features
                top_items = list(fi_data.items())[:12]
                top_items.reverse()  # For bottom-up horizontal bar chart
                f_names = [item[0].replace('num__', '').replace('cat__', '') for item in top_items]
                f_scores = [item[1] for item in top_items]

                fig_fi = go.Figure(go.Bar(
                    x=f_scores,
                    y=f_names,
                    orientation='h',
                    marker=dict(
                        color=f_scores,
                        colorscale='Tealgrn',
                        line=dict(color='rgba(255, 255, 255, 0.1)', width=1)
                    )
                ))
                fig_fi.update_layout(xaxis_title="Gini Importance Score", yaxis_title="Feature")
                apply_dark_plotly_layout(fig_fi, height=380)
                st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Feature importance data is only available for Decision Tree and Random Forest models.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 3: Class Distribution in Test Cohort
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🍰 Test Set Class Balance & Demographic Distribution</div>', unsafe_allow_html=True)
    
    d_col1, d_col2 = st.columns([1, 1.4])
    with d_col1:
        # Imbalance pie chart based on test confusion matrix totals
        # In Logistic Regression cm: [[8813, 29], [1116, 42]] -> Total 0s: 8842 (88.4%), Total 1s: 1158 (11.6%)
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Non-Default (Class 0)', 'Default (Class 1)'],
            values=[8842, 1158],
            hole=0.55,
            marker=dict(colors=['#10b981', '#f43f5e']),
            textinfo='label+percent',
            textfont=dict(family="Inter", size=13, color="#ffffff")
        )])
        apply_dark_plotly_layout(fig_pie, title="Default vs. Non-Default Distribution", height=320)
        st.plotly_chart(fig_pie, use_container_width=True)

    with d_col2:
        st.markdown("""
        <div style="padding: 1rem; color: #cbd5e1; font-size: 0.92rem; line-height: 1.7;">
            <h4 style="font-size: 1.05rem; font-weight: 700; color: #60a5fa; margin-bottom: 0.5rem;">Insights on Data Imbalance</h4>
            <p>
                The loan default dataset displays a standard class imbalance (~11.6% defaults vs. 88.4% non-defaults). 
                In credit risk modeling, capturing the minority default class with high precision is vital to prevent loan portfolio deterioration.
            </p>
            <p>
                Stratified train-test splits and cross-validation were specifically employed to prevent sample distortion, 
                ensuring test metrics realistically reflect live lending conditions.
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# PAGE 5: ABOUT PAGE
# ==========================================
def render_about():
    """Renders the About page with project credentials, methodology, and legal disclaimer."""
    st.markdown("<h2 style='font-size: 1.8rem; font-weight: 800; margin-bottom: 0.5rem;'>ℹ️ About the Loan Default Prediction Project</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 0.95rem; margin-bottom: 1.5rem;'>Background, architectural methodology, and technology specification.</p>", unsafe_allow_html=True)

    st.markdown("""
    <div class="dashboard-card">
        <div class="card-title">🎯 Project Mission & Overview</div>
        <p style="font-size: 0.94rem; color: #cbd5e1; line-height: 1.7;">
            The <strong>Loan Default Prediction Platform</strong> is an end-to-end Machine Learning web application designed to evaluate 
            consumer credit risk. By ingesting applicant demographic characteristics, financial ratios, and loan parameters, 
            the system delivers real-time probabilistic classification indicating whether an applicant is at risk of defaulting on loan obligations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c_ab1, c_ab2 = st.columns(2)

    with c_ab1:
        st.markdown("""
        <div class="dashboard-card" style="height: 100%;">
            <div class="card-title">🔬 Machine Learning Methodology</div>
            <ul style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.8; padding-left: 1.25rem;">
                <li><strong>Outlier Treatment:</strong> Interquartile Range (IQR) filtering on continuous variables (Income and Loan Amount).</li>
                <li><strong>Pipeline Transformation:</strong> Scikit-learn <code>ColumnTransformer</code> applying median imputation & standard scaling to numericals, plus most-frequent imputation & one-hot encoding to categoricals.</li>
                <li><strong>Validation Strategy:</strong> Stratified 80/20 train-test partition preserving target proportion.</li>
                <li><strong>Hyperparameter Tuning:</strong> RandomizedSearchCV with 3-fold cross-validation optimizing accuracy and stability.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c_ab2:
        st.markdown("""
        <div class="dashboard-card" style="height: 100%;">
            <div class="card-title">📦 Production Environment & Versions</div>
            <div style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.8;">
                <p><strong>Deployment Target:</strong> Streamlit Community Cloud</p>
                <div style="background: rgba(17, 24, 39, 0.7); border-radius: 8px; padding: 0.75rem 1rem; border: 1px solid rgba(255,255,255,0.06); font-family: monospace; font-size: 0.85rem; color: #60a5fa;">
                    • Streamlit: v1.51.0<br>
                    • Scikit-Learn: v1.7.2<br>
                    • Pandas: v2.3.3<br>
                    • NumPy: v2.3.5<br>
                    • Joblib: v1.5.2<br>
                    • Plotly: v6.3.0
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Disclaimer Section
    st.markdown("""
    <div class="disclaimer-box" style="margin-top: 1.5rem;">
        <h4 style="font-size: 0.95rem; font-weight: 700; color: #ffffff; margin-bottom: 0.4rem;">⚖️ Regulatory & Educational Disclaimer</h4>
        This application is developed strictly for machine learning research and educational demonstration purposes. 
        The predictions, probability metrics, and insights produced by this platform do not constitute professional financial, 
        lending, investment, or credit advisory services. Financial institutions should not rely on this demonstration application 
        for formal credit adjudication or regulatory capital calculations.
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# MAIN APPLICATION CONTROLLER
# ==========================================
def main():
    """Main routing controller for the Streamlit single-page application."""
    # 1. Apply Global Dark Theme CSS
    apply_custom_css()

    # 2. Render Global Static Navigation Bar
    render_navbar()

    # 3. Page Routing via Session State
    page = st.session_state.get("current_page", "Home")

    if page == "Home":
        render_home()
    elif page == "Prediction":
        render_prediction()
    elif page == "Model Summary":
        render_model_summary()
    elif page == "Analytics / Charts":
        render_analytics()
    elif page == "About":
        render_about()
    else:
        render_home()


if __name__ == '__main__':
    main()
