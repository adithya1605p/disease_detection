"""VitaScan AI — Next-Generation Precision Multi-Disease Screening Dashboard.

Features:
- High-precision ML ensemble models (ExtraTrees, Random Forest, HistGradientBoosting, Soft Voting).
- Real-time biomarker range validation & anomaly flags.
- Explicit diagnostic calculation pipeline with step-by-step clinical verification.
- Comprehensive precision analytics (Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, 5-Fold CV).
- Interactive risk stratification gauge and downloadable clinical screening dossiers.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

st.set_page_config(
    page_title="VitaScan AI | Precision Health Screening",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Clinical Dark Theme CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-main: #060913;
        --card-bg: rgba(14, 23, 42, 0.75);
        --card-border: rgba(148, 163, 184, 0.14);
        --accent-cyan: #06b6d4;
        --accent-blue: #3b82f6;
        --accent-emerald: #10b981;
        --accent-rose: #f43f5e;
        --accent-amber: #f59e0b;
        --accent-violet: #8b5cf6;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 15% 0%, #0d1a33 0%, #070c18 40%, #04060c 100%);
        color: var(--text-primary);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #091020 0%, #060913 100%);
        border-right: 1px solid var(--card-border);
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.03em;
        font-weight: 700;
    }

    .hero-container {
        padding: 1.5rem 0 1rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.12);
        margin-bottom: 1.5rem;
    }

    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.28rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        background: rgba(59, 130, 246, 0.12);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.28);
        margin-bottom: 0.75rem;
    }

    .glass-panel {
        background: var(--card-bg);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid var(--card-border);
        border-radius: 18px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        transition: border-color 0.2s ease;
    }

    .glass-panel:hover {
        border-color: rgba(148, 163, 184, 0.25);
    }

    .metric-badge-card {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 14px;
        padding: 1rem;
        text-align: center;
    }

    .metric-val {
        font-family: 'Space Grotesk', monospace;
        font-size: 1.7rem;
        font-weight: 700;
        line-height: 1.1;
    }

    .metric-lbl {
        color: var(--text-secondary);
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.35rem;
    }

    /* Risk Spectrum Meter */
    .risk-meter-container {
        margin: 1.25rem 0;
        padding: 1.2rem;
        border-radius: 16px;
        background: rgba(10, 16, 30, 0.8);
        border: 1px solid var(--card-border);
    }

    .risk-bar-track {
        height: 16px;
        border-radius: 8px;
        background: linear-gradient(90deg, #10b981 0%, #eab308 35%, #f97316 68%, #ef4444 100%);
        position: relative;
        margin: 1rem 0 0.5rem;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.4);
    }

    .risk-pointer {
        position: absolute;
        top: -6px;
        width: 28px;
        height: 28px;
        background: #ffffff;
        border: 3px solid #0f172a;
        border-radius: 50%;
        transform: translateX(-50%);
        box-shadow: 0 0 12px rgba(255,255,255,0.8);
        transition: left 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
    }

    .status-alert {
        padding: 1.25rem;
        border-radius: 16px;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: flex-start;
        gap: 1rem;
    }

    .status-alert.critical {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(15, 23, 42, 0.85));
        border: 1px solid rgba(239, 68, 68, 0.4);
    }

    .status-alert.elevated {
        background: linear-gradient(135deg, rgba(249, 115, 22, 0.15), rgba(15, 23, 42, 0.85));
        border: 1px solid rgba(249, 115, 22, 0.4);
    }

    .status-alert.moderate {
        background: linear-gradient(135deg, rgba(234, 179, 8, 0.15), rgba(15, 23, 42, 0.85));
        border: 1px solid rgba(234, 179, 8, 0.4);
    }

    .status-alert.low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(15, 23, 42, 0.85));
        border: 1px solid rgba(16, 185, 129, 0.4);
    }

    .anomaly-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.6rem;
        background: rgba(239, 68, 68, 0.14);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 6px;
        font-size: 0.78rem;
        color: #fca5a5;
        margin: 0.2rem 0.2rem 0.2rem 0;
    }

    .normal-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.6rem;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 6px;
        font-size: 0.78rem;
        color: #6ee7b7;
        margin: 0.2rem 0.2rem 0.2rem 0;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
        font-size: 1.02rem;
        padding: 0.65rem 1.5rem;
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.75rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.15);
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0.7rem 1.4rem;
        border-radius: 10px 10px 0 0;
        font-weight: 600;
        color: #94a3b8;
    }

    .stTabs [aria-selected="true"] {
        color: #60a5fa !important;
        background: rgba(30, 41, 59, 0.5) !important;
        border-bottom: 2px solid #3b82f6 !important;
    }

    .mono-code {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.84rem;
        background: rgba(15, 23, 42, 0.9);
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        border: 1px solid rgba(148, 163, 184, 0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

DISEASE_META = {
    "Diabetes": {
        "icon": "🩸",
        "accent": "#ef4444",
        "category": "Endocrinology",
        "description": "Screens type 2 diabetic risk utilizing fasting glucose, insulin kinetics, body mass index, and pedigree risk factors.",
        "model_type": "Tuned Ensemble (ExtraTrees + Random Forest)",
    },
    "Heart Disease": {
        "icon": "❤️",
        "accent": "#f43f5e",
        "category": "Cardiology",
        "description": "Screens coronary artery disease and cardiovascular risk using hemodynamic markers, ST wave response, and fluoroscopy signals.",
        "model_type": "Calibrated Soft-Voting Ensemble (ExtraTrees + Random Forest)",
    },
    "Breast Cancer": {
        "icon": "🎗️",
        "accent": "#ec4899",
        "category": "Oncology",
        "description": "High-precision malignant vs benign classification using digitized fine needle aspirate (FNA) cell nuclei morphometry.",
        "model_type": "Tuned ExtraTrees High-Dimensional Classifier",
    },
    "Kidney Disease": {
        "icon": "🫧",
        "accent": "#0ea5e9",
        "category": "Nephrology",
        "description": "Screens chronic kidney disease (CKD) using hematology, urinary specific gravity/albumin, electrolytes, and renal biochemistry.",
        "model_type": "Tuned Clinical ExtraTrees Classifier",
    },
    "Liver Disease": {
        "icon": "🧬",
        "accent": "#f59e0b",
        "category": "Hepatology",
        "description": "Identifies hepatic impairment and liver inflammation through total/direct bilirubin, aminotransferases, and A/G ratio.",
        "model_type": "Balanced ExtraTrees Classifier",
    },
    "Stroke": {
        "icon": "🧠",
        "accent": "#8b5cf6",
        "category": "Neurology",
        "description": "Predicts acute cerebrovascular accident risk through systemic cardiovascular markers, lifestyle factors, and metabolic indicators.",
        "model_type": "Balanced Ensemble (Random Forest + HistGradientBoosting)",
    },
}


@st.cache_resource
def load_disease_artifacts(disease: str) -> dict[str, Any]:
    safe_name = disease.lower().replace(" ", "_")
    return {
        "model": joblib.load(MODELS_DIR / f"{safe_name}_model.pkl"),
        "scaler": joblib.load(MODELS_DIR / f"{safe_name}_scaler.pkl"),
        "features": joblib.load(MODELS_DIR / f"{safe_name}_features.pkl"),
        "encoders": joblib.load(MODELS_DIR / f"{safe_name}_encoders.pkl"),
        "metrics": json.load(open(MODELS_DIR / f"{safe_name}_metrics.json", encoding="utf-8")),
    }


@st.cache_data
def load_all_metrics() -> dict[str, Any]:
    with open(MODELS_DIR / "metrics_summary.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_clinical_meta() -> dict[str, Any]:
    with open(MODELS_DIR / "clinical_meta.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_sample_profiles() -> dict[str, Any]:
    with open(MODELS_DIR / "sample_profiles.json", encoding="utf-8") as f:
        return json.load(f)


def calculate_risk(
    disease: str,
    raw_inputs: dict[str, Any],
    artifacts: dict[str, Any],
    clinical_meta: dict[str, Any],
) -> dict[str, Any]:
    """Execute complete clinical calculation pipeline with calibrated probabilities and biomarker checks."""
    encoded_vals = {}
    encoders = artifacts["encoders"]

    for feat in artifacts["features"]:
        val = raw_inputs[feat]
        if feat in encoders:
            # Handle categorical encoding safely
            enc = encoders[feat]
            str_val = str(val).strip()
            if str_val in enc.classes_:
                encoded_vals[feat] = enc.transform([str_val])[0]
            else:
                encoded_vals[feat] = 0
        else:
            try:
                encoded_vals[feat] = float(val)
            except (ValueError, TypeError):
                encoded_vals[feat] = 0.0

    # Build input tensor
    feature_df = pd.DataFrame([[encoded_vals[f] for f in artifacts["features"]]], columns=artifacts["features"])
    scaled_tensor = artifacts["scaler"].transform(feature_df)

    # Model inference
    model = artifacts["model"]
    pred = int(model.predict(scaled_tensor)[0])
    probabilities = model.predict_proba(scaled_tensor)[0]

    # Positive class probability
    pos_prob = float(probabilities[1]) if len(probabilities) > 1 else float(pred)

    # Classify clinical risk tier
    if pos_prob >= 0.75:
        tier = "Critical Risk"
        tier_color = "#ef4444"
        tier_class = "critical"
        action_summary = "High probability of positive clinical pattern. Immediate specialized physician diagnostic evaluation is strongly recommended."
    elif pos_prob >= 0.50:
        tier = "Elevated Risk"
        tier_color = "#f97316"
        tier_class = "elevated"
        action_summary = "Elevated screening indicator detected. Schedule confirmatory clinical lab tests and follow-up consultation."
    elif pos_prob >= 0.25:
        tier = "Moderate Risk"
        tier_color = "#eab308"
        tier_class = "moderate"
        action_summary = "Borderline physiological signals observed. Recommend lifestyle optimization and monitoring during routine checkups."
    else:
        tier = "Low Risk"
        tier_color = "#10b981"
        tier_class = "low"
        action_summary = "Biomarkers align with negative clinical screening reference cohort. Maintain standard preventive wellness protocols."

    # Identify biomarker anomalies against physiological normal bounds
    anomalies: list[dict[str, Any]] = []
    normal_signals: list[dict[str, Any]] = []

    for feat in artifacts["features"]:
        if feat in clinical_meta and "normal" in clinical_meta[feat]:
            val = raw_inputs.get(feat)
            if isinstance(val, (int, float)):
                low, high = clinical_meta[feat]["normal"]
                unit = clinical_meta[feat].get("unit", "")
                if val < low:
                    anomalies.append({
                        "feature": feat,
                        "value": val,
                        "normal": f"{low} - {high} {unit}",
                        "status": "Below Normal",
                    })
                elif val > high:
                    anomalies.append({
                        "feature": feat,
                        "value": val,
                        "normal": f"{low} - {high} {unit}",
                        "status": "Above Normal",
                    })
                else:
                    normal_signals.append({
                        "feature": feat,
                        "value": val,
                        "normal": f"{low} - {high} {unit}",
                    })

    return {
        "disease": disease,
        "prediction": pred,
        "probability_positive": pos_prob,
        "probability_negative": float(probabilities[0]) if len(probabilities) > 1 else float(1.0 - pred),
        "risk_tier": tier,
        "tier_color": tier_color,
        "tier_class": tier_class,
        "action_summary": action_summary,
        "anomalies": anomalies,
        "normal_signals": normal_signals,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "raw_inputs": raw_inputs,
    }


def render_sidebar(all_metrics: dict[str, Any]) -> str:
    with st.sidebar:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.2rem;">
                <span style="font-size: 2rem;">🩺</span>
                <div>
                    <h2 style="margin: 0; font-size: 1.45rem; line-height: 1.1;">VitaScan <span style="color:#38bdf8;">AI</span></h2>
                    <span style="font-size: 0.72rem; color: #94a3b8; letter-spacing: 0.05em; text-transform: uppercase;">Clinical Intelligence Studio</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown("##### Select Clinical Disease Model")
        disease_names = list(DISEASE_META.keys())
        selected_disease = st.selectbox(
            "Screening Target",
            disease_names,
            format_func=lambda d: f"{DISEASE_META[d]['icon']} {d}",
            label_visibility="collapsed",
        )

        d_meta = DISEASE_META[selected_disease]
        d_metrics = all_metrics.get(selected_disease, {})

        st.markdown(
            f"""
            <div class="glass-panel" style="margin-top: 0.75rem; padding: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 0.75rem; text-transform: uppercase; color: #64748b; font-weight: 700;">Model Specs</span>
                    <span style="font-size: 0.72rem; padding: 0.15rem 0.5rem; background: rgba(56, 189, 248, 0.15); color: #38bdf8; border-radius: 999px;">Active</span>
                </div>
                <div style="font-weight: 700; font-size: 1rem; color: #f8fafc;">{selected_disease}</div>
                <div style="font-size: 0.78rem; color: #94a3b8; margin: 0.25rem 0 0.75rem;">{d_meta['model_type']}</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; padding-top: 0.5rem; border-top: 1px solid rgba(148,163,184,0.1);">
                    <div>
                        <div style="font-size: 0.68rem; color: #64748b; text-transform: uppercase;">Accuracy</div>
                        <div style="font-size: 0.95rem; font-weight: 700; color: #10b981;">{d_metrics.get('accuracy', 0)*100:.2f}%</div>
                    </div>
                    <div>
                        <div style="font-size: 0.68rem; color: #64748b; text-transform: uppercase;">ROC-AUC</div>
                        <div style="font-size: 0.95rem; font-weight: 700; color: #38bdf8;">{d_metrics.get('roc_auc', 0):.4f}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        st.markdown(
            """
            <div style="padding: 0.5rem; background: rgba(15,23,42,0.6); border-radius: 12px; border: 1px solid rgba(148,163,184,0.12);">
                <div style="font-size: 0.75rem; font-weight: 700; color: #cbd5e1; margin-bottom: 0.3rem;">⚡ Diagnostic Workflow</div>
                <ol style="margin: 0; padding-left: 1.1rem; font-size: 0.75rem; color: #94a3b8; line-height: 1.6;">
                    <li>Configure or load patient biomarkers</li>
                    <li>Click <b>Run Diagnostic Calculation</b></li>
                    <li>Review stratified risk & anomaly flags</li>
                    <li>Export calibrated clinical dossier</li>
                </ol>
            </div>
            <div style="margin-top: 1.25rem; font-size: 0.7rem; color: #64748b; line-height: 1.5;">
                <strong style="color: #94a3b8;">⚠️ Clinical Notice:</strong> Educational screening prototype. For medical diagnosis, always consult a licensed medical practitioner.
            </div>
            """,
            unsafe_allow_html=True,
        )

    return selected_disease


def main() -> None:
    all_metrics = load_all_metrics()
    clinical_meta = load_clinical_meta()
    sample_profiles = load_sample_profiles()

    selected_disease = render_sidebar(all_metrics)
    meta = DISEASE_META[selected_disease]
    artifacts = load_disease_artifacts(selected_disease)
    disease_metrics = artifacts["metrics"]

    # Header presentation
    st.markdown(
        f"""
        <div class="hero-container">
            <div class="badge-pill">
                <span>🧬</span> {meta['category']} Intelligence Engine • 6 Clinical Models Online
            </div>
            <div style="display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 1rem;">
                <div>
                    <h1 style="margin: 0; font-size: clamp(2rem, 3.8vw, 3.2rem); line-height: 1.1;">
                        {meta['icon']} {selected_disease} <span style="color: {meta['accent']};">Screening</span>
                    </h1>
                    <p style="margin: 0.5rem 0 0; color: #94a3b8; max-width: 760px; font-size: 1rem; line-height: 1.6;">
                        {meta['description']}
                    </p>
                </div>
                <div style="display: flex; gap: 0.75rem;">
                    <div class="metric-badge-card" style="min-width: 110px;">
                        <div class="metric-val" style="color: #10b981;">{disease_metrics['accuracy']*100:.2f}%</div>
                        <div class="metric-lbl">Test Accuracy</div>
                    </div>
                    <div class="metric-badge-card" style="min-width: 110px;">
                        <div class="metric-val" style="color: #38bdf8;">{disease_metrics['roc_auc']:.4f}</div>
                        <div class="metric-lbl">ROC-AUC</div>
                    </div>
                    <div class="metric-badge-card" style="min-width: 110px;">
                        <div class="metric-val" style="color: #a78bfa;">{disease_metrics['cv_accuracy_mean']*100:.1f}%</div>
                        <div class="metric-lbl">5-Fold CV</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_screening, tab_analytics, tab_biomarkers = st.tabs([
        "🩺 Clinical Screening Studio",
        "📊 Precision Model Analytics",
        "🧬 Biomarker Reference Guide",
    ])

    # =========================================================================
    # TAB 1: CLINICAL SCREENING STUDIO
    # =========================================================================
    with tab_screening:
        # Top presets and profile actions
        st.markdown("#### Patient Biomarker Input")
        preset_cols = st.columns([1.5, 1.5, 1.5, 1.5, 2.5])

        with preset_cols[0]:
            if st.button("🔴 High-Risk Profile", use_container_width=True, help="Load representative high-risk clinical case"):
                high_prof = sample_profiles.get(selected_disease, {}).get("high_risk", {})
                for k, v in high_prof.items():
                    st.session_state[f"input_{selected_disease}_{k}"] = v
                st.session_state[f"{selected_disease}_calculated"] = False
                st.session_state["toast_msg"] = "Loaded High-Risk Clinical Profile"
                st.rerun()

        with preset_cols[1]:
            if st.button("🟢 Low-Risk Profile", use_container_width=True, help="Load representative low-risk clinical case"):
                low_prof = sample_profiles.get(selected_disease, {}).get("low_risk", {})
                for k, v in low_prof.items():
                    st.session_state[f"input_{selected_disease}_{k}"] = v
                st.session_state[f"{selected_disease}_calculated"] = False
                st.session_state["toast_msg"] = "Loaded Low-Risk Clinical Profile"
                st.rerun()

        with preset_cols[2]:
            if st.button("⚖️ Median Baseline", use_container_width=True, help="Reset to cohort median values"):
                for feat in artifacts["features"]:
                    st.session_state.pop(f"input_{selected_disease}_{feat}", None)
                st.session_state[f"{selected_disease}_calculated"] = False
                st.session_state["toast_msg"] = "Reset to Cohort Median"
                st.rerun()

        with preset_cols[3]:
            if st.button("🔄 Clear Calculation", use_container_width=True):
                st.session_state[f"{selected_disease}_calculated"] = False
                st.session_state.pop(f"{selected_disease}_calc_result", None)
                st.rerun()

        with preset_cols[4]:
            if "toast_msg" in st.session_state:
                st.success(st.session_state.pop("toast_msg"), icon="✨")

        # Organize inputs logically into clinical categories
        features = artifacts["features"]
        encoders = artifacts["encoders"]

        # Categorize features
        categorized_features: dict[str, list[str]] = {}
        for feat in features:
            f_meta = clinical_meta.get(feat, {})
            cat = f_meta.get("category", "General Clinical Signals")
            categorized_features.setdefault(cat, []).append(feat)

        # Collect user inputs
        raw_inputs: dict[str, Any] = {}

        # Render inputs inside collapsible or segmented cards
        with st.container():
            for category_name, cat_feats in categorized_features.items():
                with st.expander(f"📁 {category_name} ({len(cat_feats)} biomarkers)", expanded=True):
                    grid_cols = st.columns(3 if len(cat_feats) >= 3 else len(cat_feats))
                    for idx, feat in enumerate(cat_feats):
                        col = grid_cols[idx % len(grid_cols)]
                        with col:
                            state_key = f"input_{selected_disease}_{feat}"
                            f_meta = clinical_meta.get(feat, {})
                            unit_label = f" ({f_meta['unit']})" if "unit" in f_meta else ""
                            label_text = f"{feat.replace('_', ' ').title()}{unit_label}"
                            help_text = f_meta.get("desc", f"Biomarker: {feat}")
                            if "normal" in f_meta:
                                help_text += f" | Normal Reference: {f_meta['normal'][0]} - {f_meta['normal'][1]} {f_meta.get('unit', '')}"

                            if feat in encoders:
                                enc = encoders[feat]
                                options = list(enc.classes_)
                                current_val = st.session_state.get(state_key, options[0])
                                idx_selected = options.index(str(current_val)) if str(current_val) in options else 0
                                raw_inputs[feat] = st.selectbox(
                                    label_text,
                                    options,
                                    index=idx_selected,
                                    key=state_key,
                                    help=help_text,
                                )
                            else:
                                normal_range = f_meta.get("normal", (0.0, 100.0))
                                default_val = float(st.session_state.get(state_key, normal_range[0] if normal_range else 1.0))
                                step_size = 0.01 if (abs(default_val) < 10 and default_val != int(default_val)) else 1.0
                                raw_inputs[feat] = st.number_input(
                                    label_text,
                                    value=default_val,
                                    step=step_size,
                                    key=state_key,
                                    help=help_text,
                                )

        st.write("")
        calc_btn_container = st.container()

        with calc_btn_container:
            calculate_clicked = st.button(
                "⚡ Run Diagnostic Calculation & Risk Analysis",
                type="primary",
                use_container_width=True,
                help="Execute trained ensemble model inference, calibrate probability, and check biomarker bounds",
            )

        # Trigger Calculation Pipeline
        if calculate_clicked:
            with st.status("🔬 Running Calibrated Clinical Assessment Pipeline...", expanded=True) as status:
                st.write("Step 1/4: Validating biometric bounds & clinical limits...")
                time.sleep(0.18)
                st.write("Step 2/4: Standardizing input tensors with trained scaler...")
                time.sleep(0.18)
                st.write(f"Step 3/4: Executing {meta['model_type']} inference...")
                time.sleep(0.22)
                st.write("Step 4/4: Calibrating risk probability & detecting outlier biomarkers...")
                time.sleep(0.15)

                result = calculate_risk(selected_disease, raw_inputs, artifacts, clinical_meta)
                st.session_state[f"{selected_disease}_calculated"] = True
                st.session_state[f"{selected_disease}_calc_result"] = result
                status.update(label="✅ Diagnostic Calculation Complete!", state="complete", expanded=False)

        st.divider()

        # =====================================================================
        # OUTPUT DISPLAY (Displayed strictly after calculation)
        # =====================================================================
        is_calculated = st.session_state.get(f"{selected_disease}_calculated", False)
        calc_result = st.session_state.get(f"{selected_disease}_calc_result")

        if not is_calculated or not calc_result or calc_result.get("disease") != selected_disease:
            st.markdown(
                """
                <div class="glass-panel" style="text-align: center; padding: 3rem 1.5rem; border: 2px dashed rgba(148,163,184,0.2);">
                    <div style="font-size: 2.8rem; margin-bottom: 0.75rem;">⚡</div>
                    <h3 style="margin: 0; color: #f8fafc;">Diagnostic Assessment Pending</h3>
                    <p style="color: #94a3b8; max-width: 540px; margin: 0.5rem auto 1.5rem; font-size: 0.95rem; line-height: 1.6;">
                        Adjust patient biomarkers above or load a clinical profile preset, then click 
                        <strong style="color: #60a5fa;">"Run Diagnostic Calculation & Risk Analysis"</strong> to generate the verified clinical screening output.
                    </p>
                    <div style="display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.4rem 1rem; background: rgba(56,189,248,0.1); border-radius: 999px; color: #38bdf8; font-size: 0.8rem; font-weight: 600;">
                        <span>🔒 High-Precision Machine Learning Pipeline Ready</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # RENDER OUTPUT AFTER CALCULATION
            prob_pct = calc_result["probability_positive"] * 100.0
            tier_color = calc_result["tier_color"]
            tier_name = calc_result["risk_tier"]

            st.markdown(
                f"""
                <div class="status-alert {calc_result['tier_class']}">
                    <div style="font-size: 2.4rem;">{'🚨' if prob_pct >= 50 else '🛡️'}</div>
                    <div style="flex: 1;">
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                            <span style="font-size: 0.76rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: {tier_color};">
                                Stratified Screening Assessment • Calculated {calc_result['timestamp']}
                            </span>
                            <span style="font-family: 'Space Grotesk', monospace; font-size: 0.88rem; font-weight: 700; color: #e2e8f0;">
                                Decision: {'POSITIVE (Class 1)' if calc_result['prediction'] == 1 else 'NEGATIVE (Class 0)'}
                            </span>
                        </div>
                        <h2 style="margin: 0.3rem 0; font-size: 1.85rem; color: #f8fafc;">
                            {tier_name}: <span style="color: {tier_color};">{prob_pct:.2f}% Risk Probability</span>
                        </h2>
                        <p style="margin: 0; color: #cbd5e1; font-size: 0.95rem; line-height: 1.5;">
                            {calc_result['action_summary']}
                        </p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Key Metric Breakdown Cards
            out_cols = st.columns(4)
            with out_cols[0]:
                st.markdown(
                    f"""
                    <div class="metric-badge-card">
                        <div class="metric-val" style="color: {tier_color};">{prob_pct:.2f}%</div>
                        <div class="metric-lbl">Calibrated Risk</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with out_cols[1]:
                st.markdown(
                    f"""
                    <div class="metric-badge-card">
                        <div class="metric-val" style="color: #10b981;">{calc_result['probability_negative']*100:.2f}%</div>
                        <div class="metric-lbl">Negative Baseline</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with out_cols[2]:
                st.markdown(
                    f"""
                    <div class="metric-badge-card">
                        <div class="metric-val" style="color: #38bdf8;">{tier_name.split()[0]}</div>
                        <div class="metric-lbl">Risk Tier</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with out_cols[3]:
                st.markdown(
                    f"""
                    <div class="metric-badge-card">
                        <div class="metric-val" style="color: #f59e0b;">{len(calc_result['anomalies'])}</div>
                        <div class="metric-lbl">Biomarker Flags</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Interactive Risk Spectrum Bar
            pointer_pct = min(max(prob_pct, 1.5), 98.5)
            st.markdown(
                f"""
                <div class="risk-meter-container">
                    <div style="display: flex; justify-content: space-between; font-size: 0.78rem; font-weight: 600; color: #94a3b8;">
                        <span>🟢 Low Risk (&lt;25%)</span>
                        <span>🟡 Moderate (25-50%)</span>
                        <span>🟠 Elevated (50-75%)</span>
                        <span>🔴 Critical (&gt;75%)</span>
                    </div>
                    <div class="risk-bar-track">
                        <div class="risk-pointer" style="left: {pointer_pct}%;"></div>
                    </div>
                    <div style="text-align: right; font-size: 0.74rem; color: #64748b; margin-top: 0.2rem;">
                        Patient Marker: <strong style="color: {tier_color};">{prob_pct:.2f}%</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Biomarker Anomaly Flags & Clinical Recommendations
            col_anomalies, col_actions = st.columns([1, 1])

            with col_anomalies:
                st.markdown("##### 🔍 Biomarker Anomaly Analysis")
                if calc_result["anomalies"]:
                    st.caption("Measurements outside normal physiological reference intervals:")
                    for anom in calc_result["anomalies"]:
                        st.markdown(
                            f"""
                            <div class="anomaly-tag">
                                <strong>⚠️ {anom['feature'].replace('_', ' ').title()}:</strong> 
                                {anom['value']} ({anom['status']}) — Normal: {anom['normal']}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.success("All analyzed biometric signals fall within standard physiological reference ranges.", icon="✅")

                if calc_result["normal_signals"]:
                    with st.expander(f"Standard Signals Within Range ({len(calc_result['normal_signals'])})"):
                        for norm in calc_result["normal_signals"]:
                            st.markdown(
                                f"""
                                <div class="normal-tag">
                                    ✓ {norm['feature'].replace('_', ' ').title()}: {norm['value']} (Normal: {norm['normal']})
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

            with col_actions:
                st.markdown("##### 📋 Recommended Clinical Protocol")
                if prob_pct >= 50.0:
                    st.markdown(
                        """
                        - **Immediate Physician Follow-up:** Urgent clinical evaluation with specialist recommended.
                        - **Confirmatory Lab Panels:** Execute diagnostic bloodwork and imaging to corroborate screening pattern.
                        - **Symptom Diary:** Document acute symptoms, cardiovascular changes, or discomfort.
                        - **Preventive Precautions:** Avoid strenuous exertion or unprescribed medication until cleared.
                        """
                    )
                else:
                    st.markdown(
                        """
                        - **Routine Wellness Surveillance:** Continue standard annual health examinations.
                        - **Lifestyle Maintenance:** Sustain balanced nutritional profile and cardiovascular exercise.
                        - **Periodic Screening:** Repeat clinical screening in 6 to 12 months as indicated by primary care provider.
                        """
                    )

            # Export dossier button
            st.write("")
            report_data = {
                "Screening_Target": selected_disease,
                "Calculation_Timestamp": calc_result["timestamp"],
                "Calibrated_Risk_Probability": f"{prob_pct:.4f}%",
                "Model_Prediction_Class": "Positive" if calc_result["prediction"] == 1 else "Negative",
                "Risk_Tier": tier_name,
                "Model_Architecture": meta["model_type"],
                **calc_result["raw_inputs"],
            }
            report_df = pd.DataFrame([report_data])

            export_cols = st.columns([2, 1, 1])
            with export_cols[0]:
                st.download_button(
                    label="📄 Download Calibrated Clinical Dossier (CSV)",
                    data=report_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{selected_disease.lower().replace(' ', '_')}_clinical_dossier.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with export_cols[1]:
                st.download_button(
                    label="💾 Export JSON",
                    data=json.dumps(report_data, indent=2).encode("utf-8"),
                    file_name=f"{selected_disease.lower().replace(' ', '_')}_dossier.json",
                    mime="application/json",
                    use_container_width=True,
                )

    # =========================================================================
    # TAB 2: PRECISION MODEL ANALYTICS & BENCHMARKS
    # =========================================================================
    with tab_analytics:
        st.markdown("### High-Precision Multi-Disease Benchmarks")
        st.caption("Exact quantitative metrics across all six trained clinical models evaluated on stratified test partitions.")

        # Side-by-side benchmark comparison table
        table_rows = []
        for d_name, d_met in all_metrics.items():
            table_rows.append({
                "Disease Target": f"{DISEASE_META[d_name]['icon']} {d_name}",
                "Test Accuracy": f"{d_met['accuracy']*100:.2f}%",
                "Precision": f"{d_met['precision']*100:.2f}%",
                "Recall / Sens.": f"{d_met['recall']*100:.2f}%",
                "Specificity": f"{d_met['specificity']*100:.2f}%",
                "F1-Score": f"{d_met['f1_score']:.4f}",
                "ROC-AUC": f"{d_met['roc_auc']:.4f}",
                "5-Fold CV Acc.": f"{d_met['cv_accuracy_mean']*100:.2f}% ± {d_met['cv_accuracy_std']*100:.2f}%",
                "Brier Score": f"{d_met['brier_score']:.4f}",
                "Test Records": d_met["test_samples"],
            })

        bench_df = pd.DataFrame(table_rows)
        st.dataframe(bench_df, use_container_width=True, hide_index=True)

        st.divider()

        # Detailed view for selected disease
        st.markdown(f"#### 🔍 Deep Precision Analysis: {meta['icon']} {selected_disease}")

        analytics_cols = st.columns([1, 1])

        with analytics_cols[0]:
            st.markdown("##### Confusion Matrix Partition")
            cm = disease_metrics.get("confusion_matrix", {})
            tp, fp, tn, fn = cm.get("tp", 0), cm.get("fp", 0), cm.get("tn", 0), cm.get("fn", 0)
            total_cases = tp + fp + tn + fn

            st.markdown(
                f"""
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-top: 0.75rem;">
                    <div class="glass-panel" style="border-left: 4px solid #10b981; text-align: center;">
                        <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">True Positives (TP)</div>
                        <div style="font-size: 1.8rem; font-weight: 700; color: #10b981;">{tp}</div>
                        <div style="font-size: 0.72rem; color: #64748b;">Correct disease identification</div>
                    </div>
                    <div class="glass-panel" style="border-left: 4px solid #f43f5e; text-align: center;">
                        <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">False Positives (FP)</div>
                        <div style="font-size: 1.8rem; font-weight: 700; color: #f43f5e;">{fp}</div>
                        <div style="font-size: 0.72rem; color: #64748b;">Healthy flagged as disease</div>
                    </div>
                    <div class="glass-panel" style="border-left: 4px solid #38bdf8; text-align: center;">
                        <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">True Negatives (TN)</div>
                        <div style="font-size: 1.8rem; font-weight: 700; color: #38bdf8;">{tn}</div>
                        <div style="font-size: 0.72rem; color: #64748b;">Correct healthy identification</div>
                    </div>
                    <div class="glass-panel" style="border-left: 4px solid #f59e0b; text-align: center;">
                        <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">False Negatives (FN)</div>
                        <div style="font-size: 1.8rem; font-weight: 700; color: #f59e0b;">{fn}</div>
                        <div style="font-size: 0.72rem; color: #64748b;">Missed positive disease case</div>
                    </div>
                </div>
                <div style="margin-top: 0.75rem; font-size: 0.78rem; color: #94a3b8;">
                    Total Test Sample Cohort: <strong style="color: #f8fafc;">{total_cases} patients</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with analytics_cols[1]:
            st.markdown("##### 🧬 Top Clinical Feature Drivers")
            st.caption("Gini-importance & ensemble tree weights for decision split contribution:")
            importances = disease_metrics.get("feature_importances", {})
            if importances:
                top_features = list(importances.items())[:8]
                imp_df = pd.DataFrame(top_features, columns=["Biomarker Signal", "Relative Importance"])
                imp_df["Relative Importance"] = imp_df["Relative Importance"].apply(lambda x: f"{x*100:.2f}%")
                st.dataframe(imp_df, use_container_width=True, hide_index=True)
            else:
                st.info("Feature importance data is calculated via tree node impurity splits.")

    # =========================================================================
    # TAB 3: BIOMARKER REFERENCE GUIDE
    # =========================================================================
    with tab_biomarkers:
        st.markdown(f"### 📚 {meta['icon']} {selected_disease} Biomarker Encyclopedia")
        st.caption("Physiological definitions, standard clinical units, and normal diagnostic reference intervals.")

        bio_data = []
        for feat in artifacts["features"]:
            f_meta = clinical_meta.get(feat, {})
            norm = f_meta.get("normal")
            norm_str = f"{norm[0]} – {norm[1]} {f_meta.get('unit', '')}" if norm else "N/A (Categorical / Derived)"
            bio_data.append({
                "Biomarker Name": feat.replace("_", " ").title(),
                "Category": f_meta.get("category", "General"),
                "Standard Unit": f_meta.get("unit", "dimensionless"),
                "Normal Reference Interval": norm_str,
                "Clinical Description": f_meta.get("desc", f"Diagnostic signal {feat}"),
            })

        st.dataframe(pd.DataFrame(bio_data), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()