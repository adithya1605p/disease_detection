"""Presentation-ready Streamlit interface for the multi-disease models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

from config import DISEASE_CONFIG


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

DISEASE_META = {
    "Diabetes": {
        "icon": "🩸",
        "accent": "#ff7a66",
        "description": "Estimate diabetes risk from glucose, body composition, and age-related indicators.",
        "accuracy": "74.0%",
    },
    "Heart Disease": {
        "icon": "❤️",
        "accent": "#ff5d73",
        "description": "Screen cardiovascular risk using blood pressure, cholesterol, and heart-function measurements.",
        "accuracy": "82.0%",
    },
    "Breast Cancer": {
        "icon": "🎗️",
        "accent": "#c084fc",
        "description": "Classify breast tissue measurements into benign or malignant patterns.",
        "accuracy": "96.5%",
    },
    "Kidney Disease": {
        "icon": "🫧",
        "accent": "#38bdf8",
        "description": "Assess kidney health using blood chemistry, urine findings, and clinical indicators.",
        "accuracy": "100%",
    },
    "Liver Disease": {
        "icon": "🧬",
        "accent": "#fbbf24",
        "description": "Evaluate liver-related risk from bilirubin, enzyme, and protein measurements.",
        "accuracy": "61.5%",
    },
    "Stroke": {
        "icon": "🧠",
        "accent": "#a78bfa",
        "description": "Estimate stroke risk using demographic, lifestyle, and cardiovascular factors.",
        "accuracy": "94.8%",
    },
}


st.set_page_config(
    page_title="VitaScan | Multi-Disease Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp {
        background: radial-gradient(circle at 10% 0%, #17233d 0%, #0b1020 35%, #080c16 100%);
        color: #eef2ff;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101a31 0%, #0d1324 100%);
        border-right: 1px solid rgba(148, 163, 184, .14);
    }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.04em; }
    h1 { font-size: clamp(2.4rem, 5vw, 4.8rem) !important; line-height: 1.02 !important; }
    h2 { font-size: 2rem !important; }
    .hero {
        padding: 2.8rem 0 1.5rem;
        max-width: 850px;
    }
    .eyebrow {
        color: #8aa4ff; text-transform: uppercase; letter-spacing: .2em;
        font-size: .76rem; font-weight: 700; margin-bottom: 1rem;
    }
    .hero p { color: #aab6d3; font-size: 1.08rem; max-width: 710px; line-height: 1.7; }
    .glass-card {
        background: rgba(19, 31, 56, .72); border: 1px solid rgba(148, 163, 184, .16);
        border-radius: 20px; padding: 1.25rem; height: 100%;
        box-shadow: 0 18px 50px rgba(0, 0, 0, .15);
    }
    .disease-card {
        background: linear-gradient(145deg, rgba(30, 45, 80, .9), rgba(15, 24, 44, .9));
        border: 1px solid rgba(148, 163, 184, .15); border-radius: 18px;
        padding: 1.15rem; min-height: 150px;
    }
    .disease-icon { font-size: 1.7rem; }
    .disease-name { font-weight: 700; font-size: 1.05rem; margin-top: .65rem; }
    .muted { color: #91a0bf; font-size: .88rem; line-height: 1.5; }
    .result-card { border-radius: 22px; padding: 1.5rem; margin: .75rem 0 1.5rem; }
    .result-title { font-size: 1.75rem; font-family: 'Space Grotesk'; font-weight: 700; }
    .result-subtitle { color: #b4c0d9; margin-top: .35rem; }
    .stButton > button, .stDownloadButton > button {
        border-radius: 10px; border: 1px solid rgba(148, 163, 184, .24);
        font-weight: 600; min-height: 2.6rem;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #738cff, #a06cff); border: 0;
    }
    div[data-testid="stMetric"] {
        background: rgba(24, 37, 66, .7); border: 1px solid rgba(148, 163, 184, .13);
        padding: 1rem; border-radius: 15px;
    }
    .disclaimer { color: #8f9bb5; font-size: .78rem; line-height: 1.55; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assets(disease: str) -> dict[str, Any]:
    """Load the four artifacts produced by train_all.py."""
    safe_name = disease.lower().replace(" ", "_")
    return {
        "model": joblib.load(MODELS_DIR / f"{safe_name}_model.pkl"),
        "scaler": joblib.load(MODELS_DIR / f"{safe_name}_scaler.pkl"),
        "features": joblib.load(MODELS_DIR / f"{safe_name}_features.pkl"),
        "encoders": joblib.load(MODELS_DIR / f"{safe_name}_encoders.pkl"),
    }


@st.cache_data
def load_dataset(disease: str) -> pd.DataFrame:
    """Load the source data for sensible widget defaults and model context."""
    return pd.read_csv(ROOT / DISEASE_CONFIG[disease]["file"])


def display_label(value: Any) -> str:
    text = str(value).strip()
    return "Not recorded" if text.lower() == "nan" else text.replace("_", " ").title()


def numeric_default(series: pd.Series) -> float:
    value = pd.to_numeric(series, errors="coerce").median()
    return float(value) if pd.notna(value) else 0.0


def build_sample_values(disease: str, assets: dict[str, Any]) -> dict[str, Any]:
    data = load_dataset(disease)
    config = DISEASE_CONFIG[disease]
    values: dict[str, Any] = {}
    for feature in assets["features"]:
        encoder = assets["encoders"].get(feature)
        if encoder is not None:
            classes = [item for item in encoder.classes_ if str(item).strip().lower() != "nan"]
            values[feature] = classes[0] if classes else encoder.classes_[0]
        else:
            values[feature] = numeric_default(data[feature])
    for invalid_col in config.get("zero_invalid_cols", []):
        if invalid_col in values and values[invalid_col] == 0:
            values[invalid_col] = numeric_default(data[invalid_col])
    return values


def make_prediction(disease: str, values: dict[str, Any], assets: dict[str, Any]) -> tuple[int, float]:
    encoded = values.copy()
    for feature, encoder in assets["encoders"].items():
        if feature in encoded:
            encoded[feature] = encoder.transform([str(encoded[feature])])[0]
    frame = pd.DataFrame([[encoded[name] for name in assets["features"]]], columns=assets["features"])
    prediction = int(assets["model"].predict(assets["scaler"].transform(frame))[0])
    probabilities = assets["model"].predict_proba(assets["scaler"].transform(frame))[0]
    return prediction, float(max(probabilities))


def render_input(feature: str, disease: str, assets: dict[str, Any], data: pd.DataFrame) -> Any:
    key = f"{disease}_{feature}"
    encoder = assets["encoders"].get(feature)
    if encoder is not None:
        raw_options = list(encoder.classes_)
        return st.selectbox(
            feature.replace("_", " ").title(),
            raw_options,
            format_func=display_label,
            key=key,
        )
    series = pd.to_numeric(data[feature], errors="coerce")
    minimum = float(series.min()) if series.notna().any() else 0.0
    maximum = float(series.max()) if series.notna().any() else 100.0
    default = float(st.session_state.get(key, numeric_default(series)))
    step = 0.01 if (abs(default) < 10 and feature.lower() in {"bmi", "oldpeak", "sg"}) else 1.0
    return st.number_input(
        feature.replace("_", " ").title(),
        min_value=min(0.0, minimum),
        max_value=max(maximum, default, 1.0),
        value=default,
        step=step,
        key=key,
        help=f"Model feature: {feature}",
    )


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## 🩺 VitaScan")
        st.caption("Multi-disease screening workspace")
        st.divider()
        disease = st.selectbox(
            "Choose a screening model",
            list(DISEASE_CONFIG),
            format_func=lambda item: f"{DISEASE_META[item]['icon']}  {item}",
        )
        st.divider()
        st.markdown("### Session guide")
        st.markdown(
            "1. Select a disease model\n"
            "2. Enter patient measurements\n"
            "3. Run the screening estimate\n"
            "4. Export the result for your presentation"
        )
        st.divider()
        st.markdown(
            '<p class="disclaimer"><strong>Educational prototype</strong><br>'
            "This tool is not a diagnosis and should not replace a qualified healthcare professional.</p>",
            unsafe_allow_html=True,
        )
    return disease


def main() -> None:
    disease = render_sidebar()
    meta = DISEASE_META[disease]
    assets = load_assets(disease)
    data = load_dataset(disease)

    st.markdown(
        f'<div class="hero"><div class="eyebrow">AI-powered health screening</div>'
        f"<h1>Understand risk.<br><span style='color:{meta['accent']}'>Act earlier.</span></h1>"
        f"<p>VitaScan turns clinical measurements into an easy-to-understand screening estimate "
        f"for {disease.lower()}. Built for clear demonstrations, rapid exploration, and responsible conversations.</p></div>",
        unsafe_allow_html=True,
    )

    overview_tab, screening_tab, model_tab = st.tabs(["Overview", "Run a screening", "Model details"])
    with overview_tab:
        st.markdown("### A focused view of your health data")
        st.markdown(
            f'<div class="glass-card"><span class="disease-icon">{meta["icon"]}</span>'
            f'<h3>{disease}</h3><p class="muted">{meta["description"]}</p></div>',
            unsafe_allow_html=True,
        )
        st.write("")
        columns = st.columns(3)
        columns[0].metric("Training records", f"{len(data):,}")
        columns[1].metric("Model accuracy", meta["accuracy"])
        columns[2].metric("Input signals", len(assets["features"]))
        st.write("")
        st.markdown("### Available screening models")
        cards = st.columns(3)
        for index, item in enumerate(DISEASE_META):
            with cards[index % 3]:
                item_meta = DISEASE_META[item]
                st.markdown(
                    f'<div class="disease-card"><span class="disease-icon">{item_meta["icon"]}</span>'
                    f'<div class="disease-name">{item}</div><div class="muted">{item_meta["accuracy"]} reported accuracy</div></div>',
                    unsafe_allow_html=True,
                )

    with screening_tab:
        st.markdown(f"### {meta['icon']} {disease} screening")
        st.caption("Use the example profile for a quick demo, or enter your own values below.")
        action_col, note_col = st.columns([1, 3])
        with action_col:
            if st.button("Load example profile", use_container_width=True):
                sample = build_sample_values(disease, assets)
                for feature, value in sample.items():
                    st.session_state[f"{disease}_{feature}"] = value
                st.session_state["example_loaded"] = disease
                st.rerun()
        with note_col:
            if st.session_state.get("example_loaded") == disease:
                st.info("Example profile loaded. You can adjust any value before screening.", icon="✨")

        with st.form(f"screening_form_{disease}"):
            st.markdown("#### Patient measurements")
            input_columns = st.columns(2)
            values: dict[str, Any] = {}
            for index, feature in enumerate(assets["features"]):
                with input_columns[index % 2]:
                    values[feature] = render_input(feature, disease, assets, data)
            submitted = st.form_submit_button("Run screening estimate", type="primary", use_container_width=True)

        if submitted:
            try:
                prediction, confidence = make_prediction(disease, values, assets)
                positive = prediction == 1
                color = "#fb7185" if positive else "#34d399"
                title = "Elevated risk pattern detected" if positive else "Lower risk pattern detected"
                detail = (
                    "The model found a pattern associated with the positive class."
                    if positive
                    else "The model found a pattern associated with the negative class."
                )
                st.markdown(
                    f'<div class="result-card" style="background:linear-gradient(135deg, {color}22, #17233d); border:1px solid {color}88;">'
                    f'<div class="eyebrow" style="color:{color}">Screening result</div>'
                    f'<div class="result-title">{title}</div><div class="result-subtitle">{detail}</div></div>',
                    unsafe_allow_html=True,
                )
                result_cols = st.columns(3)
                result_cols[0].metric("Prediction", "Positive" if positive else "Negative")
                result_cols[1].metric("Model confidence", f"{confidence:.1%}")
                result_cols[2].metric("Model", "Random Forest")
                report = pd.DataFrame([{"Screening": disease, "Prediction": "Positive" if positive else "Negative",
                                        "Confidence": f"{confidence:.1%}", **values}])
                st.download_button(
                    "Download screening report",
                    report.to_csv(index=False).encode("utf-8"),
                    file_name=f"{disease.lower().replace(' ', '_')}_screening.csv",
                    mime="text/csv",
                )
                st.warning("This is a machine-learning screening estimate for demonstration only—not a medical diagnosis.", icon="⚠️")
            except (ValueError, KeyError) as error:
                st.error(f"Unable to process these inputs: {error}. Please review the values and try again.")

    with model_tab:
        st.markdown(f"### How the {disease} model works")
        st.markdown(
            '<div class="glass-card"><b>Random Forest classifier</b><p class="muted">'
            "The model combines many decision trees to identify patterns in the supplied measurements. "
            "Inputs are standardized using the same scaler used during training, and categorical values "
            "are encoded using the saved training encoders.</p></div>",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown("#### Signals used by this model")
        st.dataframe(
            pd.DataFrame({"Feature": assets["features"], "Input type": [
                "Categorical" if feature in assets["encoders"] else "Numeric"
                for feature in assets["features"]
            ]}),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()