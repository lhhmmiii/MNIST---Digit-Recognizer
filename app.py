"""
app.py — MNIST Digit Recognizer · Streamlit Demo
==================================================
Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path

import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from utils.image_processing import (
    array_to_display_image,
    preprocess_canvas,
    preprocess_upload,
)
from utils.inference import load_model, predict

# ─── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="MNIST Digit Recognizer",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e0e0e0;
    }

    /* ── Header ── */
    .hero-title {
        font-size: 3rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.25rem;
    }
    .hero-sub {
        text-align: center;
        font-size: 1rem;
        color: #9ca3af;
        margin-bottom: 2rem;
    }

    /* ── Prediction card ── */
    .pred-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(167,139,250,0.4);
        border-radius: 1.25rem;
        padding: 2rem;
        text-align: center;
        backdrop-filter: blur(12px);
        transition: all 0.3s ease;
    }
    .pred-digit {
        font-size: 7rem;
        font-weight: 900;
        background: linear-gradient(135deg, #a78bfa, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1;
    }
    .pred-label {
        font-size: 1.2rem;
        font-weight: 600;
        color: #c4b5fd;
        margin-top: 0.5rem;
    }
    .confidence-badge {
        display: inline-block;
        background: linear-gradient(90deg, #34d399, #059669);
        color: white;
        border-radius: 999px;
        padding: 0.3rem 1.2rem;
        font-weight: 700;
        font-size: 1.1rem;
        margin-top: 0.8rem;
    }
    .timing-badge {
        display: inline-block;
        background: rgba(96,165,250,0.2);
        color: #93c5fd;
        border-radius: 999px;
        padding: 0.25rem 1rem;
        font-size: 0.85rem;
        margin-top: 0.4rem;
    }

    /* ── Section headers ── */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #a78bfa;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.85);
        border-right: 1px solid rgba(167,139,250,0.2);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(90deg, #7c3aed, #2563eb);
        color: white !important;
        border: none;
        border-radius: 0.75rem;
        padding: 0.6rem 1.8rem;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.04em;
        transition: opacity 0.2s ease, transform 0.15s ease;
        width: 100%;
    }
    .stButton > button:hover {
        opacity: 0.88;
        transform: translateY(-1px);
    }

    /* ── Error banner ── */
    .error-banner {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239,68,68,0.5);
        border-radius: 0.75rem;
        padding: 1rem 1.5rem;
        color: #fca5a5;
        font-size: 0.95rem;
    }

    /* ── Probability bar colours ── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #a78bfa, #60a5fa);
    }

    /* ── Upload zone ── */
    [data-testid="stFileUploader"] label {
        color: #c4b5fd !important;
        font-weight: 600;
    }

    /* canvas container */
    .canvas-container {
        border: 2px solid rgba(167,139,250,0.5);
        border-radius: 0.75rem;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Model path ───────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "model" / "mnist_cnn.h5"


# ─── Load model (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    """Load the trained model once and cache it for the session."""
    return load_model(MODEL_PATH)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️  Settings")

    input_mode = st.radio(
        "Input method",
        ["✏️ Draw on canvas", "📤 Upload image"],
        index=0,
    )

    st.markdown("---")
    st.markdown("## 🏗️  Model Info")
    st.markdown(
        """
        **Architecture:** CNN  
        **Input:** 28 × 28 grayscale  
        **Output:** 10-class softmax  

        | Layer | Config |
        |-------|--------|
        | Conv2D | 64 filters, 3×3, ReLU |
        | MaxPool | 2×2, stride 2 |
        | Conv2D | 32 filters, 3×3, ReLU |
        | MaxPool | 2×2, stride 2 |
        | Flatten | — |
        | Dense | 10, Softmax |

        **Optimizer:** Adam  
        **Loss:** Categorical cross-entropy  
        **Expected accuracy:** ~99 %
        """
    )

    st.markdown("---")
    st.markdown("## 📚  About")
    st.markdown(
        """
        Trained on the classic
        [MNIST dataset](http://yann.lecun.com/exdb/mnist/) — 60 000 training
        images, 10 000 test images of handwritten digits 0–9.
        """
    )

# ─── Hero header ─────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">✏️ MNIST Digit Recognizer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Draw or upload a handwritten digit — the CNN predicts it instantly.</p>',
    unsafe_allow_html=True,
)

# ─── Check model exists ──────────────────────────────────────────────────────
model_missing = not MODEL_PATH.exists()

if model_missing:
    st.markdown(
        """
        <div class="error-banner">
        ⚠️  <strong>Model file not found.</strong><br><br>
        Please train the model first by running in your terminal:<br>
        <code>python train.py</code><br><br>
        This will download MNIST (~11 MB), train the CNN for 5 epochs, and
        save the weights to <code>model/mnist_cnn.h5</code>.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ─── Load model ───────────────────────────────────────────────────────────────
with st.spinner("Loading model …"):
    try:
        model = get_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        st.stop()

# ─── Layout ──────────────────────────────────────────────────────────────────
col_input, col_result = st.columns([1, 1], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN — Input
# ══════════════════════════════════════════════════════════════════════════════
with col_input:
    result = None          # will hold the inference dict
    processed_img = None   # will hold the 28×28 float array for display

    # ── Canvas mode ─────────────────────────────────────────────────────────
    if "Draw" in input_mode:
        st.markdown('<p class="section-title">🖊️ Draw your digit here</p>', unsafe_allow_html=True)
        st.caption("Use your mouse (or touch) to draw a single digit in the box below.")

        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=18,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            run_btn = st.button("🔍 Predict", key="predict_canvas")
        with btn_col2:
            clear_btn = st.button("🗑️ Clear", key="clear_canvas")

        if run_btn and canvas_result.image_data is not None:
            try:
                x = preprocess_canvas(canvas_result.image_data)
                processed_img = x[0, :, :, 0]
                result = predict(model, x)
            except ValueError as exc:
                st.warning(str(exc))
            except Exception as exc:
                st.error(f"Inference error: {exc}")
                st.code(traceback.format_exc(), language="python")

    # ── Upload mode ──────────────────────────────────────────────────────────
    else:
        st.markdown('<p class="section-title">📤 Upload an image</p>', unsafe_allow_html=True)
        st.caption("Supported formats: PNG, JPEG, BMP, TIFF, WebP")

        uploaded = st.file_uploader(
            "Choose an image …",
            type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
            label_visibility="collapsed",
        )

        if uploaded is not None:
            st.image(uploaded, caption="Uploaded image", width=200)

            if st.button("🔍 Predict", key="predict_upload"):
                try:
                    x = preprocess_upload(uploaded.read())
                    processed_img = x[0, :, :, 0]
                    result = predict(model, x)
                except Exception as exc:
                    st.error(f"Processing error: {exc}")
                    st.code(traceback.format_exc(), language="python")

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — Results
# ══════════════════════════════════════════════════════════════════════════════
with col_result:
    st.markdown('<p class="section-title">🎯 Prediction results</p>', unsafe_allow_html=True)

    if result is None:
        st.markdown(
            """
            <div style="
                background: rgba(255,255,255,0.03);
                border: 1px dashed rgba(167,139,250,0.3);
                border-radius: 1rem;
                padding: 3rem 2rem;
                text-align: center;
                color: #6b7280;
                font-size: 1.1rem;
            ">
                Draw or upload a digit, then click <strong>Predict</strong>
                to see results here.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        digit   = result["predicted_digit"]
        conf    = result["confidence"]
        probs   = result["probabilities"]
        ms      = result["processing_ms"]

        # ── Main prediction card ─────────────────────────────────────────
        st.markdown(
            f"""
            <div class="pred-card">
                <div class="pred-digit">{digit}</div>
                <div class="pred-label">Predicted Digit</div>
                <div class="confidence-badge">
                    {conf * 100:.2f}% confidence
                </div><br>
                <div class="timing-badge">⏱ {ms:.1f} ms</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── 28×28 preview ────────────────────────────────────────────────
        if processed_img is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            preview_col, _ = st.columns([1, 2])
            with preview_col:
                pil_preview = array_to_display_image(processed_img)
                st.image(pil_preview, caption="Processed (28×28)", use_container_width=False)

        # ── Probability bar chart ─────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-title">📊 Class Probabilities</p>', unsafe_allow_html=True)

        for i, prob in enumerate(probs):
            label_col, bar_col, pct_col = st.columns([0.6, 5, 1.2])
            with label_col:
                is_top = i == digit
                colour = "#a78bfa" if is_top else "#9ca3af"
                weight = "700" if is_top else "400"
                st.markdown(
                    f'<span style="color:{colour}; font-weight:{weight};">{i}</span>',
                    unsafe_allow_html=True,
                )
            with bar_col:
                st.progress(float(prob))
            with pct_col:
                pct_colour = "#34d399" if is_top else "#9ca3af"
                st.markdown(
                    f'<span style="color:{pct_colour}; font-size:0.85rem;">'
                    f'{prob * 100:.1f}%</span>',
                    unsafe_allow_html=True,
                )

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align:center; color: #4b5563; font-size:0.8rem;">
        MNIST Digit Recognizer · CNN with TensorFlow / Keras ·
        <a href="http://yann.lecun.com/exdb/mnist/" target="_blank"
           style="color:#7c3aed;">MNIST Dataset</a>
    </div>
    """,
    unsafe_allow_html=True,
)
