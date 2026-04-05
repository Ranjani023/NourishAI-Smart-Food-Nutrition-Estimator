"""
NourishAI — Smart Indian Food & Nutrition Estimator
====================================================
Run: streamlit run app_ui.py
Requirements: streamlit, pandas, numpy, pillow, tensorflow
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import re as _re
from PIL import Image
# ─────────────────────────────────────────────────────────────
# 🔥 OCR + NLP FUNCTIONS (IMPROVED)
# ─────────────────────────────────────────────────────────────
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text(img):
    """Extract text from image using OCR"""
    return pytesseract.image_to_string(img)


def extract_nutrition_info(text):
    """Extract nutrition info using flexible regex"""
    info = {}
    text = text.lower()

    # Calories (handles kcal, calories, energy)
    calories = _re.search(r'(?:energy|calories)?\s*[:\-]?\s*(\d+)\s*(kcal|cal)', text)

    # Protein (handles "5g protein", "protein: 5 g", etc.)
    protein = _re.search(r'protein\s*[:\-]?\s*(\d+)\s*g', text)

    if calories:
        info["calories"] = calories.group(1)

    if protein:
        info["protein"] = protein.group(1)

    return info

# ── TensorFlow safe import ───────────────────────────────────────
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # suppress noisy TF logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NourishAI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════
# GLOBAL CSS  — Dark botanical luxury aesthetic
# Fonts: Cormorant Garamond (editorial serif) + Syne (geometric sans)
# Palette: deep forest ink, warm gold, sage, blush terracotta
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,500;0,700;1,300;1,500&family=Syne:wght@400;500;600;700;800&display=swap');

/* ── CSS Variables ─────────────────────────────────────────────── */
:root {
    --ink:        #0D1A0F;
    --deep:       #111C13;
    --surface:    #162018;
    --panel:      #1B2B1E;
    --border:     rgba(110,180,100,0.12);
    --gold:       #D4A843;
    --gold-soft:  rgba(212,168,67,0.18);
    --sage:       #6EB464;
    --sage-soft:  rgba(110,180,100,0.14);
    --terra:      #C46B4A;
    --terra-soft: rgba(196,107,74,0.15);
    --blush:      #E8C4A8;
    --text-1:     #E8EDE6;
    --text-2:     #9DB09A;
    --text-3:     #5A7A58;
    --glow-g:     0 0 60px rgba(110,180,100,0.1);
}

/* ── Reset & Base ─────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif !important;
    color: var(--text-1) !important;
}

/* Background */
.stApp {
    background-color: var(--ink) !important;
    background-image:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(110,180,100,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 80% at 85% 90%, rgba(212,168,67,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 40% 40% at 60% 40%, rgba(196,107,74,0.04) 0%, transparent 50%);
    min-height: 100vh;
}

/* Noise grain overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.025'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1400px !important; }

/* ── Scrollbar ────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--deep); }
::-webkit-scrollbar-thumb { background: var(--text-3); border-radius: 3px; }

/* ═══════════════════════════════════════════════════════════════
   LOGIN
═══════════════════════════════════════════════════════════════ */
.login-root {
    min-height: 92vh;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
}

.login-root::before {
    content: '';
    position: fixed;
    width: 700px; height: 700px;
    border-radius: 50%;
    border: 1px solid rgba(110,180,100,0.07);
    top: 50%; left: 50%;
    transform: translate(-50%,-50%);
    pointer-events: none;
    animation: spinSlow 60s linear infinite;
}
.login-root::after {
    content: '';
    position: fixed;
    width: 480px; height: 480px;
    border-radius: 50%;
    border: 1px solid rgba(212,168,67,0.06);
    top: 50%; left: 50%;
    transform: translate(-50%,-50%);
    pointer-events: none;
    animation: spinSlow 40s linear infinite reverse;
}

@keyframes spinSlow { to { transform: translate(-50%,-50%) rotate(360deg); } }

.login-glass {
    background: rgba(27,43,30,0.75);
    backdrop-filter: blur(24px) saturate(1.4);
    -webkit-backdrop-filter: blur(24px) saturate(1.4);
    border: 1px solid rgba(110,180,100,0.18);
    border-radius: 32px;
    padding: 60px 52px 52px;
    max-width: 420px;
    width: 100%;
    box-shadow:
        0 0 0 1px rgba(110,180,100,0.06),
        0 32px 80px rgba(0,0,0,0.55),
        inset 0 1px 0 rgba(255,255,255,0.04);
    text-align: center;
    position: relative;
    z-index: 1;
    animation: riseIn 0.7s cubic-bezier(0.22,1,0.36,1) both;
}

@keyframes riseIn {
    from { opacity:0; transform: translateY(24px) scale(0.97); }
    to   { opacity:1; transform: translateY(0) scale(1); }
}

.login-emblem {
    width: 80px; height: 80px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(110,180,100,0.25), rgba(212,168,67,0.1));
    border: 1px solid rgba(110,180,100,0.3);
    display: flex; align-items: center; justify-content: center;
    font-size: 36px;
    margin: 0 auto 20px;
    box-shadow: 0 0 30px rgba(110,180,100,0.2);
    animation: pulse 3s ease-in-out infinite;
}
@keyframes pulse {
    0%,100% { box-shadow: 0 0 30px rgba(110,180,100,0.2); }
    50%      { box-shadow: 0 0 50px rgba(110,180,100,0.35), 0 0 80px rgba(110,180,100,0.1); }
}

.login-brand {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 44px !important;
    font-weight: 700 !important;
    color: var(--text-1) !important;
    letter-spacing: -0.5px;
    line-height: 1;
    margin-bottom: 6px;
}

.login-tagline {
    font-size: 12px;
    color: var(--text-3);
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-bottom: 36px;
    font-weight: 500;
}

.login-rule {
    width: 40px; height: 2px;
    background: linear-gradient(90deg, var(--sage), var(--gold));
    border-radius: 2px;
    margin: 0 auto 32px;
}

.feat-pills {
    display: flex; gap: 8px; justify-content: center;
    flex-wrap: wrap; margin-bottom: 36px;
}
.feat-pill {
    font-size: 11px; font-weight: 600;
    letter-spacing: 0.5px;
    padding: 5px 14px; border-radius: 20px;
    text-transform: uppercase;
}
.fp-sage  { background:var(--sage-soft);  color:var(--sage);  border:1px solid rgba(110,180,100,0.25); }
.fp-gold  { background:var(--gold-soft);  color:var(--gold);  border:1px solid rgba(212,168,67,0.25); }
.fp-terra { background:var(--terra-soft); color:var(--terra); border:1px solid rgba(196,107,74,0.25); }

.demo-hint {
    margin-top: 24px;
    font-size: 11px;
    color: var(--text-3);
    letter-spacing: 0.5px;
}
.demo-hint b { color: var(--gold); font-weight: 600; }

/* ═══════════════════════════════════════════════════════════════
   TOPBAR
═══════════════════════════════════════════════════════════════ */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    background: rgba(27,43,30,0.6);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 16px 28px;
    margin-bottom: 32px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3), var(--glow-g);
}
.tb-brand {
    font-family: 'Cormorant Garamond', serif;
    font-size: 26px; font-weight: 700;
    color: var(--text-1);
    display: flex; align-items: center; gap: 10px;
    letter-spacing: -0.3px;
}
.tb-brand span { color: var(--sage); }
.tb-right {
    display: flex; align-items: center; gap: 16px;
    font-size: 13px; color: var(--text-2);
}
.tb-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--sage);
    box-shadow: 0 0 8px var(--sage);
    animation: blink 2s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ═══════════════════════════════════════════════════════════════
   SECTION HEADERS
═══════════════════════════════════════════════════════════════ */
.sec-hdr {
    display: flex; align-items: baseline; gap: 14px;
    margin-bottom: 6px;
}
.sec-hdr-num {
    font-family: 'Cormorant Garamond', serif;
    font-size: 13px; color: var(--gold);
    font-style: italic; opacity: 0.8;
}
.sec-hdr-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 30px; font-weight: 700;
    color: var(--text-1); letter-spacing: -0.3px;
}
.sec-sub {
    font-size: 13px; color: var(--text-3);
    margin-bottom: 24px; font-weight: 400;
}

/* ═══════════════════════════════════════════════════════════════
   TOOL CARDS
═══════════════════════════════════════════════════════════════ */
.tool-card {
    background: rgba(27,43,30,0.5);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 28px 26px;
    height: 100%;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s;
}
.tool-card::before {
    content: '';
    position: absolute; top: -1px; left: 24px; right: 24px;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(110,180,100,0.4), transparent);
}
.tool-card:hover { border-color: rgba(110,180,100,0.25); }
.card-label {
    font-size: 10px; font-weight: 700; letter-spacing: 2.5px;
    text-transform: uppercase; color: var(--text-3);
    margin-bottom: 18px; display: flex; align-items: center; gap: 8px;
}
.card-label::before {
    content: ''; display: inline-block;
    width: 16px; height: 1px;
    background: var(--sage); opacity: 0.5;
}

/* ═══════════════════════════════════════════════════════════════
   DETECTION RESULT
═══════════════════════════════════════════════════════════════ */
.detect-result {
    background: rgba(110,180,100,0.08);
    border: 1px solid rgba(110,180,100,0.22);
    border-radius: 16px;
    padding: 14px 20px;
    display: flex; align-items: center; gap: 12px;
    margin: 14px 0;
    animation: slideUp 0.4s cubic-bezier(0.22,1,0.36,1) both;
}
@keyframes slideUp {
    from { opacity:0; transform:translateY(10px); }
    to   { opacity:1; transform:translateY(0); }
}
.detect-label {
    font-size: 10px; color: var(--sage); text-transform: uppercase;
    letter-spacing: 2px; font-weight: 600; display: block; margin-bottom: 3px;
}
.detect-name {
    font-family: 'Cormorant Garamond', serif;
    font-size: 22px; font-weight: 700; color: var(--text-1);
    letter-spacing: -0.2px;
}

/* ═══════════════════════════════════════════════════════════════
   PLATE ITEMS
═══════════════════════════════════════════════════════════════ */
.plate-row {
    background: rgba(27,43,30,0.4);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 20px 24px;
    margin-bottom: 12px;
    transition: border-color 0.25s, background 0.25s;
    position: relative;
    overflow: hidden;
}
.plate-row::after {
    content: '';
    position: absolute; left: 0; top: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, var(--sage), var(--gold));
    border-radius: 3px 0 0 3px;
}
.plate-row:hover {
    background: rgba(27,43,30,0.65);
    border-color: rgba(110,180,100,0.22);
}
.pr-name {
    font-family: 'Cormorant Garamond', serif;
    font-size: 20px; font-weight: 700; color: var(--text-1);
    margin-bottom: 10px;
}
.pr-macros {
    display: flex; gap: 20px; flex-wrap: wrap;
    font-size: 12px; color: var(--text-2);
}
.pr-macro { display: flex; align-items: center; gap: 5px; }
.pr-macro b { color: var(--text-1); font-weight: 600; font-size: 14px; }

/* ═══════════════════════════════════════════════════════════════
   MACRO BAR
═══════════════════════════════════════════════════════════════ */
.macro-wrap {
    background: rgba(27,43,30,0.45);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 22px 24px;
    margin-bottom: 20px;
}
.macro-title {
    font-size: 10px; text-transform: uppercase; letter-spacing: 2.5px;
    color: var(--text-3); font-weight: 600; margin-bottom: 14px;
}
.macro-track {
    height: 8px; border-radius: 8px;
    background: rgba(255,255,255,0.04);
    display: flex; overflow: hidden; gap: 2px;
    margin-bottom: 14px;
}
.mb-p { background: var(--sage);  height:100%; border-radius:8px; transition: width 1.2s cubic-bezier(0.22,1,0.36,1); }
.mb-c { background: var(--gold);  height:100%; border-radius:8px; transition: width 1.2s cubic-bezier(0.22,1,0.36,1); }
.mb-f { background: var(--terra); height:100%; border-radius:8px; transition: width 1.2s cubic-bezier(0.22,1,0.36,1); }
.macro-legend { display: flex; gap: 20px; flex-wrap: wrap; }
.ml-item { display: flex; align-items: center; gap: 7px; font-size: 12px; color: var(--text-2); }
.ml-dot { width:8px; height:8px; border-radius:50%; }
.ml-pct { font-weight: 700; color: var(--text-1); }

/* ═══════════════════════════════════════════════════════════════
   FEEDBACK BANNERS
═══════════════════════════════════════════════════════════════ */
.fb-banner {
    border-radius: 16px; padding: 18px 22px;
    display: flex; align-items: flex-start; gap: 14px;
    margin-bottom: 12px;
    border: 1px solid;
    animation: slideUp 0.4s cubic-bezier(0.22,1,0.36,1) both;
}
.fb-icon { font-size: 22px; flex-shrink: 0; margin-top: 1px; }
.fb-text { font-size: 14px; line-height: 1.6; }
.fb-text b { font-weight: 700; display: block; margin-bottom: 3px; }

.fb-good   { background:rgba(110,180,100,0.08); border-color:rgba(110,180,100,0.25); color:#A8D4A0; }
.fb-warn   { background:rgba(212,168,67,0.08);  border-color:rgba(212,168,67,0.25);  color:#D4B870; }
.fb-danger { background:rgba(196,107,74,0.08);  border-color:rgba(196,107,74,0.25);  color:#D4907A; }

/* ═══════════════════════════════════════════════════════════════
   INPUT OVERRIDES (dark theme)
═══════════════════════════════════════════════════════════════ */
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
    background: rgba(27,43,30,0.6) !important;
    border: 1px solid rgba(110,180,100,0.2) !important;
    border-radius: 12px !important;
    color: var(--text-1) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 14px !important;
    padding: 12px 16px !important;
    transition: border-color 0.2s !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus {
    border-color: rgba(110,180,100,0.5) !important;
    box-shadow: 0 0 0 3px rgba(110,180,100,0.08) !important;
}
div[data-testid="stTextInput"] label,
div[data-testid="stNumberInput"] label {
    color: var(--text-2) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
}

div[data-testid="stSelectbox"] > div > div {
    background: rgba(27,43,30,0.6) !important;
    border: 1px solid rgba(110,180,100,0.2) !important;
    border-radius: 12px !important;
    color: var(--text-1) !important;
    font-family: 'Syne', sans-serif !important;
}

/* Button */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, rgba(110,180,100,0.15), rgba(212,168,67,0.12)) !important;
    border: 1px solid rgba(110,180,100,0.3) !important;
    color: var(--sage) !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s !important;
    padding: 10px 24px !important;
    width: 100% !important;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, rgba(110,180,100,0.25), rgba(212,168,67,0.18)) !important;
    border-color: rgba(110,180,100,0.5) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(110,180,100,0.15) !important;
    color: #C8EDBE !important;
}

/* Slider */
div[data-testid="stSlider"] { padding-top: 0 !important; }
div[data-testid="stSlider"] label { color: var(--text-2) !important; font-size:11px !important; }

/* Alerts */
div[data-testid="stAlert"] {
    background: rgba(27,43,30,0.6) !important;
    border-color: rgba(110,180,100,0.25) !important;
    border-radius: 14px !important;
    color: var(--text-1) !important;
    font-family: 'Syne', sans-serif !important;
}

/* File uploader */
div[data-testid="stFileUploader"] {
    background: rgba(27,43,30,0.4) !important;
    border: 1px dashed rgba(110,180,100,0.25) !important;
    border-radius: 16px !important;
}
div[data-testid="stFileUploader"]:hover {
    border-color: rgba(110,180,100,0.45) !important;
}
div[data-testid="stFileUploader"] label { color: var(--text-2) !important; }
div[data-testid="stFileUploader"] p { color: var(--text-3) !important; font-family: 'Syne', sans-serif !important; }

/* Metric */
div[data-testid="stMetric"] {
    background: rgba(27,43,30,0.5) !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    padding: 18px 20px !important;
}
div[data-testid="stMetric"] label { color: var(--text-3) !important; font-size:11px !important; letter-spacing:1px !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 28px !important; color: var(--text-1) !important;
}

hr { border-color: rgba(110,180,100,0.1) !important; margin: 32px 0 !important; }

div[data-testid="stForm"] {
    border: none !important;
    padding: 0 !important;
    background: transparent !important;
}

div[data-testid="stSpinner"] p { color: var(--text-2) !important; font-family:'Syne',sans-serif !important; }
div[data-testid="stImage"] img { border-radius: 16px !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# CONFIG  — single source of truth for file paths
# ═══════════════════════════════════════════════════════════════════
CSV_FILE     = "Indian_Food_Nutrition_Processed.csv"
IMAGE_FOLDER = "archive (1)"

# ═══════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════
_defaults = {
    "logged_in": False,
    "username":  "",
    "plate":     {},
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════════
# MODEL LOADING  — robust multi-strategy loader
# Tries every known combination so version mismatches don't block load
# ═══════════════════════════════════════════════════════════════════
# All candidate paths searched in order
_MODEL_CANDIDATES = [
    "food_model.h5",
    "best_food_model.keras",
    "best_food_model.h5",
    "food_model.keras",
    "clean_food_model.h5",
]

@st.cache_resource
def load_ai_model():
    if not TF_AVAILABLE:
        st.error("❌ TensorFlow not installed. Run: pip install tensorflow")
        return None

    # Find the first model file that exists on disk
    found_path = None
    for candidate in _MODEL_CANDIDATES:
        if os.path.exists(candidate): # this os module helps to check if the file exists in system or not
            found_path = candidate
            break # if the file is found then stop checking 

    if found_path is None:
        st.error(
            "❌ No model file found. Expected one of: "
            + ", ".join(_MODEL_CANDIDATES)
            + " — place it in the same folder as app.py"
        )
        return None

    # Try loading with progressively more permissive options
    strategies = [
        # Strategy 1 — standard load, no compilation
        lambda p: tf.keras.models.load_model(p, compile=False),
        # Strategy 2 — disable safe_mode (handles quantization_config etc.)
        lambda p: tf.keras.models.load_model(p, compile=False, safe_mode=False),
        # Strategy 3 — custom_objects fallback for older saved models
        lambda p: tf.keras.models.load_model(
            p, compile=False, safe_mode=False,
            custom_objects={"preprocess_input": preprocess_input}
        ),
        # Strategy 4 — TFSMLayer for TF SavedModel format
        lambda p: tf.saved_model.load(p),
    ]

    last_error = None
    for i, strategy in enumerate(strategies, 1):
        try:
            m = strategy(found_path)
            st.success(f"✅ Model loaded: `{found_path}`")
            return m
        except Exception as e:
            last_error = e
            continue

    # All strategies failed — show detailed error to help diagnose
    st.error(
        f"❌ Could not load `{found_path}` after trying {len(strategies)} methods.\n\n"
        f"**Last error:** `{last_error}`\n\n"
        "**Common fixes:**\n"
        "- Retrain the model in Colab and re-download `food_model.h5`\n"
        "- Make sure TensorFlow versions match between Colab and local machine\n"
        "  (run `pip install tensorflow==2.15.0` to pin a version)\n"
        "- Check the file isn't corrupted (size should be > 10 MB)"
    )
    return None

model = load_ai_model()

# ═══════════════════════════════════════════════════════════════════
# cleaning and preparing class names 
# ═══════════════════════════════════════════════════════════════════
def _normalize_class(raw: str) -> str:
    """'Kofta-Resized' → 'Kofta',  'Pani-Puri-resize' → 'Pani Puri'"""
    name = _re.sub(r'[-_]re?sized?$', '', raw, flags=_re.IGNORECASE)
    name = name.replace("-", " ").replace("_", " ")
    return " ".join(name.split()).title()

if os.path.exists(IMAGE_FOLDER):#check wheather the dataset is present or not 
    RAW_CLASSES = sorted(os.listdir(IMAGE_FOLDER))
    CLASS_NAMES = [_normalize_class(c) for c in RAW_CLASSES]
else:
    # Fallback so the app runs without the image folder present
    RAW_CLASSES = [
        'Biryani-resize', 'Chole-Bhature-Resized', 'dabeli-resize',
        'dal_resized', 'dhokla-resize', 'dosa_resized', 'Jalebi-Resize',
        'kathi-resize', 'Kofta-Resized', 'Naan-Resized', 'pakora-resize',
        'Pani-Puri-resize', 'Paneer-Tikka-resized', 'Pav-Bhaji-Resized',
        'Vadapav-Resized',
    ]
    CLASS_NAMES = [_normalize_class(c) for c in RAW_CLASSES]

# ═══════════════════════════════════════════════════════════════════
# CSV LOOKUP MAP  — model class name → exact CSV "Dish Name"
# ═══════════════════════════════════════════════════════════════════
_CSV_LOOKUP: dict = {
    "Biryani":       "biryani",
    "Chole Bhature": "chole bhature",
    "Dabeli":        "dabeli",
    "Dal":           "washed moong dal (dhuli moong ki dal)",
    "Dhokla":        "dhokla",
    "Dosa":          "plain dosa",
    "Jalebi":        "jalebi",
    "Kathi":         "kathi roll",
    "Kofta":         "paneer kofta curry",
    "Naan":          "naan",
    "Pakora":        "mixed vegetable pakora/pakoda",
    "Pani Puri":     "pani puri",
    "Paneer Tikka":  "paneer tikka",
    "Pav Bhaji":     "pav bhaji",
    "Vadapav":       "vada pav",
}

# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════
@st.cache_data
def load_nutrition():
    if not os.path.exists(CSV_FILE):
        return pd.DataFrame()
    df = pd.read_csv(CSV_FILE, encoding="latin1")#df=dataframe ; opens csv and load it as table
    df.columns = df.columns.str.replace("Â", "", regex=False).str.strip()
    df = df.rename(columns={
        "Dish Name":         "name",
        "Calories (kcal)":   "calories",
        "Carbohydrates (g)": "carbs",
        "Protein (g)":       "protein",
        "Fats (g)":          "fat",
        "Free Sugar (g)":    "sugar",
        "Fibre (g)":         "fiber",
        "Sodium (mg)":       "sodium",
        "Calcium (mg)":      "calcium",
        "Iron (mg)":         "iron",
        "Vitamin C (mg)":    "vitamin_c",
        "Folate (µg)":       "folate",
    })
    df["name"] = df["name"].fillna("").str.lower().str.strip()
    for col in ["calories", "carbs", "protein", "fat", "sugar",
                "fiber", "sodium", "calcium", "iron", "vitamin_c", "folate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df

nutrition_df = load_nutrition()

# ═══════════════════════════════════════════════════════════════════
# INFERENCE HELPERS
# ═══════════════════════════════════════════════════════════════════
def predict_food(img: Image.Image) -> tuple: #takes an  image 
    """
    Run MobileNetV2 classifier.
    CRITICAL: uses preprocess_input (scales to [-1, 1]) — matches training.
    Handles both Keras Sequential/Functional models and TF SavedModels.
    Returns (class_name, confidence_pct).
    """
    if model is None or not CLASS_NAMES:
        return "Unknown", 0.0

    # Preprocess — must match training (preprocess_input, NOT /255.0)
    arr = np.array(img.resize((224, 224))).astype("float32")
    arr = preprocess_input(arr)       # scales pixels to [-1, 1]
    arr = np.expand_dims(arr, 0)      # shape: (1, 224, 224, 3)

    # Support both Keras model (.predict) and TF SavedModel (__call__)
    try:
        raw = model.predict(arr, verbose=0)
    except AttributeError:
        raw = model(arr, training=False).numpy()

    pred = np.array(raw).flatten()    # ensure 1-D array of class probs
    idx  = int(np.argmax(pred))
    conf = float(pred[idx]) * 100
    return CLASS_NAMES[idx], conf


def get_nutrition(food_name: str):
    """
    Resolve a class name to a CSV row.
    Priority: verified lookup map → exact match → partial match.
    """
    if nutrition_df.empty:
        return None

    # 1. Verified lookup map (all 15 model classes)
    csv_key = _CSV_LOOKUP.get(food_name.strip())
    if csv_key:
        row = nutrition_df[nutrition_df["name"] == csv_key]
        if not row.empty:
            return row.iloc[0]

    # 2. Exact lowercase match (handles search-box selections)
    key   = food_name.lower().strip()
    exact = nutrition_df[nutrition_df["name"] == key]
    if not exact.empty:
        return exact.iloc[0]

    # 3. Partial text search (last resort)
    partial = nutrition_df[nutrition_df["name"].str.contains(
        key, na=False, regex=False)]
    if not partial.empty:
        return partial.iloc[0]

    return None


# ═══════════════════════════════════════════════════════════════════
# ░░ PAGES ░░
# ═══════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────
# LOGIN
# ──────────────────────────────────────────────────────────────────
def login_page():
    _, mid, _ = st.columns([1, 1.05, 1])
    with mid:
        st.markdown("""
        <div class="login-glass">
            <div class="login-emblem">🌿</div>
            <div class="login-brand">NourishAI</div>
            <div class="login-tagline">Smart Indian Food Intelligence</div>
            <div class="login-rule"></div>
            <div class="feat-pills">
                <span class="feat-pill fp-sage">🤖 AI Vision</span>
                <span class="feat-pill fp-gold">🍛 15 Indian Foods</span>
                <span class="feat-pill fp-terra">📊 Diet Advisor</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        with st.form("lf", clear_on_submit=False):
            u = st.text_input("Username", placeholder="Enter username")
            p = st.text_input("Password", type="password", placeholder="Enter password")
            if st.form_submit_button("Sign In →"):
                if u == "admin" and p == "admin":
                    st.session_state.logged_in = True
                    st.session_state.username  = u
                    st.rerun()
                else:
                    st.error("Invalid credentials — use admin / admin")

        st.markdown("""
        <div class="demo-hint">Demo · <b>admin</b> / <b>admin</b></div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# TOP BAR
# ──────────────────────────────────────────────────────────────────
def topbar():
    c1, c2 = st.columns([4, 1])
    with c1:
        st.markdown(f"""
        <div class="topbar">
            <div class="tb-brand">🌿 <span>Nourish</span>AI</div>
            <div class="tb-right">
                <div class="tb-dot"></div>
                <span>{st.session_state.username}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        if st.button("Sign Out"):
            for k in _defaults:
                st.session_state[k] = _defaults[k]
            st.rerun()


# ──────────────────────────────────────────────────────────────────
# ADD FOOD SECTION
# ──────────────────────────────────────────────────────────────────
def add_food_section():
    st.markdown("""
    <div class="sec-hdr">
        <span class="sec-hdr-num">01 —</span>
        <span class="sec-hdr-title">Identify Your Food</span>
    </div>
    <div class="sec-sub">Upload a photo for AI recognition, or search the nutrition database by name</div>
    """, unsafe_allow_html=True)

    # ✅ CREATE COLUMNS
    col1, col2 = st.columns(2, gap="large")

    # ==========================================================
    # 📷 AI IMAGE + OCR SECTION
    # ==========================================================
    with col1:
        st.markdown('<div class="tool-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">📷 AI Vision · MobileNetV2</div>',
                    unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload food photo",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="uploader"
        )

        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, use_container_width=True)

            # ==============================
            # 🔥 OCR TEXT EXTRACTION
            # ==============================
            text = extract_text(img)

            st.markdown("### 📝 Extracted Text")
            if text.strip():
                st.code(text)
            else:
                st.warning("No text detected in image")

            # ==============================
            # 🧠 NLP PARSING
            # ==============================
            parsed = extract_nutrition_info(text)

            st.markdown("### 🧠 Parsed Nutrition Info")
            st.write(parsed)

            # ==============================
            # 🤖 MODEL PREDICTION
            # ==============================
            if model is None:
                st.markdown("""
                <div class="fb-banner fb-warn">
                    <div class="fb-icon">⚠️</div>
                    <div class="fb-text"><b>Model not loaded</b>
                    Place food_model.h5 in the same folder.</div>
                </div>""", unsafe_allow_html=True)

            else:
                with st.spinner("Analysing with MobileNetV2…"):
                    food, conf = predict_food(img)

                st.markdown(f"""
                <div class="detect-result">
                    <span style="font-size:28px">🎯</span>
                    <div>
                        <span class="detect-label">AI Detected · {conf:.1f}% confidence</span>
                        <span class="detect-name">{food}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ==============================
                # ➕ ADD TO PLATE
                # ==============================
                row = get_nutrition(food)

                if st.button("➕ Add to Plate", key="add_img"):
                    if row is not None:
                        st.session_state.plate[food] = {
                            "row": row,
                            "qty": 100
                        }
                        st.success(f"✅ {food} added to your plate")
                        st.rerun()
                    else:
                        st.warning(f"Nutrition data for '{food}' not found")

        else:
            st.markdown("""
            <div style="text-align:center;padding:40px 20px;color:var(--text-3);">
                <div style="font-size:40px;margin-bottom:12px;opacity:0.4;">📷</div>
                <div style="font-size:13px;">Drop a food photo here</div>
                <div style="font-size:11px;margin-top:6px;opacity:0.6;">
                    JPG, PNG · Model trained on 15 Indian dishes
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================================
    # 🔍 TEXT SEARCH SECTION
    # ==========================================================
    with col2:
        st.markdown('<div class="tool-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">⌨️ Search Database</div>',
                    unsafe_allow_html=True)

        search = st.text_input(
            "Search food", placeholder="e.g. Biryani, Dal Makhani, Dosa…")

        if search and not nutrition_df.empty:
            matches = nutrition_df[
                nutrition_df["name"].str.contains(search.lower(), na=False)]
        else:
            matches = pd.DataFrame()

        if not matches.empty:
            selected = st.selectbox(
                "Select food", matches["name"].str.title().unique())

            qty = st.number_input(
                "Quantity (grams)", min_value=10, max_value=1000,
                value=100, step=10)

            row = get_nutrition(selected)

            if row is not None:
                f = qty / 100
                cal = f * float(row.get("calories", 0))
                pr = f * float(row.get("protein", 0))
                cb = f * float(row.get("carbs", 0))
                ft = f * float(row.get("fat", 0))

                st.markdown(f"""
                🔥 {cal:.0f} kcal  
                💪 {pr:.1f}g protein  
                🌾 {cb:.1f}g carbs  
                🥑 {ft:.1f}g fat
                """)

            if st.button("➕ Add to Plate", key="add_search"):
                if row is not None:
                    st.session_state.plate[selected.title()] = {
                        "row": row,
                        "qty": qty
                    }
                    st.success(f"✅ {selected.title()} added!")
                    st.rerun()
                else:
                    st.warning("Data not found")

        elif search:
            st.warning("No matches found")

        else:
            st.info("Start typing to search food")

        st.markdown('</div>', unsafe_allow_html=True)
# ──────────────────────────────────────────────────────────────────
# PLATE + DIET ANALYSIS
# ──────────────────────────────────────────────────────────────────
def plate_section():
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div class="sec-hdr">
        <span class="sec-hdr-num">02 —</span>
        <span class="sec-hdr-title">Your Plate</span>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.plate:
        st.markdown("""
        <div style="text-align:center;padding:60px 40px;color:var(--text-3);
             background:rgba(27,43,30,0.3);
             border:1px dashed rgba(110,180,100,0.15);border-radius:24px;">
            <div style="font-size:48px;margin-bottom:16px;opacity:0.3;">🍽</div>
            <div style="font-size:15px;font-family:'Cormorant Garamond',serif;
                 color:var(--text-2);">Your plate is empty</div>
            <div style="font-size:12px;margin-top:8px;opacity:0.6;">
                Add foods using the section above
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    totals = {"cal": 0, "protein": 0, "carbs": 0,
              "fat": 0, "fiber": 0, "sodium": 0}

    for food_name, item in list(st.session_state.plate.items()):
        row = item["row"]
        c1, c2, c3 = st.columns([3, 2, 1])

        with c1:
            qty = st.slider(
                f"{food_name}", 10, 500, item["qty"],
                key=f"sl_{food_name}", label_visibility="visible"
            )
            st.session_state.plate[food_name]["qty"] = qty

        f   = qty / 100
        cal = f * float(row.get("calories", 0) or 0)
        pr  = f * float(row.get("protein",  0) or 0)
        cb  = f * float(row.get("carbs",    0) or 0)
        ft  = f * float(row.get("fat",      0) or 0)
        fb  = f * float(row.get("fiber",    0) or 0)
        na  = f * float(row.get("sodium",   0) or 0)

        with c2:
            st.markdown(f"""
            <div class="plate-row" style="margin-top:8px;">
                <div class="pr-name">{food_name}</div>
                <div class="pr-macros">
                    <span class="pr-macro">🔥 <b>{cal:.0f}</b> kcal</span>
                    <span class="pr-macro">💪 <b>{pr:.1f}g</b> prot</span>
                    <span class="pr-macro">🌾 <b>{cb:.1f}g</b> carbs</span>
                    <span class="pr-macro">🥑 <b>{ft:.1f}g</b> fat</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown("<div style='padding-top:18px'></div>",
                        unsafe_allow_html=True)
            if st.button("✕", key=f"rm_{food_name}",
                         help=f"Remove {food_name}"):
                del st.session_state.plate[food_name]
                st.rerun()

        totals["cal"]     += cal
        totals["protein"] += pr
        totals["carbs"]   += cb
        totals["fat"]     += ft
        totals["fiber"]   += fb
        totals["sodium"]  += na

    # ── MEAL SUMMARY ────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div class="sec-hdr">
        <span class="sec-hdr-num">03 —</span>
        <span class="sec-hdr-title">Meal Summary</span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("🔥 Calories", f"{totals['cal']:.0f}",  "kcal")
    with c2: st.metric("💪 Protein",  f"{totals['protein']:.1f}", "g")
    with c3: st.metric("🌾 Carbs",    f"{totals['carbs']:.1f}",   "g")
    with c4: st.metric("🥑 Fat",      f"{totals['fat']:.1f}",     "g")

    # Macro distribution bar
    total_cal_from_macros = (
        totals["protein"] * 4 + totals["carbs"] * 4 + totals["fat"] * 9)
    if total_cal_from_macros > 0:
        p_pct = round(totals["protein"] * 4 / total_cal_from_macros * 100)
        c_pct = round(totals["carbs"]   * 4 / total_cal_from_macros * 100)
        f_pct = 100 - p_pct - c_pct
        st.markdown(f"""
        <div class="macro-wrap">
            <div class="macro-title">Calorie Distribution by Macronutrient</div>
            <div class="macro-track">
                <div class="mb-p" style="width:{p_pct}%"></div>
                <div class="mb-c" style="width:{c_pct}%"></div>
                <div class="mb-f" style="width:{f_pct}%"></div>
            </div>
            <div class="macro-legend">
                <span class="ml-item">
                    <span class="ml-dot" style="background:var(--sage)"></span>
                    Protein <span class="ml-pct">&nbsp;{p_pct}%</span>
                </span>
                <span class="ml-item">
                    <span class="ml-dot" style="background:var(--gold)"></span>
                    Carbs <span class="ml-pct">&nbsp;{c_pct}%</span>
                </span>
                <span class="ml-item">
                    <span class="ml-dot" style="background:var(--terra)"></span>
                    Fat <span class="ml-pct">&nbsp;{f_pct}%</span>
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── AI DIET FEEDBACK ────────────────────────────────────────
    st.markdown("""
    <div class="sec-hdr" style="margin-top:24px;">
        <span class="sec-hdr-num">04 —</span>
        <span class="sec-hdr-title">Diet Intelligence</span>
    </div>
    """, unsafe_allow_html=True)

    cal = totals["cal"]
    if cal > 900:
        st.markdown("""
        <div class="fb-banner fb-danger"><div class="fb-icon">⚠️</div>
        <div class="fb-text"><b>High Calorie Meal</b>
        This meal exceeds 900 kcal. Consider swapping fried items for grilled
        alternatives and adding more leafy vegetables to lower calorie density.
        </div></div>""", unsafe_allow_html=True)
    elif cal < 400:
        st.markdown("""
        <div class="fb-banner fb-warn"><div class="fb-icon">💛</div>
        <div class="fb-text"><b>Low Calorie Meal</b>
        Under 400 kcal may not sustain energy needs. Add a protein source
        (paneer, dal, eggs) and complex carbs (brown rice, roti) for satiety.
        </div></div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="fb-banner fb-good"><div class="fb-icon">✅</div>
        <div class="fb-text"><b>Well-Balanced Meal</b>
        Calorie range looks great (400–900 kcal).
        Stay hydrated — aim for 2–3 glasses of water after this meal.
        </div></div>""", unsafe_allow_html=True)

    if totals["protein"] < 15:
        st.markdown("""
        <div class="fb-banner fb-warn"><div class="fb-icon">💪</div>
        <div class="fb-text"><b>Protein Deficit</b>
        Less than 15g protein detected. Add dal, paneer, chickpeas, or eggs
        to support muscle repair and keep you fuller longer.
        </div></div>""", unsafe_allow_html=True)

    if totals["sodium"] > 800:
        st.markdown("""
        <div class="fb-banner fb-danger"><div class="fb-icon">🧂</div>
        <div class="fb-text"><b>High Sodium Alert</b>
        This meal contains elevated sodium (&gt;800 mg).
        Reduce processed/pickled sides and limit added salt.
        </div></div>""", unsafe_allow_html=True)

    if totals["fiber"] >= 8:
        st.markdown("""
        <div class="fb-banner fb-good"><div class="fb-icon">🌾</div>
        <div class="fb-text"><b>Good Fibre Intake</b>
        Excellent fibre content — supports gut health and steady blood sugar.
        </div></div>""", unsafe_allow_html=True)

    # ── CLEAR PLATE ─────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑 Clear Entire Plate"):
        st.session_state.plate = {}
        st.rerun()

    return totals


# ═══════════════════════════════════════════════════════════════════
# HOME PAGE
# ═══════════════════════════════════════════════════════════════════
def home_page():
    topbar()
    add_food_section()
    plate_section()


# ═══════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    login_page()
else:
    home_page()