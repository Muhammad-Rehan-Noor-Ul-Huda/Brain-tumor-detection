import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuraScan · Brain Tumor AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Class config ───────────────────────────────────────────────────────────────
# IMPORTANT: verify this order matches what was printed at training time
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

CLASS_META = {
    "Glioma": {
        "icon": "⚠️",
        "color": "#ef4444",
        "glow": "rgba(239,68,68,0.35)",
        "desc": "A tumor originating in the glial cells of the brain or spine. Gliomas are the most common primary brain tumors and vary widely in severity.",
        "urgency": "Consult a neuro-oncologist promptly.",
    },
    "Meningioma": {
        "icon": "🔶",
        "color": "#f97316",
        "glow": "rgba(249,115,22,0.35)",
        "desc": "A tumor arising from the meninges — the protective membranes surrounding the brain and spinal cord. Most are benign and slow-growing.",
        "urgency": "Regular monitoring is typically recommended.",
    },
    "No Tumor": {
        "icon": "✅",
        "color": "#22c55e",
        "glow": "rgba(34,197,94,0.35)",
        "desc": "No visible tumor detected in the MRI scan. The tissue appears within normal parameters for AI analysis.",
        "urgency": "Continue routine check-ups as advised by your physician.",
    },
    "Pituitary": {
        "icon": "🔵",
        "color": "#3b82f6",
        "glow": "rgba(59,130,246,0.35)",
        "desc": "A tumor located in the pituitary gland at the base of the brain. Most pituitary tumors are non-cancerous and highly treatable.",
        "urgency": "Endocrinology and neurology review recommended.",
    },
}

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #060a14 !important;
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
}

[data-testid="stSidebar"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
footer, #MainMenu { display: none !important; }

.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stVerticalBlock"] { gap: 0 !important; }

/* HERO */
.hero {
    padding: 64px 60px 52px;
    background:
        radial-gradient(ellipse 80% 60% at 50% -10%, rgba(99,102,241,0.18) 0%, transparent 70%),
        radial-gradient(ellipse 40% 40% at 80% 50%, rgba(59,130,246,0.10) 0%, transparent 60%),
        #060a14;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.hero-inner {
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: center;
    gap: 40px;
    max-width: 1200px;
    margin: 0 auto;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.hero-eyebrow::before {
    content: '';
    display: block;
    width: 28px;
    height: 1px;
    background: #6366f1;
}
.hero-title {
    font-size: clamp(2.2rem, 4.5vw, 3.6rem);
    font-weight: 900;
    line-height: 1.06;
    letter-spacing: -0.03em;
    color: #f8fafc;
    margin-bottom: 18px;
}
.hero-title span {
    background: linear-gradient(135deg, #818cf8 0%, #38bdf8 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1rem;
    color: #94a3b8;
    line-height: 1.7;
    max-width: 520px;
}
.hero-stats {
    display: flex;
    flex-direction: column;
    gap: 14px;
    align-items: flex-end;
}
.stat-pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 14px 20px;
    text-align: right;
    min-width: 130px;
}
.stat-num { font-size: 1.5rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.03em; }
.stat-label { font-size: 0.68rem; color: #64748b; text-transform: uppercase; letter-spacing: 1.5px; }

/* PANEL LABEL */
.panel-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #334155;
    margin-bottom: 16px;
    font-family: 'JetBrains Mono', monospace;
}

/* IDLE STATE */
.idle-zone {
    border: 2px dashed rgba(99,102,241,0.2);
    border-radius: 16px;
    padding: 52px 32px;
    text-align: center;
    background: rgba(99,102,241,0.03);
    margin-top: 16px;
}
.idle-icon { font-size: 2.2rem; margin-bottom: 14px; display: block; opacity: 0.5; }
.idle-text { font-size: 0.88rem; color: #334155; line-height: 1.7; }

/* SCAN META */
.scan-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #334155;
    letter-spacing: 1.5px;
    margin-top: 8px;
}

/* DIAGNOSIS CARD */
.dx-card {
    border-radius: 20px;
    padding: 28px;
    border: 1px solid;
    position: relative;
    overflow: hidden;
    margin-bottom: 16px;
}
.dx-card::before {
    content: '';
    position: absolute;
    inset: 0;
    pointer-events: none;
}
.dx-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 10px;
    opacity: 0.75;
}
.dx-icon { font-size: 2.2rem; margin-bottom: 10px; display: block; }
.dx-label { font-size: 2rem; font-weight: 800; letter-spacing: -0.03em; margin-bottom: 6px; }
.dx-conf { font-size: 0.8rem; color: #475569; font-family: 'JetBrains Mono', monospace; margin-bottom: 18px; }
.dx-desc {
    font-size: 0.88rem; color: #94a3b8; line-height: 1.7;
    border-top: 1px solid rgba(255,255,255,0.06);
    padding-top: 14px; margin-bottom: 10px;
}
.dx-urgency { font-size: 0.8rem; font-weight: 500; opacity: 0.85; }

/* CONFIDENCE BARS */
.conf-block {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 22px;
    margin-bottom: 16px;
}
.conf-title {
    font-size: 0.68rem; letter-spacing: 2px; text-transform: uppercase;
    color: #334155; font-family: 'JetBrains Mono', monospace;
    margin-bottom: 18px; font-weight: 600;
}
.conf-row { margin-bottom: 13px; }
.conf-header { display: flex; justify-content: space-between; margin-bottom: 5px; }
.conf-name { font-size: 0.83rem; }
.conf-pct { font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; color: #475569; }
.conf-track { height: 4px; background: rgba(255,255,255,0.06); border-radius: 99px; overflow: hidden; }
.conf-fill { height: 100%; border-radius: 99px; }

/* DISCLAIMER */
.disclaimer {
    background: rgba(251,191,36,0.04);
    border: 1px solid rgba(251,191,36,0.12);
    border-radius: 12px;
    padding: 14px 18px;
    display: flex;
    gap: 10px;
    align-items: flex-start;
}
.disclaimer-text { font-size: 0.78rem; color: #b45309; line-height: 1.6; color: #fbbf24; opacity: 0.7; }

/* IDLE RESULT */
.result-idle {
    height: 340px;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 20px;
    background: rgba(255,255,255,0.01);
}
.result-idle-inner { text-align: center; }
.result-idle-icon { font-size: 1.8rem; opacity: 0.15; margin-bottom: 10px; }
.result-idle-text { font-size: 0.82rem; color: #1e293b; }

/* HOW IT WORKS */
.how-section {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 60px 72px;
}
.how-title {
    font-size: 0.68rem; letter-spacing: 2.5px; text-transform: uppercase;
    color: #334155; font-family: 'JetBrains Mono', monospace;
    margin-bottom: 24px; font-weight: 600;
}
.how-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
.how-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 22px;
}
.how-num { font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; color: #6366f1; letter-spacing: 2px; margin-bottom: 10px; }
.how-head { font-size: 0.92rem; font-weight: 600; color: #e2e8f0; margin-bottom: 7px; }
.how-body { font-size: 0.78rem; color: #334155; line-height: 1.65; }

.divider { height: 1px; background: rgba(255,255,255,0.05); max-width: 1200px; margin: 0 auto; }

/* Streamlit overrides */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(99,102,241,0.04) !important;
    border: 2px dashed rgba(99,102,241,0.25) !important;
    border-radius: 14px !important;
}
[data-testid="stImage"] img { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ── Model ──────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model("brain_tumor_model.h5")

with st.spinner("Initialising NeuraScan..."):
    model = load_model()


# ── HERO ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-inner">
    <div>
      <div class="hero-eyebrow">MobileNetV2 · Brain MRI Classification</div>
      <h1 class="hero-title">Detecting tumors<br><span>before time runs out</span></h1>
      <p class="hero-sub">Upload a brain MRI scan and receive an instant AI-powered classification
      across four diagnostic categories, powered by transfer learning on MobileNetV2.</p>
    </div>
    <div class="hero-stats">
      <div class="stat-pill">
        <div class="stat-num">4</div>
        <div class="stat-label">Classes</div>
      </div>
      <div class="stat-pill">
        <div class="stat-num">224px</div>
        <div class="stat-label">Input size</div>
      </div>
      <div class="stat-pill">
        <div class="stat-num">MNV2</div>
        <div class="stat-label">Backbone</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div style="height:40px"></div>', unsafe_allow_html=True)

# ── MAIN COLUMNS ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div style="padding: 0 0 0 40px;">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">MRI Scan Input</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "upload",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)
        st.markdown(
            f'<div class="scan-meta">FILE · {uploaded_file.name.upper()} · {image.size[0]}×{image.size[1]}px</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown("""
        <div class="idle-zone">
            <span class="idle-icon">🧠</span>
            <div class="idle-text">Drop a brain MRI scan here<br>
            <span style="color:#1e293b">JPG · JPEG · PNG · any resolution</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div style="padding: 0 40px 0 0;">', unsafe_allow_html=True)
    st.markdown('<div class="panel-label">AI Diagnosis</div>', unsafe_allow_html=True)

    if uploaded_file:
        with st.spinner("Analysing scan..."):
            time.sleep(0.3)
            resized = image.resize((224, 224))
            arr = np.expand_dims(np.array(resized), axis=0).astype("float32")
            prediction = model.predict(arr, verbose=0)[0]

        class_idx  = int(np.argmax(prediction))
        confidence = float(prediction[class_idx]) * 100
        label      = CLASS_NAMES[class_idx]
        meta       = CLASS_META[label]

        # diagnosis card
        st.markdown(f"""
        <div class="dx-card" style="border-color:{meta['color']}28;
             background: radial-gradient(ellipse 90% 60% at 50% 0%, {meta['glow']}, transparent 65%), rgba(255,255,255,0.015);">
            <div class="dx-eyebrow" style="color:{meta['color']};">AI Diagnosis · {confidence:.1f}% confidence</div>
            <span class="dx-icon">{meta['icon']}</span>
            <div class="dx-label" style="color:{meta['color']};">{label}</div>
            <div class="dx-conf">model confidence · {confidence:.2f}%</div>
            <div class="dx-desc">{meta['desc']}</div>
            <div class="dx-urgency" style="color:{meta['color']};">→ {meta['urgency']}</div>
        </div>
        """, unsafe_allow_html=True)

        # confidence bars
        bar_colors = {"Glioma":"#ef4444","Meningioma":"#f97316","No Tumor":"#22c55e","Pituitary":"#3b82f6"}
        rows_html  = ""
        for name, prob in zip(CLASS_NAMES, prediction):
            pct    = prob * 100
            is_top = name == label
            fill   = f"background:{bar_colors[name]};" if is_top else "background:rgba(255,255,255,0.08);"
            weight = "font-weight:700;color:#f8fafc;" if is_top else "color:#475569;"
            rows_html += f"""
            <div class="conf-row">
              <div class="conf-header">
                <span class="conf-name" style="{weight}">{name}</span>
                <span class="conf-pct">{pct:.1f}%</span>
              </div>
              <div class="conf-track">
                <div class="conf-fill" style="width:{pct:.1f}%;{fill}"></div>
              </div>
            </div>"""

        st.markdown(f"""
        <div class="conf-block">
            <div class="conf-title">Probability Breakdown</div>
            {rows_html}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer">
            <span style="font-size:0.9rem;flex-shrink:0;">⚠️</span>
            <span class="disclaimer-text">
                For educational and research purposes only. This output is not a clinical diagnosis.
                Always consult a qualified neurologist or radiologist for medical decisions.
            </span>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="result-idle">
          <div class="result-idle-inner">
            <div class="result-idle-icon">📊</div>
            <div class="result-idle-text">Diagnosis appears here after you upload a scan</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ── HOW IT WORKS ───────────────────────────────────────────────────────────────
st.markdown('<div style="height:52px"></div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div style="height:48px"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="how-section">
  <div class="how-title">How it works</div>
  <div class="how-grid">
    <div class="how-card">
      <div class="how-num">01 ·· INPUT</div>
      <div class="how-head">MRI Upload</div>
      <div class="how-body">Upload any brain MRI scan in JPG or PNG. The image is automatically resized to 224×224px for the model input layer.</div>
    </div>
    <div class="how-card">
      <div class="how-num">02 ·· PREP</div>
      <div class="how-head">Preprocessing</div>
      <div class="how-body">Pixel values are normalised via MobileNetV2's built-in preprocessing layer, baked directly into the model weights.</div>
    </div>
    <div class="how-card">
      <div class="how-num">03 ·· MODEL</div>
      <div class="how-head">Transfer Learning</div>
      <div class="how-body">MobileNetV2 pretrained on ImageNet extracts deep visual features. A fine-tuned classifier head outputs four class probabilities.</div>
    </div>
    <div class="how-card">
      <div class="how-num">04 ·· OUTPUT</div>
      <div class="how-head">Classification</div>
      <div class="how-body">The highest-probability class is returned with a full confidence breakdown for transparency, auditability, and clinical context.</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
