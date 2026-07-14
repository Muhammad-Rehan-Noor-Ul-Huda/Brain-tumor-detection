import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np

# ──────────────────────────────────────────────
# Page config (must be the first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Class labels — EDIT THESE if your training
# order was different (check your ImageDataGenerator
# / dataset folder order, e.g. class_indices)
# ──────────────────────────────────────────────
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

CLASS_INFO = {
    "Glioma": "A tumor that starts in the glial cells of the brain or spine.",
    "Meningioma": "A tumor arising from the meninges, the brain's protective lining.",
    "No Tumor": "No visible tumor detected in the scan.",
    "Pituitary": "A tumor located in the pituitary gland at the base of the brain.",
}

# ──────────────────────────────────────────────
# Custom CSS for a cleaner, more attractive look
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .result-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.6rem;
        text-align: center;
        margin-top: 1.2rem;
    }
    .result-label {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 0.3rem;
    }
    .result-desc {
        color: #cbd5e1;
        font-size: 0.95rem;
    }
    .confidence-text {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    .disclaimer {
        margin-top: 2rem;
        padding: 1rem;
        border-radius: 12px;
        background: rgba(251, 191, 36, 0.08);
        border: 1px solid rgba(251, 191, 36, 0.25);
        color: #fbbf24;
        font-size: 0.85rem;
        text-align: center;
    }
    [data-testid="stFileUploaderDropzone"] {
        border-radius: 14px !important;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 About")
    st.write(
        "This tool uses a MobileNetV2-based deep learning model to classify "
        "brain MRI scans into four categories."
    )
    st.markdown("---")
    st.markdown("### 📋 Classes")
    for name, desc in CLASS_INFO.items():
        st.markdown(f"**{name}**  \n{desc}")
    st.markdown("---")
    st.caption("Built with Streamlit · TensorFlow · MobileNetV2")

# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return tf.keras.models.load_model(
        "brain_tumor_model.keras",
        custom_objects={"preprocess_input": preprocess_input},
    )

model = load_model()

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown('<p class="main-title">🧠 Brain Tumor Classifier</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Upload an MRI scan and get an instant AI-powered classification</p>',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Upload + predict
# ──────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Drop an MRI image here, or click to browse",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1], gap="large")

    image = Image.open(uploaded_file).convert("RGB")
    resized_image = image.resize((224, 224))

    with col1:
        st.image(image, caption="Uploaded MRI Scan", use_container_width=True)

    with st.spinner("Analyzing scan..."):
        img_array = np.array(resized_image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        prediction = model.predict(img_array, verbose=0)[0]
        class_idx = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100
        predicted_class = CLASS_NAMES[class_idx]

    with col2:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">{predicted_class}</div>
                <div class="result-desc">{CLASS_INFO[predicted_class]}</div>
                <div class="confidence-text">Confidence: {confidence:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(int(confidence), 100))

        with st.expander("See full probability breakdown"):
            for name, prob in zip(CLASS_NAMES, prediction):
                st.write(f"**{name}**")
                st.progress(float(prob))
                st.caption(f"{prob * 100:.2f}%")

    st.markdown(
        """
        <div class="disclaimer">
            ⚠️ This tool is for educational purposes only and is not a substitute
            for professional medical diagnosis. Always consult a qualified
            healthcare professional for medical advice.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info("👆 Upload an MRI image to get started.")
