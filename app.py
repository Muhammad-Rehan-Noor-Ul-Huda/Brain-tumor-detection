import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠")

# Define your class names
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('brain_tumor_mobilenet.keras')

model = load_model()

st.title("🧠 Brain Tumor Classifier")
st.write("Upload an MRI scan to analyze potential tumor presence.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI scan file...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Use columns to create a clean side-by-side layout
    col1, col2 = st.columns([1, 1])

    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((224, 224))

    with col1:
        # Fixed warning from previous logs: use width='stretch' instead of use_container_width
        st.image(image, caption='Uploaded MRI Scan', width=300)

    with col2:
        # Preprocess
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Predict
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        # Display main result
        st.subheader("Analysis Result")
        st.success(f"Detected: **{class_names[class_idx].upper()}**")
        st.metric(label="Confidence", value=f"{confidence*100:.2f}%")

    # Display probabilities for all classes
    st.divider()
    st.subheader("Confidence Distribution")
    for i, class_name in enumerate(class_names):
        prob = prediction[0][i]
        st.write(f"{class_name.capitalize()}")
        st.progress(float(prob))

    st.info("⚠️ Note: This tool is for educational purposes only. Always consult a qualified medical professional for diagnosis.")
