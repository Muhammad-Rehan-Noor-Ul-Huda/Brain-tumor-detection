import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('brain_tumor_model.keras.keras')

model = load_model()

st.title("Brain Tumor Classifier")
uploaded_file = st.file_uploader("Upload an MRI scan...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((224, 224))
    st.image(image, caption='Uploaded MRI', use_container_width=True)
    
    # Preprocess
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array) 
    
    # Predict
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    
    st.write(f"Prediction Result: Class {class_idx}")
    st.write("This is only for educational purpose professionals are still recommended")
