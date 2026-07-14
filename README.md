# Brain Tumor MRI Classifier

This project is a deep learning-based web application that classifies MRI brain scans to detect the presence of tumors. It utilizes a pre-trained **MobileNetV2** model via transfer learning to provide high-accuracy predictions through a clean, intuitive web interface built with **Streamlit**.

## Project Overview
The model was trained on medical imaging datasets using Keras/TensorFlow. By leveraging transfer learning, the application can extract complex features from MRI scans effectively, allowing it to generalize well to new, unseen data.

## Key Features
- **Real-time Prediction:** Upload an MRI image and receive an immediate classification.
- **Easy Interface:** Built with Streamlit for a seamless user experience.
- **High Accuracy:** Powered by MobileNetV2, an optimized architecture for image classification.
- **Scalable:** Easy to deploy on cloud platforms like Streamlit Community Cloud or Hugging Face Spaces.

## Tech Stack
- **Language:** Python
- **Framework:** Streamlit
- **Deep Learning:** TensorFlow / Keras
- **Model Architecture:** MobileNetV2 (Transfer Learning)
- **Image Processing:** PIL / NumPy

## Setup & Installation

1. **Clone the repository** (or download your project folder):
   ```bash
   git clone [your-repository-url]
   cd brain-tumor-app
