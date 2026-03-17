import streamlit as st
import numpy as np
from PIL import Image

st.title("🚨 Crime Anomaly Detection")

IMG_SIZE = (224, 224)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

def simple_anomaly_detection(image_array):
    """
    Simple placeholder logic (since TensorFlow can't run on Streamlit Cloud)
    You can replace this later with real model logic
    """
    # Example: use pixel intensity variance as anomaly score
    score = np.var(image_array)
    return score

if uploaded_file is not None:
    # Load image
    img = Image.open(uploaded_file).resize(IMG_SIZE)
    img_array = np.array(img) / 255.0

    st.image(img, caption="Uploaded Image")

    # Get anomaly score
    score = simple_anomaly_detection(img_array)

    # Threshold (you can tune this)
    threshold = 0.05

    st.write(f"Anomaly Score: {score:.5f}")

    if score > threshold:
        st.error("🚨 ANOMALOUS")
    else:
        st.success("✅ NORMAL")