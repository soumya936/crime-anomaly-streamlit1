import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

st.title("🚨 Crime Anomaly Detection")

# Load models
feature_model = load_model("cnn_feature_extractor.h5")
autoencoder = load_model("anomaly_autoencoder.h5")

IMG_SIZE = (224,224)

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    img = load_img(uploaded_file, target_size=IMG_SIZE)
    img_array = img_to_array(img)/255.0

    st.image(img, caption="Uploaded Image")

    feat = feature_model.predict(np.expand_dims(img_array, axis=0))
    recon = autoencoder.predict(feat)

    error = np.mean(np.square(feat - recon))
    threshold = 0.01

    if error > threshold:
        st.error("🚨 ANOMALOUS")
    else:
        st.success("✅ NORMAL")