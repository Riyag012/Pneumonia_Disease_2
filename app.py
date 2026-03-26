import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input

# 1. Load the trained DenseNet121 model
@st.cache_resource
def load_model():
    # Fixed the .keras.keras typo from your snippet
    return tf.keras.models.load_model("pneumonia_densenet_v2.keras")

model = load_model()

# 2. Preprocessing Logic
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)
    # Critical: DenseNet requires its specific preprocess_input function, 
    # not just a simple 1/255 division.
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# 3. Inference Logic
def predict_pneumonia(img):
    processed_img = preprocess_image(img)
    # model.predict returns a 2D array [[probability]]
    prediction_prob = model.predict(processed_img)[0][0]
    
    # Thresholding logic (0.5)
    label = "Pneumonia" if prediction_prob > 0.5 else "Normal"
    return label, prediction_prob

# 4. Streamlit UI
st.set_page_config(page_title="Pneumonia Detection", page_icon="🩺")
st.title("🩺 Pneumonia Detection from Chest X-ray")

st.write("Upload a chest X-ray image to identify potential pneumonia patterns with **96% accuracy**.")

uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded X-ray", use_container_width=True)
    
    with st.spinner("Analyzing medical features..."):
        label, confidence = predict_pneumonia(img)

    # Final Display with logic for confidence presentation
    if label == "Pneumonia":
        st.error(f"🚨 **Result: {label} Detected**")
        st.write(f"**Confidence Score:** {confidence:.2%}")
    else:
        st.success(f"✅ **Result: {label} (Healthy)**")
        # For Normal, the confidence in being "Normal" is (1 - prediction_prob)
        st.write(f"**Confidence Score:** {(1 - confidence):.2%}")