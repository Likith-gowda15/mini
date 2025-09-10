import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Creative gradient background and custom CSS
st.markdown("""
    <style>
        /* Gradient background */
        body, .stApp {
            background: linear-gradient(120deg, #f8ffae, #43c6ac, #191654);
        }
        /* Sidebar Styling */
        .css-1d391kg {background: #eee; border-radius: 16px;}
        .css-hxt7ib {border-right: 2px solid #43c6ac;}
        /* Footer Styling */
        .footer {
            background: #191654;
            color: #f8ffae;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation with emoji icons
st.sidebar.markdown("### 🧭 Navigation", unsafe_allow_html=True)
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Jump to:",
    ["🏠 Home", "📤 Upload & Predict", "ℹ️ About"],
    index=0
)

# Model download and load logic
FILE_ID = "124Fo29-Vt7UVeCLdRnJl75dZda3wRn9X"
URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
OUTPUT_PATH = "model.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(OUTPUT_PATH):
        st.info("📥 Downloading model...")
        try:
            gdown.download(URL, OUTPUT_PATH, quiet=False)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Model download failed: {e}")
            return None
    if os.path.exists(OUTPUT_PATH):
        model = tf.keras.models.load_model(OUTPUT_PATH)
        st.success(f"✅ Model loaded! Expected input: (None, 48, 48, 3)")
        return model
    else:
        st.error("❌ Model file not found after download.")
        return None

model = load_model()
if model is None:
    st.stop()

CLASS_NAMES = ["Benign", "Malignant"]

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((48, 48))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

if page == "🏠 Home":
    st.title("🏠 Breast Cancer Classifier")
    st.markdown("""
        <div style='background: #fffbe6; padding: 16px; border-radius: 12px; border-left: 6px solid #43c6ac'>
            Welcome! This app classifies breast cancer cell images as <b>Benign</b> or <b>Malignant</b>.<br>
            <ul>
                <li>🔸 <b>Upload</b> an image of a breast cancer cell</li>
                <li>🔸 The model <b>analyzes</b> the image</li>
                <li>🔸 It predicts if the cell is <b>Benign</b> or <b>Malignant</b></li>
            </ul>
        </div>
        <br>
        👉 <b>Click</b> <span style="background: #43c6ac; color: white; border-radius: 8px; padding: 2px 8px">Upload & Predict</span> in the sidebar to start!
    """, unsafe_allow_html=True)

elif page == "📤 Upload & Predict":
    st.title("📤 Upload & Predict")
    with st.expander("How to use this tool", expanded=False):
        st.write("""
            Upload a clear image of a breast cancer cell to receive quick predictions.<br>
            Supported formats: JPG, PNG, JPEG.
        """)
    uploaded_file = st.file_uploader
