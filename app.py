import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import gdown

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Global Rock & Mineral Expert | AI",
    page_icon="💎",
    layout="wide"
)

# --- 2. Load Model Function (Using Base Model ID) ---
@st.cache_resource
def load_rock_model():
    # معرف الملف الأساسي الجديد الذي أرسلته
    file_id = '1tOsn8F5Bspr4xYiM5LmoA4Dj0EAbIJ8v'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'base_rock_model.h5'
    
    if not os.path.exists(output) or os.path.getsize(output) < 1000000:
        with st.spinner('Downloading Base AI Model for Testing...'):
            try:
                gdown.download(url, output, quiet=False)
            except Exception as e:
                st.error(f"Download Error: {e}")
    
    # تحميل الموديل
    return tf.keras.models.load_model(output, compile=False)

# --- 3. Custom CSS ---
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; background-color: #004b87; color: white; font-weight: bold; }
    .designer-credit { font-size: 1.1rem; color: #004b87; font-weight: bold; border-left: 5px solid #004b87; padding-left: 15px; }
</style>
""", unsafe_allow_html=True)

# --- 4. Header ---
h1, h2 = st.columns([3, 1])
with h1:
    st.title("🔬 Global Rock & Mineral Expert AI")
    st.markdown("#### *Basic Testing Interface - SPC Project*")
with h2:
    st.markdown('<div class="designer-credit">Designed & Developed by:<br>Eng. Solaiman Kudaimi</div>', unsafe_allow_html=True)

st.divider()

# --- 5. Logic Section ---
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 📂 Input Sample")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Sample", use_container_width=True)

with col_right:
    st.markdown("### 📊 AI Diagnostic Results")
    if uploaded_file:
        try:
            model = load_rock_model()
            with st.spinner('Processing...'):
                # معالجة الصورة
                img = image.resize((224, 224))
                img_array = np.array(img)
                if img_array.shape[-1] == 4: img_array = img_array[..., :3]
                img_array = img_array.astype('float32') / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # التنبؤ البسيط (Standard Prediction)
                predictions = model.predict(img_array)
                
                # قائمة الأصناف الافتراضية (تأكد من مطابقتها للموديل الأساسي)
                labels = ['Igneous Rock', 'Metamorphic Rock', 'Sedimentary Rock', 'Mineral Sample']
                
                idx = np.argmax(predictions[0])
                conf = np.max(predictions[0]) * 100
                
                # عرض النتائج
                st.success("Analysis Completed Successfully")
                st.metric(label="Classification", value=f"{labels[idx]}")
                st.write(f"**Confidence:** {conf:.2f}%")
                st.progress(int(conf))
                
        except Exception as e:
            st.error(f"Technical Error: {e}")
            st.info("Note: If 'dense_1' error persists, the Base Model might also have architectural complexity.")
    else:
        st.warning("Awaiting sample input...")

# --- 6. Footer ---
st.divider()
st.markdown("<center><p style='color: #888;'>All Technical Rights Reserved © 2026 | <b>Eng. Solaiman Kudaimi</b></p></center>", unsafe_allow_html=True)
