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

# --- 2. Intelligent Model Loader ---
@st.cache_resource
def load_rock_model():
    file_id = '1WtLpd9NpOmJ3o0bpUYEtE-1eH6jzPNTS'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'rock_model.h5'
    
    if not os.path.exists(output) or os.path.getsize(output) < 1000000:
        with st.spinner('Downloading AI Engine from Cloud...'):
            try:
                gdown.download(url, output, quiet=False)
            except Exception as e:
                st.error(f"Download Error: {e}")
    
    # تحميل الموديل بدون تجميع (لحل مشاكل الأبعاد)
    return tf.keras.models.load_model(output, compile=False)

# --- 3. Professional Styling ---
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button { 
        width: 100%; border-radius: 8px; height: 3.5em; 
        background-color: #004b87; color: white; font-weight: bold;
    }
    .designer-credit { 
        font-size: 1.1rem; color: #004b87; font-weight: bold; 
        border-left: 5px solid #004b87; padding-left: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --- 4. Header ---
h1, h2 = st.columns([3, 1])
with h1:
    st.title("🔬 Global Rock & Mineral Expert AI")
    st.markdown("#### *Advanced Lithology Classification System for SPC Operations*")
with h2:
    st.markdown('<div class="designer-credit">Designed & Developed by:<br>Eng. Solaiman Kudaimi</div>', unsafe_allow_html=True)

st.divider()

# --- 5. Main Application Logic ---
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 📂 Input Sample")
    uploaded_file = st.file_uploader("Upload Image (Cuttings/Cores)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Sample", use_container_width=True)

with col_right:
    st.markdown("### 📊 AI Analysis")
    if uploaded_file:
        try:
            model = load_rock_model()
            with st.spinner('Analyzing...'):
                # معالجة الصورة
                img = image.resize((224, 224))
                img_array = np.array(img)
                if img_array.shape[-1] == 4: img_array = img_array[..., :3]
                img_array = img_array.astype('float32') / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # --- حل مشكلة dense_1 (استدعاء الموديل مباشرة كدالة) ---
                # نستخدم training=False لضمان عدم تفعيل Dropout أو BatchNormalization
                predictions = model(img_array, training=False)
                
                # تحويل النتيجة إلى مصفوفة numpy للتعامل معها
                if hasattr(predictions, "numpy"):
                    predictions = predictions.numpy()
                
                # التأكد من الحصول على آخر طبقة (في حال كان الموديل يعيد مخرجات متعددة)
                if isinstance(predictions, list):
                    predictions = predictions[-1]

                labels = ['Igneous Rock', 'Metamorphic Rock', 'Sedimentary Rock', 'Mineral Sample']
                idx = np.argmax(predictions[0])
                conf = np.max(predictions[0]) * 100
                
                # العرض
                st.success("Analysis Completed")
                st.metric(label="Classification", value=f"{labels[idx]}")
                st.write(f"**Confidence:** {conf:.2f}%")
                st.progress(int(conf))
                
                with st.expander("🔍 Technical Notes"):
                    if idx == 2:
                        st.write("Identified as **Sedimentary**. Essential for reservoir characterization.")
                    else:
                        st.write("Target identified based on trained geological patterns.")
                        
        except Exception as e:
            st.error(f"Operational Error: {e}")
            st.info("Technical Note: This error usually relates to model input/output layer mismatch.")
    else:
        st.warning("Awaiting sample input...")

# --- 6. Footer ---
st.divider()
st.markdown("<center><p style='color: #888;'>All Rights Reserved © 2026 | <b>Eng. Solaiman Kudaimi</b><br>Syrian Petroleum Company (SPC)</p></center>", unsafe_allow_html=True)
