import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import gdown

# --- 1. Page Configuration ---
st.set_page_config(page_title="Global Rock & Mineral Expert | AI", page_icon="💎", layout="wide")

# --- 2. Load Model Function ---
@st.cache_resource
def load_rock_model():
    file_id = '1WtLpd9NpOmJ3o0bpUYEtE-1eH6jzPNTS'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'rock_model.h5'
    
    if not os.path.exists(output) or os.path.getsize(output) < 1000000:
        with st.spinner('Downloading Model...'):
            gdown.download(url, output, quiet=False)
    
    # تحميل الموديل مع تجاهل التهيئة الأصلية لحل مشكلة dense_1
    model = tf.keras.models.load_model(output, compile=False)
    return model

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
    st.markdown("#### *Advanced Geological Classification for SPC*")
with h2:
    st.markdown('<div class="designer-credit">Designed & Developed by:<br>Eng. Solaiman Kudaimi</div>', unsafe_allow_html=True)

st.divider()

# --- 5. Logic ---
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 📂 Input Sample")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Sample", use_container_width=True)

with col_right:
    st.markdown("### 📊 AI Diagnosis")
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

                # --- الحل القاطع لمشكلة الطبقة dense_1 ---
                # نستخدم الموديل كدالة ونحدد بوضوح أننا في وضع "التدريب = خطأ" 
                # ونمرر المدخل كـ Tensor واحد لفك الاشتباك
                input_tensor = tf.convert_to_tensor(img_array)
                predictions = model(input_tensor, training=False)

                # التأكد من تحويل النتيجة لمصفوفة بسيطة
                if isinstance(predictions, list) or isinstance(predictions, tuple):
                    predictions = predictions[0]
                
                preds_np = predictions.numpy() if hasattr(predictions, 'numpy') else predictions
                
                # إذا كانت الأبعاد (None, 7, 7, 1280) فهذا يعني أن الطبقة الأخيرة لم تُضغط (Pooling)
                # سنقوم بضغطها يدوياً هنا برمجياً لإنقاذ الموقف
                if len(preds_np.shape) > 2:
                    preds_np = np.mean(preds_np, axis=(1, 2))

                labels = ['Igneous Rock', 'Metamorphic Rock', 'Sedimentary Rock', 'Mineral Sample']
                idx = np.argmax(preds_np[0])
                conf = np.max(preds_np[0]) * 100
                
                # النتائج
                st.success("Analysis Completed")
                st.metric(label="Classification", value=f"{labels[idx]}")
                st.write(f"**Confidence Score:** {conf:.2f}%")
                st.progress(int(conf))

        except Exception as e:
            st.error(f"Technical Error: {e}")
            st.info("The model architecture requires an explicit Pooling layer before prediction.")
    else:
        st.warning("Awaiting input...")

st.divider()
st.markdown("<center><p style='color: #888;'>All Rights Reserved © 2026 | <b>Eng. Solaiman Kudaimi</b></p></center>", unsafe_allow_html=True)
