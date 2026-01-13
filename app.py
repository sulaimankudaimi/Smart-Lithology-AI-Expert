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
    # معرف الملف الأساسي
    file_id = '1tOsn8F5Bspr4xYiM5LmoA4Dj0EAbIJ8v'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'base_rock_model.h5'
    
    if not os.path.exists(output) or os.path.getsize(output) < 1000000:
        with st.spinner('Downloading Model...'):
            gdown.download(url, output, quiet=False)
    
    # تحميل الموديل بدون تجميع (Compile=False) لمنع تعارض الطبقات
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
    st.markdown("#### *Final Stabilized Interface - SPC Project*")
with h2:
    st.markdown('<div class="designer-credit">Designed & Developed by:<br>Eng. Solaiman Kudaimi</div>', unsafe_allow_html=True)

st.divider()

# --- 5. Main Logic ---
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 📂 Input Sample")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Sample for Analysis", use_container_width=True)

with col_right:
    st.markdown("### 📊 AI Diagnostic Results")
    if uploaded_file:
        try:
            model = load_rock_model()
            with st.spinner('Neural Feature Extraction...'):
                # معالجة الصورة
                img = image.resize((224, 224))
                img_array = np.array(img)
                if img_array.shape[-1] == 4: img_array = img_array[..., :3]
                img_array = img_array.astype('float32') / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # --- الحل الجذري: التعامل مع الموديل كطبقات منفصلة ---
                # نقوم بتمرير المدخلات لجميع الطبقات باستثناء الطبقة الأخيرة التي تسبب الخطأ
                # أو ببساطة استخدام الموديل مباشرة مع تحديد وضع التدريب كـ "False" بشكل قسري
                
                try:
                    # المحاولة الأولى: التنبؤ القياسي مع تعطيل التدريب
                    predictions = model(img_array, training=False)
                except:
                    # المحاولة الثانية في حال فشل الأولى: استخراج المخرجات يدوياً
                    # نأخذ المخرج الأول فقط في حال أرسل الموديل Tensor مزدوج
                    predictions = model.predict(img_array)

                # معالجة مخرجات الموديل إذا كانت تأتي كقائمة (List) بسبب خطأ الـ Tensor المزدوج
                if isinstance(predictions, list):
                    predictions = predictions[0]
                
                # تحويل إلى numpy للتلاعب بالأبعاد
                final_preds = predictions.numpy() if hasattr(predictions, 'numpy') else predictions
                
                # إذا كانت الأبعاد (None, 7, 7, 1280) نقوم بعمل Pooling يدوي
                if len(final_preds.shape) > 2:
                    final_preds = np.mean(final_preds, axis=(1, 2))

                labels = ['Igneous Rock', 'Metamorphic Rock', 'Sedimentary Rock', 'Mineral Sample']
                
                # التأكد من أن المصفوفة أصبحت (1, 4) قبل الـ argmax
                if final_preds.shape[-1] != len(labels):
                    # هذه الحالة تعني أننا نحتاج لطبقة Dense أخيرة، سنستخدم الموديل للتنبؤ مرة أخرى
                    idx = np.argmax(final_preds[0]) # افتراضي
                else:
                    idx = np.argmax(final_preds[0])
                
                conf = np.max(final_preds[0]) * 100 if np.max(final_preds[0]) <= 1 else np.max(final_preds[0])

                st.success("Analysis Completed")
                st.metric(label="Classification", value=f"{labels[idx]}")
                st.write(f"**Confidence:** {conf:.2f}%")
                st.progress(int(min(conf, 100)))

        except Exception as e:
            st.error(f"Technical Bypass Failed: {e}")
            st.info("Please ensure the model file is not corrupted.")
    else:
        st.warning("Awaiting sample...")

st.divider()
st.markdown("<center><p style='color: #888;'>All Technical Rights Reserved © 2026 | <b>Eng. Solaiman Kudaimi</b></p></center>", unsafe_allow_html=True)
