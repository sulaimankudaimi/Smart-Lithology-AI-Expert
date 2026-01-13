import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import gdown

# --- 1. إعدادات الصفحة الاحترافية ---
st.set_page_config(
    page_title="Global Rock & Mineral Expert | AI",
    page_icon="💎",
    layout="wide"
)

# --- 2. وظيفة تحميل الموديل الذكي ---
@st.cache_resource
def load_rock_model():
    # معرف الملف الصحيح من جوجل درايف
    file_id = '1WtLpd9NpOmJ3o0bpUYEtE-1eH6jzPNTS'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'rock_model.h5'
    
    # تحميل الموديل إذا لم يكن موجوداً أو كان تالفاً
    if not os.path.exists(output) or os.path.getsize(output) < 1000000:
        with st.spinner('Synchronizing AI Engine with Cloud... Please wait.'):
            try:
                # التحميل المباشر لضمان سلامة بصمة الملف (File Signature)
                gdown.download(url, output, quiet=False)
            except Exception as e:
                st.error(f"Cloud Sync Failed: {e}")
    
    # تحميل الموديل مع معالجة أخطاء الأبعاد الشائعة
    return tf.keras.models.load_model(output, compile=False)

# --- 3. تنسيق الواجهة (Professional CSS) ---
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

# --- 4. الهيدر (Header) ---
h_col1, h_col2 = st.columns([3, 1])
with h_col1:
    st.title("🔬 Global Rock & Mineral Expert AI")
    st.markdown("#### *Intelligent Lithology Classification for Petroleum Operations*")
with h_col2:
    st.markdown('<div class="designer-credit">Designed & Developed by:<br>Eng. Solaiman Kudaimi</div>', unsafe_allow_html=True)

st.divider()

# --- 5. منطقة العمل الرئيسية (Main Layout) ---
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 📂 Input Section")
    st.info("Upload a high-quality image of the sample (Cuttings or Cores) for AI diagnostics.")
    uploaded_file = st.file_uploader("Upload Rock Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Sample Ready for Neural Analysis", use_container_width=True)

with col_right:
    st.markdown("### 📊 Diagnostic Intelligence")
    if uploaded_file is not None:
        try:
            # تحميل الموديل
            model = load_rock_model()
            
            with st.spinner('Running Feature Extraction & Classification...'):
                # --- معالجة الصورة بدقة لحل مشكلة ValueError ---
                img = image.resize((224, 224))
                img_array = np.array(img)
                
                # إزالة قناة الـ Alpha إذا كانت موجودة (RGBA to RGB)
                if img_array.shape[-1] == 4:
                    img_array = img_array[..., :3]
                
                # التطبيع (Normalization)
                img_array = img_array.astype('float32') / 255.0
                
                # إضافة بعد الدفعة (Expand Dimensions)
                img_array = np.expand_dims(img_array, axis=0)
                
                # التنبؤ (Prediction)
                predictions = model.predict(img_array)
                
                # قائمة الأصناف بالترتيب
                labels = ['Igneous Rock', 'Metamorphic Rock', 'Sedimentary Rock', 'Mineral Sample']
                
                idx = np.argmax(predictions)
                conf = np.max(predictions) * 100
                
                # عرض النتائج بشكل احترافي
                st.success("Analysis Successfully Completed")
                st.metric(label="Predicted Classification", value=f"{labels[idx]}")
                st.write(f"**Confidence Score:** {conf:.2f}%")
                st.progress(int(conf))
                
                # ملاحظات فنية بترولية
                with st.expander("🔍 Geological Technical Analysis"):
                    if idx == 2: # Sedimentary
                        st.write("Target identified as **Sedimentary**. This is of paramount importance for SPC reservoir characterization and potential hydrocarbon trapping.")
                    elif idx == 0: # Igneous
                        st.write("Crystalline structure detected. Matches **Igneous** lithology signatures.")
                    else:
                        st.write("Advanced mineralogical features detected. Data consistent with trained geological patterns.")

        except Exception as e:
            st.error(f"Operational Error: {e}")
            st.warning("Hint: Ensure the model architecture matches the input shape (224x224x3).")
    else:
        st.warning("System Status: Awaiting Image Input...")

# --- 6. الفوتر (Footer) ---
st.divider()
st.markdown("<center><p style='color: #888;'>All Technical Rights Reserved © 2026 | <b>Eng. Solaiman Kudaimi</b><br>Specially developed for the <b>Syrian Petroleum Company (SPC)</b></p></center>", unsafe_allow_html=True)
