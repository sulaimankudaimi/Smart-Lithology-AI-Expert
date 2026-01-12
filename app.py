import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import gdown

# --- Professional Page Configuration ---
st.set_page_config(
    page_title="Global Rock & Mineral Expert | AI",
    page_icon="💎",
    layout="wide"
)

# --- Integrated Model Downloader & Loader ---
@st.cache_resource
def load_rock_model():
    # Correct File ID from your provided link
    file_id = '1WtLpd9NpOmJ3o0bpUYEtE-1eH6jzPNTS'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'rock_model.h5'
    
    # Check if the model exists or is corrupted
    if not os.path.exists(output) or os.path.getsize(output) < 1000000:
        with st.spinner('Synchronizing AI Engine with Cloud... Please wait.'):
            try:
                # Using gdown for reliable direct download
                gdown.download(url, output, quiet=False)
            except Exception as e:
                st.error(f"Sync Failed: {e}")
    
    return tf.keras.models.load_model(output)

# --- Clean CSS Styling ---
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

# --- Header ---
h_col1, h_col2 = st.columns([3, 1])
with h_col1:
    st.title("🔬 Global Rock & Mineral Expert AI")
    st.markdown("#### *The Intelligent Frontier for SPC Geological Classification*")
with h_col2:
    st.markdown('<div class="designer-credit">Designed & Developed by:<br>Eng. Solaiman Kudaimi</div>', unsafe_allow_html=True)

st.divider()

# --- Application Core ---
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 📂 Input Section")
    st.info("Upload a clear image of the sample (Limestone, Basalt, etc.) for instant identification.")
    uploaded_file = st.file_uploader("Upload Rock Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Sample Ready for Processing", use_container_width=True)

with col_right:
    st.markdown("### 📊 Diagnostic Intelligence")
    if uploaded_file is not None:
        try:
            model = load_rock_model()
            with st.spinner('Neural Network Analysis in progress...'):
                # Image processing
                img = image.resize((224, 224))
                img_array = np.array(img) / 255.0
                if img_array.shape[-1] == 4: img_array = img_array[..., :3]
                img_array = np.expand_dims(img_array, axis=0)
                
                # Prediction
                predictions = model.predict(img_array)
                labels = ['Igneous Rock', 'Metamorphic Rock', 'Sedimentary Rock', 'Mineral Sample']
                
                idx = np.argmax(predictions)
                conf = np.max(predictions) * 100
                
                # Results Display
                st.success("Analysis Successfully Completed")
                st.metric(label="Predicted Classification", value=f"{labels[idx]}")
                st.write(f"**Confidence Level:** {conf:.2f}%")
                st.progress(int(conf))
                
                with st.expander("🔍 Geological Technical Notes"):
                    if idx == 2:
                        st.write("Identified as **Sedimentary**. Critical for reservoir characterization in SPC fields.")
                    else:
                        st.write("Target features extracted. Matches defined lithological signatures.")

        except Exception as e:
            st.error(f"Operational Error: {e}")
    else:
        st.warning("System Status: Awaiting Image Input...")

# --- Footer ---
st.divider()
st.markdown("<center><p style='color: #888;'>All Technical Rights Reserved © 2026 | <b>Eng. Solaiman Kudaimi</b><br>Prepared for the Syrian Petroleum Company (SPC)</p></center>", unsafe_allow_html=True)
