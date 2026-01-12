import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
import os

# --- Professional Page Configuration ---
st.set_page_config(
    page_title="Global Rock & Mineral Expert | AI",
    page_icon="💎",
    layout="wide"
)

# --- Function to Load Model from Google Drive ---
@st.cache_resource
def load_rock_model():
    # Replace 'YOUR_DRIVE_FILE_ID_HERE' with your actual file ID
    file_id = 'YOUR_DRIVE_FILE_ID_HERE' 
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'rock_model.h5'
    
    if not os.path.exists(output):
        with st.spinner('Downloading AI Engine from Cloud... Please wait.'):
            response = requests.get(url, stream=True)
            with open(output, 'wb') as f:
                f.write(response.content)
    
    return tf.keras.models.load_model(output)

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #004b87; color: white; font-weight: bold; }
    .designer-credit { font-size: 1.1rem; color: #004b87; font-weight: bold; border-left: 4px solid #004b87; padding-left: 10px; }
    .result-card { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0px 4px 6px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_content_type=True)

# --- Header Section ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🔬 Global Rock & Mineral Expert AI")
    st.subheader("Next-Gen Geological Analysis Platform")
with col2:
    st.markdown('<p class="designer-credit">Designed & Developed by:<br>Eng. Solaiman Kudaimi</p>', unsafe_content_type=True)

st.divider()

# --- Main Application Logic ---
col_upload, col_result = st.columns([1, 1])

with col_upload:
    st.markdown("### 📂 Sample Upload")
    st.info("Upload a high-quality image of the rock sample (Field or Microscopic)")
    uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Sample for Analysis", use_container_width=True)

with col_result:
    st.markdown("### 📊 AI Diagnostic Results")
    if uploaded_file is not None:
        try:
            model = load_rock_model()
            
            with st.spinner('Analyzing Geological Features...'):
                # Image Pre-processing
                img = image.resize((224, 224)) # Adjust to your training size
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Model Prediction
                predictions = model.predict(img_array)
                # Ensure the order matches your training labels
                class_names = ['Igneous Rock', 'Metamorphic Rock', 'Sedimentary Rock', 'Mineral Sample']
                
                result_idx = np.argmax(predictions)
                confidence = np.max(predictions) * 100
                
                # Display Results in a Clean Layout
                st.success("✅ Analysis Complete")
                st.metric(label="Predicted Classification", value=f"{class_names[result_idx]}")
                st.progress(int(confidence))
                st.write(f"**Confidence Level:** {confidence:.2f}%")
                
                # Geological Insights Expander
                with st.expander("🔍 Technical Insights"):
                    if result_idx == 2: # Sedimentary
                        st.write("Target identified as Sedimentary. High importance for hydrocarbon potential and reservoir characterization.")
                    elif result_idx == 0: # Igneous
                        st.write("Crystalline structure detected. Likely associated with basement complex or volcanic activity.")
                    else:
                        st.write("Detailed mineralogical features identified based on deep learning feature extraction.")
                        
        except Exception as e:
            st.error(f"Error during processing: {e}")
    else:
        st.warning("Awaiting sample upload to initiate AI scanning...")

# --- Footer Section ---
st.divider()
st.markdown("""
    <center>
    <p style='color: #888;'>All Rights Reserved &copy; 2026 | Eng. Solaiman Kudaimi</p>
    <p style='color: #004b87; font-weight: bold;'>Prepared for: Syrian Petroleum Company (SPC)</p>
    </center>
    """, unsafe_content_type=True)