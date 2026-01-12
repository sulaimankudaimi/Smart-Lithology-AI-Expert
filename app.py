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
    # Replace the ID below with your actual Google Drive File ID
    file_id = 'YOUR_DRIVE_FILE_ID_HERE' 
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'rock_model.h5'
    
    if not os.path.exists(output):
        with st.spinner('Accessing AI Engine from Cloud... Please wait.'):
            try:
                response = requests.get(url, stream=True)
                with open(output, 'wb') as f:
                    f.write(response.content)
            except Exception as e:
                st.error(f"Failed to download model: {e}")
    
    return tf.keras.models.load_model(output)

# --- Custom CSS for Styling (Fixed & Clean) ---
custom_css = """
<style>
    .main { 
        background-color: #f8f9fa; 
    }
    .stButton>button { 
        width: 100%; 
        border-radius: 8px; 
        height: 3.5em; 
        background-color: #004b87; 
        color: white; 
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #003366;
        color: #ffcc00;
    }
    .designer-credit { 
        font-size: 1.1rem; 
        color: #004b87; 
        font-weight: bold; 
        border-left: 5px solid #004b87; 
        padding-left: 15px;
        margin-top: 10px;
    }
    .footer-text {
        text-align: center;
        color: #666;
        font-size: 0.9rem;
        margin-top: 50px;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- Header Section ---
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title("🔬 Global Rock & Mineral Expert AI")
    st.markdown("#### *The Intelligent Frontier for Geological Lithology Classification*")
with header_col2:
    st.markdown(f"""
    <div class="designer-credit">
        Designed & Developed by:<br>
        Eng. Solaiman Kudaimi
    </div>
    """, unsafe_allow_html=True)

st.divider()

# --- Main Application Layout ---
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 📂 Input Section")
    st.info("Please upload a clear image of a Rock or Mineral sample for instant AI identification.")
    
    uploaded_file = st.file_uploader("Upload Image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Current Sample Status: Ready for Analysis", use_container_width=True)

with col_right:
    st.markdown("### 📊 Diagnostic Intelligence")
    
    if uploaded_file is not None:
        try:
            # Load the model
            model = load_rock_model()
            
            with st.spinner('Running Neural Network Analysis...'):
                # 1. Pre-processing (Ensure 224x224 matches your training size)
                img = image.resize((224, 224))
                img_array = np.array(img) / 255.0
                if img_array.shape[-1] == 4:  # Handle RGBA images
                    img_array = img_array[..., :3]
                img_array = np.expand_dims(img_array, axis=0)
                
                # 2. Prediction
                predictions = model.predict(img_array)
                
                # IMPORTANT: Labels must follow your model's training order
                labels = ['Igneous Rock', 'Metamorphic Rock', 'Sedimentary Rock', 'Mineral Sample']
                
                result_index = np.argmax(predictions)
                confidence_score = np.max(predictions) * 100
                
                # 3. Visualizing Results
                st.success("Analysis Successfully Completed")
                st.metric(label="Detected Classification", value=f"{labels[result_index]}")
                
                # Confidence Meter
                st.write(f"**Confidence Level:** {confidence_score:.2f}%")
                st.progress(int(confidence_score))
                
                # Technical Notes Expander
                with st.expander("🔍 Geological Context"):
                    if result_index == 2: # Sedimentary
                        st.write("This sample is classified as **Sedimentary**. These rocks are vital for reservoir studies and hydrocarbon exploration within the SPC fields.")
                    elif result_index == 0: # Igneous
                        st.write("The AI detected an **Igneous** structure. This is often associated with basement complexes or volcanic intrusions.")
                    elif result_index == 1: # Metamorphic
                        st.write("Target identified as **Metamorphic**. Represents rocks altered by high pressure and temperature.")
                    else:
                        st.write("Specific mineralogical features have been detected. Further thin-section analysis may complement this AI diagnostic.")

        except Exception as e:
            st.error(f"Operational Error: {e}")
    else:
        st.warning("System Status: Awaiting Image Input...")
        st.write("Once an image is uploaded, the AI engine will automatically begin the classification process.")

# --- Footer ---
st.divider()
st.markdown(f"""
    <div class="footer-text">
        All Technical Rights Reserved © 2026 | <b>Eng. Solaiman Kudaimi</b><br>
        This AI system is specifically engineered for the <b>Syrian Petroleum Company (SPC)</b>
    </div>
    """, unsafe_allow_html=True)
