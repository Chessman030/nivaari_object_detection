import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image
import tempfile
from hybrid_pipeline import HybridDetector

# ----- Page Configuration -----
st.set_page_config(
    page_title="Nivaari | Object Detection",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----- Custom CSS for Nivaari Theme -----
st.markdown("""
<style>
    :root {
        --primary-color: #2E7D32;
        --secondary-color: #1565C0;
        --background-color: #F1F8E9;
    }
    
    .main {
        background-color: #F8F9FA;
    }
    
    .stApp header {
        background-color: transparent !important;
    }
    
    h1, h2, h3 {
        color: #1565C0 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .stButton>button {
        background-color: #2E7D32;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #1B5E20;
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    .nivaari-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #1565C0, #2E7D32);
        color: white;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .nivaari-header h1 {
        color: white !important;
        margin: 0;
        font-size: 3rem;
    }
    
    .nivaari-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid #2E7D32;
        color: #212529; /* Force dark text regardless of dark mode */
    }
    
    .metric-card h4 {
        color: #1565C0 !important;
        margin-top: 0;
    }
    
    .metric-card p {
        color: #212529 !important;
    }
</style>
""", unsafe_allow_html=True)

# ----- Header -----
st.markdown("""
<div class="nivaari-header">
    <h1>🌍 NIVAARI</h1>
    <p>Civic Issue Detection System (Hybrid YOLO + SVM)</p>
</div>
""", unsafe_allow_html=True)

# ----- Sidebar Configuration -----
st.sidebar.title("⚙️ Configuration")
st.sidebar.info("Configure the hybrid detection pipeline here.")

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
default_yolo = os.path.join(base_dir, "weights", "best.pt")
default_svm = os.path.join(base_dir, "svm_hybrid", "svm_model.pkl")

conf_threshold = st.sidebar.slider("YOLO Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

# ----- Main App State -----
@st.cache_resource
def load_model(yolo_path, svm_path):
    try:
        return HybridDetector(yolo_path, svm_path)
    except Exception as e:
        return str(e)

detector = load_model(default_yolo, default_svm)

if isinstance(detector, str):
    st.error(f"Error loading models: {detector}")
    st.warning("Note: You must train the SVM model first using `svm_classifier.py` before inference can run!")
else:
    if not detector.svm_ready:
        st.warning("⚠️ The SVM Model is not trained yet! The pipeline cannot classify objects. Please train the SVM model first.")
        
    st.subheader("📤 Upload Image for Inspection")
    uploaded_file = st.file_uploader("Choose an image representing a civic issue...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Layout columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Original Image")
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_container_width=True)
            
        with st.spinner("Analyzing image using YOLO + SVM pipeline..."):
            # Reset file pointer since Image.open() read to the end
            uploaded_file.seek(0)
            
            # Save uploaded image to temp file for OpenCV
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') 
            tfile.write(uploaded_file.read())
            tfile.close()  # MUST close the file on Windows so OpenCV can read it!
            
            if detector.svm_ready:
                # Run inference
                try:
                    img_result, detections = detector.detect_and_classify(tfile.name, conf_threshold)
                    img_annotated = detector.draw_results(img_result, detections)
                    
                    with col2:
                        st.markdown("### Detected Issues")
                        # Convert BGR to RGB for Streamlit
                        img_annotated_rgb = cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB)
                        st.image(img_annotated_rgb, use_container_width=True)
                        
                    # Show metrics and results
                    st.markdown("---")
                    st.subheader("📊 Analysis Report")
                    
                    if detections:
                        st.success(f"Identified {len(detections)} civic issue(s).")
                        
                        # Display results in a grid
                        cols = st.columns(3)
                        for i, det in enumerate(detections):
                            with cols[i % 3]:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h4>Issue {i+1}</h4>
                                    <p><b>Class:</b> {det['class_name']}</p>
                                    <p><b>SVM Confidence:</b> {det['svm_confidence']:.1%}</p>
                                    <p><b>YOLO (Loc) Confidence:</b> {det['yolo_confidence']:.1%}</p>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.info("No civic issues detected in this image.")
                        
                except Exception as e:
                    st.error(f"Error processing image: {e}")
            else:
                with col2:
                    st.markdown("### Processed Image")
                    st.info("Pipeline paused: SVM model missing.")
