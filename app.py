import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
from PIL import Image
import time
import altair as alt
import pandas as pd

# Set page configuration with a modern, clean look
st.set_page_config(
    page_title="Construction Safety - Helmet Detection System",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize webcam session state (simple)
if 'webcam_running' not in st.session_state:
    st.session_state.webcam_running = False

# Custom CSS for a visually appealing UI
st.markdown("""
<style>
    body {
        font-family: 'Roboto', sans-serif;
    }
    .main {
        background-color: #f9f7f0;
        background-image: url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M11 18c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm48 25c3.866 0 7-3.134 7-7s-3.134-7-7-7-7 3.134-7 7 3.134 7 7 7zm-43-7c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm63 31c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM34 90c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zm56-76c1.657 0 3-1.343 3-3s-1.343-3-3-3-3 1.343-3 3 1.343 3 3 3zM12 86c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm28-65c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm23-11c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-6 60c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm29 22c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zM32 63c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm57-13c2.76 0 5-2.24 5-5s-2.24-5-5-5-5 2.24-5 5 2.24 5 5 5zm-9-21c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM60 91c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM35 41c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2zM12 60c1.105 0 2-.895 2-2s-.895-2-2-2-2 .895-2 2 .895 2 2 2z' fill='%23f0d678' fill-opacity='0.1' fill-rule='evenodd'/%3E%3C/svg%3E");
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .stApp {
        background: linear-gradient(120deg, #f9f7f0 0%, #f5f0e0 100%);
    }
    .title-container {
        background: linear-gradient(90deg, #ff6b35 0%, #f7c59f 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2.5rem;
        color: white;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
        border-top: 5px solid #ff6b35;
        animation: pulse 2s infinite;
        position: relative;
        overflow: hidden;
    }
    .title-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url("data:image/svg+xml,%3Csvg width='40' height='40' viewBox='0 0 40 40' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%23ffffff' fill-opacity='0.1' fill-rule='evenodd'%3E%3Cpath d='M0 40L40 0H20L0 20M40 40V20L20 40'/%3E%3C/g%3E%3C/svg%3E");
        z-index: 0;
    }
    .title-container * {
        position: relative;
        z-index: 1;
    }
    @keyframes pulse {
        0% {
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
        }
        50% {
            box-shadow: 0 8px 25px rgba(255, 107, 53, 0.3);
        }
        100% {
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
        }
    }
    .title-container h1 {
        color: white !important;
        margin: 0;
        font-size: 2.6rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    .title-container p {
        font-size: 1.3rem;
        opacity: 0.9;
        margin-top: 0.8rem;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }
    .card {
        background: rgb(255,255,255);
        background: linear-gradient(135deg, rgba(255,255,255,1) 0%, rgba(255,250,240,1) 100%);
        border-radius: 15px;
        padding: 1.8rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.8rem;
        border-top: 5px solid #ff6b35;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.12);
    }
    .stButton > button {
        background: linear-gradient(90deg, #ff6b35 0%, #f7c59f 100%);
        color: white;
        border: none;
        padding: 0.7rem 1.5rem;
        border-radius: 30px;
        font-weight: bold;
        box-shadow: 0 4px 10px rgba(255, 107, 53, 0.3);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #ff5a1f 0%, #ff6b35 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(255, 107, 53, 0.4);
    }
    .progress-bar {
        height: 14px;
        border-radius: 7px;
        margin-top: 0.8rem;
        overflow: hidden;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    .divider {
        height: 4px;
        background: linear-gradient(90deg, #ff6b35 0%, #f7c59f 100%);
        margin: 1.5rem 0;
        border-radius: 2px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #495057;
        padding: 1.8rem;
        background: linear-gradient(135deg, rgba(255,255,255,1) 0%, rgba(255,250,240,1) 100%);
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
        border-top: 5px solid #ff6b35;
    }
    /* Safety status specific styles */
    .status-safe {
        background: linear-gradient(90deg, #2d936c 0%, #64c897 100%);
        color: white;
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        box-shadow: 0 6px 15px rgba(45, 147, 108, 0.3);
        margin: 20px 0;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }
    .status-safe::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url("data:image/svg+xml,%3Csvg width='40' height='40' viewBox='0 0 40 40' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%23ffffff' fill-opacity='0.1' fill-rule='evenodd'%3E%3Cpath d='M0 40L40 0H20L0 20M40 40V20L20 40'/%3E%3C/g%3E%3C/svg%3E");
        z-index: 0;
    }
    .status-safe * {
        position: relative;
        z-index: 1;
    }
    .status-unsafe {
        background: linear-gradient(90deg, #c82333 0%, #dc3545 100%);
        color: white;
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        box-shadow: 0 6px 15px rgba(220, 53, 69, 0.3);
        margin: 20px 0;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }
    .status-unsafe::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url("data:image/svg+xml,%3Csvg width='40' height='40' viewBox='0 0 40 40' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%23ffffff' fill-opacity='0.1' fill-rule='evenodd'%3E%3Cpath d='M0 40L40 0H20L0 20M40 40V20L20 40'/%3E%3C/g%3E%3C/svg%3E");
        z-index: 0;
    }
    .status-unsafe * {
        position: relative;
        z-index: 1;
    }
    .status-neutral {
        background: linear-gradient(90deg, #4b6584 0%, #778ca3 100%);
        color: white;
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
        box-shadow: 0 6px 15px rgba(75, 101, 132, 0.3);
        margin: 20px 0;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }
    .status-neutral::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url("data:image/svg+xml,%3Csvg width='40' height='40' viewBox='0 0 40 40' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%23ffffff' fill-opacity='0.1' fill-rule='evenodd'%3E%3Cpath d='M0 40L40 0H20L0 20M40 40V20L20 40'/%3E%3C/g%3E%3C/svg%3E");
        z-index: 0;
    }
    .status-neutral * {
        position: relative;
        z-index: 1;
    }
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,1) 0%, rgba(255,250,240,1) 100%);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: fadeIn 0.5s ease-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.12);
    }
    .metric-value {
        font-size: 3rem;
        font-weight: bold;
        line-height: 1.2;
        margin: 10px 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    .metric-helmet {
        border-top: 5px solid #2d936c;
        color: #2d936c;
    }
    .metric-helmet .metric-value {
        color: #2d936c;
    }
    .metric-head {
        border-top: 5px solid #dc3545;
        color: #dc3545;
    }
    .metric-head .metric-value {
        color: #dc3545;
    }
    /* Improved headings */
    h1, h2, h3, h4 {
        color: #ff6b35 !important;
        font-weight: 700 !important;
    }
    h3 {
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
        font-size: 1.5rem !important;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid rgba(255, 107, 53, 0.2);
    }
    /* Image container */
    .image-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
        border: 3px solid #ffffff;
        margin-bottom: 1rem;
    }
    /* Detection result highlight */
    .detection-highlight {
        background-color: rgba(255, 107, 53, 0.1);
        border-left: 4px solid #ff6b35;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    /* Safety banner */
    .safety-banner {
        background: linear-gradient(135deg, #ff6b35 0%, #f7c59f 100%);
        border-radius: 12px;
        padding: 15px;
        color: white;
        margin: 15px 0;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(255, 107, 53, 0.2);
    }
    .safety-banner::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url("data:image/svg+xml,%3Csvg width='52' height='26' viewBox='0 0 52 26' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.1'%3E%3Cpath d='M10 10c0-2.21-1.79-4-4-4-3.314 0-6-2.686-6-6h2c0 2.21 1.79 4 4 4 3.314 0 6 2.686 6 6 0 2.21 1.79 4 4 4 3.314 0 6 2.686 6 6 0 2.21 1.79 4 4 4v2c-3.314 0-6-2.686-6-6 0-2.21-1.79-4-4-4-3.314 0-6-2.686-6-6zm25.464-1.95l8.486 8.486-1.414 1.414-8.486-8.486 1.414-1.414z' /%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        z-index: 0;
    }
    .safety-banner-content {
        position: relative;
        z-index: 1;
        display: flex;
        align-items: center;
    }
    .safety-icon {
        font-size: 2rem;
        margin-right: 15px;
        min-width: 40px;
    }
    .safety-text {
        flex: 1;
        font-size: 1.1rem;
    }
    .safety-title {
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Header with custom styling
st.markdown('<div class="title-container">', unsafe_allow_html=True)
st.markdown("<h1><span style='font-size:3rem;'>🪖</span> Construction Safety Monitor</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size:1.4rem;'><strong>Helmet Detection System</strong> - Advanced AI-powered safety monitoring for construction sites</p>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Load the YOLOv8n model
@st.cache_resource
def load_model():
    """Load and cache the detection model"""
    try:
        # Path to the best trained model weights
        model_path = "runs/train/helmet_detection3/weights/best.pt"
        
        # Check if the trained model exists
        if os.path.exists(model_path):
            return YOLO(model_path)
        # Try loading ONNX format (faster for CPU)
        elif os.path.exists("best.onnx"):
            return YOLO("best.onnx")
        # Fall back to other model paths
        elif os.path.exists("best.pt"):
            return YOLO("best.pt")
        else:
            # Use pretrained YOLOv8n if no custom model is found
            return YOLO("yolov8n.pt")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Sidebar configuration
with st.sidebar:
    # Add locally available hardhat icon for safety controls
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <div style="font-size: 64px; margin-bottom: 10px;">🪖</div>
        <h3 style="margin: 0; color: #ff6b35;">Safety Controls</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Safety fact banner in sidebar
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ff6b35 0%, #f7c59f 100%); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <div style="font-weight: bold; color: white; margin-bottom: 5px;">SAFETY FACT:</div>
        <div style="color: white; font-size: 0.9rem;">According to OSHA, head protection is required for workers in areas where there is a potential for injury from falling objects. Proper helmet use reduces head injury risk by up to 50%.</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Detection Settings - with gradient background instead of card
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(255,255,255,0.8) 0%, rgba(255,250,240,0.8) 100%); 
                border-radius: 10px; padding: 15px; margin-bottom: 15px; border-left: 4px solid #ff6b35;">
        <h4 style="margin-top: 0; color: #ff6b35;">⚙️ Detection Settings</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Detection confidence slider
    detection_confidence = st.slider(
        "Detection Confidence",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Adjust how confident the model must be to report a detection"
    )
    
    # Input selection - with colored gradient separator instead of divider
    st.markdown('<div style="height: 2px; background: linear-gradient(90deg, #ff6b35 0%, rgba(247, 197, 159, 0.3) 100%); margin: 15px 0;"></div>', unsafe_allow_html=True)
    input_source = st.radio(
        "Select Input Source",
        ["📷 Image", "🎥 Video", "📹 Webcam"],
        help="Choose the type of input you want to analyze"
    )
    
    # About section with visual elements - with gradient background instead of card
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(255,255,255,0.8) 0%, rgba(255,250,240,0.8) 100%); 
                border-radius: 10px; padding: 15px; margin-top: 20px; border-left: 4px solid #ff6b35;">
        <div style="display: flex; align-items: center; margin-bottom: 15px;">
            <div style="font-size: 28px; margin-right: 10px;">🪖</div>
            <h4 style="margin: 0; color: #ff6b35;">About This System</h4>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("""
    This system uses computer vision to detect helmet usage on construction sites:
    
    * Detection powered by YOLOv8 object detection
    * Trained on construction site safety images
    * Shows real-time detection results
    * Identifies proper helmet use and possible violations
    """)

# Load model with a spinner
with st.spinner("Loading helmet detection model..."):
    model = load_model()

# Display error if model fails to load
if model is None:
    st.error("Failed to load the detection model. Please check the model files.")
    st.stop()

# Process uploaded image/video/webcam and perform detection
def process_image(image, confidence):
    """Run detection on an image and return the annotated result"""
    # Configure custom plotting parameters
    plot_args = {
        'line_width': 2,            # Thicker lines for better visibility
        'boxes': True,              # Show bounding boxes
        'conf': True,               # Show confidence scores
        'labels': True,             # Show class labels
        'font_size': 14,            # Larger font size
    }
    
    # Custom colors for different classes (class_id -> RGB color)
    class_colors = {
        0: (0, 200, 0),    # helmet: green
        1: (255, 50, 50),  # head: red
        2: (0, 150, 255)   # person: blue
    }
    
    # Run inference
    results = model(image, conf=confidence)
    
    # Custom visualization (if needed)
    annotated_img = results[0].plot(**plot_args)
    
    return annotated_img, results[0]

# Display detection statistics in a visually appealing way
def display_detection_stats(results):
    """Create a nice visual display of detection results"""
    if results is None or len(results.boxes) == 0:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.warning("No detections found in this image.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Count instances of each class
    classes = results.boxes.cls.cpu().numpy()
    names = results.names
    confidences = results.boxes.conf.cpu().numpy()
    
    # Get class counts
    class_counts = {names[int(cls)]: int(np.sum(classes == cls)) for cls in np.unique(classes)}
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Detection Results")
    
    # Display class counts as metrics in a row
    cols = st.columns(len(class_counts))
    for i, (cls, count) in enumerate(class_counts.items()):
        with cols[i]:
            # Choose icon based on class name
            icon = "🪖" if "helmet" in cls.lower() else "👷" if "person" in cls.lower() else "👤" if "head" in cls.lower() else "⚠️"
            st.metric(f"{icon} {cls}", f"{count} detected")
    
    # Show safety status summary
    helmet_count = class_counts.get("helmet", 0) if "helmet" in names else 0
    head_count = class_counts.get("head", 0) if "head" in names else 0
    
    if helmet_count > 0 and head_count == 0:
        safety_status = "✅ SAFE: All personnel wearing helmets"
        status_class = "status-safe"
    elif head_count > 0:
        safety_status = "⚠️ UNSAFE: Personnel without helmets detected"
        status_class = "status-unsafe"
    else:
        safety_status = "ℹ️ No helmet status to report"
        status_class = "status-neutral"
    
    st.markdown(f"<div class='{status_class}'>{safety_status}</div>", unsafe_allow_html=True)
    
    # Show confidence levels with visual progress bars
    st.markdown("### Confidence Levels")
    
    # Sort boxes by confidence for better presentation
    indices = np.argsort([-box.conf[0].item() for box in results.boxes])
    
    for idx in indices:
        box = results.boxes[idx]
        cls = int(box.cls[0].item())
        conf = box.conf[0].item()
        class_name = names[cls]
        
        # Get box coordinates for reference
        if hasattr(box, 'xyxy'):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            coords = f"[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
        else:
            coords = "[N/A]"
        
        # Create a colored progress bar based on confidence
        conf_pct = conf * 100
        color = "#28a745" if conf >= 0.7 else "#ffc107" if conf >= 0.5 else "#dc3545"
        
        st.markdown(f"""
        <div style="margin-bottom: 15px; background-color: white; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="font-weight: bold;">{class_name} #{idx+1}</span>
                <span style="font-weight: bold; color: {color};">{conf_pct:.1f}%</span>
            </div>
            <div class="progress-bar" style="background-color: #e9ecef; height: 12px;">
                <div style="width: {conf_pct}%; height: 100%; background-color: {color}; border-radius: 5px;"></div>
            </div>
            <div style="font-size: 0.8em; color: #6c757d; margin-top: 5px;">Coordinates: {coords}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Add recommendations based on detections
    if head_count > 0:
        st.markdown("<div style='margin-top: 20px; padding: 15px; background-color: #f8d7da; border-left: 5px solid #dc3545; border-radius: 5px;'><strong>🚨 Safety Alert:</strong> <span style='color: #721c24;'>Detected personnel without helmets. Please ensure all workers wear proper safety equipment.</span></div>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main app logic based on input source selection
if "📷 Image" in input_source:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📷 Image Upload")
    
    uploaded_file = st.file_uploader(
        "Upload an image containing people with or without helmets",
        type=["jpg", "jpeg", "png"]
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Educational banner about image analysis
    st.markdown("""
    <div class="safety-banner" style="background: linear-gradient(135deg, #4b6584 0%, #778ca3 100%);">
        <div class="safety-banner-content">
            <div class="safety-icon">📸</div>
            <div class="safety-text">
                <div class="safety-title">IMAGE ANALYSIS:</div>
                <div>Our AI detects both proper helmet usage and safety violations in static images. Regular site image audits can help identify safety compliance trends and training needs.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        image_np = np.array(image.convert('RGB'))
        
        # Process the image
        with st.spinner("Detecting helmets..."):
            processed_img, results = process_image(image_np, detection_confidence)
        
        # Display results in a two-column layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Original Image")
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image_np, channels="RGB", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Detection Results")
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(processed_img, channels="RGB", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Display detection statistics
        display_detection_stats(results)

elif "🎥 Video" in input_source:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🎥 Video Upload")
    
    uploaded_file = st.file_uploader(
        "Upload a video containing people with or without helmets",
        type=["mp4", "avi", "mov", "mkv"]
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Educational banner about video monitoring
    st.markdown("""
    <div class="safety-banner" style="background: linear-gradient(135deg, #4b6584 0%, #778ca3 100%);">
        <div class="safety-banner-content">
            <div class="safety-icon">🎬</div>
            <div class="safety-text">
                <div class="safety-title">VIDEO MONITORING:</div>
                <div>Video analysis allows for comprehensive safety monitoring over time. Studies show continuous monitoring can reduce safety incidents by up to 70% on construction sites.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Save video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        tfile.close()
        
        # Open the video file
        cap = cv2.VideoCapture(tfile.name)
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Display video information
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Video Information")
        
        # Display video details in a clean layout
        info_cols = st.columns(4)
        with info_cols[0]:
            st.metric("Resolution", f"{frame_width}x{frame_height}")
        with info_cols[1]:
            st.metric("FPS", fps)
        with info_cols[2]:
            st.metric("Duration", f"{total_frames/fps:.1f}s")
        with info_cols[3]:
            st.metric("Total Frames", total_frames)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Create placeholders for video display and progress tracking
        video_placeholder = st.empty()
        progress_placeholder = st.empty()
        results_placeholder = st.empty()
        
        # Start video processing
        if st.button("Process Video", key="process_video"):
            # Create placeholders inside cards
            progress_placeholder.markdown('<div class="card">', unsafe_allow_html=True)
            progress_bar = progress_placeholder.progress(0)
            status_text = progress_placeholder.empty()
            progress_placeholder.markdown('</div>', unsafe_allow_html=True)
            
            video_placeholder.markdown('<div class="card">', unsafe_allow_html=True)
            video_display = video_placeholder.empty()
            video_placeholder.markdown('</div>', unsafe_allow_html=True)
            
            # Track detections
            helmet_count = 0
            head_count = 0
            person_count = 0
            detection_history = []
            
            # Process frames
            frame_count = 0
            sample_rate = max(1, int(fps / 10))  # Process 10 frames per second for performance
            
            start_time = time.time()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every nth frame for better performance
                if frame_count % sample_rate == 0:
                    # Update progress
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {frame_count}/{total_frames} ({progress*100:.1f}%)")
                    
                    # Convert frame to RGB and process
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    processed_frame, results = process_image(frame_rgb, detection_confidence)
                    
                    # Update counters
                    if len(results.boxes) > 0:
                        classes = results.boxes.cls.cpu().numpy()
                        names = results.names
                        
                        # Count each class
                        for cls in classes:
                            class_name = names[int(cls)]
                            if "helmet" in class_name.lower():
                                helmet_count += 1
                            elif "head" in class_name.lower():
                                head_count += 1
                            elif "person" in class_name.lower():
                                person_count += 1
                        
                        # Add to detection history for timeline
                        time_point = frame_count / fps
                        for cls in np.unique(classes):
                            class_name = names[int(cls)]
                            count = int(np.sum(classes == cls))
                            detection_history.append({
                                "time": time_point,
                                "frame": frame_count,
                                "class": class_name,
                                "count": count
                            })
                    
                    # Display the current frame
                    video_display.markdown('<div class="image-container">', unsafe_allow_html=True)
                    video_display.image(processed_frame, channels="RGB", use_container_width=True)
                    video_display.markdown('</div>', unsafe_allow_html=True)
                
                frame_count += 1
                
                # Stop if taking too long (optional safety feature)
                if time.time() - start_time > 300:  # 5 minute timeout
                    status_text.text("Processing timeout reached. Video too long or processing too slow.")
                    break
            
            # Complete progress and clean up
            progress_bar.progress(1.0)
            status_text.text(f"Processing complete! Analyzed {frame_count} frames.")
            cap.release()
            
            # Display detection summary
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Detection Summary")
            
            # Create improved metric cards with animations and gradients
            st.markdown(f"""
            <div class="row" style="display: flex; gap: 15px; margin-top: 20px;">
                <div class="metric-card metric-helmet" style="flex: 1; animation-delay: 0.1s;">
                    <div style="font-size: 1.2rem; margin-bottom: 5px;">🪖 Helmets Detected</div>
                    <div class="metric-value" style="background: -webkit-linear-gradient(45deg, #28a745, #5cb85c); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{helmet_count}</div>
                </div>
                <div class="metric-card metric-head" style="flex: 1; animation-delay: 0.3s;">
                    <div style="font-size: 1.2rem; margin-bottom: 5px;">👤 Safety Violations</div>
                    <div class="metric-value" style="background: -webkit-linear-gradient(45deg, #dc3545, #f86b7a); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{head_count}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show overall safety compliance percentage
            if (helmet_count + head_count) > 0:
                safety_compliance = (helmet_count / (helmet_count + head_count)) * 100
                if safety_compliance >= 90:
                    status_class = "status-safe"
                    icon = "✅"
                    message = "EXCELLENT SAFETY COMPLIANCE"
                elif safety_compliance >= 70:
                    status_class = "status-safe"
                    icon = "✅"
                    message = "GOOD SAFETY COMPLIANCE"
                else:
                    status_class = "status-unsafe"
                    icon = "⚠️"
                    message = "POOR SAFETY COMPLIANCE - ACTION REQUIRED"
                    
                st.markdown(f"""<div class="{status_class}" style="margin-top: 15px;">
                    <div style="font-size: 1.6rem; margin-bottom: 5px;">{icon} {message}</div>
                    <div style="font-size: 1.2rem;">Safety Score: {safety_compliance:.1f}%</div>
                </div>""", unsafe_allow_html=True)
            
            # Create timeline visualization if we have detections
            if detection_history:
                st.subheader("Detection Timeline")
                
                # Convert to DataFrame for visualization
                df = pd.DataFrame(detection_history)
                
                # Create a line chart showing detections over time
                time_chart = alt.Chart(df).mark_line(point=True).encode(
                x=alt.X('time:Q', title='Time (seconds)'),
                y=alt.Y('count:Q', title='Number of Detections'),
                color=alt.Color('class:N', 
                scale=alt.Scale(
                domain=['helmet', 'head', 'person'],
                range=['#2d936c', '#dc3545', '#4b6584']
                )),
                tooltip=['time:Q', 'class:N', 'count:Q', 'frame:Q']
                ).properties(
                    width=600,
                height=300,
                title='Detections Over Time'
                ).interactive()
                
            # Add a safety ratio chart (helmets vs heads)
            safety_df = df.pivot_table(
                index='time', 
                columns='class', 
                values='count',
                aggfunc='sum'
            ).reset_index().fillna(0)
            
            # Check if we have both helmet and head data
            if 'helmet' in safety_df.columns and 'head' in safety_df.columns:
                # Calculate safety ratio (helmets / (helmets + heads))
                safety_df['safety_ratio'] = safety_df['helmet'] / (safety_df['helmet'] + safety_df['head'] + 0.0001) * 100
                
                safety_chart = alt.Chart(safety_df).mark_area().encode(
                    x=alt.X('time:Q', title='Time (seconds)'),
                    y=alt.Y('safety_ratio:Q', title='Safety Compliance (%)', scale=alt.Scale(domain=[0, 100])),
                    color=alt.value('#2d936c'),
                    opacity=alt.value(0.7),
                    tooltip=['time:Q', 'safety_ratio:Q']
                ).properties(
                    width=600,
                    height=200,
                    title='Safety Compliance Over Time (% of Personnel with Helmets)'
                ).interactive()
                
                # Display both charts
                st.altair_chart(time_chart, use_container_width=True)
                st.altair_chart(safety_chart, use_container_width=True)
            else:
                # Just show the time chart if we don't have both classes
                st.altair_chart(time_chart, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Clean up the temporary file
            os.unlink(tfile.name)

elif "📹 Webcam" in input_source:
    st.markdown('<div style="background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,250,240,0.9) 100%); border-radius: 15px; padding: 20px; margin-bottom: 20px; border-top: 5px solid #ff6b35; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">', unsafe_allow_html=True)
    st.subheader("📹 Live Webcam Detection")
    
    st.warning("Note: This requires camera permissions. Please grant access when prompted.")
    
    # Create placeholders for webcam feed and controls
    webcam_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    st.subheader("📹 Live Webcam Detection")
            
    # Add video upload alternative
    with st.expander("Problems with webcam? Use a video file instead"):
        use_uploaded_video = st.checkbox("Use video file instead of webcam")
        
        if use_uploaded_video:
            uploaded_video = st.file_uploader("Upload a video to use instead of webcam", type=["mp4", "avi", "mov", "mkv"])
            if uploaded_video is not None:
                # Save uploaded video for webcam simulation
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_video.read())
                tfile.close()
                st.session_state['video_path'] = tfile.name
                st.success("Video uploaded successfully! Click 'Start Webcam' to begin.")
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Simple placeholder for webcam - no container wrapping
    webcam_container = st.empty()
    
    # Stats container
    st.markdown('''
    <div style="
        background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,250,240,0.9) 100%); 
        border-radius: 15px; 
        padding: 20px; 
        margin: 20px 0; 
        border-left: 5px solid #ff6b35;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    ">
        <h3 style="color: #ff6b35; margin-bottom: 15px;">Detection Stats</h3>
    ''', unsafe_allow_html=True)
    
    stats_placeholder = st.empty()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle webcam start/stop with simple buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Webcam"):
            st.session_state.webcam_running = True
    with col2:
        if st.button("⏹️ Stop Webcam"):
            st.session_state.webcam_running = False
    
    # Function to find available camera
    def find_working_camera():
        # Try multiple camera indices
        for index in [0, 1, -1, 2, 3]:  # Try more camera indices
            try:
                cap = cv2.VideoCapture(index)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        cap.release()
                        return index
                cap.release()
            except Exception:
                continue
        return None
    
    # Check if we should run webcam detection
    if st.session_state.webcam_running:
        # Check if we have an uploaded video to use instead
        use_uploaded = 'video_path' in st.session_state and use_uploaded_video
        
        if use_uploaded:
            try:
                # Use uploaded video as webcam feed
                video_path = st.session_state['video_path']
                cap = cv2.VideoCapture(video_path)
                
                if not cap.isOpened():
                    st.error("Failed to open uploaded video. The file might be corrupted.")
                    st.session_state.webcam_running = False
                else:
                    # Add note that we're using uploaded video
                    st.info("Using uploaded video for detection instead of webcam.")
                    
                    # Show webcam feed
                    webcam_feed = webcam_container.empty()
                    stats_display = stats_placeholder.empty()
                    
                    # Initialize detection counters
                    helmet_count = 0
                    head_count = 0
                    person_count = 0
                    
                    # Main video processing loop
                    while st.session_state.webcam_running:
                        ret, frame = cap.read()
                        
                        # Loop video when it reaches the end
                        if not ret:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame = cap.read()
                            if not ret:
                                st.error("Failed to loop video.")
                                break
                        
                        # Process frame
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        processed_frame, results = process_image(frame_rgb, detection_confidence)
                        
                        # Update detection counts
                        if len(results.boxes) > 0:
                            classes = results.boxes.cls.cpu().numpy()
                            names = results.names
                            
                            # Count detections by class
                            current_helmet = 0
                            current_head = 0
                            current_person = 0
                            
                            for cls in np.unique(classes):
                                class_name = names[int(cls)]
                                count = int(np.sum(classes == cls))
                                
                                if "helmet" in class_name.lower():
                                    current_helmet = count
                                elif "head" in class_name.lower():
                                    current_head = count
                                elif "person" in class_name.lower():
                                    current_person = count
                            
                            # Update counts
                            helmet_count = current_helmet
                            head_count = current_head
                            person_count = current_person
                        
                        # Display the processed frame directly
                        webcam_feed.image(processed_frame, channels="RGB", use_container_width=True)
                        
                        # Show safety status
                        if helmet_count > 0 and head_count == 0:
                            safety_status = "✅ SAFE: All personnel wearing helmets"
                            status_class = "status-safe"
                        elif head_count > 0:
                            safety_status = "⚠️ UNSAFE: Personnel without helmets detected"
                            status_class = "status-unsafe"
                        else:
                            safety_status = "ℹ️ No helmet status to report"
                            status_class = "status-neutral"
                        
                        stats_display.markdown(f"<div class='{status_class}'>{safety_status}</div>", unsafe_allow_html=True)
                        
                        # Add a small delay
                        time.sleep(0.1)
                    
                    # Clean up
                    cap.release()
            
            except Exception as e:
                st.error(f"Error processing video: {e}")
                st.session_state.webcam_running = False
        
        else:  # Use webcam
            # First check if we can find a working camera
            camera_index = find_working_camera()
            
            if camera_index is None:
                st.error("No working webcam found. Please check your camera connection and permissions.")
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-top: 15px;">
                    <h4 style="color: #ff6b35;">Troubleshooting Steps:</h4>
                    <ol>
                        <li>Verify your webcam is properly connected</li>
                        <li>Check if other applications can access your webcam</li>
                        <li>Restart your computer to reset the webcam</li>
                        <li>Ensure Streamlit has permission to access your camera in your browser settings</li>
                        <li>Try a different USB port if using an external webcam</li>
                        <li>Or use the "Use video file instead of webcam" option above</li>
                    </ol>
                </div>
                """, unsafe_allow_html=True)
                st.session_state.webcam_running = False
            else:
                try:
                    # Open webcam with the working index
                    cap = cv2.VideoCapture(camera_index)
                    
                    if not cap.isOpened():
                        st.error(f"Failed to open webcam (index {camera_index}). Please check your camera connection.")
                        st.session_state.webcam_running = False
                    else:
                        # Use direct references to the containers
                        webcam_feed = webcam_container
                        stats_display = stats_placeholder
                
                        # Initialize detection counters
                        helmet_count = 0
                        head_count = 0
                        person_count = 0
                
                        # Main webcam loop
                        while st.session_state.webcam_running:
                            # Read frame
                            ret, frame = cap.read()
                            
                            # Handle video loop if using uploaded video
                            if not ret:
                                if use_uploaded:
                                    # Reset video to beginning
                                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                    ret, frame = cap.read()
                                    if not ret:
                                        st.error("Failed to loop video. The file might be corrupted.")
                                        break
                                else:
                                    st.error("Failed to capture frame from webcam.")
                                    break
                            
                            # Convert and process frame
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            processed_frame, results = process_image(frame_rgb, detection_confidence)
                            
                            # Update detection counts
                            if len(results.boxes) > 0:
                                classes = results.boxes.cls.cpu().numpy()
                                names = results.names
                                
                                # Reset counts for this frame
                                current_helmet = 0
                                current_head = 0
                                current_person = 0
                                
                                # Count detections
                                for cls in np.unique(classes):
                                    class_name = names[int(cls)]
                                    count = int(np.sum(classes == cls))
                                    
                                    if "helmet" in class_name.lower():
                                        current_helmet = count
                                    elif "head" in class_name.lower():
                                        current_head = count
                                    elif "person" in class_name.lower():
                                        current_person = count
                                
                                # Update total counts
                                helmet_count = current_helmet
                                head_count = current_head
                                person_count = current_person
                            
                            # Display the processed frame directly
                            webcam_feed.image(processed_frame, channels="RGB", use_container_width=True)
                            
                            # Display live stats with fancy gradient text
                            stats_display.markdown(f"""
                            <div class="row" style="display: flex; gap: 15px; margin-top: 20px;">
                                <div class="metric-card metric-helmet" style="flex: 1; animation-delay: 0.1s;">
                                    <div style="font-size: 1.2rem; margin-bottom: 5px;">🪖 Helmets Detected</div>
                                    <div class="metric-value" style="background: -webkit-linear-gradient(45deg, #28a745, #5cb85c); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{helmet_count}</div>
                                </div>
                                <div class="metric-card metric-head" style="flex: 1; animation-delay: 0.3s;">
                                    <div style="font-size: 1.2rem; margin-bottom: 5px;">👤 Safety Violations</div>
                                    <div class="metric-value" style="background: -webkit-linear-gradient(45deg, #dc3545, #f86b7a); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{head_count}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show safety status
                            if helmet_count > 0 and head_count == 0:
                                safety_status = "✅ SAFE: All personnel wearing helmets"
                                status_class = "status-safe"
                            elif head_count > 0:
                                safety_status = "⚠️ UNSAFE: Personnel without helmets detected"
                                status_class = "status-unsafe"
                            else:
                                safety_status = "ℹ️ No helmet status to report"
                                status_class = "status-neutral"
                            
                            stats_display.markdown(f"<div class='{status_class}'>{safety_status}</div>", unsafe_allow_html=True)
                            
                            # Small delay to prevent UI freezing
                            time.sleep(0.05)
                
                        # Release webcam when stopped
                        cap.release()
                        st.session_state.webcam_running = False
        
                except Exception as e:
                    st.error(f"Error with webcam: {e}")
                    st.session_state.webcam_running = False

# Add more construction safety information 
st.markdown("""
<div class="safety-banner" style="background: linear-gradient(135deg, #2d936c 0%, #64c897 100%);">
    <div class="safety-banner-content">
        <div class="safety-icon">💡</div>
        <div class="safety-text">
            <div class="safety-title">SAFETY TIP:</div>
            <div>Regular safety training and properly fitted helmets can significantly reduce the risk of head injuries. Helmets should be replaced after any significant impact, even if no visible damage is present.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown(f"""
<div style="display: flex; align-items: center; justify-content: center; margin-bottom: 15px;">
    <span style="font-size: 2.5rem; margin-right: 10px;">🪖</span>
    <div>
        <div style="font-size: 1.4rem; font-weight: bold; background: -webkit-linear-gradient(45deg, #ff6b35, #f7c59f); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Construction Safety Monitor</div>
        <div style="font-size: 0.9rem; color: #666;">Powered by YOLOv8 | Model Performance: 95.7% mAP</div>
    </div>
</div>
<div style="margin-top: 10px; font-size: 0.9rem;">Developed by Selma B. | © 2025 | Version 1.0</div>
<div style="margin-top: 15px; font-size: 0.85rem;">
    <span style="background-color: #f1f3f5; padding: 5px 10px; border-radius: 20px; margin: 0 5px;">#SafetyFirst</span>
    <span style="background-color: #f1f3f5; padding: 5px 10px; border-radius: 20px; margin: 0 5px;">#AISafety</span>
    <span style="background-color: #f1f3f5; padding: 5px 10px; border-radius: 20px; margin: 0 5px;">#ConstructionTech</span>
</div>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
