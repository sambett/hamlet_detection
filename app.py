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
    page_title="Helmet Detection System",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a visually appealing UI
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stApp {
        background: linear-gradient(to right, #f8f9fa, #e9ecef);
    }
    .title-container {
        background: linear-gradient(to right, #007bff, #6610f2);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .title-container h1 {
        color: white !important;
        margin: 0;
    }
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    .stats-card {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton > button {
        background-color: #007bff;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0069d9;
    }
    .progress-bar {
        height: 10px;
        border-radius: 5px;
        margin-top: 0.5rem;
    }
    .divider {
        height: 3px;
        background: linear-gradient(to right, #007bff, #6610f2);
        margin: 1rem 0;
        border-radius: 2px;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #6c757d;
    }
    .stProgress > div > div > div {
        background-color: #007bff;
    }
</style>
""", unsafe_allow_html=True)

# Header with custom styling
st.markdown('<div class="title-container">', unsafe_allow_html=True)
st.title("🪖 Helmet Detection System")
st.markdown("A fast, lightweight system to detect people wearing (or not wearing) helmets")
st.markdown('</div>', unsafe_allow_html=True)

# Load the YOLOv8n model
@st.cache_resource
def load_model():
    """Load and cache the detection model"""
    try:
        # Try loading ONNX format first (faster for CPU)
        if os.path.exists("best.onnx"):
            return YOLO("best.onnx")
        # Fall back to PT format if ONNX isn't available
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
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⚙️ Detection Settings")
    
    # Detection confidence slider
    detection_confidence = st.slider(
        "Detection Confidence",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Adjust how confident the model must be to report a detection"
    )
    
    # Class filter
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Input selection
    input_source = st.radio(
        "Select Input Source",
        ["📷 Image", "🎥 Video", "📹 Webcam"],
        help="Choose the type of input you want to analyze"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # About section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("ℹ️ About")
    st.markdown("""
    This app uses YOLOv8n, a CPU-friendly object detection model to detect people wearing or not wearing helmets in images and videos.
    
    * Fast and lightweight
    * Runs on CPU
    * Built with Ultralytics YOLOv8
    * Processes images, videos, and webcam streams
    """)
    st.markdown('</div>', unsafe_allow_html=True)

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
    results = model(image, conf=confidence)
    annotated_img = results[0].plot()
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
    
    # Show confidence levels with visual progress bars
    st.markdown("### Confidence Levels")
    
    for i, box in enumerate(results.boxes):
        cls = int(box.cls[0].item())
        conf = box.conf[0].item()
        class_name = names[cls]
        
        # Create a colored progress bar based on confidence
        conf_pct = conf * 100
        color = "green" if conf >= 0.7 else "orange" if conf >= 0.5 else "red"
        
        st.markdown(f"""
        <div style="margin-bottom: 10px;">
            <div style="display: flex; justify-content: space-between;">
                <span>{class_name} #{i+1}</span>
                <span>{conf_pct:.1f}%</span>
            </div>
            <div class="progress-bar" style="background-color: #e9ecef;">
                <div style="width: {conf_pct}%; height: 100%; background-color: {color}; border-radius: 5px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
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
            st.image(image_np, channels="RGB", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Detection Results")
            st.image(processed_img, channels="RGB", use_column_width=True)
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
                    video_display.image(processed_frame, channels="RGB", use_column_width=True)
                
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
            
            # Create a row of metrics
            summary_cols = st.columns(3)
            with summary_cols[0]:
                st.metric("🪖 Helmets", helmet_count)
            with summary_cols[1]:
                st.metric("👤 Heads", head_count)
            with summary_cols[2]:
                st.metric("👷 People", person_count)
            
            # Create timeline visualization if we have detections
            if detection_history:
                st.subheader("Detection Timeline")
                
                # Convert to DataFrame for visualization
                df = pd.DataFrame(detection_history)
                
                # Create a line chart showing detections over time
                chart = alt.Chart(df).mark_circle(size=100).encode(
                    x=alt.X('time:Q', title='Time (seconds)'),
                    y=alt.Y('class:N', title='Detection Class'),
                    color=alt.Color('class:N', 
                                   scale=alt.Scale(
                                       domain=['Helmet', 'Head', 'Person'],
                                       range=['#28a745', '#dc3545', '#17a2b8']
                                   )),
                    size=alt.Size('count:Q', title='Count', scale=alt.Scale(range=[50, 200])),
                    tooltip=['time:Q', 'class:N', 'count:Q', 'frame:Q']
                ).properties(
                    width=600,
                    height=300,
                    title='Detections Over Time'
                ).interactive()
                
                st.altair_chart(chart, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Clean up the temporary file
            os.unlink(tfile.name)

elif "📹 Webcam" in input_source:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📹 Live Webcam Detection")
    
    st.warning("Note: This requires camera permissions. Please grant access when prompted.")
    
    # Create placeholders for webcam feed and controls
    webcam_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    # Control buttons for webcam
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("▶️ Start Webcam")
    with col2:
        stop_button = st.button("⏹️ Stop Webcam")
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize webcam session state
    if 'webcam_running' not in st.session_state:
        st.session_state.webcam_running = False
    
    # Handle webcam start/stop
    if start_button:
        st.session_state.webcam_running = True
    
    if stop_button:
        st.session_state.webcam_running = False
        st.success("Webcam stopped")
    
    # Run webcam detection
    if st.session_state.webcam_running:
        try:
            # Open webcam
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Failed to open webcam. Please check your camera connection.")
            else:
                # Create placeholders within cards
                webcam_placeholder.markdown('<div class="card">', unsafe_allow_html=True)
                webcam_feed = webcam_placeholder.empty()
                webcam_placeholder.markdown('</div>', unsafe_allow_html=True)
                
                stats_placeholder.markdown('<div class="card">', unsafe_allow_html=True)
                stats_display = stats_placeholder.empty()
                stats_placeholder.markdown('</div>', unsafe_allow_html=True)
                
                # Initialize detection counters
                helmet_count = 0
                head_count = 0
                person_count = 0
                
                # Main webcam loop
                while st.session_state.webcam_running:
                    # Read frame
                    ret, frame = cap.read()
                    if not ret:
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
                    
                    # Display the processed frame
                    webcam_feed.image(processed_frame, channels="RGB", use_column_width=True)
                    
                    # Display live stats
                    metric_cols = stats_display.columns(3)
                    with metric_cols[0]:
                        metric_cols[0].metric("🪖 Helmets", helmet_count)
                    with metric_cols[1]:
                        metric_cols[1].metric("👤 Heads", head_count)
                    with metric_cols[2]:
                        metric_cols[2].metric("👷 People", person_count)
                    
                    # Small delay to prevent UI freezing
                    time.sleep(0.05)
                
                # Release webcam when stopped
                cap.release()
                st.session_state.webcam_running = False
        
        except Exception as e:
            st.error(f"Error with webcam: {e}")
            st.session_state.webcam_running = False

# Footer
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("Helmet Detection System | Built with YOLOv8n and Streamlit")
st.markdown("Optimized for CPU usage and real-time detection")
st.markdown('</div>', unsafe_allow_html=True)