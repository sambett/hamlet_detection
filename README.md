# 🪖 Helmet Detection System

A fast, lightweight, and accurate helmet detection system powered by YOLOv8n. This system identifies whether people in images or video streams are wearing helmets, with a clean and modern Streamlit interface.

## ✨ Features

- **Lightweight & Fast**: Uses YOLOv8n nano model for efficient CPU inference
- **Multiple Input Types**: Supports image upload, video processing, and webcam streams
- **Modern UI**: Clean, intuitive Streamlit interface with real-time detection
- **Detection Analytics**: Visualizes detection results with confidence scores
- **Performance Optimized**: Sample rate adjustment for smooth video processing

## 🚀 Quick Setup

### Prerequisites

- Python 3.8+
- Pip package manager

### Installation Steps

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the YOLOv8n model:
   ```bash
   python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

4. Access the interface in your web browser at http://localhost:8501

## 📋 Usage Guide

### Image Detection

1. Select "📷 Image" from the sidebar
2. Upload an image containing people with or without helmets
3. Adjust the detection confidence threshold if needed
4. View the annotated image and detection statistics

### Video Detection

1. Select "🎥 Video" from the sidebar
2. Upload a video file
3. View the video information
4. Click "Process Video" to start detection
5. Monitor progress and view the detection timeline

### Webcam Detection

1. Select "📹 Webcam" from the sidebar
2. Click "▶️ Start Webcam" to begin live detection
3. Allow camera permissions when prompted
4. View real-time detection results
5. Click "⏹️ Stop Webcam" when finished

## 🏋️‍♂️ Training (Optional)

To fine-tune the model on the Helmet Detection dataset:

1. Create a training script:
   ```bash
   python train.py --dataset [path-to-dataset] --epochs 20 --export
   ```

2. Test the model:
   ```bash
   python test.py --image [path-to-test-image]
   ```

## 🔧 Technical Details

### Dataset Information

This system works with the Helmet Detection dataset containing 5,000 images with bounding box annotations for three classes:
- Helmet
- Person
- Head

The dataset uses PASCAL VOC format and provides a balanced distribution of examples showing people with and without helmets in various scenarios.

### Model Information

This project uses YOLOv8n (nano), a lightweight object detection model from Ultralytics:

- **Model Size**: ~6MB
- **Inference Speed**: Optimized for CPU usage
- **Input Resolution**: 640x640 pixels
- **Backbone**: CSPDarknet

### Performance Optimization

- **ONNX Export**: Model is exported to ONNX format for faster CPU inference
- **Frame Sampling**: Processes subset of video frames for smoother performance
- **Confidence Threshold**: Adjustable to filter out low-confidence detections