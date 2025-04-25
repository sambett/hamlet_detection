# 🪖 Construction Safety - Helmet Detection System

![Helmet Detection](https://img.shields.io/badge/Safety-Helmet%20Detection-orange)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

An advanced computer vision system that monitors construction sites for proper helmet usage, providing real-time safety detection and analytics.

## 🔍 Overview

This construction safety system uses YOLOv8 object detection to identify workers with and without helmets, helping enforce safety regulations and prevent workplace accidents. The system processes images, videos, and webcam feeds with a modern, intuitive interface.

### Key Results

| Class   | mAP50 | mAP50-95 |
|---------|-------|----------|
| Helmet  | 0.957 | 0.636    |
| Head    | 0.928 | 0.619    |
| Person  | 0.024 | 0.012    |
| Overall | 0.636 | 0.422    |

## ✨ Features

- **High-accuracy Detection**: >95% mAP for helmet and head detection
- **Multi-input Support**: Process images, videos, and live webcam feeds
- **Safety Analytics**: Real-time safety compliance tracking and alerts
- **Modern Interface**: Clean, responsive Streamlit web application
- **Comprehensive Visualization**: Confidence scores, bounding boxes, and timeline analytics
- **Optimized Performance**: Efficient processing for real-time monitoring

## 🔧 Installation

### Prerequisites

- Python 3.8+
- Git
- CPU or NVIDIA GPU (optional, for faster processing)

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/sambett/hamlet_detection.git
   cd hamlet_detection
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python run_app.py
   ```

4. Access the web interface at http://localhost:8501

## 🚀 Usage Guide

### Image Detection

1. Select "📷 Image" from the sidebar
2. Upload an image containing people with or without helmets
3. View safety status and detailed detection information

![Image Detection](assets/image_detection_example.jpg)

### Video Analysis

1. Select "🎥 Video" from the sidebar
2. Upload a video file (MP4, AVI, MOV, or MKV)
3. Click "Process Video" to analyze
4. Review the safety timeline and detection statistics

![Video Detection](assets/video_detection_example.jpg)

### Live Monitoring

1. Select "📹 Webcam" from the sidebar
2. Click "Start Webcam" to begin real-time detection
3. Monitor safety compliance in real-time

![Webcam Detection](assets/webcam_detection_example.jpg)

## 📊 Technical Details

### Model Information

- **Architecture**: YOLOv8n (nano)
- **Training Dataset**: 5,000 annotated construction site images
- **Classes**: Helmet, Head, Person
- **Input Size**: 640×640 pixels
- **Model Size**: ~6MB
- **Training Duration**: 50 epochs

### System Requirements

- **Minimum**: 2GB RAM, 2-core CPU
- **Recommended**: 8GB RAM, 4-core CPU or NVIDIA GPU
- **Disk Space**: ~100MB

## 🧠 How It Works

The system follows this processing pipeline:

1. **Input Acquisition**: Load image/video or capture webcam frame
2. **Inference**: Process through YOLOv8 model to detect objects
3. **Classification**: Identify helmets, heads, and people
4. **Safety Analysis**: Determine if safety violations exist
5. **Visualization**: Render detections with appropriate highlighting
6. **Analytics**: Calculate safety metrics and generate reports

## 🔄 Project Structure

```
helmet_detection/
├── app.py                     # Main Streamlit application
├── train.py                   # YOLOv8 model training script
├── prepare_dataset.py         # Dataset preparation utilities
├── model_utils.py             # Model loading and inference utilities
├── run_app.py                 # Application launcher
├── requirements.txt           # Python dependencies
├── dataset_yolo/              # Processed YOLO format dataset
│   ├── data.yaml              # Dataset configuration
│   ├── images/                # Training images
│   └── labels/                # YOLO format annotations
├── runs/                      # Training outputs and model weights
│   └── train/
│       └── helmet_detection3/
│           └── weights/
│               ├── best.pt    # Best model weights
│               └── last.pt    # Latest model weights
└── assets/                    # Images and resources
```

## 🌟 Future Enhancements

- Export to ONNX format for faster CPU inference
- Add other PPE detection capabilities (vests, gloves, etc.)
- Implement alert notification system (email, SMS)
- Add historical analytics and reporting features
- Support for multiple camera inputs and multi-threading

## 🔗 Related Resources

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OSHA Construction Safety Standards](https://www.osha.gov/construction)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions or feedback, please contact:
- GitHub: [@sambett](https://github.com/sambett)
