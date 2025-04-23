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

1. Clone or download this repository:
   ```bash
   git clone [repository-url]
   cd helmet-detection
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the model and setup the project:
   ```bash
   python setup.py
   ```

5. If you have a dataset path, run:
   ```bash
   python setup.py --dataset [path/to/your/dataset]
   ```

## 🏃‍♂️ Running the Application

Start the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default browser, typically at http://localhost:8501.

## 🏋️‍♂️ Training (Optional)

If you want to train the model on your dataset:

```bash
python train.py --dataset [path/to/your/dataset] --epochs 20 --batch 16 --export
```

Arguments:
- `--dataset`: Path to the dataset directory (required)
- `--epochs`: Number of training epochs (default: 20)
- `--batch`: Batch size (default: 16)
- `--img-size`: Image size (default: 640)
- `--export`: Export model to ONNX format after training

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

## 🔧 Technical Details

### Model Information

This project uses YOLOv8n (nano), a lightweight object detection model from Ultralytics. Key specifications:

- **Model Size**: ~6MB
- **Inference Speed**: Optimized for CPU usage
- **Input Resolution**: 640x640 pixels
- **Backbone**: CSPDarknet
- **Classes**: Detects Helmet, Head, and Person

### Dataset Requirements

The expected dataset format is YOLO format with the following structure:
```
dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

The `data.yaml` file should contain class information, for example:
```yaml
names:
  0: Helmet
  1: Person
  2: Head
```

### Performance Optimization

- **ONNX Export**: Model is exported to ONNX format for faster CPU inference
- **Frame Sampling**: Processes subset of video frames for smoother performance
- **Confidence Threshold**: Adjustable to filter out low-confidence detections

## 🤝 License

This project is licensed under the MIT License.