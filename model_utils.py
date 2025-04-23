import os
import yaml
from ultralytics import YOLO
import cv2
import numpy as np

def download_model(model_name="yolov8n.pt"):
    """Download YOLOv8 model if needed"""
    if not os.path.exists(model_name):
        print(f"Downloading {model_name}...")
        model = YOLO(model_name)
        print(f"Model downloaded: {model_name}")
        return model
    else:
        print(f"Model already exists: {model_name}")
        return YOLO(model_name)

def train_model(data_yaml_path, model_name="yolov8n.pt", epochs=20, batch_size=16, image_size=640):
    """Train YOLOv8 model on custom dataset"""
    model = YOLO(model_name)
    
    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        patience=5,  # Early stopping patience
        save=True,
        name="helmet_detection_model",
        device='cpu'  # Use CPU for training
    )
    
    # Return path to best weights
    return model

def export_model(model, format="onnx"):
    """Export model to specified format for faster inference"""
    # Export the model to ONNX format
    print(f"Exporting model to {format} format...")
    export_path = model.export(format=format, dynamic=True)
    print(f"Model exported to: {export_path}")
    return export_path

def setup_dataset(dataset_path):
    """Setup and verify the dataset"""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    # Check for data.yaml
    yaml_path = os.path.join(dataset_path, "data.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"data.yaml not found in dataset directory: {dataset_path}")
    
    # Load and print dataset info
    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print("Dataset Information:")
    print(f"- Train Images: {data_config.get('train', 'Not specified')}")
    print(f"- Validation Images: {data_config.get('val', 'Not specified')}")
    print(f"- Test Images: {data_config.get('test', 'Not specified')}")
    print(f"- Classes: {data_config.get('names', {})}")
    
    return yaml_path, data_config

def get_test_images(dataset_path, num_images=5):
    """Get some test images from the dataset"""
    test_dir = os.path.join(dataset_path, "test", "images")
    if not os.path.exists(test_dir):
        test_dir = os.path.join(dataset_path, "valid", "images")
        if not os.path.exists(test_dir):
            return []
    
    test_images = []
    for file in os.listdir(test_dir)[:num_images]:
        if file.endswith((".jpg", ".jpeg", ".png")):
            test_images.append(os.path.join(test_dir, file))
    
    return test_images

def run_inference(model, image_path, conf=0.5):
    """Run inference on an image and return results"""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = model(img_rgb, conf=conf)
    
    # Return results and annotated image
    return results[0], results[0].plot()