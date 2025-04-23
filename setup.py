import os
import sys
import subprocess
import argparse
from model_utils import download_model, setup_dataset

def check_requirements():
    """Check if all required packages are installed"""
    print("Checking requirements...")
    try:
        import ultralytics
        import streamlit
        import cv2
        import numpy
        import PIL
        import yaml
        print("All required packages are installed.")
        return True
    except ImportError as e:
        print(f"Missing required package: {e}")
        return False

def install_requirements():
    """Install required packages from requirements.txt"""
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Requirements installed successfully.")

def setup_model(dataset_path=None):
    """Download model and prepare dataset"""
    # Download YOLOv8n model
    model = download_model("yolov8n.pt")
    
    # If dataset path is provided, setup the dataset
    if dataset_path and os.path.exists(dataset_path):
        yaml_path, data_config = setup_dataset(dataset_path)
        print(f"Dataset setup complete. YAML path: {yaml_path}")
    
    print("Model setup complete.")
    return model

def main():
    parser = argparse.ArgumentParser(description="Setup Helmet Detection Project")
    parser.add_argument("--dataset", default=None, help="Path to the dataset directory")
    parser.add_argument("--install", action="store_true", help="Install requirements")
    args = parser.parse_args()
    
    # Check if requirements are installed
    if args.install or not check_requirements():
        install_requirements()
    
    # Download and setup model
    setup_model(args.dataset)
    
    print("\nSetup completed successfully!")
    print("To run the Streamlit app, use the command: streamlit run app.py")

if __name__ == "__main__":
    main()