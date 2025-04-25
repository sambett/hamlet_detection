from ultralytics import YOLO
import argparse
import os
import time
import cv2
import numpy as np

def test_model(model_path, data_yaml, imgsz, device):
    """
    Test a YOLOv8 model on the validation set
    """
    # Load the model
    model = YOLO(model_path)
    
    # Validate the model
    print(f"Testing model {model_path}...")
    results = model.val(data=data_yaml, imgsz=imgsz, device=device)
    
    print("Testing completed!")
    return results

def test_on_image(model_path, image_path, conf=0.25, save=True, imgsz=640):
    """
    Run inference on a single image
    """
    # Load the model
    model = YOLO(model_path)
    
    # Run inference
    print(f"Running inference on {image_path}...")
    results = model.predict(source=image_path, conf=conf, save=save, imgsz=imgsz)
    
    # Display results
    for r in results:
        print(f"Detected {len(r.boxes)} objects:")
        
        # Get class counts
        classes = r.boxes.cls.cpu().numpy().astype(int)
        unique_classes, counts = np.unique(classes, return_counts=True)
        
        for i, cls in enumerate(unique_classes):
            class_name = model.names[cls]
            count = counts[i]
            print(f"  - {class_name}: {count}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test YOLOv8 model for helmet detection')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.pt file)')
    parser.add_argument('--data', type=str, default='dataset_yolo/data.yaml', help='Path to data.yaml file')
    parser.add_argument('--image', type=str, help='Path to test image (optional)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='0', help='Device to use (0, 0,1,2,3, cpu)')
    
    args = parser.parse_args()
    
    # Make paths absolute if they're relative
    if not os.path.isabs(args.data):
        args.data = os.path.join(os.getcwd(), args.data)
    
    if args.image:
        # Run inference on a single image
        if not os.path.isabs(args.image):
            args.image = os.path.join(os.getcwd(), args.image)
        
        start_time = time.time()
        results = test_on_image(args.model, args.image, args.conf, save=True, imgsz=args.imgsz)
        end_time = time.time()
        
        print(f"Inference time: {(end_time - start_time):.4f} seconds")
    else:
        # Run validation on test/val set
        start_time = time.time()
        results = test_model(args.model, args.data, args.imgsz, args.device)
        end_time = time.time()
        
        print(f"Validation time: {(end_time - start_time):.4f} seconds")
        print(f"Results: mAP50 = {results.box.map50:.4f}, mAP50-95 = {results.box.map:.4f}")
