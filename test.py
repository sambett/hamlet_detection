import os
import argparse
import cv2
import matplotlib.pyplot as plt
from model_utils import download_model, run_inference

def test_model(image_path, confidence=0.5, save_results=True):
    """Test the model on a single image"""
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    # Load model
    print("Loading model...")
    model = download_model("yolov8n.pt")
    if os.path.exists("best.pt"):
        print("Found trained model weights, using them instead.")
        model = download_model("best.pt")
    
    # Run inference
    print(f"Running inference on {image_path}...")
    results, annotated_img = run_inference(model, image_path, conf=confidence)
    
    # Display detection details
    print("\nDetection Results:")
    if len(results.boxes) == 0:
        print("No objects detected.")
    else:
        for i, box in enumerate(results.boxes):
            cls = int(box.cls[0].item())
            conf = box.conf[0].item()
            class_name = results.names[cls]
            coords = box.xyxy[0].tolist()
            print(f"{i+1}. {class_name}: {conf:.2f} at {[round(c, 2) for c in coords]}")
    
    # Save and display results
    if save_results:
        if not os.path.exists("results"):
            os.makedirs("results")
        
        # Create a figure with original and annotated images side by side
        plt.figure(figsize=(16, 8))
        
        # Original image
        plt.subplot(1, 2, 1)
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title("Original Image")
        plt.axis("off")
        
        # Annotated image
        plt.subplot(1, 2, 2)
        plt.imshow(annotated_img)
        plt.title("Detection Results")
        plt.axis("off")
        
        # Add detection details as text
        detection_text = "Detections:\n"
        if len(results.boxes) == 0:
            detection_text += "None found"
        else:
            for i, box in enumerate(results.boxes):
                cls = int(box.cls[0].item())
                conf = box.conf[0].item()
                class_name = results.names[cls]
                detection_text += f"{class_name}: {conf:.2f}\n"
        
        plt.figtext(0.02, 0.02, detection_text, fontsize=10, wrap=True)
        
        # Save the figure
        output_path = os.path.join("results", os.path.basename(image_path))
        plt.savefig(output_path)
        print(f"Results saved to {output_path}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Test Helmet Detection Model")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    args = parser.parse_args()
    
    test_model(args.image, confidence=args.conf, save_results=not args.no_save)

if __name__ == "__main__":
    main()