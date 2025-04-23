import os
import argparse
import matplotlib.pyplot as plt
from model_utils import download_model, train_model, export_model, setup_dataset, get_test_images, run_inference

def visualize_results(results, test_images, model):
    """Visualize detection results on test images"""
    print("Visualizing detection results on test images...")
    
    if not os.path.exists("results"):
        os.makedirs("results")
    
    plt.figure(figsize=(20, 20))
    for i, img_path in enumerate(test_images):
        # Run inference on test image
        results, annotated_img = run_inference(model, img_path, conf=0.3)
        
        # Save annotated image
        img_name = os.path.basename(img_path)
        plt.subplot(len(test_images), 2, i*2+1)
        plt.imshow(plt.imread(img_path))
        plt.title(f"Original: {img_name}")
        plt.axis("off")
        
        plt.subplot(len(test_images), 2, i*2+2)
        plt.imshow(annotated_img)
        plt.title(f"Detected: {img_name}")
        plt.axis("off")
        
        # Print detection details
        print(f"\nResults for {img_name}:")
        boxes = results.boxes
        for box in boxes:
            cls = int(box.cls[0].item())
            conf = box.conf[0].item()
            class_name = results.names[cls]
            print(f"- Detected: {class_name} with confidence {conf:.2f}")
    
    # Save the visualization
    plt.tight_layout()
    plt.savefig("results/test_detections.png")
    plt.close()
    print("Results saved to 'results/test_detections.png'")

def main():
    parser = argparse.ArgumentParser(description="Train Helmet Detection Model")
    parser.add_argument("--dataset", required=True, help="Path to the dataset directory")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument("--export", action="store_true", help="Export model to ONNX format")
    args = parser.parse_args()
    
    # Setup dataset
    yaml_path, data_config = setup_dataset(args.dataset)
    
    # Download or load model
    base_model = download_model("yolov8n.pt")
    
    # Train model
    print(f"Training model for {args.epochs} epochs...")
    trained_model = train_model(
        yaml_path, 
        model_name="yolov8n.pt", 
        epochs=args.epochs,
        batch_size=args.batch,
        image_size=args.img_size
    )
    
    # Export model if requested
    if args.export:
        export_path = export_model(trained_model, format="onnx")
        print(f"Model exported to: {export_path}")
    
    # Get some test images to visualize results
    test_images = get_test_images(args.dataset, num_images=5)
    if test_images:
        visualize_results(None, test_images, trained_model)
    
    print("\nTraining completed successfully!")
    print("To run the Streamlit app, use the command: streamlit run app.py")

if __name__ == "__main__":
    main()