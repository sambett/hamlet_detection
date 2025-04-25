from ultralytics import YOLO
import argparse
import os
import time

def train_model(data_yaml, model_name, epochs, batch_size, imgsz, device, project, name):
    """
    Train a YOLOv8 model
    """
    # Load the model
    model = YOLO(model_name)
    
    # Train the model
    print(f"Starting training for {epochs} epochs...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        verbose=True
    )
    
    print("Training completed!")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLOv8 model for helmet detection')
    parser.add_argument('--data', type=str, default='dataset_yolo/data.yaml', help='Path to data.yaml file')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Model size (yolov8n.pt, yolov8s.pt, etc.)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='0', help='Device to use (0, 0,1,2,3, cpu)')
    parser.add_argument('--project', type=str, default='runs/train', help='Project name')
    parser.add_argument('--name', type=str, default='helmet_detection', help='Experiment name')
    
    args = parser.parse_args()
    
    # Make data.yaml path absolute if it's relative
    if not os.path.isabs(args.data):
        args.data = os.path.join(os.getcwd(), args.data)
    
    start_time = time.time()
    
    # Train the model
    results = train_model(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Best model saved to: {results.best}")
