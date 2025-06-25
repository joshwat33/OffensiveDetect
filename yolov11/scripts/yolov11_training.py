import torch
from ultralytics import YOLO

def train_model(data_path, epochs=100, batch_size=8, img_size=512, workers=2):
    # Ensure GPU is used if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load YOLO model (adjust to your version, e.g., yolov8x.pt)
    model = YOLO("yolov12.pt").to(device)
    
    # Start training with optimizations
    model.train(
        data=data_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        workers=workers,
        device=device,
        half=True  # Enables mixed precision training
    )
    
if __name__ == "__main__":
    data_path = "C:/yolov11/datasets/data.yaml"  # Path to dataset
    train_model(data_path)
