import torch
from ultralytics import YOLO

def evaluate_model(weights_path, data_path):
    model = YOLO(weights_path)
    results = model.val(data=data_path)

    print("Evaluation Metrics:")
    print(f"mAP@0.5: {results.box.map50:.4f}")  
    print(f"mAP@0.5-0.95: {results.box.map:.4f}")  
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")

if __name__ == "__main__":
    weights_path = r"runs/detect/train37/weights/best.pt"  # Set default weights
    data_path = r"C:\yolov11\datasets\data.yaml"  # Set default dataset path
    
    evaluate_model(weights_path, data_path)
