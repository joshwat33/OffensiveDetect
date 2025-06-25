import cv2
import pytesseract
from ultralytics import YOLO

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load Model
model = YOLO(r"C:\yolov11\runs\detect\train38\weights\best.pt")


# Run Detection
image_path = r"C:\Users\sketc\Downloads\61GhJxSN5pL.jpg"

results = model(image_path)

# Extract Text
image = cv2.imread(image_path)
for r in results:
    for box in r.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        roi = image[y1:y2, x1:x2]
        text = pytesseract.image_to_string(roi)
        print("Detected Text:", text)
