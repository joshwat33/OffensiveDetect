import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="en")  # Enable angle classification for curved text

# Load YOLO Model
model = YOLO(r"C:\yolov11\runs\detect\train9\weights\best.pt")

# Run Detection
image_path = r"C:\yolov11\datasets\test\images\img39_jpg.rf.70d2a58076b70743bb03db4386f614b3.jpg"

results = model(image_path)

# Read Image
image = cv2.imread(image_path)

for r in results:
    for box in r.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        roi = image[y1:y2, x1:x2]

        # Convert ROI to grayscale
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # OCR on detected region
        result = ocr.ocr(roi_gray, cls=True)

        # Print detected text
        for line in result:
            for word in line:
                print("Detected Text:", word[1][0])  # Extract text from OCR result
