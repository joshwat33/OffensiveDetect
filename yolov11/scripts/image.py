import os

# Path to the dataset
train_path = "C:/yolov11/datasets/train/images"
valid_path = "C:/yolov11/datasets/valid/images"
test_path = "C:/yolov11/datasets/test/images"

# Count the number of images in each folder
train_images = len([f for f in os.listdir(train_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
valid_images = len([f for f in os.listdir(valid_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
test_images = len([f for f in os.listdir(test_path) if f.endswith(('.jpg', '.png', '.jpeg'))])

print(f"Training Images: {train_images}")
print(f"Validation Images: {valid_images}")
print(f"Test Images: {test_images}")
print(f"Total Images: {train_images + valid_images + test_images}")
