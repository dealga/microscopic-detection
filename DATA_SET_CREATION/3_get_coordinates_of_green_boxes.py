import cv2
import numpy as np
import os

# Base folder path containing the images
base_path = os.environ['BASE_PATH']
base_folder_path = os.path.join(base_path, "temp_data", "images", "train")

# Directory to save labels
output_labels_dir = os.path.join(base_path, "data", "labels", "train")
os.makedirs(output_labels_dir, exist_ok=True)

# Process only TIFF images in the folder
image_files = [f for f in os.listdir(base_folder_path) if f.endswith('.tiff')]

for image_file in image_files:

    image_path = os.path.join(base_folder_path, image_file)

    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image {image_file}")
        continue

    # Define green color range in BGR format
    green_lower = np.array([0, 255, 0], dtype=np.uint8)  # Lower bound for green
    green_upper = np.array([0, 255, 0], dtype=np.uint8)  # Upper bound for green
    green_mask = cv2.inRange(image, green_lower, green_upper)

    # Find contours in the green mask
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"No green bounding boxes found in {image_file}.")
        continue

    # Save bounding boxes to the labels folder
    txt_filename = os.path.join(output_labels_dir, f"{os.path.splitext(image_file)[0]}.txt")
    with open(txt_filename, 'w') as txt_file:
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w < 5 or h < 5:  # Skip very small detections
                continue

            # Normalize YOLO-style bounding box coordinates
            center_x = (x + w / 2) / image.shape[1]
            center_y = (y + h / 2) / image.shape[0]
            width = w / image.shape[1]
            height = h / image.shape[0]

            txt_file.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

    print(f"Processed {image_file}: Bounding boxes saved to {txt_filename}  ")

print("\nProcessing completed for all images.")
