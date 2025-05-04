import cv2
import numpy as np
import os

# Base folder path containing the images
base_path = os.environ['BASE_PATH']
base_folder_path = os.path.join(base_path, "temp_data", "images", "train")  # Folder for image data
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

    # Define yellow color range in BGR format
    yellow_lower = np.array([0, 255, 255], dtype=np.uint8)  # Lower bound for yellow
    yellow_upper = np.array([0, 255, 255], dtype=np.uint8)  # Upper bound for yellow
    yellow_mask = cv2.inRange(image, yellow_lower, yellow_upper)

    # Find contours in the yellow mask
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"No yellow bounding boxes found in {image_file}.")
        continue

    # Get the label file path
    txt_filename = os.path.join(output_labels_dir, f"{os.path.splitext(image_file)[0]}.txt")

    # Determine file mode: 'a' (append) if file exists, otherwise 'w' (create new file)
    file_mode = 'a' if os.path.exists(txt_filename) else 'w'

    # Save bounding boxes to the labels folder
    with open(txt_filename, file_mode) as txt_file:
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w < 5 or h < 5:  # Skip very small detections
                continue

            # Normalize YOLO-style bounding box coordinates
            center_x = (x + w / 2) / image.shape[1]
            center_y = (y + h / 2) / image.shape[0]
            width = w / image.shape[1]
            height = h / image.shape[0]

            txt_file.write(f"1 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

    print(f"Processed {image_file}: Bounding boxes {'appended' if file_mode == 'a' else 'saved'} to {txt_filename}")

print("\nProcessing completed for all images.")
