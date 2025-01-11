import cv2
import numpy as np
import os

# Folder path containing the TIFF images
folder_path = r'C:\Users\user\Desktop\image-viewer\frames'
# Output text file path
output_file = os.path.join(folder_path, 'green_box_coordinates.txt')
coordinate_file = os.path.join(folder_path, 'coordinates.txt')

# Create/overwrite the output file with a header
with open(output_file, 'w') as f:
    f.write("Green Box Normalized values as \n")
    f.write("===================\n")
    f.write("[class_id, x_center, y_center, width, height]\n")
    f.write("===================\n")

with open(coordinate_file, 'w') as f:
    f.write("Green Box Coordinates\n")
    f.write("===================\n")
    f.write("[min_x, min_y, max_x, max_y]\n")
    f.write("===================\n")

# Get all TIFF files in the folder
tiff_files = [f for f in os.listdir(folder_path) if f.endswith('.tiff')]

# Process each TIFF file
for image_file in tiff_files:
    image_path = os.path.join(folder_path, image_file)
    
    # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image was loaded correctly
    if image is None:
        print(f"Error: Could not load image {image_file}")
        continue

    # Define the green color in BGR format

    image_height, image_width, _ = image.shape
    green_bgr = (0, 255, 0)

    # Find the coordinates where the pixel value is green
    green_pixels = np.where(np.all(image == green_bgr, axis=-1))

    # Extract the coordinates (row, column) of the green pixels
    coordinates = list(zip(green_pixels[0], green_pixels[1]))

    # If green pixels are found, calculate the bounding box
    if coordinates:
        # Get the minimum and maximum coordinates
        min_y, min_x = np.min(coordinates, axis=0)
        max_y, max_x = np.max(coordinates, axis=0)

        # # Append the results to the text file
        # NOT NORMALIZED
        with open(coordinate_file, 'a') as f:
             f.write(f'{image_file}: [{min_x}, {min_y}, {max_x}, {max_y}]\n')

        # Calculate YOLO-style bounding box coordinates (normalized)
        center_x = (min_x + max_x) / 2 / image_width
        center_y = (min_y + max_y) / 2 / image_height
        width = (max_x - min_x) / image_width
        height = (max_y - min_y) / image_height

        # Write the results to the output file in YOLO format
        with open(output_file, 'a') as f:
            f.write(f'{image_file}: 0 {center_x} {center_y} {width} {height}\n')

        print(f"Processed {image_file}: Found green box")

        # Draw the bounding box (optional)

        ##FOR DEBUGGING PURPOSES
        
        # cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

        # Save the image with bounding box (optional)
        # output_image_path = os.path.join(folder_path, f'bbox_{image_file}')
        # cv2.imwrite(output_image_path, image)

    else:
        print(f"Processed {image_file}: No green box found")

print(f"\nResults have been saved to: {output_file}")