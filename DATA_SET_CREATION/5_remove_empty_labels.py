import os

base_path = os.environ['BASE_PATH']
base_folder_path = os.path.join(os.environ['BASE_PATH'], "data/images/train")
output_labels_dir = os.path.join(os.environ['BASE_PATH'], "data/labels/train")


# Process only TXT files in the labels directory
label_files = [f for f in os.listdir(output_labels_dir) if f.endswith('.txt')]

for label_file in label_files:
    label_path = os.path.join(output_labels_dir, label_file)

    # Check if the label file is empty
    if os.path.getsize(label_path) == 0:
        print(f"Empty label file found: {label_file}. Removing corresponding image and label.")

        # Remove the label file
        os.remove(label_path)

        # Find corresponding image (assuming .tiff format)
        image_filename = os.path.splitext(label_file)[0] + ".tiff"
        image_path = os.path.join(base_folder_path, image_filename)

        # Remove the image if it exists
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Deleted: {image_path}")
        else:
            print(f"Warning: Image {image_filename} not found.")

print("Cleanup complete. Empty labels and their images have been removed.")
