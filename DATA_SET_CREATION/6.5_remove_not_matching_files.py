import os

# Define paths
base_path = os.environ['BASE_PATH']
image_dir = os.path.join(base_path, "data", "images", "train")
label_dir = os.path.join(base_path, "data", "labels", "train")

# Get base filenames without extensions
image_files = {os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith('.tiff')}
label_files = {os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.txt')}

# Files to delete
images_to_delete = image_files - label_files
labels_to_delete = label_files - image_files

# Delete unmatched image files
for base in images_to_delete:
    image_path = os.path.join(image_dir, base + '.tiff')
    if os.path.exists(image_path):
        os.remove(image_path)
        print(f"Deleted image: {image_path}")

# Delete unmatched label files
for base in labels_to_delete:
    label_path = os.path.join(label_dir, base + '.txt')
    if os.path.exists(label_path):
        os.remove(label_path)
        print(f"Deleted label: {label_path}")

print("\nCleanup complete.")
