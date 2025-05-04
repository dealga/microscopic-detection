import os
import shutil
import random
import math



# Paths
base_path = os.environ['BASE_PATH']
image_dir = os.path.join(base_path, "data", "images", "train")
label_dir = os.path.join(base_path, "data", "labels", "train")

base_image_dir = os.path.join(base_path, "data", "images")
base_label_dir = os.path.join(base_path, "data", "labels")

# Create val and test directories if not exist
for split in ['val', 'test']:
    os.makedirs(os.path.join(base_image_dir, split), exist_ok=True)
    os.makedirs(os.path.join(base_label_dir, split), exist_ok=True)

# Get all image files
all_images = [f for f in os.listdir(image_dir) if f.endswith('.tiff')]
random.shuffle(all_images)

total = len(all_images)

# Calculate at least 10% for val and test (rounded up)
val_size = max(1, math.ceil(0.1 * total))
test_size = max(1, math.ceil(0.1 * total))
train_size = total - val_size - test_size

# Edge case: not enough images
if train_size < 0:
    train_size = 0
    val_size = total // 2
    test_size = total - val_size

# Split the data
train_images = all_images[:train_size]
val_images = all_images[train_size:train_size + val_size]
test_images = all_images[train_size + val_size:]

# Function to move files
def move_files(file_list, split):
    for file in file_list:
        image_path = os.path.join(image_dir, file)
        label_path = os.path.join(label_dir, file.replace('.tiff', '.txt'))

        dest_image = os.path.join(base_image_dir, split, file)
        dest_label = os.path.join(base_label_dir, split, file.replace('.tiff', '.txt'))

        if os.path.exists(image_path):
            shutil.move(image_path, dest_image)
        if os.path.exists(label_path):
            shutil.move(label_path, dest_label)

# Move files
move_files(val_images, 'val')
move_files(test_images, 'test')

print(f"Total images: {total}")
print(f"Moved to validation: {len(val_images)}")
print(f"Moved to test: {len(test_images)}")
print(f"Remaining in train: {len(train_images)}")
