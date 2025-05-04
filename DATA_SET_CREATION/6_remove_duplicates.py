import os
import re
import shutil

base_path = os.environ['BASE_PATH']

image_dir = os.path.join(base_path, "data", "images", "train")
label_dir = os.path.join(base_path, "data", "labels", "train")

copy_image_dir = os.path.join(image_dir, "copies")
copy_label_dir = os.path.join(label_dir, "copies")

# Ensure copy directories exist
os.makedirs(copy_image_dir, exist_ok=True)
os.makedirs(copy_label_dir, exist_ok=True)

# Regular expression to extract image number and frame number
pattern = re.compile(r'(\d{3})_frame_(\d{5})\.tiff')

# Dictionary to store results
results = {}


# Iterate over all image files
for filename in sorted(os.listdir(image_dir)):
    match = pattern.match(filename)
    if match:
        image_num, frame_num = match.groups()
        frame_num = int(frame_num)

        if image_num not in results:
            results[image_num] = []
        results[image_num].append(frame_num)

# Filter frames with a step of 7
filtered_results = {}
total_duplicate_count = 0

for image_num, frames in results.items():
    consecutive_frames = [frames[0]]
    for i in range(1, len(frames)):
        if frames[i] - consecutive_frames[-1] == 7:
            consecutive_frames.append(frames[i])
        else:
            if len(consecutive_frames) > 1:
                selected_frames = consecutive_frames[::2]
                filtered_results.setdefault(image_num, []).extend(selected_frames)
                total_duplicate_count += len(consecutive_frames) - len(selected_frames)
            consecutive_frames = [frames[i]]
    if len(consecutive_frames) > 1:
        selected_frames = consecutive_frames[::2]
        filtered_results.setdefault(image_num, []).extend(selected_frames)
        total_duplicate_count += len(consecutive_frames) - len(selected_frames)

# Move the images and corresponding labels
for image_num, frames in filtered_results.items():
    for frame in frames:
        image_name = f"{image_num}_frame_{str(frame).zfill(5)}.tiff"
        label_name = image_name.replace('.tiff', '.txt')

        image_path = os.path.join(image_dir, image_name)
        label_path = os.path.join(label_dir, label_name)

        copy_image_path = os.path.join(copy_image_dir, image_name)
        copy_label_path = os.path.join(copy_label_dir, label_name)

        if os.path.exists(image_path):
            shutil.move(image_path, copy_image_path)
        if os.path.exists(label_path):
            shutil.move(label_path, copy_label_path)

# Delete the 'copies' directories and their contents
shutil.rmtree(copy_image_dir)
shutil.rmtree(copy_label_dir)

print(f"Total duplicate count: {total_duplicate_count}")
print("Temporary 'copies' directories have been deleted.")
