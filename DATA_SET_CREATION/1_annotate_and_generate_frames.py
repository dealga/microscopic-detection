import os
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

class TIFFScanner:
    def __init__(self, slide_path, metadata=None, annotations=None, categories=None):
        path = Path(slide_path)
        if path.suffix.lower() not in ['.tif', '.tiff']:
            raise ValueError("Only TIFF formats are supported")

        self.slide = Image.open(slide_path)
        self.dimensions = self.slide.size

        self.metadata = metadata
        self.annotations = annotations or []
        self.categories = categories or {}

    def has_annotation_in_region(self, x, y, width, height):
        region_bounds = {
            'left': x,
            'top': y,
            'right': x + width,
            'bottom': y + height
        }

        for annotation in self.annotations:
            bbox = annotation["bbox"]
            ann_bounds = {
                'left': bbox[0],
                'top': bbox[1],
                'right': bbox[2],
                'bottom': bbox[3]
            }

            if not (region_bounds['right'] < ann_bounds['left'] or
                    region_bounds['left'] > ann_bounds['right'] or
                    region_bounds['bottom'] < ann_bounds['top'] or
                    region_bounds['top'] > ann_bounds['bottom']):
                return True
        return False

    def visualize_annotations(self, output_path):
        draw = ImageDraw.Draw(self.slide)
        for annotation in self.annotations:
            bbox = annotation["bbox"]
            category_id = annotation["category_id"]

            if category_id == 1:
                box_color = (0, 255, 0)  # Green
            elif category_id == 2:
                box_color = (255, 255, 0)  # Yellow
            else:
                box_color = (255, 0, 0)  # Red

            draw.rectangle(bbox, outline=box_color, width=2)
        self.slide.save(output_path)

    def smooth_scan(self, output_dir, window_size=(256, 256), speed=20):
        os.makedirs(output_dir, exist_ok=True)
        x_steps = np.arange(0, self.dimensions[0] - window_size[0] + 1, speed)
        y_steps = np.arange(0, self.dimensions[1] - window_size[1] + 1, speed * 8)

        tiff_base_name = Path(self.slide.filename).stem
        frame_count = 0
        saved_frames = 0
        total_frames = len(x_steps) * len(y_steps)

        for y in y_steps:
            for x in x_steps:
                frame_count += 1
                if self.has_annotation_in_region(x, y, *window_size):
                    if frame_count % 7 == 1 or frame_count == 1:
                        region = self.slide.crop((x, y, x + window_size[0], y + window_size[1]))
                        frame = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2BGR)

                        frame_path = os.path.join(output_dir, f"{tiff_base_name}_frame_{frame_count:05d}.tiff")
                        cv2.imwrite(frame_path, frame)
                        saved_frames += 1

                if frame_count % 100 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames. Saved {saved_frames}.")

        print(f"Scan complete. Total processed: {frame_count}, saved: {saved_frames} in {output_dir}")

def load_metadata_and_annotations(json_file, file_name):
    with open(json_file, 'r') as f:
        data = json.load(f)

    metadata = next((img for img in data.get("images", []) if img["file_name"] == file_name), None)
    if not metadata:
        raise ValueError(f"No metadata found for file: {file_name}")

    annotations = [ann for ann in data.get("annotations", []) if ann["image_id"] == metadata["id"]]
    categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
    return metadata, annotations, categories

def process_all_tiffs_in_folder(input_folder, json_path, output_folder):
    tiff_files = [f for f in os.listdir(input_folder) if f.endswith(('.tif', '.tiff'))]

    train_dir = os.path.join(output_folder, "images", "train")
    os.makedirs(train_dir, exist_ok=True)

    annotated_dir = os.path.join(os.environ['BASE_PATH'], "annotated_frames/temp_data")
    os.makedirs(annotated_dir, exist_ok=True)
    
    for tiff_file in tiff_files:
        print(f"\nProcessing {tiff_file}")
        tiff_path = os.path.join(input_folder, tiff_file)

        try:
            metadata, annotations, categories = load_metadata_and_annotations(json_path, tiff_file)
            scanner = TIFFScanner(tiff_path, metadata, annotations, categories)
        except (OSError, ValueError) as e:
            print(f"Skipping {tiff_file} - {e}")
            continue

        annotated_image_path = os.path.join(annotated_dir, f"annotated_{Path(tiff_file).stem}.tiff")
        scanner.visualize_annotations(annotated_image_path)
        scanner.smooth_scan(output_dir=train_dir)

if __name__ == "__main__":
    input_folder = os.path.join(os.environ['BASE_PATH'])
    json_path = os.path.join(os.environ['BASE_PATH'], "MIDOG.json")
    output_folder = os.path.join(os.environ['BASE_PATH'], "temp_data")
    os.makedirs(output_folder, exist_ok=True)

    process_all_tiffs_in_folder(input_folder, json_path, output_folder)

