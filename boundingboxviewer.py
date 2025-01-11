import json
import os
import time
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

        # Magnification settings
        self.native_mag = 20
        self.level_downsamples = [1, 4, 16]
        self.level_dimensions = [
            (self.dimensions[0] // downsample, self.dimensions[1] // downsample)
            for downsample in self.level_downsamples
        ]

        # Metadata, annotations, categories
        self.metadata = metadata
        self.annotations = annotations or []
        self.categories = categories or {}

        self.print_level_info()
        self.print_annotations()

    def print_level_info(self):
        print(f"Available levels: {len(self.level_downsamples)}")
        print(f"Dimensions at each level: {self.level_dimensions}")
        print(f"Downsampling factors: {self.level_downsamples}")
        print(f"Available magnifications: {[self.native_mag / d for d in self.level_downsamples]}")

    def print_annotations(self):
        if not self.annotations:
            print("No annotations available.")
            return

        print("\nAnnotations:")
        for annotation in self.annotations:
            bbox = annotation["bbox"]
            category_name = self.categories.get(annotation["category_id"], "Unknown")
            print(f"  - BBox: {bbox}, Category: {category_name}, ID: {annotation['id']}")

    def get_magnification_level(self, target_mag):
        for level, downsample in enumerate(self.level_downsamples):
            current_mag = self.native_mag / downsample
            if current_mag <= target_mag:
                return level
        return len(self.level_downsamples) - 1

    def read_region(self, location, level, size):
        downsample = self.level_downsamples[level]
        x, y = location
        width, height = size

        # Calculate region
        left = int(x)
        top = int(y)
        right = int(left + width * downsample)
        bottom = int(top + height * downsample)

        # Ensure bounds
        left = max(0, min(left, self.dimensions[0]))
        top = max(0, min(top, self.dimensions[1]))
        right = max(0, min(right, self.dimensions[0]))
        bottom = max(0, min(bottom, self.dimensions[1]))

        return self.slide.crop((left, top, right, bottom))

    def smooth_scan(self, target_mag=20, window_size=(256, 256), speed=20, display=True, save_video=False,
                save_frames=False, frames_dir="frames"):
        level = self.get_magnification_level(target_mag)
        downsample = self.level_downsamples[level]

        print(f"Scanning at {target_mag}x magnification (Level {level})")
        x_steps = np.arange(0, self.dimensions[0] - window_size[0] * downsample + 1, speed * downsample)
        y_steps = np.arange(0, self.dimensions[1] - window_size[1] * downsample + 1, speed * downsample * 12.5)

        if save_video:
            os.makedirs("videos", exist_ok=True)
            video_path = f"videos/tiff_scan.mp4"
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, window_size)
            print(f"Video saved at {video_path}")

        if save_frames:
            os.makedirs(frames_dir, exist_ok=True)

        start_time = time.time()
        frame_count = 0
        current_step = 0

        try:
            for y in y_steps:
                for x in x_steps:
                    current_step += 1
                    frame_count += 1

                    # Read the region (frame)
                    region = self.read_region(
                        location=(x, y),
                        level=level,
                        size=window_size
                    )
                    frame = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2BGR)

                    # Save frame to the directory
                    if save_frames and (frame_count % 7 == 1 or frame_count == 1):
                        frame_path = os.path.join(frames_dir, f"frame_{current_step:05d}.tiff")
                        cv2.imwrite(frame_path, frame)
                        #print(f"Saved frame {current_step} to {frame_path}")

                    # FPS Calculation: Every 100 frames, calculate FPS
                    if current_step % 100 == 0:
                        elapsed_time = time.time() - start_time
                        fps = current_step / elapsed_time
                        print(f"FPS: {fps:.2f}")

                    # Display the frame (optional)
                    if display:
                        cv2.imshow("TIFF Scan", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            raise KeyboardInterrupt

                    # Save video (optional)
                    if save_video:
                        out.write(frame)

                    time.sleep(0.01)  # Delay to control speed of scanning

        except KeyboardInterrupt:
            print("\nScan interrupted by user")
        finally:
            print("\nScan complete")
            if save_video:
                out.release()
            if display:
                cv2.destroyAllWindows()



    def visualize_annotations(self):
        draw = ImageDraw.Draw(self.slide)
        for annotation in self.annotations:
            bbox = annotation["bbox"]
            category_name = self.categories.get(annotation["category_id"], "Unknown")
            draw.rectangle(bbox, outline="red", width=2)
            draw.text((bbox[0], bbox[1]), category_name, fill="yellow")
        self.slide.show()
        self.slide.save("annotated_tiff.tiff")

    def intersects_window(self, bbox, x_window, y_window, window_size=256):
        x_min, y_min, x_max, y_max = bbox
        intersects = not (x_max < x_window or x_min > x_window + window_size or
                          y_max < y_window or y_min > y_window + window_size)

        if intersects:
            new_x_min = max(0, x_min - x_window)
            new_y_min = max(0, y_min - y_window)
            new_x_max = min(window_size, x_max - x_window)
            new_y_max = min(window_size, y_max - y_window)
            return True, (new_x_min, new_y_min, new_x_max, new_y_max)
        return False, None


def load_metadata_and_annotations(json_file, file_name):
    with open(json_file, 'r') as f:
        data = json.load(f)

    metadata = next((img for img in data.get("images", []) if img["file_name"] == file_name), None)
    if not metadata:
        raise ValueError(f"No metadata found for file: {file_name}")

    annotations = [ann for ann in data.get("annotations", []) if ann["image_id"] == metadata["id"]]
    categories = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
    return metadata, annotations, categories


def view_tiff_with_annotations(slide_path, json_path, magnification=20):
    file_name = Path(slide_path).name
    metadata, annotations, categories = load_metadata_and_annotations(json_path, file_name)
    scanner = TIFFScanner(slide_path, metadata, annotations, categories)
    scanner.visualize_annotations()
    scanner.smooth_scan(target_mag=magnification, save_video=True, save_frames=True)


# Example usage
if __name__ == "__main__":
    tiff_path = r"C:\Users\user\Desktop\image-viewer\001.tiff"
    json_path = r"C:\Users\user\Desktop\image-viewer\MIDOG.json"
    view_tiff_with_annotations(tiff_path, json_path, magnification=20)