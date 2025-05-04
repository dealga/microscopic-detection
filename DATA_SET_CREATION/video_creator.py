import os
import cv2
import time
import numpy as np
from pathlib import Path
from PIL import Image


class TIFFScanner:
    def __init__(self, slide_path):
        path = Path(slide_path)
        if path.suffix.lower() not in ['.tif', '.tiff']:
            raise ValueError("Only TIFF formats are supported")

        self.slide = Image.open(slide_path)
        self.dimensions = self.slide.size

    def smooth_scan(self, output_dir, window_size=(256, 256), speed=20):
        print(f"Scanning with window size {window_size}")
        x_steps = np.arange(0, self.dimensions[0] - window_size[0] + 1, speed)
        y_steps = np.arange(0, self.dimensions[1] - window_size[1] + 1, speed * 12.5)

        # Prepare output directories
        video_path = os.path.join(output_dir, "tiff_scan.mp4")
        # frames_dir = os.path.join(output_dir, "frames")
        # os.makedirs(frames_dir, exist_ok=True)

        # Initialize video writer
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, window_size)

        frame_count = 0
        for step_y, y in enumerate(y_steps):
            for step_x, x in enumerate(x_steps):
                frame_count += 1

                region = self.slide.crop((
                    int(x), int(y),
                    int(x + window_size[0]), int(y + window_size[1])
                ))

                frame = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2BGR)

                # # Save one frame every ~7 to keep storage manageable
                # if frame_count % 7 == 1 or frame_count == 1:
                #     frame_path = os.path.join(frames_dir, f"frame_{frame_count:05d}.tiff")
                #     cv2.imwrite(frame_path, frame)

                out.write(frame)

        out.release()
        print(f"Saved video to {video_path}")


def process_all_tiffs_in_folder(input_folder, output_folder):
    tiff_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.tif', '.tiff'))]
    for tiff_file in tiff_files:
        tiff_path = os.path.join(input_folder, tiff_file)
        output_dir = os.path.join(output_folder, Path(tiff_file).stem)
        os.makedirs(output_dir, exist_ok=True)

        scanner = TIFFScanner(tiff_path)
        scanner.smooth_scan(output_dir=output_dir)


# Example usage
if __name__ == "__main__":
    input_folder = r"C:\Users\srivatsa gubbi\OneDrive\Desktop\Hackathon"
    output_folder = r"C:\Users\srivatsa gubbi\OneDrive\Desktop\Hackathon\trialvideo"

    process_all_tiffs_in_folder(input_folder, output_folder)