import os
import cv2
import json
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

        # Magnification settings
        self.native_mag = 20
        self.level_downsamples = [1, 4, 16]
        self.level_dimensions = [
            (self.dimensions[0] // downsample, self.dimensions[1] // downsample)
            for downsample in self.level_downsamples
        ]

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

def extract_frame_info(frame_name):
    """
    Extract the base name and frame number from frame filename
    Example: 001_frame_00561.tiff -> ('001', 561)
    """
    parts = frame_name.split('_frame_')
    if len(parts) != 2:
        return None, None
    
    base_name = parts[0]
    try:
        frame_num = int(parts[1].split('.')[0])
        return base_name, frame_num
    except ValueError:
        return None, None

def extract_clean_frames(input_folder, annotated_frames_folder, output_folder, magnification=20, window_size=(256, 256)):
    """
    Extract clean frames from original WSI files only if corresponding annotated frames exist
    
    Args:
        input_folder: Folder containing original WSI TIFF files
        annotated_frames_folder: Folder containing annotated frames (output\images\train)
        output_folder: Folder to save clean frames
        magnification: Target magnification
        window_size: Size of frames to extract
    """
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all annotated frames
    annotated_frames = [f for f in os.listdir(annotated_frames_folder) if f.endswith(('.tif', '.tiff'))]
    
    # Map each base name to a dict of frame numbers and locations
    frame_mapping = {}
    for frame_name in annotated_frames:
        base_name, frame_num = extract_frame_info(frame_name)
        if base_name and frame_num is not None:
            if base_name not in frame_mapping:
                frame_mapping[base_name] = set()
            frame_mapping[base_name].add(frame_num)
    
    # Process each WSI file
    wsi_files = [f for f in os.listdir(input_folder) if f.endswith(('.tif', '.tiff'))]
    total_wsi = len(wsi_files)
    frames_extracted = 0
    
    for i, wsi_file in enumerate(wsi_files, 1):
        base_name = Path(wsi_file).stem
        print(f"Processing WSI {i}/{total_wsi}: {wsi_file}")
        
        # Skip if no frames from this WSI
        if base_name not in frame_mapping:
            print(f"No annotated frames found for {base_name}, skipping")
            continue
            
        frame_numbers = frame_mapping[base_name]
        print(f"Found {len(frame_numbers)} annotated frames for {base_name}")
        
        try:
            wsi_path = os.path.join(input_folder, wsi_file)
            scanner = TIFFScanner(wsi_path)
            level = scanner.get_magnification_level(magnification)
            downsample = scanner.level_downsamples[level]
            
            # Calculate scanning parameters (same as in original code)
            x_steps = np.arange(0, scanner.dimensions[0] - window_size[0] * downsample + 1, 20 * downsample)
            y_steps = np.arange(0, scanner.dimensions[1] - window_size[1] * downsample + 1, 20 * downsample * 8)
            
            # Scan each frame
            frame_count = 0
            for y in y_steps:
                for x in x_steps:
                    frame_count += 1
                    
                    # Only process frames that match annotated frame numbers
                    if frame_count in frame_numbers:
                        # Extract region
                        region = scanner.read_region(
                            location=(x, y),
                            level=level,
                            size=window_size
                        )
                        frame = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2BGR)
                        
                        # Save with the same naming convention
                        output_path = os.path.join(output_folder, f"{base_name}_frame_{frame_count:05d}.tiff")
                        cv2.imwrite(output_path, frame)
                        frames_extracted += 1
                        
                        # Print progress every 10 frames
                        if frames_extracted % 10 == 0:
                            print(f"Extracted {frames_extracted} frames so far")
            
        except Exception as e:
            print(f"Error processing {wsi_file}: {e}")
            continue
    
    print(f"\nExtraction complete. Total clean frames extracted: {frames_extracted}")

if __name__ == "__main__":
    # Paths
    original_wsi_folder = os.path.join(os.environ['BASE_PATH'])  # Original WSI files
    annotated_frames_folder = os.path.join(os.environ['BASE_PATH'], "temp_data", "images", "train")  # Folder with annotated frames
    clean_frames_output = os.path.join(os.environ['BASE_PATH'], "data", "images", "train")  # Output folder for clean frames

    # Create the clean frames output folder if it doesn't exist
    os.makedirs(clean_frames_output, exist_ok=True)
    # Extract clean frames
    extract_clean_frames(
        input_folder=original_wsi_folder,
        annotated_frames_folder=annotated_frames_folder,
        output_folder=clean_frames_output,
        magnification=20,
        window_size=(256, 256)
    )