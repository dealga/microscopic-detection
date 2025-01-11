import numpy as np
from PIL import Image
import cv2
import time
from pathlib import Path
import os

class TIFFScanner:
    def __init__(self, slide_path):
        """
        Initialize the TIFF scanner
        
        Args:
            slide_path (str): Path to the TIFF file
        """
        # Verify file format
        path = Path(slide_path)
        if path.suffix.lower() not in ['.tif', '.tiff']:
            raise ValueError("Only TIFF formats are supported")
            
        self.slide = Image.open(slide_path)
        self.dimensions = self.slide.size
        
        # Match OpenSlide-like levels exactly
        self.native_mag = 20  # Base magnification
        self.level_downsamples = [1, 4, 16]  # Matches 20x, 5x, 1.25x magnifications
        self.level_dimensions = [
            (self.dimensions[0] // downsample, self.dimensions[1] // downsample)
            for downsample in self.level_downsamples
        ]

        self.print_level_info()
        
    def print_level_info(self):
        """
        Print information about available levels, their dimensions, and downsampling factors.
        """
        print(f"Available levels: {len(self.level_downsamples)}")
        print(f"Dimensions at each level: {self.level_dimensions}")
        print(f"Downsampling factors: {self.level_downsamples}")
        print(f"Available magnifications: {[self.native_mag / d for d in self.level_downsamples]}")

    def get_magnification_level(self, target_mag):
        """
        Get the appropriate level for desired magnification
        """
        for level, downsample in enumerate(self.level_downsamples):
            current_mag = self.native_mag / downsample
            if current_mag <= target_mag:
                return level
        return len(self.level_downsamples) - 1

    def read_region(self, location, level, size):
        """
        Read a region from the TIFF file at the specified level
        """
        downsample = self.level_downsamples[level]
        x, y = location
        width, height = size
        
        # Calculate the region to extract
        left = int(x)
        top = int(y)
        right = int(left + width * downsample)
        bottom = int(top + height * downsample)
        
        # Ensure coordinates are within bounds
        left = max(0, min(left, self.dimensions[0]))
        top = max(0, min(top, self.dimensions[1]))
        right = max(0, min(right, self.dimensions[0]))
        bottom = max(0, min(bottom, self.dimensions[1]))
        
        # Extract region from base image
        region = self.slide.crop((left, top, right, bottom))
        
        # Resize to target size if necessary
        # if downsample != 1:
        #     region = region.resize(size, Image.Resampling.LANCZOS)
            
        return region

    def smooth_scan(self, target_mag=20, window_size=(1024, 768), 
                speed=100, display=True, save_video=False, save_frames=False, frames_dir="frames"):
        """
        Perform smooth scanning of the TIFF and optionally save frames as individual image files.
        """
        level = self.get_magnification_level(target_mag)
        downsample = self.level_downsamples[level]
        
        # Print scanning information
        print(f"\nScanning with:")
        print(f"Target magnification: {target_mag}x")
        print(f"Using level: {level} (downsample: {downsample}x)")
        print(f"Effective magnification: {self.native_mag/downsample}x")
        print(f"Full image size: {self.dimensions}")
        print(f"Window size: {window_size}")
        
        if save_video:
            base_dir = r'C:\Users\user\Desktop\image-viewer\videos'
            base_filename = 'tiff_scan'
            extension = '.mp4'
            filename = os.path.join(base_dir, f"{base_filename}{extension}")
            
            counter = 1
            while os.path.exists(filename):
                filename = os.path.join(base_dir, f"{base_filename}_{counter}{extension}")
                counter += 1
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, 30.0, window_size)
            print(f"Video will be saved to: {filename}")

        # Prepare folder for saving frames
        if save_frames:
            os.makedirs(frames_dir, exist_ok=True)
            print(f"Frames will be saved to folder: {frames_dir}")
        
        # Calculate scanning steps in base coordinates
        x_steps = np.arange(0, self.dimensions[0] - window_size[0] * downsample + 1, speed * downsample)
        y_steps = np.arange(0, self.dimensions[1] - window_size[1] * downsample + 1, speed * downsample * 12.5)    ##12.5 - 13 works best for 256 x 256 window size 
        
        total_steps = len(x_steps) * len(y_steps)
        current_step = 0

        # save_first_frame = True  # Flag to save only the first frame
    
        # try:
        #     for y in [0]:  # Scanning the first row only
        #         for x in [0]:  # Scanning the first column only
        #             region = self.read_region(
        #                 location=(x, y),
        #                 level=level,
        #                 size=window_size
        #             )
                    
        #             # Save the first frame in TIFF format
        #             if save_first_frame:
        #                 region.save("first_frame2.tiff", format='TIFF')
        #                 print("First frame saved as 'first_frame.tiff' with original resolution.")
        #                 save_first_frame = False

        #             # Optionally display the frame
        #             if display:
        #                 frame = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2BGR)
        #                 cv2.imshow('TIFF Scan - First Frame', frame)
        #                 if cv2.waitKey(0) & 0xFF == ord('q'):
        #                     raise KeyboardInterrupt
        #             return  # Exit after saving and displaying the first frame
        
        # except KeyboardInterrupt:
        #     print("\nScan interrupted by user")
        # finally:
        #     print("\nScan complete")
        #     if display:
        #         cv2.destroyAllWindows()



        ##to calculate fps

        # start_time = time.time()  # Start time for FPS calculation

        # try:
        #     for y in y_steps:
        #         for x in x_steps:
        #             current_step += 1
                    
        #             # Read the region (frame)
        #             region = self.read_region(
        #                 location=(x, y),
        #                 level=level,
        #                 size=window_size
        #             )
                    
        #             frame = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2BGR)
                    
        #             # FPS Calculation: Every 100 frames, calculate FPS
        #             if current_step % 100 == 0:
        #                 elapsed_time = time.time() - start_time  # Time elapsed
        #                 fps = current_step / elapsed_time  # Calculate FPS
        #                 print(f"FPS: {fps:.2f}")
                    
        #             # Display the frame (optional)
        #             if display:
        #                 cv2.imshow('TIFF Scan', frame)
        #                 if cv2.waitKey(1) & 0xFF == ord('q'):
        #                     raise KeyboardInterrupt
                        
        #             # Save video (optional)
        #             if save_video:
        #                 out.write(frame)
                        
        #             time.sleep(0.01)  # Delay to control speed of scanning
                
        # except KeyboardInterrupt:
        #     print("\nScan interrupted by user")
        # finally:
        #     print("\nScan complete")
        #     if save_video:
        #         out.release()
        #     if display:
        #         cv2.destroyAllWindows()

        start_time = time.time()
        frame_count = 0

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
                        print(f"Saved frame {current_step} to {frame_path}")
                    
                    # FPS Calculation: Every 100 frames, calculate FPS
                    if current_step % 100 == 0:
                        elapsed_time = time.time() - start_time  # Time elapsed
                        fps = current_step / elapsed_time  # Calculate FPS
                        print(f"FPS: {fps:.2f}")
                    
                    # Display the frame (optional)
                    if display:
                        cv2.imshow('TIFF Scan', frame)
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


    def get_available_magnifications(self):
        """
        Returns list of available magnification levels
        """
        return [self.native_mag / downsample for downsample in self.level_downsamples]

def view_tiff(slide_path, magnification=20):
    """
    View TIFF with smooth scanning motion
    """
    scanner = TIFFScanner(slide_path)
    
    print(f"Available magnifications: {scanner.get_available_magnifications()}")
    print("Starting smooth scan... Press 'q' to exit")
    
    scanner.smooth_scan(
        target_mag=magnification,
        window_size=(256, 256),
        speed=20,  ##20 is good for smoothness
        display=True,
        save_video=True,
        save_frames=True,
        frames_dir=r"C:\Users\user\Desktop\image-viewer\frames"
    )

# Example usage:
if __name__ == "__main__":
    view_tiff(r"C:\Users\user\Desktop\image-viewer\001.tiff", magnification=20)  # 20x = no downsample


##change timeout and speed for more smoothness
##change window size to match your display resolution
##