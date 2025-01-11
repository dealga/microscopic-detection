import openslide
import numpy as np
from PIL import Image
import cv2
import time
from pathlib import Path

class SmoothWSIScanner:
    def __init__(self, slide_path):
        """
        Initialize the WSI scanner for .tif or .svs files
        
        Args:
            slide_path (str): Path to the WSI file
        """
        # Verify file format
        path = Path(slide_path)
        if path.suffix.lower() not in ['.tif', '.svs']:
            raise ValueError("Only .tif and .svs formats are supported")
            
        self.slide = openslide.OpenSlide(slide_path)
        self.dimensions = self.slide.dimensions
        self.properties = self.slide.properties
        
        # Get native magnification
        self.native_mag = float(self.properties.get('openslide.objective-power', 40))
        self.level_downsamples = self.slide.level_downsamples

        self.print_level_info()
        
    def print_level_info(self):
        """
        Print information about available levels, their dimensions, and downsampling factors.
        """
        print(f"Available levels: {self.slide.level_count}")
        print(f"Dimensions at each level: {self.slide.level_dimensions}")
        print(f"Downsampling factors: {self.level_downsamples}")

        
    def get_magnification_level(self, target_mag):
        """
        Get the appropriate level for desired magnification
        
        Args:
            target_mag (float): Desired magnification
            
        Returns:
            int: Best matching level
        """
        for level, downsample in enumerate(self.level_downsamples):
            current_mag = self.native_mag / downsample
            if current_mag <= target_mag:
                return level
        return len(self.level_downsamples) - 1

    def smooth_scan(self, target_mag=20, window_size=(1024, 768), 
                   speed=100, display=True, save_video=False):
        """
        Perform smooth scanning of the WSI like a video
        
        Args:
            target_mag (float): Desired magnification level
            window_size (tuple): Size of viewing window (width, height)
            speed (int): Pixels per frame to move (higher = faster)
            display (bool): Whether to display the scan in a window
            save_video (bool): Whether to save the scan as a video
            
        Returns:
            None, displays or saves video of the scan
        """
        level = self.get_magnification_level(target_mag)
        level_dimensions = self.slide.level_dimensions[level]
        downsample = self.level_downsamples[level]
        
        # Initialize video writer if saving
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('wsi_scan.mp4', fourcc, 30.0, window_size)
        
        # Calculate steps for smooth motion
        x_steps = np.arange(0, level_dimensions[0] - window_size[0], speed)
        y_steps = np.arange(0, level_dimensions[1] - window_size[1], speed)
        
        try:
            # Scan pattern: left to right, top to bottom
            for y in y_steps:
                # Forward scan (left to right)
                for x in x_steps:
                    region = self.slide.read_region(
                        location=(int(x * downsample), int(y * downsample)),
                        level=level,
                        size=window_size
                    )
                    
                    # Convert to RGB and then BGR for OpenCV
                    frame = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2BGR)
                    
                    if display:
                        cv2.imshow('WSI Scan', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            raise KeyboardInterrupt
                    
                    if save_video:
                        out.write(frame)
                    
                    # Add small delay for smooth motion
                    time.sleep(0.03)
                
                # Move to next row
                if display:
                    cv2.imshow('WSI Scan', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        raise KeyboardInterrupt
                
        except KeyboardInterrupt:
            print("\nScan interrupted by user")
        finally:
            if save_video:
                out.release()
            if display:
                cv2.destroyAllWindows()

    # def smooth_scan(self, target_mag=20, window_size=(1024, 768), 
    #            speed=100, display=True, save_video=False):
    #     """
    #     Perform smooth scanning of the WSI like a video
        
    #     Args:
    #         target_mag (float): Desired magnification level
    #         window_size (tuple): Size of viewing window (width, height)
    #         speed (int): Pixels per frame to move (higher = faster)
    #         display (bool): Whether to display the scan in a window
    #         save_video (bool): Whether to save the scan as a video
            
    #     Returns:
    #         None, displays or saves video of the scan
    #     """
    #     level = self.get_magnification_level(target_mag)
    #     level_dimensions = self.slide.level_dimensions[level]
    #     downsample = self.level_downsamples[level]

    #     # Initialize video writer if saving
    #     if save_video:
    #         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #         out = cv2.VideoWriter('wsi_scan.mp4', fourcc, 30.0, window_size)

    #     # Calculate steps for smooth motion
    #     x_steps = np.arange(0, level_dimensions[0] - window_size[0], speed)
    #     y_steps = np.arange(0, level_dimensions[1] - window_size[1], speed)

    #     save_first_frame = True  # Flag to save only the first frame

    #     try:
    #         # Scan pattern: left to right, top to bottom
    #         for y in y_steps:
    #             for x in x_steps:
    #                 region = self.slide.read_region(
    #                     location=(int(x * downsample), int(y * downsample)),
    #                     level=level,
    #                     size=window_size
    #                 )

    #                 # Save the first frame in TIFF format
    #                 if save_first_frame:
    #                     region.save("first_frame.tiff", format='TIFF')
    #                     print("First frame saved as 'first_frame.tiff' with original resolution.")
    #                     save_first_frame = False  # Ensure this runs only once

    #                 # Convert to RGB and then BGR for OpenCV
    #                 frame = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2BGR)

    #                 if display:
    #                     cv2.imshow('WSI Scan', frame)
    #                     if cv2.waitKey(1) & 0xFF == ord('q'):
    #                         raise KeyboardInterrupt

    #                 if save_video:
    #                     out.write(frame)

    #                 # Add small delay for smooth motion
    #                 #time.sleep(0.03)

    #     except KeyboardInterrupt:
    #         print("\nScan interrupted by user")
    #     finally:
    #         if save_video:
    #             out.release()
    #         if display:
    #             cv2.destroyAllWindows()

    def get_available_magnifications(self):
        """
        Returns list of available magnification levels
        
        Returns:
            list: Available magnification levels
        """
        return [self.native_mag / downsample for downsample in self.level_downsamples]

# Example usage
def view_wsi(slide_path, magnification=20):
    """
    View WSI with smooth scanning motion
    
    Args:
        slide_path (str): Path to .tif or .svs file
        magnification (float): Desired magnification level
    """
    scanner = SmoothWSIScanner(slide_path)
    
    print(f"Available magnifications: {scanner.get_available_magnifications()}")
    print("Starting smooth scan... Press 'q' to exit")
    
    # Start smooth scanning
    scanner.smooth_scan(
        target_mag=magnification,
        window_size=(256, 256),  # Adjustable window size
        speed=20,                # Adjustable speed
        display=True,             # Show live view
        save_video=True          # Save as video file
    )

# Example:

view_wsi(r"C:\Users\user\Desktop\image-viewer\CMU-1-JP2K-33005.svs", magnification=1)   ##can set magnifications 20, or 5 or 1.25
                                                                             ##at 20, no downsample
                                                                            ##at 5, downsample 4x
                                                                            ##at 1.25, downsample 16x