import os
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('/content/drive/MyDrive/frompc/runs/detect/train31/weights/best.pt')  # Your trained model path

# Set the video source
video_path = '/content/tiff_scan.mp4'  # Your video file path

# Set output directories
output_dir_mitotic = '/content/output7/output_mitotic/'
output_dir_non_mitotic = '/content/output7/output_non_mitotic/'
output_debug_mitotic = '/content/output7/output_debug/output_mitotic/'
output_debug_non_mitotic = '/content/output7/output_debug/output_non_mitotic/'

# Create directories if they don't exist
for directory in [output_dir_mitotic, output_dir_non_mitotic, output_debug_mitotic, output_debug_non_mitotic]:
    os.makedirs(directory, exist_ok=True)

# Set the confidence threshold
conf_threshold = 0.7  # Adjust based on your model's performance

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get video dimensions
ret, first_frame = cap.read()
if not ret:
    print("Error reading video file")
    exit()
    
height, width = first_frame.shape[:2]

# Define the vertical line position (in the middle of the frame)
line_x = width // 2
# Define the gap (5% from top and bottom)
gap_top = int(height * 0.05)
gap_bottom = int(height * 0.05)
line_start = (line_x, gap_top)
line_end = (line_x, height - gap_bottom)

# Counters for objects crossing the line
mitotic_count = 0
non_mitotic_count = 0
frame_count = 0

# Reset video capture to start
cap.release()
cap = cv2.VideoCapture(video_path)

# Dictionary to track objects and their paths
# Format: {object_id: {'positions': [], 'crossed': False, 'class': 0/1}}
objects_track = {}  
next_id = 0
max_disappeared = 15  # Frames before we forget an object
iou_threshold = 0.3  # Threshold for matching boxes between frames

# Colors for visualization
COLOR_MITOTIC = (0, 255, 0)       # Green for mitotic
COLOR_NON_MITOTIC = (255, 165, 0)  # Orange for non-mitotic
COLOR_CROSSED = (0, 0, 255)       # Red for crossed line

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0

# For each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Make a copy of the frame for visualization
    vis_frame = frame.copy()
    
    # Create a copy for the debug output (with line)
    debug_frame = frame.copy()
    
    # Draw the vertical line for visualization (only on debug frame)
    cv2.line(debug_frame, line_start, line_end, COLOR_CROSSED, 2)
    
    # Perform inference
    results = model(frame)
    
    # Mark all objects as not found in this frame
    for obj_id in objects_track:
        objects_track[obj_id]['found'] = False
    
    # Check for detections
    for box in results[0].boxes:
        confidence = box.conf.item()
        class_id = int(box.cls.item())
        
        if confidence >= conf_threshold:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            current_box = [x1, y1, x2, y2]
            
            # Determine if this is mitotic or non-mitotic
            is_mitotic = class_id == 1  # Change this if your class mappings are different
            
            # Set color based on class
            color = COLOR_MITOTIC if is_mitotic else COLOR_NON_MITOTIC
            
            # Draw bounding box on both frames
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
            
            # Add label only to debug frame
            label = f'{"Mitotic" if is_mitotic else "Non-Mitotic"} {confidence:.2f}'
            cv2.putText(debug_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Calculate center of the box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Skip objects that are significantly outside the valid vertical region
            if y2 < gap_top or y1 > (height - gap_bottom):
                continue
                
            # Try to match with existing objects
            matched = False
            best_match_id = None
            best_iou = 0
            
            for obj_id, obj_data in objects_track.items():
                if obj_data['found'] or obj_data['class'] != class_id:
                    # Skip already matched objects or objects of different class
                    continue
                    
                if len(obj_data['positions']) > 0:
                    last_pos = obj_data['positions'][-1]
                    curr_iou = calculate_iou(last_pos, current_box)
                    
                    if curr_iou > iou_threshold and curr_iou > best_iou:
                        best_match_id = obj_id
                        best_iou = curr_iou
            
            if best_match_id is not None:
                # Update the existing object
                objects_track[best_match_id]['positions'].append(current_box)
                objects_track[best_match_id]['found'] = True
                objects_track[best_match_id]['disappeared'] = 0
                
                # Check if it has crossed the line from right to left
                if not objects_track[best_match_id]['crossed']:
                    # Get previous and current positions
                    positions = objects_track[best_match_id]['positions']
                    
                    if len(positions) >= 2:
                        prev_box = positions[-2]
                        curr_box = positions[-1]
                        prev_center_x = (prev_box[0] + prev_box[2]) // 2
                        curr_center_x = (curr_box[0] + curr_box[2]) // 2
                        
                        # Check if it crossed the line from right to left
                        if prev_center_x > line_x and curr_center_x <= line_x:
                            # Increment appropriate counter
                            if objects_track[best_match_id]['class'] == 1:  # Mitotic
                                mitotic_count += 1
                                counter = mitotic_count
                                output_dir = output_dir_mitotic
                                debug_dir = output_debug_mitotic
                                count_type = "Mitotic"
                            else:  # Non-mitotic
                                non_mitotic_count += 1
                                counter = non_mitotic_count
                                output_dir = output_dir_non_mitotic
                                debug_dir = output_debug_non_mitotic
                                count_type = "Non-Mitotic"
                                
                            objects_track[best_match_id]['crossed'] = True
                            
                            # Draw crossing indicators (only on debug frame)
                            cv2.circle(debug_frame, (center_x, center_y), 8, COLOR_CROSSED, -1)
                            cv2.putText(debug_frame, f"CROSS", (center_x, center_y - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_CROSSED, 2)
                            
                            # Save the clean frame with just the bounding box
                            clean_filename = os.path.join(output_dir, f'{count_type.lower()}_crossing_{counter:04d}_frame_{frame_count:04d}.jpg')
                            cv2.imwrite(clean_filename, vis_frame)
                            
                            # Save the debug frame with all visualization
                            debug_filename = os.path.join(debug_dir, f'{count_type.lower()}_crossing_{counter:04d}_frame_{frame_count:04d}_debug.jpg')
                            
                            # Add count information to the debug frame
                            info_text = f"Mitotic: {mitotic_count} | Non-Mitotic: {non_mitotic_count}"
                            cv2.putText(debug_frame, info_text, (10, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                            
                            cv2.imwrite(debug_filename, debug_frame)
                            print(f'{count_type} figure crossed the line! Count: {counter}. Frame: {frame_count}')
            else:
                # Create a new object
                objects_track[next_id] = {
                    'positions': [current_box],
                    'crossed': False,
                    'found': True,
                    'disappeared': 0,
                    'class': class_id  # Store the class ID
                }
                # For new objects starting on the left side of the line
                center_x = (x1 + x2) // 2
                if center_x <= line_x:
                    # Already crossed when first detected, mark as crossed
                    objects_track[next_id]['crossed'] = True
                next_id += 1
    
    # Update tracking - increment disappeared counter for objects not found
    object_ids_to_delete = []
    
    for obj_id, obj_data in objects_track.items():
        if not obj_data['found']:
            obj_data['disappeared'] += 1
            
            # If object has been missing for too long, mark it for deletion
            if obj_data['disappeared'] > max_disappeared:
                object_ids_to_delete.append(obj_id)
    
    # Remove objects that have disappeared for too long
    for obj_id in object_ids_to_delete:
        del objects_track[obj_id]
    
    # Draw tracking information (only on debug frame)
    for obj_id, obj_data in objects_track.items():
        if len(obj_data['positions']) > 0 and obj_data['disappeared'] < 3:
            last_box = obj_data['positions'][-1]
            x1, y1, x2, y2 = last_box
            
            # Color based on class and crossed status
            if obj_data['crossed']:
                color = COLOR_CROSSED
            else:
                color = COLOR_MITOTIC if obj_data['class'] == 1 else COLOR_NON_MITOTIC
                
            # Draw tracking box and ID on debug frame only
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 1)
            class_label = "M" if obj_data['class'] == 1 else "N"
            cv2.putText(debug_frame, f"{class_label}:{obj_id}", (x1, y1-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    frame_count += 1

print(f"Total counts:")
print(f"- Mitotic figures: {mitotic_count}")
print(f"- Non-mitotic figures: {non_mitotic_count}")
print(f"- Total figures: {mitotic_count + non_mitotic_count}")
cap.release()