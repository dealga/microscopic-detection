from PIL import Image
from PIL.TiffTags import TAGS

with Image.open(r'C:\Users\user\Desktop\image-viewer\001.tiff') as img:
    meta_dict = {TAGS[key] : img.tag[key] for key in img.tag.keys()}

print(meta_dict)



from PIL import Image

# Open the TIFF file
image = Image.open(r"C:\Users\user\Desktop\image-viewer\001.tiff")

# Get dimensions
width, height = image.size

print(f"Width: {width}px, Height: {height}px")


import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from PIL import Image
import numpy as np

# Open a .tiff image and convert it into a numpy array
image_path = r'C:\Users\user\Desktop\image-viewer\frames\frame_00001.tiff'  # Replace with your .tiff image path
image = Image.open(image_path)
image_array = np.array(image)

# Create a Plotly figure
fig = go.Figure()

# Add the image as a layout image in Plotly
fig.add_layout_image(
    dict(
        source=image_path,  # Image path (it can be local or URL)
        x=0,
        y=1,
        xref="paper",
        yref="paper",
        sizex=1,
        sizey=1,
        opacity=1,
        layer="below"
    )
)

# Dash App setup
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='image-plot', figure=fig),
    html.Div(id='cursor-position')
])

@app.callback(
    Output('cursor-position', 'children'),
    Input('image-plot', 'relayoutData')
)
def update_cursor_position(relayoutData):
    # Check if relayoutData is not None
    if relayoutData and 'xaxis.range' in relayoutData:
        return f"Cursor position: {relayoutData.get('xaxis.range', '')}, {relayoutData.get('yaxis.range', '')}"
    return "No cursor position detected."


if __name__ == '__main__':
    app.run_server(debug=True)




import cv2
import numpy as np

# Load the .tiff image (use your file path)
image = cv2.imread(r'C:\Users\user\Desktop\image-viewer\frames\frame_01023.tiff')

# Check if the image was loaded correctly
if image is None:
    print("Error: Image not found.")
    exit()

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to get a binary image
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours in the thresholded image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop over all contours and get bounding box coordinates
for contour in contours:
    # Get the bounding box coordinates (x, y, width, height)
    x, y, w, h = cv2.boundingRect(contour)

    # Print the coordinates
    print(f'Bounding Box Coordinates: x={x}, y={y}, width={w}, height={h}')
    
    # Optionally, draw the bounding box on the original image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Save or display the image with bounding boxes
cv2.imwrite('output_with_boxes.tiff', image)  # Save as .tiff
cv2.imshow('Bounding Boxes', image)  # Display the image with bounding boxes
cv2.waitKey(0)
cv2.destroyAllWindows()




import cv2
import numpy as np
import os

# Load the .tiff image (use your file path)
image = cv2.imread(r'C:\Users\user\Desktop\image-viewer\frames\frame_01079.tiff')

# Check if the image was loaded correctly
if image is None:
    print("Error: Image not found.")
    exit()

# Define the green color in BGR format (since OpenCV uses BGR)
green_bgr = (0, 255, 0)

# Find the coordinates where the pixel value is green (0, 255, 0)
green_pixels = np.where(np.all(image == green_bgr, axis=-1))

# Extract the coordinates (row, column) of the green pixels
coordinates = list(zip(green_pixels[0], green_pixels[1]))

# If green pixels are found, calculate the bounding box
if coordinates:
    # Get the minimum and maximum coordinates to form the bounding box
    min_y, min_x = np.min(coordinates, axis=0)
    max_y, max_x = np.max(coordinates, axis=0)

    # Calculate width and height of the bounding box
    width = max_x - min_x
    height = max_y - min_y

    # Calculate the center of the bounding box
    x_center = (min_x + max_x) / 2
    y_center = (min_y + max_y) / 2

    # Normalize the coordinates by the image size
    image_height, image_width = image.shape[:2]
    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height

    # Print the YOLO format coordinates
    print(f'YOLO format: 0 {x_center} {y_center} {width} {height}')

    # Save the annotation to a text file in YOLO format
    annotation_dir = "annotations"  # Create a directory to save annotations
    if not os.path.exists(annotation_dir):
        os.makedirs(annotation_dir)
    
    # Save as 'image_name.txt' (replace extension with .txt)
    image_name = os.path.splitext(os.path.basename(r'C:\Users\user\Desktop\image-viewer\frames\frame_01079.tiff'))[0]
    annotation_path = os.path.join(annotation_dir, f"{image_name}.txt")
    
    with open(annotation_path, 'w') as file:
        file.write(f"0 {x_center} {y_center} {width} {height}\n")

    # Optionally, draw the bounding box on the original image (for visualization)
    cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

else:
    print("No green pixels found.")

# Save or display the image with bounding box
cv2.imwrite('output_with_bounding_box.tiff', image)  # Save as .tiff
cv2.imshow('Bounding Box', image)  # Display the image with bounding box
cv2.waitKey(0)
cv2.destroyAllWindows()



import cv2
import numpy as np
import os

# Load the .tiff image (use your file path)
image = cv2.imread(r'C:\Users\user\Desktop\image-viewer\frames\frame_01023.tiff')

# Check if the image was loaded correctly
if image is None:
    print("Error: Image not found.")
    exit()

# Define the green color in BGR format (since OpenCV uses BGR)
green_bgr = (0, 255, 0)

# Find the coordinates where the pixel value is green (0, 255, 0)
green_pixels = np.where(np.all(image == green_bgr, axis=-1))

# Extract the coordinates (row, column) of the green pixels
coordinates = list(zip(green_pixels[0], green_pixels[1]))

# If green pixels are found, calculate the bounding box
if coordinates:
    # Get the minimum and maximum coordinates to form the bounding box
    min_y, min_x = np.min(coordinates, axis=0)
    max_y, max_x = np.max(coordinates, axis=0)

    # Print the bounding box coordinates in the requested format
    print(f'BBox: [{min_x}, {min_y}, {max_x}, {max_y}]')

    # Optionally, draw the bounding box on the original image (for visualization)
    cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

    # Save or display the image with bounding box
    cv2.imwrite('output_with_bounding_box.tiff', image)  # Save as .tiff
    cv2.imshow('Bounding Box', image)  # Display the image with bounding box
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("No green pixels found.")









import cv2
import numpy as np
import os

# Folder path containing the TIFF images
folder_path = r'C:\Users\user\Desktop\image-viewer\frames'
# Output text file path
output_file = os.path.join(folder_path, 'green_box_coordinates.txt')
coordinate_file = os.path.join(folder_path, 'coordinates.txt')

# Create/overwrite the output file with a header
with open(output_file, 'w') as f:
    f.write("Green Box Normalized values as \n")
    f.write("===================\n")
    f.write("[class_id, x_center, y_center, width, height]\n")
    f.write("===================\n")

with open(coordinate_file, 'w') as f:
    f.write("Green Box Coordinates\n")
    f.write("===================\n")
    f.write("[min_x, min_y, max_x, max_y]\n")
    f.write("===================\n")

# Get all TIFF files in the folder
tiff_files = [f for f in os.listdir(folder_path) if f.endswith('.tiff')]

# Process each TIFF file
for image_file in tiff_files:
    image_path = os.path.join(folder_path, image_file)
    
    # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image was loaded correctly
    if image is None:
        print(f"Error: Could not load image {image_file}")
        continue

    # Define the green color in BGR format

    image_height, image_width, _ = image.shape
    green_bgr = (0, 255, 0)

    # Find the coordinates where the pixel value is green
    green_pixels = np.where(np.all(image == green_bgr, axis=-1))

    # Extract the coordinates (row, column) of the green pixels
    coordinates = list(zip(green_pixels[0], green_pixels[1]))

    # If green pixels are found, calculate the bounding box
    if coordinates:
        # Get the minimum and maximum coordinates
        min_y, min_x = np.min(coordinates, axis=0)
        max_y, max_x = np.max(coordinates, axis=0)

        # # Append the results to the text file
        # NOT NORMALIZED
        with open(coordinate_file, 'a') as f:
             f.write(f'{image_file}: [{min_x}, {min_y}, {max_x}, {max_y}]\n')

        # Calculate YOLO-style bounding box coordinates (normalized)
        center_x = (min_x + max_x) / 2 / image_width
        center_y = (min_y + max_y) / 2 / image_height
        width = (max_x - min_x) / image_width
        height = (max_y - min_y) / image_height

        # Write the results to the output file in YOLO format
        with open(output_file, 'a') as f:
            f.write(f'{image_file}: 0 {center_x} {center_y} {width} {height}\n')

        print(f"Processed {image_file}: Found green box")

        # Draw the bounding box (optional)

        ##FOR DEBUGGING PURPOSES
        
        # cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

        # Save the image with bounding box (optional)
        # output_image_path = os.path.join(folder_path, f'bbox_{image_file}')
        # cv2.imwrite(output_image_path, image)

    else:
        print(f"Processed {image_file}: No green box found")

print(f"\nResults have been saved to: {output_file}")