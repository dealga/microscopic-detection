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
