DATA CREATION README

make sure all the scripts in the zip file, including the MIDOG.json is inside the base_path 's folder. 

put all the input images into a folder named "wsi" inside the base_path 's folder. 

run the create_data.py script and you'll get the relevant frames and labels in a directory named "data"

"annotated_frames" consists the annotated wsi, i.e. the whole image with the bounding boxes drawn on them. 

after the script is run, temp_data and annotated_frames could be deleted