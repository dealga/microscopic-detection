import os
import subprocess

# ==== Prompt User for Base Path ====
base_path = '/home/swathi/rashmi/Experiments/ultralytics' 

# Check if the base path exists
if not os.path.exists(base_path):
    raise ValueError(f"Base path {base_path} does not exist. Please provide a valid path.")

# ==== All Paths (based on base path) ====
paths = {
    "annotated_dir": os.path.join(base_path, "annotated_frames/temp_data"),
    "input_folder": base_path,
    "json_path": os.path.join(base_path, "MIDOG.json"),
    "output_folder": os.path.join(base_path, "temp_data"),
    
    "annotated_frames_folder": os.path.join(base_path, "temp_data/images/train"),
    "clean_frames_output": os.path.join(base_path, "data/images/train"),
    
    "image_dir": os.path.join(base_path, "data/images/train"),
    "label_dir": os.path.join(base_path, "data/labels/train"),
    
    "base_folder_path_temp": os.path.join(base_path, "temp_data/images/train"),
    "output_labels_dir": os.path.join(base_path, "data/labels/train"),
    
    "base_image_dir": os.path.join(base_path, "data/images"),
    "base_label_dir": os.path.join(base_path, "data/labels")
}

# ==== List of scripts in order ====
scripts = [
    "1_annotate_and_generate_frames.py",
    "2_extract_the_same_frames_but_not_annotated.py",
    "3_get_coordinates_of_green_boxes.py",
    "4_get_coordinates_of_yellow_boxes.py",
    "5_remove_empty_labels.py",
    "6_remove_duplicates.py",
    "6.5_remove_not_matching_files.py",
    "7_test_val_split.py"
]

# ==== Function to run each script ====
def run_script(script_name, env_vars):
    script_path = os.path.join(base_path, script_name)  # Full path to script
    print(f"\n📜 Running: {script_path}")
    env = os.environ.copy()
    env.update(env_vars)
    result = subprocess.run(["python", script_path], env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Script {script_name} failed with exit code {result.returncode}")

# ==== Set environment variables for scripts ====
env_vars = {
    "BASE_PATH": base_path, 
    "ANNOTATED_DIR": paths["annotated_dir"],
    "INPUT_FOLDER": paths["input_folder"],
    "JSON_PATH": paths["json_path"],
    "OUTPUT_FOLDER": paths["output_folder"],

    "ANNOTATED_FRAMES_FOLDER": paths["annotated_frames_folder"],
    "CLEAN_FRAMES_OUTPUT": paths["clean_frames_output"],

    "BASE_FOLDER_TEMP": paths["base_folder_path_temp"],
    "OUTPUT_LABELS_DIR": paths["output_labels_dir"],

    "IMAGE_DIR": paths["image_dir"],
    "LABEL_DIR": paths["label_dir"],

    "BASE_IMAGE_DIR": paths["base_image_dir"],
    "BASE_LABEL_DIR": paths["base_label_dir"]
}

# ==== Run all scripts ====
for script in scripts:
    run_script(script, env_vars)

print("\n✅ All scripts executed successfully.")
