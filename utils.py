import os
import yaml

def read_annotation_file(file_path):
    annotations = {}
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                anno = line.split(",")
                frame_id = int(anno[0])
                if frame_id not in annotations:
                    annotations[frame_id] = []
                annotations[frame_id].append(anno)
    return annotations

def create_yaml_file(yaml_path, content):
    with open(yaml_path, "w") as f:
        yaml.dump(content, f, default_flow_style=False)
    print(f"Created {yaml_path}")

def verify_preprocessed_data(folder_path):
    image_count = len([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
    label_count = len([f for f in os.listdir(folder_path) if f.endswith('.txt')])
    print(f"Number of images in {folder_path}: {image_count}")
    print(f"Number of label files in {folder_path}: {label_count}")
    if image_count == 0 or label_count == 0:
        raise ValueError("No images or labels found in the preprocessed folder.")