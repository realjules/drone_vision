import os
import cv2
import numpy as np
from utils import read_annotation_file
from augmentation import apply_augmentation

def resize_image_and_annotations(image, annotations, target_size=(640, 640)):
    h, w = image.shape[:2]
    resized_image = cv2.resize(image, target_size)
    scale_x, scale_y = target_size[0] / w, target_size[1] / h
    resized_annotations = []
    for anno in annotations:
        obj_id, x, y, w, h = int(anno[1]), float(anno[2]), float(anno[3]), float(anno[4]), float(anno[5])
        new_x, new_y = x * scale_x, y * scale_y
        new_w, new_h = w * scale_x, h * scale_y
        new_x = max(0, min(new_x, target_size[0] - 1))
        new_y = max(0, min(new_y, target_size[1] - 1))
        new_w = max(1, min(new_w, target_size[0] - new_x))
        new_h = max(1, min(new_h, target_size[1] - new_y))
        resized_annotations.append([anno[0], obj_id, new_x, new_y, new_w, new_h] + anno[6:])
    return resized_image, resized_annotations

def convert_to_yolo_format(annotations, image_width, image_height):
    yolo_annotations = []
    for anno in annotations:
        obj_id, x, y, w, h = int(anno[1]), float(anno[2]), float(anno[3]), float(anno[4]), float(anno[5])
        x_center = (x + w / 2) / image_width
        y_center = (y + h / 2) / image_height
        width = w / image_width
        height = h / image_height
        yolo_annotations.append(f"{obj_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return yolo_annotations

def preprocess_dataset(sequences_folder, annotations_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    processed_count = 0
    error_count = 0

    for sequence_folder in os.listdir(sequences_folder):
        if sequence_folder.endswith('.cache'):
            continue
        sequence_path = os.path.join(sequences_folder, sequence_folder)
        image_folder = sequence_path
        annotation_file = os.path.join(annotations_folder, f"{sequence_folder}.txt")

        if not os.path.exists(annotation_file):
            print(f"Warning: Annotation file not found for {sequence_folder}")
            continue

        annotations_dict = read_annotation_file(annotation_file)

        for image_file in os.listdir(image_folder):
            if image_file.endswith(".jpg"):
                frame_id = int(image_file.split(".")[0])
                image_path = os.path.join(image_folder, image_file)
                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        raise ValueError(f"Failed to read image: {image_path}")

                    if frame_id in annotations_dict:
                        annotations = annotations_dict[frame_id]
                        resized_image, resized_annotations = resize_image_and_annotations(image, annotations)
                        augmented_image, augmented_annotations = apply_augmentation(resized_image, resized_annotations)
                        yolo_annotations = convert_to_yolo_format(augmented_annotations, augmented_image.shape[1], augmented_image.shape[0])

                        output_image_path = os.path.join(output_folder, f"{sequence_folder}_{image_file}")
                        cv2.imwrite(output_image_path, augmented_image)

                        output_anno_path = os.path.join(output_folder, f"{sequence_folder}_{image_file.rsplit('.', 1)[0]}.txt")
                        with open(output_anno_path, 'w') as f:
                            for anno in yolo_annotations:
                                f.write(f"{anno}\n")

                        processed_count += 1
                    else:
                        print(f"Warning: No annotations found for frame {frame_id} in {sequence_folder}")
                except Exception as e:
                    print(f"Error processing image {image_file} in {sequence_folder}: {e}")
                    error_count += 1
                    continue

    print(f"Preprocessing complete. Processed {processed_count} images. Encountered {error_count} errors.")
    return output_folder