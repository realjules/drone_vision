import os

# Dataset paths
DATASET_FOLDER = "dataset"
TRAIN_FOLDER = os.path.join(DATASET_FOLDER, "train")
VAL_FOLDER = os.path.join(DATASET_FOLDER, "val")

# Model parameters
MODEL_PATH = 'yolov8s.pt'
IMG_SIZE = 640
BATCH_SIZE = 16
EPOCHS = 100

# Class names
CLASS_NAMES = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']

# Output paths
OUTPUT_FOLDER = "output"
PREPROCESSED_TRAIN = os.path.join(OUTPUT_FOLDER, "preprocessed_train")
PREPROCESSED_VAL = os.path.join(OUTPUT_FOLDER, "preprocessed_val")