import os
from data_preprocessing import preprocess_dataset
from model import load_model, train_model, run_inference
from evaluation import calculate_metrics, per_class_metrics, confusion_matrix
from visualization import plot_overall_metrics, plot_per_class_metrics, plot_confusion_matrix, plot_sample_predictions
from utils import create_yaml_file, verify_preprocessed_data
import config

def main():
    # Preprocess data
    preprocess_dataset(os.path.join(config.TRAIN_FOLDER, "sequences"), 
                       os.path.join(config.TRAIN_FOLDER, "annotations"), 
                       config.PREPROCESSED_TRAIN)
    preprocess_dataset(os.path.join(config.VAL_FOLDER, "sequences"), 
                       os.path.join(config.VAL_FOLDER, "annotations"), 
                       config.PREPROCESSED_VAL)
    
    # Verify preprocessed data
    verify_preprocessed_data(config.PREPROCESSED_TRAIN)
    verify_preprocessed_data(config.PREPROCESSED_VAL)
    
    # Create YAML file
    yaml_content = {
        "path": os.path.abspath(config.OUTPUT_FOLDER),
        "train": os.path.basename(config.PREPROCESSED_TRAIN),
        "val": os.path.basename(config.PREPROCESSED_VAL),
        "nc": len(config.CLASS_NAMES),
        "names": config.CLASS_NAMES
    }
    create_yaml_file("dataset.yaml", yaml_content)
    
    # Load and train model
    model = load_model(config.MODEL_PATH)
    results = train_model(model, "dataset.yaml", config.EPOCHS, config.BATCH_SIZE, config.IMG_SIZE)
    
    # Evaluate model
    metrics = calculate_metrics(results)
    class_metrics = per_class_