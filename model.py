from ultralytics import YOLO

def load_model(model_path):
    return YOLO(model_path)

def train_model(model, data_yaml, epochs, batch_size, img_size):
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name='yolov8s_visdrone_preprocessed'
    )
    return results

def run_inference(model, image_path):
    results = model(image_path)
    return results