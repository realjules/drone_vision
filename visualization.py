import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_overall_metrics(metrics):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
    plt.title('Overall Model Performance')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    for i, v in enumerate(metrics.values()):
        plt.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('overall_metrics.png')
    plt.close()

def plot_per_class_metrics(class_metrics, class_names):
    plt.figure(figsize=(12, 6))
    sns.barplot(x=class_names, y=class_metrics)
    plt.title('mAP50 by Class')
    plt.xlabel('Class')
    plt.ylabel('mAP50')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('map50_by_class.png')
    plt.close()

def plot_confusion_matrix(conf_matrix, class_names):
    conf_matrix_norm = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_sample_predictions(model, image_paths):
    for i, img_path in enumerate(image_paths):
        results = model(img_path)
        res_plotted = results[0].plot()
        plt.figure(figsize=(10, 10))
        plt.imshow(res_plotted)
        plt.axis('off')
        plt.title(f"Sample Prediction {i+1}")
        plt.tight_layout()
        plt.savefig(f"sample_prediction_{i+1}.png")
        plt.close()