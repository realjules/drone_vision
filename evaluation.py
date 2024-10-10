import numpy as np

def calculate_metrics(results):
    metrics = results.results_dict
    overall_metrics = {
        'precision': metrics['metrics/precision(B)'],
        'recall': metrics['metrics/recall(B)'],
        'mAP50': metrics['metrics/mAP50(B)'],
        'mAP50-95': metrics['metrics/mAP50-95(B)']
    }
    return overall_metrics

def per_class_metrics(results):
    return results.maps

def confusion_matrix(results):
    return results.confusion_matrix.matrix