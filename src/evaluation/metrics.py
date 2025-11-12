"""Evaluation metrics"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def calculate_metrics(y_true, y_pred, threshold=0.5):
    """Calculate precision, recall, F1"""
    
    # Binarize predictions
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred_binary, average='micro', zero_division=0)
    recall = recall_score(y_true, y_pred_binary, average='micro', zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, average='micro', zero_division=0)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def calculate_weighted_f1(y_true, y_pred, ia_weights, threshold=0.5):
    """Calculate IA-weighted F1 score"""
    
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Calculate TP, FP, FN per term
    tp = ((y_true == 1) & (y_pred_binary == 1)).sum(axis=0)
    fp = ((y_true == 0) & (y_pred_binary == 1)).sum(axis=0)
    fn = ((y_true == 1) & (y_pred_binary == 0)).sum(axis=0)
    
    # Precision and recall per term
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    
    # Weight by IA
    weighted_precision = (precision * ia_weights).sum() / ia_weights.sum()
    weighted_recall = (recall * ia_weights).sum() / ia_weights.sum()
    
    # F1
    if weighted_precision + weighted_recall == 0:
        return 0.0
    
    f1 = 2 * (weighted_precision * weighted_recall) / (weighted_precision + weighted_recall)
    
    return f1
