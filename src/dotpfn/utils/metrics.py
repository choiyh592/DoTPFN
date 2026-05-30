import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

def compute_metrics(y_true, y_prob, threshold=0.5):
    """Computes performance metrics: AUROC, AUPRC, Accuracy, F1.
    Gracefully handles edge cases like single-class labels to avoid crashes.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return {
            "AUROC": 0.0,
            "AUPRC": 0.0,
            "Acc": float(accuracy_score(y_true, y_pred)),
            "F1": 0.0
        }
        
    return {
        "AUROC": float(roc_auc_score(y_true, y_prob)),
        "AUPRC": float(average_precision_score(y_true, y_prob)),
        "Acc": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0))
    }
