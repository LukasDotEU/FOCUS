from sklearn.calibration import label_binarize
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    roc_auc_score,
    confusion_matrix
)
import numpy as np

class Evaluator:
    """
    Aggregates a set of metrics for multiclass classification.
    If `y_score` is provided (softmax probabilities or raw scores), compute AUC (one-vs-rest).
    """

    def __init__(self, average='macro'):
        """
        `average`: how to average multiclass metrics (e.g. 'micro', 'macro', 'weighted').
        """
        self.average = average

    def compute_metrics(self, y_true, y_pred, y_score=None, test_pred=False):
        """
        y_true: np.array [N]
        y_pred: np.array [N]
        y_score: np.array [N, num_classes] or None
        Returns a dict with:
          - accuracy, balanced_acc, f1, precision, recall, cohen_kappa, confusion matrix (if test_pred=True), auc (if test_pred=True and y_score not None)
        """
        results = {}
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['balanced_acc'] = balanced_accuracy_score(y_true, y_pred)
        results['f1'] = f1_score(y_true, y_pred, average=self.average)
        results['precision'] = precision_score(y_true, y_pred, average=self.average, zero_division=0)
        results['recall'] = recall_score(y_true, y_pred, average=self.average, zero_division=0)
        results['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)

        if test_pred:
            results['confusion_matrix'] = confusion_matrix(y_true, y_pred)

        # AUC should only be computed once at end of epoch for test set as quite computationally expensive
        if test_pred and y_score is not None:
            # For multiclass AUC, use one-vs-rest (requires binary indicator matrix)
            try:
                # Efficiently build binary indicator matrix
                classes = np.unique(y_true)
                y_true_bin = label_binarize(y_true, classes=classes)

                # If y_score columns mismatch, ensure alignment
                if y_true_bin.shape[1] != y_score.shape[1]:
                    raise ValueError(
                        f"Mismatch: got {y_score.shape[1]} score columns for {y_true_bin.shape[1]} classes"
                    )
                # Compute multiclass AUC using one-vs-rest
                auc_val = roc_auc_score(y_true_bin, y_score, average=self.average, multi_class='ovr')
            except Exception:
                auc_val = np.nan

            results['auc'] = auc_val
        else:
            results['auc'] = np.nan

        return results
