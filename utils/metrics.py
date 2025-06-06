from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    roc_auc_score
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

    def compute_metrics(self, y_true, y_pred, y_score=None):
        """
        y_true: np.array [N]
        y_pred: np.array [N]
        y_score: np.array [N, num_classes] or None
        Returns a dict with:
          - accuracy, f1, precision, recall, cohen_kappa, auc (if y_score not None)
        """
        results = {}
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['f1'] = f1_score(y_true, y_pred, average=self.average)
        results['precision'] = precision_score(y_true, y_pred, average=self.average, zero_division=0)
        results['recall'] = recall_score(y_true, y_pred, average=self.average, zero_division=0)
        results['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)

        if y_score is not None:
            # For multiclass AUC, use one-vs-rest (requires binary indicator matrix)
            try:
                # First, binarize y_true
                classes = np.unique(y_true)
                # If fewer classes than columns in y_score, we assume the columns correspond
                # to sorted classes. Else, raise an error.
                n_classes = y_score.shape[1]
                if len(classes) != n_classes:
                    # We can optionally do label_binarize, but we assume classes are 0..C-1
                    pass
                y_true_bin = np.zeros((y_true.size, n_classes))
                for i, c in enumerate(classes):
                    y_true_bin[:, i] = (y_true == c).astype(int)
                # Compute multiclass AUC using one-vs-rest
                auc_val = roc_auc_score(y_true_bin, y_score, average=self.average, multi_class='ovr')
            except Exception:
                auc_val = float('nan')
            results['auc'] = auc_val
        else:
            results['auc'] = float('nan')

        return results
