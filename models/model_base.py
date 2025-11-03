import torch
import torch.nn as nn

from utils.metrics import Evaluator

class BaseModel(nn.Module):
    """
    Abstract base class providing a common interface. Subclasses must implement:
      - build_model(**kwargs): Construct model layers.
      - forward(batch): Return raw outputs (logits or embeddings).
      - compute_loss(batch, outputs): Compute and return a scalar loss.
      - predict(batch): Return predicted labels and (optional) confidence scores or embeddings.
    """

    def __init__(self, num_classes: int, device='cuda', **kwargs):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.build_model(**kwargs)
        self.to(self.device)

    def build_model(self, **kwargs):
        """
        Defines self.model (e.g. an nn.Module) and any other layers.
        Subclasses should override this method.
        """
        raise NotImplementedError

    def forward(self, batch: dict):
        """
        batch is a dict with keys: 'eeg' (Tensor), 'class_idx' (int),
        'image_idx' (int), 'subject' (int), and optionally 'image' (Tensor).
        Returns:
          - If softmax model: logits (Tensor of shape [batch_size, num_classes]).
          - If embedding model: embedding tensor (Tensor of shape [batch_size, embed_dim]).
        """
        raise NotImplementedError

    def compute_loss(self, batch: dict, outputs):
        """
        Given a batch and forward outputs, computes and returns a scalar loss (Tensor).
        """
        raise NotImplementedError

    def predict(self, batch: dict):
        """
        Runs inference on a batch (in eval mode)
        """
        raise NotImplementedError
    
    def compute_predictions(self, logits_or_eegfeatures):
        """
        Compute predictions on output of forward (in eval mode)
        """
        raise NotImplementedError
    
    def optimize(self) -> None:
        """
        run optimizer steps
        """
        self.optimizer.step()

    def count_params(self):
        """
        Returns a tuple (total_params, trainable_params).
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def train_one_epoch(self, dataloader, evaluator: Evaluator):
        """
        Performs a basic training loop for one epoch.
        Returns the average loss over batches.
        """
        self.train()
        running_loss = 0.0
        n_batches = 0

        all_preds = []
        all_labels = []
        all_scores = []

        for batch in dataloader:
            # Move tensors to the correct device.
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device, non_blocking=True)

            # TODO: check order of zero_grad and forward
            self.optimizer.zero_grad()
            output = self.forward(batch)
            loss = self.compute_loss(batch, output)
            loss.backward()
            self.optimize()

            running_loss += loss.item()
            n_batches += 1

            with torch.no_grad():
                preds, scores = self.compute_predictions(output)

            all_preds.append(preds)
            all_labels.append(batch['class_idx'])
            if scores is not None:
                all_scores.append(scores)

        all_preds = torch.cat(all_preds).cpu().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()
        all_scores = torch.cat(all_scores).cpu().numpy() if all_scores else None
        results = evaluator.compute_metrics(
            y_true=all_labels,
            y_pred=all_preds,
            y_score=all_scores
        )

        avg_loss = running_loss / n_batches
        return avg_loss, results

    def evaluate_on_dataloader(self, dataloader, evaluator: Evaluator, test_pred: bool = False):
        """
        Runs inference over the dataloader, collects predictions, labels, and scores 
        and uses the evaluator to compute and return metrics as a dict.
        """
        self.eval()
        running_loss = 0.0
        n_batches = 0

        all_preds = []
        all_labels = []
        all_scores = []    # e.g., softmax probabilities

        # Prepare per-sequence tracking if requested
        per_seq_counts = {} if test_pred else None

        with torch.no_grad():
            for batch in dataloader:
                # Move tensors to the target device.
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device, non_blocking=True)

                # Expect predict() to return a tuple: (preds, scores, loss)
                # TODO: refactor to do forward in here and give output to compute metrics and computer loss
                preds, scores, loss = self.predict(batch)

                running_loss += loss.item()
                n_batches += 1

                all_preds.append(preds)
                all_labels.append(batch['class_idx'])
                if scores is not None:
                    all_scores.append(scores)

                # Update per-sequence correctness counters when requested and when sequence_index exists
                if test_pred and 'sequence_index' in batch:
                    # preds and class_idx are tensors of shape [batch_size]; iterate per example
                    seq_idxs = batch['sequence_index']
                    labels = batch['class_idx']
                    for i in range(preds.size(0)):
                        seq = int(seq_idxs[i].item())
                        pred_i = int(preds[i].item())
                        label_i = int(labels[i].item())
                        if seq not in per_seq_counts:
                            per_seq_counts[seq] = {'correct': 0, 'total': 0}
                        per_seq_counts[seq]['total'] += 1
                        if pred_i == label_i:
                            per_seq_counts[seq]['correct'] += 1

        # Concatenate tensors and convert to numpy arrays when applicable.
        all_preds = torch.cat(all_preds).cpu().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()
        all_scores = torch.cat(all_scores).cpu().numpy() if all_scores else None

        results = evaluator.compute_metrics(
            y_true=all_labels,
            y_pred=all_preds,
            y_score=all_scores,
            test_pred=test_pred
        )

        # Add per-sequence accuracy info to results if requested
        if test_pred and per_seq_counts is not None:
            per_sequence = {}
            for seq, counts in per_seq_counts.items():
                total = counts['total']
                correct = counts['correct']
                per_sequence[int(seq)] = {
                    'correct': int(correct),
                    'total': int(total),
                    'accuracy': float(correct) / float(total) if total > 0 else 0.0
                }
            results['per_sequence'] = per_sequence

        avg_loss = running_loss / n_batches
        return avg_loss, results
