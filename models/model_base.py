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
        Runs inference on a batch (in eval mode) and returns:
          - preds: Tensor of shape [batch_size] (int labels);
          - optionally: confidence scores or embeddings.
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

    def evaluate_on_dataloader(self, dataloader, evaluator: Evaluator):
        """
        Runs inference over the dataloader, collects predictions, labels,
        (and optionally scores or embeddings) and uses the evaluator
        to compute and return metrics as a dict.
        """
        self.eval()
        all_preds = []
        all_labels = []
        all_scores = []    # e.g., softmax probabilities

        with torch.no_grad():
            for batch in dataloader:
                # Move tensors to the target device.
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device, non_blocking=True)

                # Expect predict() to return a tuple: (preds, labels, scores)
                preds, labels, scores = self.predict(batch)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                if scores is not None:
                    all_scores.append(scores.cpu())

        # Concatenate tensors and convert to numpy arrays when applicable.
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_scores = torch.cat(all_scores).numpy() if all_scores else None

        results = evaluator.compute_metrics(
            y_true=all_labels,
            y_pred=all_preds,
            y_score=all_scores
        )

        return results
