# Roughly based on https://www.sciencedirect.com/science/article/abs/pii/S174680942030313X

import torch
import torch.nn as nn
import torchvision
from .model_base import BaseModel


class BiLSTM(BaseModel):
    def __init__(self, num_classes, device='cuda', **kwargs):
        super().__init__(num_classes, device=device, **kwargs)

    def build_model(self,
                    time_steps: int = 440,
                    num_electrodes: int = 128,
                    hidden_channels: list[int] = [128],
                    lr: float = 1e-3,
                    weight_decay: float = 1e-4):
        # Save hyperparameters.
        self.chunk_size = time_steps
        self.num_electrodes = num_electrodes
        self.lr = lr
        self.weight_decay = weight_decay
        self.hidden_channels = hidden_channels
        self.hidden_channels.append(self.num_classes)

        self.lstm = nn.LSTM(
            input_size=self.num_electrodes,
            hidden_size=self.num_electrodes,
            bidirectional=True,
            batch_first=True
        )
        self.mlp = torchvision.ops.misc.MLP(in_channels=self.num_electrodes * 2,
                                            hidden_channels=self.hidden_channels,
                                            norm_layer=nn.BatchNorm1d,
                                            dropout=0.2)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                          lr=self.lr, weight_decay=self.weight_decay)
    
    def forward(self, batch):
        """
        Forward pass for the BiLSTM.
        Expects 'eeg' in batch, and returns logits.
        """
        eeg = batch['eeg'].transpose(1,2)  # [B, C, T]
        x, _ = self.lstm(eeg)
        x = x[:, -1, :]
        logits = self.mlp(x)
        return logits

    def compute_loss(self, batch, logits):
        """
        Computes cross-entropy loss between logits and labels.
        Expects 'class_idx' in batch.
        """
        class_idx = batch['class_idx']
        loss = self.loss_fn(logits, class_idx)
        return loss
    
    def predict(self, batch):
        """
        Performs inference and returns:
          - preds: Tensor [B] (predicted class labels),
          - scores: Tensor [B, num_classes] (softmax probabilities),
          - loss
        """
        logits = self.forward(batch)
        preds, scores = self.compute_predictions(logits)
        loss = self.compute_loss(batch, logits)
        return preds, scores, loss
    
    def compute_predictions(self, logits):
        """
        Compute predictions from logits.
        """
        scores = torch.softmax(logits, dim=1)
        preds = torch.argmax(scores, dim=1)
        return preds, scores