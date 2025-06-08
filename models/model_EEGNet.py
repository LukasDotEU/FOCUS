# using torcheeg implementation from https://github.com/torcheeg/torcheeg/blob/main/torcheeg/models/cnn/eegnet.py
# Originally: https://arxiv.org/abs/1611.08024

import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_base import BaseModel


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: int = 1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Renormalize weights with a maximum norm.
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNet(BaseModel):
    r'''
    A compact convolutional neural network (EEGNet). For more details, please refer to the following information.

    - Paper: Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    - URL: https://arxiv.org/abs/1611.08024
    - Related Project: https://github.com/braindecode/braindecode/tree/master/braindecode

    Below is a recommended suite for use in emotion recognition tasks:

    .. code-block:: python

        from torcheeg.datasets import DEAPDataset
        from torcheeg import transforms
        from torcheeg.models import EEGNet
        from torch.utils.data import DataLoader

        dataset = DEAPDataset(root_path='./data_preprocessed_python',
                              online_transform=transforms.Compose([
                                  transforms.To2d(),
                                  transforms.ToTensor(),
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select('valence'),
                                  transforms.Binary(5.0),
                              ]))

        model = EEGNet(chunk_size=128,
                       num_electrodes=32,
                       dropout=0.5,
                       kernel_1=64,
                       kernel_2=16,
                       F1=8,
                       F2=16,
                       D=2,
                       num_classes=2)

        x, y = next(iter(DataLoader(dataset, batch_size=64)))
        model(x)

    Args:
        time_steps (int): formerly chunk_size. Number of data points included in each EEG chunk, i.e., :math:`T` in the paper. (default: :obj:`151`)
        num_electrodes (int): The number of electrodes, i.e., :math:`C` in the paper. (default: :obj:`60`)
        F1 (int): The filter number of block 1, i.e., :math:`F_1` in the paper. (default: :obj:`8`)
        F2 (int): The filter number of block 2, i.e., :math:`F_2` in the paper. (default: :obj:`16`)
        D (int): The depth multiplier (number of spatial filters), i.e., :math:`D` in the paper. (default: :obj:`2`)
        num_classes (int): The number of classes to predict, i.e., :math:`N` in the paper. (default: :obj:`2`)
        kernel_1 (int): The filter size of block 1. (default: :obj:`64`)
        kernel_2 (int): The filter size of block 2. (default: :obj:`64`)
        dropout (float): Probability of an element to be zeroed in the dropout layers. (default: :obj:`0.25`)
    '''
    def __init__(self, num_classes, device='cuda', **kwargs):
        super().__init__(num_classes, device=device, **kwargs)

    def build_model(self,
                    time_steps: int = 151,
                    num_electrodes: int = 64,
                    F1: int = 8,
                    F2: int = 16,
                    D: int = 2,
                    kernel_1: int = 64,
                    kernel_2: int = 16,
                    dropout: float = 0.25,
                    learning_rate: float = 1e-3):
        # Save hyperparameters.
        self.chunk_size = time_steps
        self.num_electrodes = num_electrodes
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout
        self.learning_rate = learning_rate

        # Block 1: Temporal convolution and depthwise spatial convolution.
        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernel_1), stride=1, padding=(0, self.kernel_1 // 2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            Conv2dWithConstraint(self.F1,
                                 self.F1 * self.D, (self.num_electrodes, 1),
                                 max_norm=1,
                                 stride=1,
                                 padding=(0, 0),
                                 groups=self.F1,
                                 bias=False),
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),
            nn.Dropout(self.dropout)
        )

        # Block 2: Separable convolution.
        self.block2 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D,
                      self.F1 * self.D, (1, self.kernel_2),
                      stride=1,
                      padding=(0, self.kernel_2 // 2),
                      bias=False,
                      groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(self.dropout)
        )

        self.lin = nn.Linear(self.feature_dim(), self.num_classes, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                          lr=self.learning_rate)

    # TODO: remove this method, retrieve the output size of the second sequential block differently
    def feature_dim(self):
        """
        Determines the dimensionality of the features output by block2.
        """
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)
            mock_eeg = self.block1(mock_eeg)
            mock_eeg = self.block2(mock_eeg)
        return self.F2 * mock_eeg.shape[3]
    
    def forward(self, batch):
        """
        Forward pass for the EEGNet.
        Expects 'eeg' in batch, converts it to shape [B, 1, C, T],
        and returns logits.
        """
        eeg = batch['eeg'].unsqueeze(1)  # [B, 1, C, T]
        x = self.block1(eeg)
        x = self.block2(x)
        x = x.flatten(start_dim=1)
        logits = self.lin(x)
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
          - labels: Tensor [B] (ground truth),
          - scores: Tensor [B, num_classes] (softmax probabilities),
          - embeddings: None (not used),
          - subjects: list of subject IDs.
        """
        labels = batch['class_idx']
        subjects = list(batch['subject'])
        logits = self.forward(batch)
        scores = torch.softmax(logits, dim=1)
        preds = torch.argmax(scores, dim=1)
        return preds, labels, scores, None, subjects