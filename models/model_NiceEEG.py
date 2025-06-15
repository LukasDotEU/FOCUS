# Model taken and adpated from: https://github.com/eeyhsong/NICE-EEG

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor

from einops.layers.torch import Rearrange

from models.model_base import BaseModel

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

class PatchEmbedding(nn.Module):
    def __init__(self, k=40, m1=25, m2=51, s=5, ch=63):
        super().__init__()
        # revised from shallownet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, k, (1, m1), (1, 1)),
            nn.AvgPool2d((1, m2), (1, s)),
            nn.BatchNorm2d(k),
            nn.ELU(),
            nn.Conv2d(k, k, (ch, 1), (1, 1)),
            nn.BatchNorm2d(k),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(k, k, (1, 1), stride=(1, 1)),  
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = self.tsconv(x)
        x = self.projection(x)
        return x

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x
        
class NiceEEG(BaseModel):
    def __init__(self, num_classes, device='cuda', **kwargs):
        super().__init__(num_classes, device=device, **kwargs)
    
    def build_model(self, time_steps: int, num_electrodes: int, clip_centers_file: str, 
                    img_embedding_dim: int = 768, proj_dim: int = 768, 
                    k: int = 40, m1:int = 25, m2:int = 51, s:int = 5, 
                    lr: int = 2e-4, b1: float = 0.5, b2: float = 0.999):
        self.time_steps = time_steps
        self.num_electrodes = num_electrodes

        self.Proj_img = nn.Sequential(
            nn.Linear(img_embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(0.3),)),
            nn.LayerNorm(proj_dim),
        ).apply(weights_init_normal)

        self.Enc_eeg = nn.Sequential(
            PatchEmbedding(k=k, m1=m1, m2=m2, s=s, ch=self.num_electrodes),
            FlattenHead()
        ).apply(weights_init_normal)

        # calculate the embedding dimension of EEG after EEG encoder
        # k: number of filters, m1: kernel size, m2: pooling size, s: stride
        eeg_embedding_dim = int(k * ((self.time_steps - m1 - m2 + 1)/s + 1))
        self.Proj_eeg = nn.Sequential(
            nn.Linear(eeg_embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(0.5),)),
            nn.LayerNorm(proj_dim),
        ).apply(weights_init_normal)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=lr, betas=(b1, b2)
        )

        load_dir = os.path.join(clip_centers_file)
        feature_center_names = np.load(load_dir, allow_pickle=True).item()
        self.feature_centers = torch.from_numpy(feature_center_names['clip_center_features']).to(self.device)

    def forward(self, batch):
        eeg = batch['eeg'].unsqueeze(1)  # [B, 1, C, T]
        img_features = batch['image']    # [B, proj_dim]
        
        eeg_features = self.Enc_eeg(eeg)
        eeg_features = self.Proj_eeg(eeg_features)
        eeg_features = F.normalize(eeg_features, dim=-1)

        img_features = self.Proj_img(img_features)
        img_features = F.normalize(img_features, dim=-1)

        # cosine similarity as the logits
        logit_scale = self.logit_scale.exp()
        logits_per_eeg = logit_scale * eeg_features @ img_features.t()
        logits_per_img = logits_per_eeg.t()

        # return of eeg_features is dirty hack to compute predictions later
        return [logits_per_eeg, logits_per_img, eeg_features]
    
    def compute_loss(self, batch, logits):
        logits_per_eeg, logits_per_img, _ = logits
        labels = torch.arange(batch['eeg'].shape[0]).to(self.device)  # used for the loss

        loss_eeg = self.loss_fn(logits_per_eeg, labels)
        loss_img = self.loss_fn(logits_per_img, labels)
        loss_cos = (loss_eeg + loss_img) / 2
        return loss_cos
    
    # TODO: make sure that order of all_center is THE SAME as the order of labels
    def predict(self, batch):
        subjects = list(batch['subject'])
        labels = batch['class_idx']
        eeg = batch['eeg'].unsqueeze(1)  # [B, 1, C, T]

        eeg_features = self.Proj_eeg(self.Enc_eeg(eeg))
        eeg_features = F.normalize(eeg_features, dim=-1)
        
        preds, scores = self.compute_predictions(eeg_features)
        return preds, labels, scores, None, subjects
    
    def compute_predictions(self, eeg_features):
        # eeg_features through logits is dirty hack to compute predictions for training
        if isinstance(eeg_features, list):
            eeg_features = eeg_features[2]
        proj_feature_centers = self.Proj_img(self.feature_centers)
        proj_feature_centers = F.normalize(proj_feature_centers, dim=-1)
        scores = (100.0 * eeg_features @ proj_feature_centers.t()).softmax(dim=-1)  # no use 100?
        preds = torch.argmax(scores, dim=1)
        return preds, scores