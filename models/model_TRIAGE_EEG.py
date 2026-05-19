# Model taken and adpated from: https://github.com/eeyhsong/NICE-EEG

import os
from typing import List, Optional, Tuple
import numpy as np
from math import ceil, floor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops.layers.torch import Rearrange

from models.model_base import BaseModel

def all_conv_kernel_stride_solutions(
    L_in: int,
    T: int,
    search_slack: int = 5,
    prefer_stride_ge_2: bool = True
) -> List[Tuple[int, int, int, int]]:
    """
    Return all (K, S, T_out, err) solutions for a valid conv that maps
    input length L_in to output length as close as possible to T.

    Output tuple format:
        (K, S, T_out, err)

    Constraints:
      - padding = 0, dilation = 1
      - prefer S >= 2
      - include S = 1 solutions only if no S >= 2 solutions exist

    Solutions are sorted by:
      1) increasing err
      2) increasing K
      3) increasing S
    """
    if not (isinstance(L_in, int) and L_in >= 1):
        raise ValueError("L_in must be integer >= 1")
    if not (isinstance(T, int) and T >= 1):
        raise ValueError("T must be integer >= 1")

    def generate(min_stride: int):
        solutions = []

        if T == 1:
            S = max(min_stride, 1)
            return [(L_in, S, 1, 0)]

        denom = T - 1
        if denom > 0:
            S_base = ceil((L_in - 1) / denom)
        else:
            S_base = min_stride

        S_max = max(min_stride, S_base + search_slack)

        for S in range(min_stride, S_max + 1):
            # Ideal kernel for exact match (may be non-integer or invalid)
            K_ideal = L_in - S * (T - 1)

            # Try nearby integer kernels to ensure we don't miss candidates
            for K in {floor(K_ideal), ceil(K_ideal), round(K_ideal)}:
                if not isinstance(K, int):
                    continue
                if K < 1 or K > L_in:
                    continue

                T_out = (L_in - K) // S + 1
                err = abs(T_out - T)

                solutions.append((K, S, T_out, err))

        return solutions

    # 1) try with S >= 2
    solutions = generate(min_stride=2)

    # 2) fallback: allow S = 1 only if necessary
    if not solutions and prefer_stride_ge_2:
        solutions = generate(min_stride=1)

    # Deduplicate
    solutions = list(set(solutions))

    # Sort: smallest error, then smallest K, then smallest S
    solutions.sort(key=lambda x: (x[3], x[0], x[1]))

    return solutions

class PatchEmbedding(nn.Module):
    """
    Patch embedding that is insensitive to input time length by adaptive pooling.
    Input expected: x: [B, 1, C, T]
    Output: tokens [B, N, k] where N = target_tokens
    """
    def __init__(self, k=40, timepoints=250, target_tokens=35, ch=63, dropout=0.5, batch_norm=True):
        super().__init__()
        self.k = k
        self.target_tokens = target_tokens
        self.ch = ch
        self.timepoints = timepoints

        if self.timepoints >= 400:
            kernel_stride_solutions = all_conv_kernel_stride_solutions(timepoints, 250)
            # Use solution with highest stride, as long as kernel is at least 3.
            kernel = kernel_stride_solutions[0][0]
            stride = kernel_stride_solutions[0][1]
            if kernel < 3:
                kernel = kernel_stride_solutions[1][0]
                stride = kernel_stride_solutions[1][1]
            self.preconv = nn.Sequential(
                nn.Conv2d(1, k, (1, kernel), (1, stride), bias=False),
                nn.BatchNorm2d(k) if batch_norm else nn.GroupNorm(8, k),
                nn.ELU(),
            )

        #kernel = min(1, timepoints - 250 + 1)
        #kernel = 2 * round((kernel + 1) / 2) - 1 #ensure uneven

        self.tsconv = nn.Sequential(
            # temporal conv (1 x m1 (25))
            nn.Conv2d(k if self.timepoints >= 300 else 1, k, (1, 25), (1, 1), bias=False),
            nn.BatchNorm2d(k) if batch_norm else nn.GroupNorm(8, k),
            nn.ELU(),

            # spatial conv: mix across electrodes (kernel height = ch)
            # this reduces height to 1
            nn.Conv2d(k, k, (ch, 1), (1, 1), bias=False),
            nn.BatchNorm2d(k) if batch_norm else nn.GroupNorm(8, k),
            nn.ELU(),

            # now height is 1; adaptively pool time dimension to target_tokens
            nn.AdaptiveAvgPool2d((1, target_tokens)),
            nn.Dropout(dropout),
        )

        self.projection = nn.Sequential(
            Rearrange('B k 1 T -> B T k'),
            nn.Linear(k, k),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.timepoints >= 400:
            x = self.preconv(x)
        x = self.tsconv(x)          # [B, k, 1, T']
        x = self.projection(x)      # [B, N, k]
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
        
class TRIAGE_EEG(BaseModel):
    def __init__(self, num_classes, device='cuda', **kwargs):
        super().__init__(num_classes, device=device, **kwargs)

    def build_model(self, time_steps: int, num_electrodes: int, clip_centers_file: str,
                    img_embedding_dim: int = 768, proj_dim: int = 768,
                    k: int = 40, target_tokens: int = 35, weight_decay: float = 1e-4,
                    lr: float = 2e-4, b1: float = 0.5, b2: float = 0.999,
                    batch_norm: bool = True, clip_grad_norm: Optional[float] = 1.0, 
                    cls_label_smoothing: float = 0.1, linear_bias: bool = False):
        self.time_steps = time_steps
        self.num_electrodes = num_electrodes

        eeg_embedding_dim = int(k * target_tokens)

        self.Proj_img = nn.Sequential(
            nn.Linear(img_embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim, bias=linear_bias),
                nn.Dropout(0.3),)),
            nn.LayerNorm(proj_dim),
        )

        self.Enc_eeg = nn.Sequential(
            PatchEmbedding(k=k, timepoints=self.time_steps, target_tokens=target_tokens, 
                           ch=self.num_electrodes, batch_norm=batch_norm),
            FlattenHead(),
            nn.LayerNorm(eeg_embedding_dim),
        )

        hidden_dim = max(256, eeg_embedding_dim // 2)

        self.classification_head = nn.Sequential(
            nn.Linear(eeg_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, self.num_classes)
        )

        self.Proj_eeg = nn.Sequential(
            nn.Linear(eeg_embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim, bias=linear_bias),
                nn.Dropout(0.5),)),
            nn.LayerNorm(proj_dim),
        )

        # temperature scalars
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1/0.07)))
        self.logit_scale_cls = nn.Parameter(torch.tensor(np.log(1/0.07)))

        # gate for combining center logits and classification head
        self.cls_mix_logits = nn.Parameter(torch.zeros(self.num_classes))

        self.loss_weights = nn.Parameter(torch.ones(3))

        # losses
        self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=cls_label_smoothing)
        self.loss_fn_no_smoothing = torch.nn.CrossEntropyLoss(label_smoothing=0.0)

        # gradient clipping hyperparam (used by optimize)
        self.clip_grad_norm = clip_grad_norm

        # optimizer must be created after new params are present so they are included
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=lr, betas=(b1, b2), weight_decay=weight_decay
        )

        # load class centers
        load_dir = os.path.join(clip_centers_file)
        feature_center_names = np.load(load_dir, allow_pickle=True).item()
        self.feature_centers = torch.from_numpy(feature_center_names['clip_center_features']).to(self.device)

        self.return_loss_components = True

    def forward(self, batch):
        eeg = batch['eeg'].unsqueeze(1)  # [B, 1, C, T]
        img_features = batch['image']    # [B, proj_dim]

        eeg_feat_raw = self.Enc_eeg(eeg)

        logit_scale_cls = torch.clamp(self.logit_scale_cls.exp(), max=100.0)
        eeg_classification_logits = self.classification_head(eeg_feat_raw)
        eeg_classification_logits = logit_scale_cls * eeg_classification_logits

        eeg_features = self.Proj_eeg(eeg_feat_raw)
        eeg_features = F.normalize(eeg_features, dim=-1)

        img_features = self.Proj_img(img_features)
        img_features = F.normalize(img_features, dim=-1)

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        logits_per_eeg = logit_scale * eeg_features @ img_features.t()
        logits_per_img = logits_per_eeg.t()


        proj_centers = self.Proj_img(self.feature_centers)
        proj_centers = F.normalize(proj_centers, dim=-1)
        center_logits = logit_scale * eeg_features @ proj_centers.t()   # (B×C)

        # Cosine similarity to all centers
        sim = img_features @ proj_centers.t()   # [B, num_classes]

        return logits_per_eeg, logits_per_img, eeg_classification_logits, eeg_features, center_logits, sim

    def compute_loss(self, batch, forward_out):
        logits_per_eeg, logits_per_img, eeg_classification_logits, _, center_logits, sim = forward_out

        B = batch['eeg'].shape[0]
        C = self.num_classes
        labels = torch.arange(B).to(self.device)
        class_idx = batch['class_idx']

        # ---- 1) instance contrastive loss (paired image/eeg) ----
        # CrossEntropyLoss averages over the batch dimension by default.
        loss_inst = (self.loss_fn_no_smoothing(logits_per_eeg, labels) +
                     self.loss_fn_no_smoothing(logits_per_img, labels)) / 2.0

        # ---- center loss (class centers vs eeg features) ----
        loss_center = self.loss_fn(center_logits, class_idx)

        # ---- combined classification (gate between center_logits and classification head) ----
        gate = torch.sigmoid(self.cls_mix_logits).unsqueeze(0)  # shape [1, C]
        combined_logits = gate * center_logits + (1 - gate) * eeg_classification_logits
        loss_cls = self.loss_fn(combined_logits, class_idx)

        α, β, γ = torch.softmax(self.loss_weights, dim=0)
        total_loss = α * loss_inst + β * loss_center + γ * loss_cls

        gate_mean = gate.mean().detach().cpu().item()

        if self.return_loss_components:
            loss_dict = {
                'loss_total': total_loss.detach().cpu().item(),
                'loss_instance': loss_inst.detach().cpu().item(),
                'loss_center': loss_center.detach().cpu().item(),
                'loss_cls': loss_cls.detach().cpu().item(),
                'weight_instance': α.detach().cpu().item(),
                'weight_center': β.detach().cpu().item(),
                'weight_cls': γ.detach().cpu().item(),
                'gate_mean': gate_mean,
                'logit_scale': torch.clamp(self.logit_scale.exp(), max=100.0).detach().cpu().item(),
                'logit_scale_cls': torch.clamp(self.logit_scale_cls.exp(), max=100.0).detach().cpu().item(),
            }
            return total_loss, loss_dict
        else:
            return total_loss

    def optimize(self):
        # Clip gradients only for this model
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)
        self.optimizer.step()

    def predict(self, batch):
        forward_out = self.forward(batch)
        preds, scores = self.compute_predictions(forward_out)
        if self.return_loss_components:
            loss, loss_dict = self.compute_loss(batch, forward_out)
            return preds, scores, loss, loss_dict
        else:
            loss = self.compute_loss(batch, forward_out)
            return preds, scores, loss

    def compute_predictions(self, forward_out):
        _, _, eeg_classification_logits, _, center_logits, _ = forward_out

        gate = torch.sigmoid(self.cls_mix_logits).unsqueeze(0)
        combined_logits = gate * center_logits + (1 - gate) * eeg_classification_logits
        combined_probs = torch.softmax(combined_logits, dim=1)

        preds = torch.argmax(combined_probs, dim=1)
        return preds, combined_probs

