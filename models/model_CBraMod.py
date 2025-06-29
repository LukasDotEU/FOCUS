# Model taken from https://github.com/wjq-learning/CBraMod

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

import copy

from models.model_base import BaseModel

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, x: Tensor) -> Tensor:
        for mod in self.layers:
            x = mod(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn_s = nn.MultiheadAttention(d_model//2, nhead // 2, dropout=dropout, batch_first=True)
        self.self_attn_t = nn.MultiheadAttention(d_model//2, nhead // 2, dropout=dropout, batch_first=True)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.gelu

    def forward(self, src: Tensor) -> Tensor:
        x = src
        x = x + self._sa_block(self.norm1(x))
        x = x + self._ff_block(self.norm2(x))
        return x

    # self-attention block
    def _sa_block(self, x: Tensor) -> Tensor:
        bz, ch_num, patch_num, patch_size = x.shape

        xs = x[:, :, :, :patch_size // 2]
        xs = xs.transpose(1, 2).contiguous().view(bz*patch_num, ch_num, patch_size // 2)
        xs = self.self_attn_s(xs, xs, xs, need_weights=False)[0]
        xs = xs.contiguous().view(bz, patch_num, ch_num, patch_size//2).transpose(1, 2)

        xt = x[:, :, :, patch_size // 2:]
        xt = xt.contiguous().view(bz*ch_num, patch_num, patch_size // 2)
        xt = self.self_attn_t(xt, xt, xt, need_weights=False)[0]
        xt = xt.contiguous().view(bz, ch_num, patch_num, patch_size//2)

        x = torch.concat((xs, xt), dim=3)
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class PatchEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(19, 7), stride=(1, 1), padding=(9, 3),
                      groups=d_model),
        )

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(101, d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        bz, ch_num, patch_num, patch_size = x.shape

        x = x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        patch_emb = self.proj_in(x)
        patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, self.d_model)

        x = x.contiguous().view(bz*ch_num*patch_num, patch_size)
        spectral = torch.fft.rfft(x, dim=-1, norm='forward')
        spectral = torch.abs(spectral).contiguous().view(bz, ch_num, patch_num, 101)
        spectral_emb = self.spectral_proj(spectral)
        patch_emb = patch_emb + spectral_emb

        positional_embedding = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)

        patch_emb = patch_emb + positional_embedding

        return patch_emb

class CBraMod(BaseModel):
    def __init__(self, num_classes, device='cuda', **kwargs):
        super().__init__(num_classes, device=device, **kwargs)

    def build_model(self, time_steps, num_electrodes, dropout, use_pretrained, classifier,
                    lr, weight_decay, label_smoothing, epochs, train_size, clip_value,
                    d_model=200, dim_feedforward=800, n_layer=12, nhead=8, num_patches=None):
        self.clip_value = clip_value
        self.d_model = d_model
        self.use_pretrained = use_pretrained
        self.num_electrodes = num_electrodes
        self.time_steps = time_steps

        self.patch_embedding = PatchEmbedding(d_model)

        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=n_layer)

        if self.use_pretrained:
            loaded_state_dict = torch.load("models/model_CBraMod_pretrained_weights.pth", map_location='cpu')
            model_dict = self.state_dict()
            filtered_dict = {k: v for k, v in loaded_state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
            
            model_dict.update(filtered_dict) # makes sure parameters not in loaded dict stay intact
            self.load_state_dict(model_dict) # actually apply the new parameters
        else:
            self.apply(_weights_init)

        self.num_patches = num_patches
        if self.num_patches is None:
            patch_count = self.time_steps // d_model
        else:
            patch_count = self.num_patches

        if classifier == 'avgpooling_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b d c s'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(self.d_model, self.num_classes),
            )
        elif classifier == 'all_patch_reps_onelayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(self.num_electrodes * patch_count * self.d_model, self.num_classes),
            )
        elif classifier == 'all_patch_reps_twolayer':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(self.num_electrodes * patch_count * self.d_model, self.d_model),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(self.d_model, self.num_classes),
            )
        elif classifier == 'all_patch_reps':
            self.classifier = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.Linear(self.num_electrodes * patch_count * self.d_model, patch_count * self.d_model),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(patch_count * self.d_model, self.d_model),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(self.d_model, self.num_classes),
            )

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()),
                                          lr=lr, weight_decay=weight_decay)
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs * train_size, eta_min=1e-6
        )

    def forward(self, batch):
        # Taking the middle is not clean dividable by 200 is my idea
        def patchify_eeg(eeg: Tensor, patch_length: int = 200) -> Tensor:
            """
            Split (B, C, T) into (B, C, P, patch_length), trimming equally at
            start/end if T % patch_length != 0.
            """
            B, C, T = eeg.shape
            assert T >= patch_length, f"EEG length {T} is too short for patch size {patch_length}"
            num_patches = T // patch_length
            rem = T - num_patches * patch_length
            if rem:
                cut_start = rem // 2
                cut_end   = rem - cut_start
                eeg = eeg[:, :, cut_start : T - cut_end]
            # now shape is (B, C, num_patches * patch_length)
            return eeg.view(B, C, num_patches, patch_length)
        
        # This is my own idea, it's not from CBraMod
        def patchify_eeg_overlap(eeg: torch.Tensor, num_patches: int, patch_length: int = 200) -> torch.Tensor:
            """
            Splits (B, C, T) into (B, C, P, patch_length) with P = num_patches,
            using overlapping windows equally spaced across the time axis.
            Only pads if T < patch_length to ensure one full patch.
            """
            B, C, T = eeg.shape
            assert num_patches > 1, f"Number of patches {num_patches} must be >1"
            assert T <= num_patches * patch_length, \
                f"EEG length {T} is to big to be covered by {num_patches} patches with a size of {patch_length} each."
            assert T >= patch_length, f"EEG length {T} is too short for patch size {patch_length}"

            max_start = T - patch_length
            # linspace from 0 to max_start in num_patches steps
            floats = torch.linspace(0, max_start, steps=num_patches)
            starts = [int(round(x.item())) for x in floats]

            # Slice each overlapping patch
            patches = []
            for s in starts:
                patch = eeg[:, :, s : s + patch_length]  # (B, C, patch_length)
                patches.append(patch.unsqueeze(2))        # → (B, C, 1, patch_length)

            return torch.cat(patches, dim=2)              # → (B, C, num_patches, patch_length)
        
        eeg = batch['eeg']
        if self.num_patches is None:
            x = patchify_eeg(eeg, patch_length=self.d_model)
        else:
            x = patchify_eeg_overlap(eeg, num_patches=self.num_patches, patch_length=self.d_model)

        patch_emb = self.patch_embedding(x)
        feats = self.encoder(patch_emb)
        out = self.classifier(feats)
        return out
    
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
        preds, scores = self.compute_predictions(logits)
        return preds, labels, scores, None, subjects
    
    def compute_predictions(self, logits):
        """
        Compute predictions from logits.
        """
        scores = torch.softmax(logits, dim=1)
        preds = torch.argmax(scores, dim=1)
        return preds, scores
    
    def optimize(self) -> None:
        """
        run optimizer steps + other
        """
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.clip_value)
        self.optimizer.step()
        self.optimizer_scheduler.step()
