# model taken from: https://github.com/ncclab-sustech/EEG_Image_decode (Only Change: removal of Horovod package)
# Changes done:
# Removed never used code: Subject depandant layers, Masking in Transformer, different Embeddings
# ---> ALTOUGH Postional Embedding is mentioned in paper but was never actually used...
# .... TODO: Mention in paper/thesis
# Changed custom implemented transformer to use native Transformer
# Changed to use OpenClips ClipLoss (as it was basically identical)

import os
import torch
import torch.nn as nn
from torch import Tensor
from einops.layers.torch import Rearrange
import numpy as np
from open_clip.loss import ClipLoss
from transformers import CLIPModel

from models.model_base import BaseModel

    
# -------------- from subject_layers/Embed.py

class SubjectEmbedding(nn.Module):
    def __init__(self, num_subjects, d_model):
        super().__init__()
        self.subject_embedding = nn.Embedding(num_subjects, d_model)
        self.shared_embedding = nn.Parameter(torch.randn(1, d_model))  # Shared token for unknown subjects

    def forward(self, subject_ids):
        if subject_ids[0] is None or torch.any(subject_ids >= self.subject_embedding.num_embeddings):
            batch_size = subject_ids.size(0)
            return self.shared_embedding.expand(batch_size, 1, -1)
        else:
            return self.subject_embedding(subject_ids).unsqueeze(1)
    
class DataEmbedding(nn.Module):
    def __init__(self, time_steps, d_model, dropout=0.1, num_subjects=None):
        super().__init__()

        # TODO: Check if want to bring back as this was only used when not joint trained 
        self.value_embedding = nn.Linear(time_steps, d_model)  # 如果没有指定subjects，则使用单一的value embedding

        self.dropout = nn.Dropout(p=dropout)
        self.subject_embedding = SubjectEmbedding(num_subjects, d_model) if num_subjects is not None else None
        
    def forward(self, x, subject_ids=None):
        x = self.value_embedding(x)

        if self.subject_embedding is not None:
            subject_emb = self.subject_embedding(subject_ids)  # (batch_size, 1, d_model)
            x = torch.cat([subject_emb, x], dim=1)  # 在序列维度上拼接 (batch_size, seq_len + 1, d_model)

        return self.dropout(x)
    
    
# -------------- from ATMS_retrieval.py
# TODO: Add num_subjects into config file... (dependant on evaluations split type...)
class iTransformer(nn.Module):
    def __init__(self, time_steps: int = 250, d_model: int = 250, dropout: float = 0.25, 
                 n_heads: int = 4, e_layers: int = 1, d_ff: int = 256, 
                 activation: str = 'gelu', num_subjects: int =10):
        super().__init__()
        self.d_model = d_model
        # Embedding remains the same
        #configs.seq_len, configs.embed, configs.freq, joint_train=False,
        self.enc_embedding = DataEmbedding(
            time_steps=time_steps,
            d_model=d_model,
            dropout=dropout,
            num_subjects=num_subjects
        )
        # Native TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=False  # or True for pre-norm behavior
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=e_layers,
            norm=nn.LayerNorm(d_model)
        )

    def forward(self, x_enc: Tensor, subject_ids=None) -> Tensor:
        enc_out = self.enc_embedding(x_enc, subject_ids)
        # mask or src_key_padding_mask can be passed here if needed
        enc_out = self.encoder(enc_out)  # returns (B, seq_len+1, d_model)
        # slice off subject token if present and reduce to num_channels
        print(enc_out.shape)
        enc_out = enc_out[:, :63, :]
        return enc_out


# Same as NiceEEG except .unsqueeze()
class PatchEmbedding(nn.Module):
    def __init__(self, k=40, m1=25, m2=51, s=5, ch=63):
        super().__init__()
        # Revised from ShallowNet
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
        x = x.unsqueeze(1)
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

class ATMS(BaseModel):
    def __init__(self, num_classes, device='cuda', **kwargs):
        super().__init__(num_classes, device=device, **kwargs)

    def build_model(self, time_steps: int, num_electrodes: int, clip_centers_file: str, 
                    proj_dim: int = 1024, k: int = 40, m1:int = 25, m2:int = 51, s:int = 5, lr:float = 3e-4,
                    **kwargs):
        self.time_steps = time_steps
        self.num_electrodes = num_electrodes

        self.Enc_img = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", cache_dir=".cache")
        # disable grad on every CLIP parameter:
        for p in self.Enc_img.parameters():
            p.requires_grad = False

        self.encoder = iTransformer(self.time_steps, **kwargs)
        
        # Same as NiceEEG
        self.enc_eeg = nn.Sequential(
            PatchEmbedding(k=k, m1=m1, m2=m2, s=s, ch=self.num_electrodes),
            FlattenHead()
        )

        # Same as NiceEEG  
        # calculate the embedding dimension of EEG after EEG encoder
        # k: number of filters, m1: kernel size, m2: pooling size, s: stride
        eeg_embedding_dim = int(k * ((self.time_steps - m1 - m2 + 1)/s + 1))
        self.proj_eeg = nn.Sequential(
            nn.Linear(eeg_embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(0.5),)),
            nn.LayerNorm(proj_dim),
        )
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.loss_fn = ClipLoss()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)

        load_dir = os.path.join(clip_centers_file)
        feature_center_names = np.load(load_dir, allow_pickle=True).item()
        self.feature_centers = torch.from_numpy(feature_center_names['clip_center_features']).to(self.device)
         
    def forward(self, batch):
        eeg = batch['eeg'] # [B, C, T]
        img_features = batch['image'] # [B, 3, H, W]
        subject_ids = batch['subject']

        # ensure encoder is in eval (dropout off, batchnorm stats frozen)
        self.Enc_img.eval()
        with torch.no_grad():
            img_features = self.Enc_img.get_image_features(img_features)

        eeg_features = self.encoder(eeg, subject_ids)
        eeg_embedding = self.enc_eeg(eeg_features)
        eeg_projection = self.proj_eeg(eeg_embedding)

        return [eeg_projection, img_features]
    
    # batch has to stay as it's used from other models...
    def compute_loss(self, batch, forward_out):
        eeg_projection, img_features = forward_out

        img_loss = self.loss_fn(eeg_projection, img_features, self.logit_scale)
        # That was in original code but since alpha is 0.99, the text features barely have an influence.
        # The usage of text features for training is also not mentioned in paper.
        #text_loss = self.loss_func(eeg_projection, text_features, self.logit_scale)
        #loss = img_loss*0.99 + text_loss*(1-0.99)

        return img_loss
    
    # make sure that order of all_center is THE SAME as the order of labels
    def predict(self, batch):
        subject_ids = batch['subject']
        labels = batch['class_idx']
        eeg = batch['eeg']  # [B, C, T]

        eeg_features = self.encoder(eeg, subject_ids)
        eeg_embedding = self.enc_eeg(eeg_features)
        eeg_projection = self.proj_eeg(eeg_embedding)
        
        preds, scores = self.compute_predictions(eeg_projection)
        return preds, labels, scores, None, list(subject_ids)
    
    def compute_predictions(self, eeg_features):
        # eeg_features through list with img_features or just eeg_features is dirty hack to compute predictions
        if isinstance(eeg_features, list):
            eeg_features = eeg_features[0]
        scores = (self.logit_scale * eeg_features @ self.feature_centers.t()).softmax(dim=-1)
        preds = torch.argmax(scores, dim=1)
        return preds, scores
