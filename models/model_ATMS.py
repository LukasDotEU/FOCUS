# model taken from: https://github.com/ncclab-sustech/EEG_Image_decode (Only Change: removal of Horovod package)
# Changes done:
# Removed never used code: Masking in Transformer, different Embeddings
# Changed to use OpenClips ClipLoss (as it was basically identical)

from math import floor
import os
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import numpy as np
from open_clip.loss import ClipLoss

from models.model_base import BaseModel


# -------------- from subject_layers/Transformer_EncDec.py
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        new_x = self.attention(x, x, x)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        # x [B, L, D]
        for attn_layer in self.attn_layers:
            x = attn_layer(x)

        if self.norm is not None:
            x = self.norm(x)

        return x
    
# -------------- from subject_layers/SelfAttention_Family.py


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super().__init__()

        d_keys = d_model // n_heads
        d_values = d_model // n_heads

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values
        )
        out = out.view(B, L, -1)

        return self.out_projection(out)

class FullAttention(nn.Module):
    def __init__(self, attention_dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        scale = 1. / np.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()
    
# -------------- from subject_layers/Embed.py

class SubjectEmbedding(nn.Module):
    def __init__(self, num_subjects, d_model):
        super().__init__()
        self.num_subjects = num_subjects
        # Only create per-subject embeddings if more than one
        if self.num_subjects > 1:
            self.subject_embedding = nn.Embedding(num_subjects, d_model)
        # Shared embedding always exists
        self.shared_embedding = nn.Parameter(torch.randn(1, d_model))

    def forward(self, subject_ids):
        # If only one subject ever, always use shared
        if self.num_subjects <= 1:
            batch_size = subject_ids.size(0)
            return self.shared_embedding.expand(batch_size, 1, -1)

        # Otherwise, per-sample: if sid < num_subjects use learned, else shared
        emb_list = []
        for sid in subject_ids:
            if sid.item() < self.num_subjects:
                emb_list.append(self.subject_embedding(sid))
            else:
                emb_list.append(self.shared_embedding.squeeze(0))
        emb = torch.stack(emb_list, dim=0).unsqueeze(1)  # (batch_size, 1, d_model)
        return emb
    
class DataEmbedding(nn.Module):
    def __init__(self, time_steps, d_model, num_subjects, dropout=0.1, subject_dropout=0.1):
        super().__init__()
        self.num_subjects = num_subjects
        self.subject_dropout = subject_dropout

        # Create per-subject value embeddings only if >1
        if self.num_subjects > 1:
            self.subject_value_embeddings = nn.ModuleDict({
                str(sub): nn.Linear(time_steps, d_model)
                for sub in range(self.num_subjects)
            })
        # Shared value embedding
        self.value_embedding = nn.Linear(time_steps, d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.subject_embedding = SubjectEmbedding(self.num_subjects, d_model)
        
    def forward(self, x, subject_ids):
        batch_size = x.size(0)

        # If only one subject ever, skip per-subject and dropout logic
        if self.num_subjects <= 1:
            # Shared path for all
            x_emb = self.value_embedding(x)             # (batch_size, seq_len, d_model)
            subject_emb = self.subject_embedding(subject_ids) # (batch_size, 1, d_model)
            out = torch.cat([subject_emb, x_emb], dim=1) # (batch_size, seq_len+1, d_model)
            return self.dropout(out)

        # 1) Subject dropout: randomly mark some as unknown
        if self.training and self.subject_dropout > 0:
            mask = torch.rand(batch_size, device=subject_ids.device) < self.subject_dropout
            subject_ids = subject_ids.clone()
            subject_ids[mask] = self.num_subjects

        # 2) Value embedding per sample
        value_list = []
        for i in range(batch_size):
            sid = subject_ids[i].item()
            # x[i]: (seq_len, time_steps)
            if sid < self.num_subjects:
                value_list.append(self.subject_value_embeddings[str(sid)](x[i])) # (seq_len, d_model)
            else:
                value_list.append(self.value_embedding(x[i]))                    # (seq_len, d_model)
        x_emb = torch.stack(value_list, dim=0) # (batch_size, seq_len, d_model)

        # 3) Subject token embedding (handles per-sample shared vs specific)
        subject_emb = self.subject_embedding(subject_ids)  # (batch_size, 1, d_model)

        # 4) Combine token + data
        out = torch.cat([subject_emb, x_emb], dim=1)      # (batch_size, seq_len+1, d_model)
        return self.dropout(out)
    
    
# -------------- from ATMS_retrieval.py
# iTransformer is from a paper about inversed Transformer. Git rep exists for that.
# There is also a OpenSource Recreation (with improvements?) that is available via pip.
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
        #encoder_layer = nn.TransformerEncoderLayer(
        #    d_model=d_model,
        #    nhead=n_heads,
        #    dim_feedforward=d_ff,
        #    dropout=dropout,
        #    activation=activation,
        #    batch_first=True,
        #    norm_first=False  # or True for pre-norm behavior
        #)
        #self.encoder = nn.TransformerEncoder(
        #    encoder_layer,
        #    num_layers=e_layers,
        #    norm=nn.LayerNorm(d_model)
        #)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(attention_dropout=dropout),
                        d_model, n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

    def forward(self, x_enc: Tensor, subject_ids) -> Tensor:
        enc_out = self.enc_embedding(x_enc, subject_ids)
        enc_out = self.encoder(enc_out)  # returns (B, ch+1, d_model)
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
                    d_model: int = 250, **kwargs):
        self.time_steps = time_steps
        self.num_electrodes = num_electrodes
        self.d_model = d_model

        self.encoder = iTransformer(time_steps=self.time_steps, d_model=self.d_model, **kwargs)
        
        # Same as NiceEEG
        self.enc_eeg = nn.Sequential(
            # 1 more channel because subject token
            PatchEmbedding(k=k, m1=m1, m2=m2, s=s, ch=self.num_electrodes + 1),
            FlattenHead()
        )

        # Same as NiceEEG  
        # calculate the embedding dimension of EEG after EEG encoder
        # k: number of filters, m1: kernel size, m2: pooling size, s: stride
        eeg_embedding_dim = int(k * floor(( (self.d_model - m1 + 1) - m2 ) / s + 1))
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
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)

        load_dir = os.path.join(clip_centers_file)
        feature_center_names = np.load(load_dir, allow_pickle=True).item()
        self.feature_centers = torch.from_numpy(feature_center_names['clip_center_features']).to(self.device)
         
    def forward(self, batch):
        eeg = batch['eeg'] # [B, C, T]
        subject_ids = batch['subject']

        eeg_features = self.encoder(eeg, subject_ids)
        eeg_embedding = self.enc_eeg(eeg_features)
        eeg_projection = self.proj_eeg(eeg_embedding)

        return eeg_projection
    
    def compute_loss(self, batch, forward_out):
        img_loss = self.loss_fn(forward_out, batch['image'], self.logit_scale)
        # That was in original code but since alpha is 0.99, the text features barely have an influence.
        # The usage of text features for training is also not mentioned in paper.
        #text_loss = self.loss_func(eeg_projection, text_features, self.logit_scale)
        #loss = img_loss*0.99 + text_loss*(1-0.99)
        return img_loss
    
    def predict(self, batch):
        eeg_projection = self.forward(batch)
        preds, scores = self.compute_predictions(eeg_projection)
        loss = self.compute_loss(batch, eeg_projection)
        return preds, scores, loss
    
    # make sure that order of all_center is THE SAME as the order of labels
    def compute_predictions(self, eeg_features):
        scores = (self.logit_scale * eeg_features @ self.feature_centers.t()).softmax(dim=-1)
        preds = torch.argmax(scores, dim=1)
        return preds, scores
