# model taken from: https://github.com/ncclab-sustech/EEG_Image_decode (Only Change: removal of Horovod package)
# Changes done:
# Removed never used code: Subject depandant layers, Masking in Transformer, different Embeddings
# ---> ALTOUGH Postional Embedding is mentioned in paper but was never actually used...
# .... TODO: Mention in paper/thesis
# Changed custom implemented transformer to use native Transformer

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch import distributed as dist
from einops.layers.torch import Rearrange
import numpy as np

    
# -------------- from subject_layers/Embed.py

class SubjectEmbedding(nn.Module):
    def __init__(self, num_subjects, d_model):
        super(SubjectEmbedding, self).__init__()
        self.subject_embedding = nn.Embedding(num_subjects, d_model)
        self.shared_embedding = nn.Parameter(torch.randn(1, d_model))  # Shared token for unknown subjects
        self.mask_embedding = nn.Parameter(torch.randn(1, d_model))  # Mask token embedding

    def forward(self, subject_ids):
        if subject_ids[0] is None or torch.any(subject_ids >= self.subject_embedding.num_embeddings):
            batch_size = subject_ids.size(0)
            return self.shared_embedding.expand(batch_size, 1, -1)
        else:
            return self.subject_embedding(subject_ids).unsqueeze(1)
    
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1, num_subjects=None):
        super(DataEmbedding, self).__init__()

        self.value_embedding = nn.Linear(c_in, d_model)  # 如果没有指定subjects，则使用单一的value embedding

        self.dropout = nn.Dropout(p=dropout)
        self.subject_embedding = SubjectEmbedding(num_subjects, d_model) if num_subjects is not None else None
        self.mask_token = nn.Parameter(torch.randn(1, d_model))  # Mask token embedding
        
    def forward(self, x, subject_ids=None, mask=None):
        x = self.value_embedding(x)

        if mask is not None:
            x = x * (~mask.bool()) + self.mask_token * mask.float()

        if self.subject_embedding is not None:
            subject_emb = self.subject_embedding(subject_ids)  # (batch_size, 1, d_model)
            x = torch.cat([subject_emb, x], dim=1)  # 在序列维度上拼接 (batch_size, seq_len + 1, d_model)

        return self.dropout(x)
    
# -------------- from loss.py

def gather_features(
    image_features,
    text_features,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
):
    # We gather tensors from all gpus
    if gather_with_grad:
        all_image_features = torch.cat(
            torch.distributed.nn.all_gather(image_features), dim=0
        )
        all_text_features = torch.cat(
            torch.distributed.nn.all_gather(text_features), dim=0
        )
    else:
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features

class ClipLoss(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                self.local_loss,
                self.gather_with_grad,
                self.rank,
                self.world_size,
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = (
                    logit_scale * all_image_features @ all_text_features.T
                )
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (
            F.cross_entropy(logits_per_image, labels)
            + F.cross_entropy(logits_per_text, labels)
        ) / 2
        return total_loss
    
# -------------- from ATMS_retrieval.py

class iTransformer(nn.Module):
    def __init__(self, configs, num_subjects=10):
        super().__init__()
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        # Embedding remains the same
        self.enc_embedding = DataEmbedding(
            c_in=configs.enc_in,
            d_model=configs.d_model,
            dropout=configs.dropout,
            num_subjects=num_subjects
        )
        # Native TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=configs.d_model,
            nhead=configs.n_heads,
            dim_feedforward=configs.d_ff,
            dropout=configs.dropout,
            activation=configs.activation,
            batch_first=True,
            norm_first=False  # or True for pre-norm behavior
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=configs.e_layers,
            norm=nn.LayerNorm(configs.d_model)
        )

    def forward(self, x_enc: Tensor, subject_ids=None) -> Tensor:
        # x_enc: (B, seq_len, c_in)
        enc_out = self.enc_embedding(x_enc, subject_ids)
        # mask or src_key_padding_mask can be passed here if needed
        enc_out = self.encoder(enc_out)  # returns (B, seq_len+1, d_model)
        # slice off subject token if present and reduce to num_channels
        enc_out = enc_out[:, :63, :] #configs.enc_in
        return enc_out


# Same as NiceEEG except .unsqueeze()
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # Revised from ShallowNet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = x.unsqueeze(1)     
        # print("x", x.shape)   
        x = self.tsconv(x)
        # print("tsconv", x.shape)   
        x = self.projection(x)
        # print("projection", x.shape)  
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


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )

        
class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

class Config:
    def __init__(self):
        self.seq_len = 250                 # Sequence length
        self.output_attention = False      # Whether to output attention weights
        self.d_model = 250                 # Model dimension
        self.dropout = 0.25                # Dropout rate
        self.factor = 1                    # Attention scaling factor
        self.n_heads = 4                   # Number of attention heads
        self.e_layers = 1                  # Number of encoder layers
        self.d_ff = 256                    # Feedforward network dimension
        self.activation = 'gelu'           # Activation function
        self.enc_in = 63                   # Encoder input dimension (example value)


class ATMS(nn.Module):    
    def __init__(self, num_channels=63, sequence_length=250, num_subjects=2, num_features=64, num_latents=1024, num_blocks=1):
        super(ATMS, self).__init__()
        default_config = Config()
        self.encoder = iTransformer(default_config)
        self.enc_eeg = Enc_eeg() # Same as NiceEEG
        self.proj_eeg = Proj_eeg() # Same as NiceEEG    
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()       
         
    def forward(self, x, subject_ids):
        x = self.encoder(x, subject_ids)
        eeg_embedding = self.enc_eeg(x)
        
        out = self.proj_eeg(eeg_embedding)
        return out  