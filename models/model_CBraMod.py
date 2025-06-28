# Model taken from https://github.com/wjq-learning/CBraMod

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import copy

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

class CBraMod(nn.Module):
    def __init__(self, out_dim=200, d_model=200, dim_feedforward=800, n_layer=12, nhead=8):
        super().__init__()
        self.patch_embedding = PatchEmbedding(d_model)
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.proj_out = nn.Sequential(
            nn.Linear(d_model, out_dim),
        )
        self.apply(_weights_init)

    def forward(self, x):
        patch_emb = self.patch_embedding(x)
        feats = self.encoder(patch_emb)

        out = self.proj_out(feats)

        return out
