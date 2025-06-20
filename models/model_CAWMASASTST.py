# model taken from https://github.com/ljbuaa/VisualDecoding
# could have also taken a different model from: https://github.com/JeniferK28/Learning-Spatiotemporal-Graph-Representations-for-Visual-Perception-using-EEG-Signals but would have needed to have functionality connectivity matrix

# Also thanks to this fork!: https://github.com/busiqiao/CAW-MASA-STST

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange

from models.model_base import BaseModel

class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
        )

class GCN(nn.Module):
    def __init__(self, chns,feas):
        super().__init__()  
        self.a=nn.Parameter(torch.rand((chns,chns)))
        self.k=2
        self.num_of_filters=feas
        self.Theta=nn.Parameter(torch.randn((self.k,feas, self.num_of_filters)))
        self.adj=None
    def forward(self, x):
        x = x.squeeze()
        b, c, l = x.size()
        fea_matrix = x
        # Similarity Matrix
        self.diff = (x.expand([c,b,c,l]).permute(2,1,0,3)-x).permute(1,0,2,3)
        self.diff=torch.abs(self.diff).sum(-1)
        self.diff=F.normalize(self.diff,dim=0)
        tmpS = torch.exp(torch.relu((1-self.diff)*self.a))
        # Laplacian matrix 
        self.adj = tmpS / torch.sum(tmpS,axis=1,keepdims=True)
        D = torch.diag_embed(torch.sum(self.adj,axis=1))
        L = D - self.adj
        # Chebyshev graph convolution
        firstOrder=torch.eye(c).cuda()
        lambda_max = 2.0
        L_t = (2 * L) / lambda_max - firstOrder
        cheb_polynomials = [firstOrder, L_t]
        for i in range(2, self.k):
            cheb_polynomials.append(2 * torch.matmul(L_t, cheb_polynomials[i - 1]) - cheb_polynomials[i - 2])
            # cheb_polynomials.append(2 * L_t * cheb_polynomials[i - 1] - cheb_polynomials[i - 2]) Chebyshev recurrence requires matrix multiplication?
        output = torch.zeros(b, c, self.num_of_filters).cuda()
        for kk in range(self.k):
            T_k = cheb_polynomials[kk].expand([b,c,c])
            rhs = torch.bmm(T_k.permute(0, 2, 1), fea_matrix)
            output = output + torch.matmul(rhs, self.Theta[kk])
        output=torch.relu(output)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) 

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)") 
        out = self.projection(out)
        return out

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

# Channel Attention Weighting
class CAW(nn.Module):
    def __init__(self, channel, reduction = 1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ELU(inplace  = True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        if len(x.shape)==3:
            b, c, t = x.size()
            xstd=((x-x.mean(-1).view(b,c,1))**2)
            xstd = F.normalize(xstd.sum(-1),dim=-1)
            attn = self.fc(xstd).view(b, c, 1)
        else:
            b, c, s, t = x.size()
            xstd=((x-x.mean(-1).view(b,c,s,1))**2)
            xstd = F.normalize(xstd.sum(-1),dim=-1)
            attn = self.fc(xstd).view(b, c, s, 1)
        out = x * attn.expand_as(x)
        return out, attn


class STSTransformerBlock(nn.Module):
    def __init__(self, emb_size1, emb_size2):
        super().__init__()
        drop_p = 0.5
        self.emb_size = emb_size1
        self.att_drop1 = nn.Dropout(drop_p)
        self.projection1 = nn.Linear(emb_size1, emb_size1)
        self.projection2 = nn.Linear(emb_size1, emb_size1)
        self.drop1=nn.Dropout(drop_p)
        self.drop2=nn.Dropout(drop_p)

        self.layerNorm1=nn.LayerNorm(emb_size1)
        self.layerNorm2=nn.LayerNorm(emb_size2)
 
        self.queries1 = nn.Linear(emb_size1, emb_size1)
        self.values1 = nn.Linear(emb_size1, emb_size1)
        self.keys2 = nn.Linear(emb_size2, emb_size2)
        self.values2 = nn.Linear(emb_size2, emb_size2)

        self.layerNorm3=nn.LayerNorm(emb_size1+emb_size2)
        self.mha=MultiHeadAttention(emb_size1+emb_size2, 5, 0.5)
        self.drop3=nn.Dropout(drop_p)

        self.ffb=nn.Sequential(
            nn.LayerNorm(emb_size1+emb_size2),
            FeedForwardBlock(
                emb_size1+emb_size2, expansion=4, drop_p=0.5),
            nn.Dropout(drop_p)
        )

    def forward(self, x1, x2):
        x1=rearrange(x1, 'b e (h) (w) -> b (h w) e ')
        x2=rearrange(x2, 'b e (h) (w) -> b (h w) e ')
        res1=x1
        res2=x2

        x1 = self.layerNorm1(x1)
        x2 = self.layerNorm2(x2)
        queries1 = self.queries1(x1) 
        values1 = self.values1(x1)
        keys2 = self.keys2(x2)
        values2 = self.values2(x2)

        energy = torch.einsum('bqd, bkd -> bqk', keys2, queries1)
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop1(att)

        out1 = torch.einsum('bal, blv -> bav ', att, values1)
        out1 = self.projection1(out1)
        x1 = self.drop1(out1)
        x1+=res1

        out2 = torch.einsum('bal, blv -> bav ', att, values2)
        out2 = self.projection2(out2)
        x2 = self.drop2(out2)
        x2+=res2

        x=torch.cat((x1,x2),dim=-1)
        res = x
        x=self.layerNorm3(x)
        x=self.mha(x)
        x=self.drop3(x)
        x += res

        res = x
        x = self.ffb(x)
        x += res
        x = rearrange(x, 'b t e -> b e 1 t')
        return x
    
class CAWMASASTST(BaseModel):
    def __init__(self, num_classes, device='cuda', **kwargs):
        super().__init__(num_classes, device=device, **kwargs)

    def build_model(self, time_steps: int = 32, num_electrodes: int = 124, spectral_channels = 25, lr=1e-3):
        self.time_steps = time_steps
        self.num_electrodes = num_electrodes
        self.spectral_channels = spectral_channels
        self.lr = lr

        # Spectral
        self.chunks=5
        self.speWidth=self.spectral_channels//self.chunks

        self.asa_modules = nn.ModuleList([
            #ASA
            nn.Sequential(
                ConvBlock(self.num_electrodes, 25, (self.speWidth,1)),
                Rearrange("a b c d -> a (c d) b"),
                GCN(self.time_steps,25),
                Rearrange("a b (c d) -> a c d b",d=1),
                ConvBlock(25, 2, (1,1)),
            )
            for _ in range(self.chunks)
        ])
        self.fullband_convolution = ConvBlock(self.num_electrodes, 30, (self.spectral_channels, 1))
        self.spe2 = nn.Sequential(
            ConvBlock(40,30,(1,13)),
            ConvBlock(30,10,(1,11)),
            nn.AdaptiveAvgPool2d((1,8)),
            nn.Dropout2d(0.5)
        )

        # Spatial
        self.caw=CAW(self.num_electrodes,2)
        self.spa1 = ConvBlock(1,40,(self.num_electrodes,1))
        self.spa2 = nn.Sequential(
            ConvBlock(40,30,(1,13)),
            ConvBlock(30,10,(1,11)),
            nn.AdaptiveAvgPool2d((1,8)),
            nn.Dropout2d(0.5)
        )

        # Feature Fusion
        self.feature_fusion = nn.Sequential(
            STSTransformerBlock(40,40),
            ConvBlock(80,70,(1,13)),
            ConvBlock(70,80,(1,11)),
            nn.AdaptiveAvgPool2d((1,8)),
            nn.Dropout2d(0.5)
        )

        self.feaLen=(100)*8
        self.classify = nn.Sequential(
            nn.Linear(self.feaLen, self.num_classes),
            nn.Softmax(dim=1),
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                          lr=self.lr)

    def forward(self, batch):
        x = batch['eeg']
        xcwt = batch['cwt']
        # Spectral (MASA)
        xcwts=xcwt.chunk(self.chunks,dim=2)
        asa_features=[]
        for i, asa_modules in enumerate(self.asa_modules):
            asa_features.append(asa_modules(xcwts[i]))
        asa_features=torch.cat(asa_features,1)

        fullband_features = self.fullband_convolution(xcwt) 
        spectral_features = torch.cat((fullband_features,asa_features), dim=1) # also used for feature fusion

        spectral_out = self.spe2(spectral_features).squeeze() # spectral features go through conv before being concatenated with other features

        # Spatial
        x, _ = self.caw(x)
        spa1_out= self.spa1(x.unsqueeze(1)) # also used for feature fusion

        spa_out= self.spa2(spa1_out).squeeze() # spatial features go through conv before being concatenated with other features

        # Feature Fusion
        fuse_out = self.feature_fusion(spectral_features,spa1_out).squeeze()

        fused_features = torch.cat((spectral_out,spa_out,fuse_out),dim=1)

        # Classification Unit
        out = self.classify(fused_features.reshape(-1,self.feaLen))
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
