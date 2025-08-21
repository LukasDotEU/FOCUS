# based on implementation from
# https://github.com/prajwalsingh/EEGStyleGAN-ADA/tree/main/EEGClip

import torch
from torch import nn

from models.model_base import BaseModel


class EEGClip(BaseModel):
    def __init__(self, num_classes, device="cuda", **kwargs):
        super().__init__(num_classes, device=device, **kwargs)

    def build_model(
        self,
        time_steps=440,
        num_electrodes=128,
        num_layers=1,
        mlp_inter=256,
        pretrain_lr=3e-4,
        lr=1e-4,
    ):
        self.time_steps = time_steps
        self.num_electrodes = num_electrodes
        self.n_features = self.num_electrodes
        self.embedding_dim = self.num_electrodes * 2
        self.num_layers = num_layers
        self.mlp_inter = mlp_inter

        self.eeg_encoder = EEG_Encoder(
            in_channels=self.num_electrodes,
            n_features=self.n_features,
            projection_dim=self.embedding_dim,
            num_layers=self.num_layers,
            device=self.device,
        )

        self.image_encoder_fc = nn.Sequential(
            nn.ReLU(), nn.Linear(2048, self.embedding_dim, bias=False)
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, self.mlp_inter),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.mlp_inter, self.num_classes),
            nn.Softmax(dim=1) # not best practice but logits are already softmaxed in original model
        )

        self.pretrain_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=pretrain_lr
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=lr
        )

    def pretrain_forward(self, batch):
        eeg = batch["eeg"]  # [B, C, T]
        eeg = eeg.permute(0, 2, 1)  # [B, T, C]
        img_features_resnet50 = batch["image"]
        eeg_feat = self.eeg_encoder(eeg)
        img_feat = self.image_encoder_fc(img_features_resnet50)  # [B, embedding_dim]
        eeg_embed = torch.nn.functional.normalize(eeg_feat, dim=-1)
        image_embed = torch.nn.functional.normalize(img_feat, dim=-1)
        return eeg_embed, image_embed, eeg_feat, img_feat

    def pretrain_one_epoch(self, dataloader):
        self.train()
        running_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            # Move tensors to the correct device.
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device, non_blocking=True)

            self.pretrain_optimizer.zero_grad()
            eeg_embed, image_embed, _, _ = self.pretrain_forward(batch)
            logits = (eeg_embed @ image_embed.T) * torch.exp(torch.tensor(0.5))

            labels = torch.arange(image_embed.shape[0]).to(self.device)

            loss_i = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
            loss_t = torch.nn.functional.cross_entropy(
                logits.T, labels, reduction="none"
            )

            loss = (loss_i + loss_t) / 2.0
            loss = loss.mean()  # average the loss over the batch

            loss.backward()
            self.pretrain_optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / n_batches
        return avg_loss

    def forward(self, batch):
        eeg = batch["eeg"]  # [B, C, T]
        eeg = eeg.permute(0, 2, 1)  # [B, T, C]
        eeg_features = self.eeg_encoder(eeg)
        logits = self.mlp(eeg_features)
        return logits

    def compute_loss(self, batch, logits):
        class_idx = batch["class_idx"]
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
        #scores = torch.softmax(logits, dim=1) # already done in model -> logits are already scores...
        preds = torch.argmax(logits, dim=1)
        return preds, logits.detach()


class EEG_Encoder(nn.Module):
    def __init__(self, in_channels, n_features, projection_dim, num_layers, device="cuda"):
        super(EEG_Encoder, self).__init__()
        self.hidden_size = n_features
        self.num_layers = num_layers
        self.device = device

        self.encoder = nn.LSTM(
            input_size=in_channels,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(
            in_features=n_features, out_features=projection_dim, bias=False
        )

    def forward(self, x):
        h_n = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c_n = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        _, (h_n, c_n) = self.encoder(x, (h_n, c_n))

        feat = h_n[-1]
        x = self.fc(feat)
        return x
