import argparse

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from scipy.stats import pearsonr

from data import *


# WIN_LEN = 4
# NUM_FEATURES = 20


class LinBnDrop(nn.Sequential):
    "Module grouping `BatchNorm1d`, `Dropout` and `Linear` layers"

    def __init__(self, n_in, n_out, bn=True, p=0.0, act=None, lin_first=False):
        layers = [nn.BatchNorm1d(n_out if lin_first else n_in)] if bn else []
        if p != 0:
            layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not bn)]
        if act is not None:
            lin.append(act)
        layers = lin + layers if lin_first else layers + lin
        super().__init__(*layers)


class MLP(nn.Module):
    def __init__(
        self,
        c_in,
        seq_len,
        c_out=1,
        n_hidden=[256, 128, 64],
        dropout=[0.2, 0.2, 0.2, 0.2],
        act=nn.ReLU(),
        autoencoder=None,
        feature_dim = 64
    ):

        super().__init__()
        self.autoencoder = autoencoder
        dim_autoencoder = 0 if self.autoencoder is None else autoencoder.bottleneck_dim
        self.feature_dim = feature_dim
        self.feature_extractor = nn.Conv1d(c_in, self.feature_dim, 1)

        dims_in = [c_in * seq_len + dim_autoencoder + self.feature_dim*seq_len] + n_hidden[:-1]
        

        layers = [nn.BatchNorm1d(dims_in[0])]

        for n_in, n_out, drop in zip(dims_in, n_hidden, dropout):
            layers.append(LinBnDrop(n_in, n_out, p=drop, act=act, lin_first=True))
        layers.append(nn.Linear(n_hidden[-1], c_out))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, time):
        # print(x)
        # if self.autoencoder is not None:
        #     encoded = self.autoencoder.encoder(x)
        #     x = torch.cat([x, encoded])
        if len(time.shape) == 2:
            time = time.unsqueeze(-2)
        # print(x.shape)
        features = self.feature_extractor(time.permute(0, -1, -2))
        x = x.reshape(x.shape[0], -1)
        features = features.reshape(features.shape[0], -1)
        # print(features.shape)
        # return features, x
        x = torch.cat([x, features], 1)
        # print(x.shape)

        return self.layers(x)


class Extractor(nn.Module):
    def __init__(self, c_in, feature_dim, kernel_size, kernel_size_pool=None):
        super().__init__()
        self.conv = nn.Conv1d(c_in, feature_dim, kernel_size, padding='same')
        self.pool = None
        if kernel_size_pool > 1:
            self.pool = nn.MaxPool1d(kernel_size_pool)

    def forward(self, x):
        x = self.conv(x)
        if self.pool is not None:
            x = self.pool(F.relu(x))
        return x




class MLP_Time(nn.Module):
    def __init__(
        self,
        c_in,
        seq_len,
        time_win_len,
        c_out=1,
        n_hidden=[256, 128, 64],
        dropout=[0.2, 0.2, 0.2, 0.2],
        act=nn.ReLU(),
        autoencoder=None,
        feature_dim = 64,
        kernel_size=3,
        kernel_size_pool=1
    ):

        super().__init__()
        self.autoencoder = autoencoder
        dim_autoencoder = 0 if self.autoencoder is None else autoencoder.bottleneck_dim
        self.feature_dim = feature_dim
        # self.feature_extractor = nn.Conv1d(c_in, self.feature_dim, kernel_size, padding='same')
        self.feature_extractor = Extractor(c_in, self.feature_dim, kernel_size, kernel_size_pool)

        dims_in = [c_in * seq_len + dim_autoencoder + self.feature_dim*(time_win_len//kernel_size_pool)] + n_hidden[:-1]
        
        layers = [nn.BatchNorm1d(dims_in[0])]

        for n_in, n_out, drop in zip(dims_in, n_hidden, dropout):
            layers.append(LinBnDrop(n_in, n_out, p=drop, act=act, lin_first=True))
        layers.append(nn.Linear(n_hidden[-1], c_out))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, time):
        # print(x)
        # if self.autoencoder is not None:
        #     encoded = self.autoencoder.encoder(x)
        #     x = torch.cat([x, encoded])
        if len(time.shape) == 2:
            time = time.unsqueeze(-2)
        # print(x.shape, time.shape)
        features = self.feature_extractor(time.permute(0, -1, -2))
        x = x.reshape(x.shape[0], -1)
        features = features.reshape(features.shape[0], -1)
        # print(features.shape)
        # return features, x
        x = torch.cat([x, features], 1)
        # print(x.shape)

        return self.layers(x)

class AutoEncoder(nn.Module):
    def __init__(
        self,
        c_in,
        seq_len,
        noise=0.2,
        encoder_layers=[256, 128],
        decoder_layers=[128, 256],
        bottleneck_dim=64,
    ):

        super().__init__()
        self.noise = noise
        self.bottleneck_dim = bottleneck_dim
        self.encoder = MLP(c_in, seq_len, n_hidden=encoder_layers, c_out=bottleneck_dim)
        self.decoder = MLP(bottleneck_dim, seq_len, n_hidden=decoder_layers, c_out=c_in)

    def forward(self, x):
        noise = torch.randn_like(x) * self.noise
        x = x * noise
        encoded = self.encoder(x)
        # print(encoded.shape)
        decoded = self.decoder(encoded)
        return decoded


def pretrain(autoencoder, train_dl, lr=1e-3, n_epochs=10):
    autoencoder.train()
    opt = torch.optim.AdamW(autoencoder.parameters(), lr=lr)
    for i in range(n_epochs):
        losses = []
        for xb, _ in tqdm(train_dl):
            opt.zero_grad()
            decoded = autoencoder(xb)
            loss = F.mse_loss(xb, decoded)
            losses.append(loss)
            loss.backward()
            opt.step()

        print(torch.tensor(losses).mean())


def pearson_coef(data):
    return data.corr()["target"]["preds"]


def default_lm_args():
    return argparse.Namespace(loss_fn=F.mse_loss, lr=1e-3)


class LitModel(pl.LightningModule):
    def __init__(self, model, args: argparse.Namespace):
        super().__init__()
        args = vars(args) if args is not None else None
        # print(args)
        self.model = model
        self.loss_fn = args.get("loss_fn")
        self.df_valid = args.get("df_valid")  # for time_id pearson corr
        self.lr = args.get("lr")  # for time_id pearson corr

    def forward(self, x, t):
        return self.model(x, t)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        xb, tb, yb = batch
        logits = self.model(xb, tb)
        loss = self.loss_fn(logits.flatten(), yb)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        xb, tb, yb = batch
        logits = self.model(xb, tb)
        loss = self.loss_fn(logits.flatten(), yb)
        pearson = pearsonr(
            logits.flatten().detach().cpu().numpy(), yb.flatten().detach().cpu().numpy()
        )
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("pearson", pearson[0], on_epoch=True, prog_bar=True)
        return logits.view(xb.size(0), -1)

    def validation_epoch_end(self, outputs):

        preds = torch.cat(outputs).squeeze()
        if self.df_valid is not None:
            if preds.shape[0] == self.df_valid.shape[0]:
                self.df_valid["preds"] = preds.detach().cpu().numpy()
                metric = np.mean(
                    self.df_valid[["time_id", "target", "preds"]]
                    .groupby("time_id")
                    .apply(pearson_coef)
                )

                self.log("pearson_full", metric, prog_bar=True)

    # def predict_step(self, batch, batch_idx, dataloader_idx=None):
    #     xb, yb = batch
    #     return self(xb).squeeze()
