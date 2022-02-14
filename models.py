import argparse

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.beta import Beta

import pytorch_lightning as pl
from scipy.stats import pearsonr

from data import *


# WIN_LEN = 4
# NUM_FEATURES = 20
LOSS = F.mse_loss
LR = 1e-3


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
        x = x.reshape(x.shape[0], -1)
        if time.shape[1]>0:
            if len(time.shape) == 2:
                time = time.unsqueeze(-2)
            # print(x.shape)
            features = self.feature_extractor(time.permute(0, -1, -2))
            features = features.reshape(features.shape[0], -1)
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
        dropout=[0.2, 0.2, 0.2],
        act='relu',
        autoencoder=False,
        feature_dim = 64,
        kernel_size=3,
        kernel_size_pool=1,
        layer_bn=True,
    ):

        super().__init__()
        if act=='relu':
            act_fn=nn.ReLU()
        elif act=='gelu':
            act_fn=nn.GELU()
        else:
            act_fn=nn.GELU()
        assert len(n_hidden)==len(dropout)
        if autoencoder is True:
            self.autoencoder = AutoEncoder(c_in, seq_len, time_win_len)
        else:
            self.autoencoder = None
        dim_autoencoder = 0 if self.autoencoder is None else self.autoencoder.bottleneck_dim
        self.feature_dim = feature_dim
        # self.feature_extractor = nn.Conv1d(c_in, self.feature_dim, kernel_size, padding='same')
        if time_win_len>0:
            self.feature_extractor = Extractor(c_in, self.feature_dim, kernel_size, kernel_size_pool)

            dims_in = [c_in * seq_len + dim_autoencoder + self.feature_dim*(time_win_len//kernel_size_pool)] + n_hidden[:-1]
        else:
            self.feature_extractor = None
            dims_in = [c_in * seq_len + dim_autoencoder] + n_hidden[:-1]

        
        layers = [nn.BatchNorm1d(dims_in[0])]

        for n_in, n_out, drop in zip(dims_in, n_hidden, dropout):
            layers.append(LinBnDrop(n_in, n_out, p=drop, act=act_fn, lin_first=True,
                bn=layer_bn))
        layers.append(nn.Linear(n_hidden[-1], c_out))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, time):
        # print(x)
        x_in = x
        if self.autoencoder is not None:
            encoded = self.autoencoder.encoder(x, time)
            # print(x.shape, encoded.unsqueeze(-2).shape)
            x = torch.cat([x, encoded.unsqueeze(-2)], dim=-1)
        x = x.reshape(x.shape[0], -1)
        # print(x.shape, time.shape)
        if time.shape[1]>0:
            if len(time.shape) == 2:
                time = time.unsqueeze(-2)
            # print(x.shape, time.shape)
            features = self.feature_extractor(time.permute(0, -1, -2))
            features = features.reshape(features.shape[0], -1)
            x = torch.cat([x, features], 1)
        # print(features.shape)
        # return features, x
        # print(x.shape)
        if self.autoencoder is not None:
            return self.layers(x), self.autoencoder(x_in, time)  

        return self.layers(x)

class AutoEncoder(nn.Module):
    def __init__(
        self,
        c_in,
        seq_len,
        time_win_len,
        noise=0.2,
        encoder_layers=[256, 128],
        decoder_layers=[128, 256],
        dropout = [.2, .1],
        bottleneck_dim=64,
    ):

        super().__init__()
        self.noise = noise
        self.bottleneck_dim = bottleneck_dim
        self.encoder = MLP_Time(c_in, seq_len, time_win_len, n_hidden=encoder_layers, c_out=bottleneck_dim, 
                dropout=dropout)
        self.decoder = MLP_Time(bottleneck_dim, seq_len, time_win_len, n_hidden=decoder_layers, c_out=c_in,
                dropout=dropout)

    def forward(self, x, time):
        noise = torch.randn_like(x) * self.noise
        x = x * noise
        encoded = self.encoder(x, time)
        # print(encoded.shape)
        decoded = self.decoder(encoded, time)
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
    return argparse.Namespace(loss_fn=F.mse_loss, lr=1e-3, alpha=.4, 
            n_epochs=10)


def weighted_mse_loss(logits, yb, weights):
    if weights is None:
        return F.mse_loss(logits, yb)
    return (weights * (logits - yb) ** 2).mean()

def unsqueeze(x, dim=-1, n=1):
    "Same as `torch.unsqueeze` but can add `n` dims"
    for _ in range(n): x = x.unsqueeze(dim)
    return x

def mixup_batch(batch, alpha, device='cuda'):
    xb, tb, yb = batch
    bs = yb.shape[0]
    shuffle = torch.randperm(bs).to(device)
    # distrib = Beta(alpha, alpha)
    # lam = distrib.sample((bs,)).to(device)

    lam = np.random.beta(alpha, alpha)
    # xb_mixed = torch.lerp(xb, xb[shuffle], weight=unsqueeze(lam, n=2))
    xb_mixed = lam*xb + (1-lam)*xb[shuffle]
    if tb.shape[1]>0:
        tb_mixed = lam*tb + (1-lam)*tb[shuffle]
        # tb_mixed = torch.lerp(tb, tb[shuffle], weight=unsqueeze(lam, n=2))
    else:
        tb_mixed = tb
    yb_1 = yb
    yb_2 = yb[shuffle]
    return xb_mixed, tb_mixed, yb_1, yb_2, lam



class LitModel(pl.LightningModule):
    def __init__(self, model, args: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters(ignore=['loss_fn'])
        args = vars(args) if args is not None else None
        # print(args)
        self.model = model
        self.loss_fn = args.get("loss_fn")
        self.df_valid = args.get("df_valid")  # for time_id pearson corr
        self.lr = args.get("lr")  # for time_id pearson corr
        self.alpha = args.get("alpha")
        #for schedulers
        self.max_epochs = args.get('max_epochs')
        self.steps_per_epoch = args.get('steps_per_epoch')
        self.n_restarts = args.get('n_restarts', 2)
        print(self.alpha, self.steps_per_epoch)

    def forward(self, x, t):
        return self.model(x, t)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        if self.steps_per_epoch is None:
            return optimizer

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, self.steps_per_epoch*self.max_epochs//self.n_restarts)
        
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer=optimizer, max_lr=self.lr, epochs=self.n_epochs, steps_per_epoch=self.steps_per_epoch, pct_start=self.one_cycle_pct_start
        #     # steps_per_epoch=15, epochs=10
        #     )
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer=optimizer, max_lr=self.lr, epochs=self.n_epochs, steps_per_epoch=self.steps_per_epoch, pct_start=self.one_cycle_pct_start
            # steps_per_epoch=15, epochs=10
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer=optimizer, max_lr=self.lr,
        #     # epochs=self.n_epochs, steps_per_epoch=self.steps_per_epoch,
        #     total_steps = self.n_epochs*self.steps_per_epoch,
        #     pct_start=self.one_cycle_pct_start
        #     # steps_per_epoch=15, epochs=10
        #     )
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer=optimizer, max_lr=self.one_cycle_max_lr,
        #      total_steps=self.one_cycle_total_steps, pct_start=self.one_cycle_pct_start
        #     # steps_per_epoch=15, epochs=10
        #     )
        lr_dict = {"scheduler": scheduler,
                "interval":"step", "monitor": "val_loss", "strict":False}
        # print(vars(optimizer))
        print(vars(scheduler))
        print(lr_dict)
        return {"optimizer": optimizer, 'lr_scheduler': lr_dict}

    def training_step(self, batch, batch_idx):
        if self.alpha is None:
            xb, tb, yb = batch
            logits = self.model(xb, tb)
            loss = self.loss_fn(logits.flatten(), yb)
        else: #mixup
            xb, tb, yb_1, yb_2, lam = mixup_batch(batch, self.alpha)
            # print(xb.shape, tb.shape)
            logits = self.model(xb, tb)
            loss = self.loss_fn(logits.flatten(), yb_1)*lam + self.loss_fn(logits.flatten(), yb_2)*(1-lam) 
            # loss = self.loss_fn(logits, yb_1, weights=(1-lam)) + self.loss_fn(logits, yb_2, weights=(lam))  
        if self.lr_schedulers() is not None:
            scheduler = self.lr_schedulers()
            self.log("lr", scheduler.get_last_lr()[0], on_step=True)
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

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--optimizer", type=str, default=torch.optim.AdamW, help="optimizer class from torch.optim")
        parser.add_argument("--lr", type=float, default=LR)
        parser.add_argument("--weight_decay", type=float, default=3e-2)
        parser.add_argument("--alpha", type=float)
        parser.add_argument("--n_restarts", type=int, default=2)
        # parser.add_argument("--one_cycle_max_lr", type=float, default=None)
        # parser.add_argument("--one_cycle_total_steps", type=int, default=ONE_CYCLE_TOTAL_STEPS)
        # parser.add_argument("--one_cycle_epochs", type=int, default=1)
        # parser.add_argument("--one_cycle_steps_per_epoch", type=int, default=10)
        parser.add_argument("--one_cycle_pct_start", type=float, default=.3)
        parser.add_argument("--loss_fn", type=str, default=LOSS, help="loss function from torch.nn.functional")
        parser.add_argument("--scheduler", type=str, default='one_cycle', help='one_cycle|mult_step use flat lr scheduler with one 10x decrease after 80 epochs')
        parser.add_argument("--n_epochs", type=int, default=10)
        parser.add_argument("--n_hidden", type=int, nargs="+", default=[512, 256, 128])
        parser.add_argument("--dropout", type=float, nargs="+", default=[.2, .2, .2])
        parser.add_argument("--layer_bn", dest='layer_bn', default=False, action='store_true')
        parser.add_argument("--act", type=str, default='relu')
        parser.add_argument("--autoencoder", dest='autoencoder', default=False, action='store_true')
        return parser
    # def predict_step(self, batch, batch_idx, dataloader_idx=None):
    #     xb, yb = batch
    #     return self(xb).squeeze()
