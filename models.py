import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from scipy.stats import pearsonr

from data import *



WIN_LEN = 4
# NUM_FEATURES = 20

class LinBnDrop(nn.Sequential):
    "Module grouping `BatchNorm1d`, `Dropout` and `Linear` layers"
    def __init__(self, n_in, n_out, bn=True, p=0., act=None, lin_first=False):
        layers = [nn.BatchNorm1d(n_out if lin_first else n_in)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not bn)]
        if act is not None: lin.append(act)
        layers = lin+layers if lin_first else layers+lin
        super().__init__(*layers)

class MLP(nn.Module):

    def __init__(self, c_in, seq_len, n_hidden = [256, 128, 64],
            dropout=[.2, .2 ,.2, .2], act=nn.ReLU()):
        super().__init__()
        layers = [nn.BatchNorm1d(c_in*seq_len)]
        dims_in = [c_in*seq_len] + n_hidden[:-1]

        for n_in, n_out, drop in zip(dims_in, n_hidden, dropout):
            layers.append(LinBnDrop(n_in, n_out, p=drop, act=act, lin_first=True))
        layers.append(nn.Linear(n_hidden[-1], 1))
        self.layers = nn.Sequential(*layers)


    def forward(self, x):
        # print(x)
        x = x.reshape(x.shape[0], -1)
        return self.layers(x)

def pearson_coef(data):
    return data.corr()['target']['preds']

class LitModel(pl.LightningModule):
    
    def __init__(self, model, df_valid):
        super().__init__()
        self.model = model
        self.loss_fn = F.mse_loss
        self.df_valid = df_valid

    def forward(self, x):
        return self.model(x)


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)


    def training_step(self, batch, batch_idx):
        xb, yb = batch
        logits = self.model(xb)
        loss = self.loss_fn(logits.flatten(), yb)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        xb, yb = batch
        logits = self.model(xb)
        loss = self.loss_fn(logits.flatten(), yb)
        pearson = pearsonr(logits.flatten().detach().cpu().numpy(), 
                yb.flatten().detach().cpu().numpy())
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('pearson', pearson[0], on_epoch=True, prog_bar=True)
        return logits.view(xb.size(0),-1)

    def validation_epoch_end(self, outputs):

        preds = torch.cat(outputs).squeeze()
        if preds.shape[0] == self.df_valid.shape[0]:
            self.df_valid['preds'] = preds.detach().cpu().numpy()
            metric = np.mean(self.df_valid[['time_id', 'target', 'preds']].groupby('time_id').apply(pearson_coef))
            
            self.log('pearson_full', metric, prog_bar=True)

    # def predict_step(self, batch, batch_idx, dataloader_idx=None):
    #     xb, yb = batch
    #     return self(xb).squeeze()
