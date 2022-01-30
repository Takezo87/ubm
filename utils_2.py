import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import feature_cols



def window_dict(df, win_len=4):
    groups = df.groupby('investment_id').groups
    d = {}
    for group in tqdm(groups.values()):
        for i, idx in enumerate(group):
            d[idx] = group[max(0, i-win_len+1):i+1]
    return d



def prepend_slice(ids, df, win_len):
    df_ids = df.loc[df.investment_id.isin(ids)]
    groups = df_ids.groupby('investment_id').groups
    idxs = np.concatenate([i.values[(-1)*(win_len-1):] for i in groups.values()])
    df_slice = df_ids.loc[idxs]
    return df_slice

class WinDataset(Dataset):

    def __init__(self, df, win_len=4, prepend_df=None):
        
        # groups = df.groupby('investment_id').groups
        if prepend_df is not None:
            df_slice = prepend_slice(df.investment_id.unique(), prepend_df, win_len)
            df = pd.concat([df_slice, df])
            df.reset_index(inplace=True, drop=True)
            n_prepend_rows = df_slice.shape[0]


        self.n_prepend_rows = 0 if prepend_df is None else n_prepend_rows


        self.win_len = win_len
        self.win_dict = window_dict(df, win_len=self.win_len) 
        
        self.df_index = df.index
        self.features = df[feature_cols()].values
        self.targets = df['target']

    def __getitem__(self, i):
        if self.n_prepend_rows>0:
            i = i+self.n_prepend_rows
        features = self.features[self.win_dict[i]]
        n_rows = features.shape[0]
        if n_rows < self.win_len:
            # features = np.pad(features, ((self.win_len-n_rows,0), (0, 0)), mode='constant',
            # constant_values = (-1, -1))
            features = np.pad(features, ((self.win_len-n_rows,0), (0, 0)), mode='mean')
        target = self.targets[i]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

    def __len__(self):
        return self.targets.shape[0] - self.n_prepend_rows


class TimeDataset(Dataset):

    def __init__(self, df, win_len=2, prepend_df=None):
        
        # groups = df.groupby('investment_id').groups
        if prepend_df is not None:
            df_slice = prepend_slice(df.investment_id.unique(), prepend_df, win_len)
            df = pd.concat([df_slice, df])
            df.reset_index(inplace=True, drop=True)
            n_prepend_rows = df_slice.shape[0]


        self.n_prepend_rows = 0 if prepend_df is None else n_prepend_rows
        time_df = df.groupby('time_id')[feature_cols()].mean()


        self.win_len = win_len
        self.win_dict = window_dict(df, win_len=self.win_len) 
        
        self.df_index = df.index
        self.time_ids = df.time_id.values
        
        self.time_map = {k:v for v,k in enumerate(df.time_id.unique())}
        self.time_features = time_df.astype('float32').values
        self.features = df[feature_cols()].values
        self.targets = df['target']

    def __getitem__(self, i):
        if self.n_prepend_rows>0:
            i = i+self.n_prepend_rows
        features = self.features[self.win_dict[i]]
        # print(features)
        n_rows = features.shape[0]
        if n_rows < self.win_len:
            # features = np.pad(features, ((self.win_len-n_rows,0), (0, 0)), mode='constant',
            # constant_values = (-1, -1))
            features = np.pad(features, ((self.win_len-n_rows,0), (0, 0)), mode='mean')
        time_features = self.time_features[[self.time_map[self.time_ids[i]]]]
        # print(time_features)
        features = np.concatenate([features, time_features])
        target = self.targets[i]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

    def __len__(self):
        return self.targets.shape[0] - self.n_prepend_rows

class LinBnDrop(nn.Sequential):
    "Module grouping `BatchNorm1d`, `Dropout` and `Linear` layers"
    def __init__(self, n_in, n_out, bn=True, p=0., act=None, lin_first=False):
        layers = [nn.BatchNorm1d(n_out if lin_first else n_in)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not bn)]
        if act is not None: lin.append(act)
        layers = lin+layers if lin_first else layers+lin
        super().__init__(*layers)


def predict(model, test_dl, device):
    model.to(device)
    model.eval()
    val_preds, ys = [], []
    for xb, yb in tqdm(test_dl):
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        val_preds.append(logits.squeeze().reshape(-1).detach().numpy())
        ys.append(yb.squeeze().reshape(-1).detach().numpy())
    return np.concatenate(val_preds), np.concatenate(ys)


def do_predict(model, df, df_test, win_len=4):
    test_ds = WinDataset(df_test, prepend_df=df, win_len=win_len)
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False)
    preds, ys = predict(model, test_dl, torch.device('cpu'))
    torch.cuda.empty_cache()
    return preds


def do_iterative_prediction(model, df, df_test, win_len=4):
    time_id_groups = df_test.groupby('time_id').groups
    all_preds = []
    for group in time_id_groups.values():
        df_t = df_test.loc[group]
        # df_t['time_id'] = df_t['row_id'].transform(lambda x: int(x.split('_'[0])))
        preds = do_predict(model, df, df_t, win_len)
        all_preds.append(preds)
        if df is not None:
            df = pd.concat([df, df_t])
            df.reset_index(inplace=True, drop=True)
        
    return np.concatenate(all_preds)






def run_training_2(model, train_dl, valid_dl, device):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), 1e-2)
    loss_fn = F.mse_loss
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-5, epochs=20, steps_per_epoch=1)
    for i in range(50):
        losses=[]
        for xb, yb in tqdm(train_dl):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            # logits = model(xb)
            logits = model(xb)
            # weights = calc_weights(yb, seq_lens, seq_limits)
            # print(weights.shape)
            loss = loss_fn(logits.squeeze(-1), yb.squeeze())
            # print(loss)
            losses.append(loss)
            # print(torch.tensor(losses).mean())
            loss.backward()
            opt.step()
        # scheduler.step()
        # print(opt.state_dict)
        # print(scheduler.get_last_lr())
        with torch.no_grad():
            print(f'train loss {torch.tensor(losses).mean()}')
            # val_losses = []
            for xb, yb, seq_lens, seq_limits in valid_dl:
                logits = model(xb)
                weights = calc_weights(yb.data, seq_lens, seq_limits)
                val_losses.append([loss_fn(logits, yb.data, weights)])
            # preds, y_true = predict(model, valid_dl)
            # print(f'val_loss: {weighted_mse(preds, y_true)}, pearson: {pearsonr(preds.detach().numpy(), y_true.detach().numpy())[0]}')
            print(f'valid loss {torch.tensor(val_losses).mean()}')
