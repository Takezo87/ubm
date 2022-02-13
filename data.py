import argparse
import gc

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pytorch_lightning as pl


NUM_FEATURES_DEFAULT = 20


def feature_cols(n_features=NUM_FEATURES_DEFAULT):
    col_subset = [f"f_{i}" for i in range(0, n_features)]
    return col_subset


def idx_cols():
    col_subset = ["time_id", "investment_id"]
    return col_subset


def load_df(
    idx_cols, feature_cols, target="target", fn="train_low_mem.parquet", small=False
):
    train = pd.read_parquet(fn, columns=idx_cols + feature_cols + ["target"])
    if small:
        train = train.loc[train.investment_id <= 100]
        train.reset_index(inplace=True, drop=True)
    return train


def window_dict(df, win_len=4):
    groups = df.groupby("investment_id").groups
    d = {}
    for group in tqdm(groups.values()):
        for i, idx in enumerate(group):
            d[idx] = np.array(group[max(0, i - win_len + 1) : i + 1])
    return d


def prepend_slice(ids, df, win_len):
    df_ids = df.loc[df.investment_id.isin(ids)]
    groups = df_ids.groupby("investment_id").groups
    idxs = np.concatenate([i.values[(-1) * (win_len - 1) :] for i in groups.values()])
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

        # self.df_index = df.index
        self.features = df[feature_cols()].values
        if "target" in list(df.columns):
            self.targets = df["target"]
            self.with_targets = True
        else:
            self.with_targets = False

    def __getitem__(self, i):
        if self.n_prepend_rows > 0:
            i = i + self.n_prepend_rows
        features = self.features[self.win_dict[i]]
        n_rows = features.shape[0]
        if n_rows < self.win_len:
            # features = np.pad(features, ((self.win_len-n_rows,0), (0, 0)), mode='constant',
            # constant_values = (-1, -1))
            features = np.pad(
                features, ((self.win_len - n_rows, 0), (0, 0)), mode="mean"
            )
        if self.with_targets:
            target = self.targets[i]
            return torch.tensor(features, dtype=torch.float32), torch.tensor(
                target, dtype=torch.float32
            )
        else:
            return torch.tensor(features, dtype=torch.float32)

    def __len__(self):
        return self.features.shape[0] - self.n_prepend_rows


class TimeDataset(Dataset):
    def __init__(self, df, win_len=2, prepend_df=None):

        # groups = df.groupby('investment_id').groups
        if prepend_df is not None:
            df_slice = prepend_slice(df.investment_id.unique(), prepend_df, win_len)
            df = pd.concat([df_slice, df])
            df.reset_index(inplace=True, drop=True)
            n_prepend_rows = df_slice.shape[0]

        self.n_prepend_rows = 0 if prepend_df is None else n_prepend_rows
        time_df = df.groupby("time_id")[feature_cols()].mean()

        self.win_len = win_len
        self.win_dict = window_dict(df, win_len=self.win_len)

        self.df_index = df.index
        self.time_ids = df.time_id.values

        self.time_map = {k: v for v, k in enumerate(df.time_id.unique())}
        self.time_features = torch.tensor(
            time_df.astype("float32").values, dtype=torch.float32
        )
        self.features = torch.tensor(df[feature_cols()].values, dtype=torch.float32)
        if "target" in list(df.columns):
            self.targets = torch.tensor(df["target"], dtype=torch.float32)
            self.with_targets = True
        else:
            self.with_targets = False

    def __getitem__(self, i):
        if self.n_prepend_rows > 0:
            i = i + self.n_prepend_rows
        features = self.features[self.win_dict[i]]
        # print(features)
        n_rows = features.shape[0]
        if n_rows < self.win_len:
            # features = np.pad(features, ((self.win_len-n_rows,0), (0, 0)), mode='constant',
            # constant_values = (-1, -1))
            # features = np.pad(
            #     features, ((self.win_len - n_rows, 0), (0, 0)), mode="mean"
            features = F.pad(
                features, (0, 0, self.win_len - n_rows, 0), mode="constant"
            )  # )
        time_features = self.time_features[[self.time_map[self.time_ids[i]]]]
        # print(time_features)
        # features = np.concatenate([features, time_features])
        features = torch.cat([features, time_features])
        if self.with_targets:
            target = self.targets[i]

            # return torch.tensor(features, dtype=torch.float32), torch.tensor(
            #     target, dtype=torch.float32
            # )
            return features, target
        else:
            # return torch.tensor(features, dtype=torch.float32)
            return features

    def __len__(self):
        return self.features.shape[0] - self.n_prepend_rows


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
    preds, ys = predict(model, test_dl, torch.device("cpu"))
    torch.cuda.empty_cache()
    return preds


def do_iterative_prediction(model, df, df_test, win_len=4):
    time_id_groups = df_test.groupby("time_id").groups
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


class WinDS(Dataset):
    def __init__(self, features, idcs, win_dict, targets=None, win_len=1, **kwargs):
        self.features, self.idcs, self.win_dict = features, idcs, win_dict
        self.targets, self.win_len = targets, win_len

    def __getitem__(self, i):
        features = self.features[self.win_dict[self.idcs[i]]]
        if features.shape[0] < self.win_len:
            features = np.pad(
                features, ((self.win_len - features.shape[0], 0), (0, 0)), mode="mean"
            )
        if self.targets is None:
            return features
        else:
            return features, self.targets[self.idcs[i]]

    def __len__(self):
        return len(self.idcs)


def get_win_features(features, win_dict, i, win_len=4):
    win_features = features[win_dict[i]]
    if win_features.shape[0] < win_len:
        win_features = np.pad(
            win_features, ((win_len - win_features.shape[0], 0), (0, 0)), mode="mean"
        )
    if win_len == 1:
        win_features = np.expand_dims(win_features, axis=0)
    return win_features


def get_time_features(time_features, time_ids, time_map, i, time_win_len=1):
    if time_win_len == 0:
        return np.array([], dtype="float32")
    time_idx = time_map[time_ids[i]]
    time_win_features = time_features[
        max(0, time_idx - time_win_len + 1) : time_idx + 1
    ]
    if time_win_features.shape[0] < time_win_len:
        time_win_features = np.pad(
            time_win_features,
            ((time_win_len - time_win_features.shape[0], 0), (0, 0)),
            mode="mean",
        )
    return time_win_features


class TimeDS(Dataset):
    def __init__(
        self,
        features,
        time_features,
        idcs,
        win_dict,
        time_ids,
        time_map,
        targets=None,
        win_len=1,
        time_win_len=1,
        **kwargs,
    ):
        self.features, self.time_features, self.targets = (
            features,
            time_features,
            targets,
        )
        self.idcs = idcs
        self.win_dict, self.win_len = win_dict, win_len
        self.time_ids, self.time_map, self.time_win_len = (
            time_ids,
            time_map,
            time_win_len,
        )

    def __getitem__(self, i):
        df_idx = self.idcs[i]
        # print(df_idx)
        features = get_win_features(self.features, self.win_dict, df_idx, self.win_len)
        # time_step aggregated features

        time_features = get_time_features(
            self.time_features, self.time_ids, self.time_map, df_idx, self.time_win_len
        )
        # print(features)
        # print(time_features)
        # features = np.concatenate([features, time_features])
        # features_full = np.concatenate([features, time_features])
        if self.targets is not None:
            target = self.targets[df_idx]
            return features, time_features, target
        else:
            return features, time_features

    def __len__(self):
        return len(self.idcs)


def default_args():
    args = argparse.Namespace(
        df_path="train_low_mem.parquet",
        win_len=1,
        time_win_len=0,
        dset_type="time_ds",
        num_features=20,
        split_time_id=1000,
        batch_size=256,
        num_workers=0,
        small_df=True,
        pin_memory=False,
    )
    return args


class WinDM(pl.LightningDataModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        args = vars(args) if args is not None else None

        self.num_features = args.get("num_features")
        self.df_path = args.get("df_path")
        self.win_len = args.get("win_len")
        self.time_win_len = args.get("time_win_len")
        self.dset_type = args.get("dset_type")
        self.split_time_id = args.get("split_time_id")
        self.batch_size = args.get("batch_size")
        self.num_workers = args.get("num_workers")
        self.small_df = args.get("small_df")
        self.pin_memory = args.get('pin_memory', False)

    def setup(self):
        self.feature_cols = feature_cols(self.num_features)
        df = load_df(
            idx_cols(), self.feature_cols, fn=self.df_path, small=self.small_df
        )
        # self.df = df
        self.features = df[self.feature_cols].values.astype("float32")
        if "target" in df.columns:
            self.targets = df["target"].values
        else:
            self.targets = None
        if self.win_len > 1:
            self.win_dict = window_dict(df, self.win_len)
        else:
            self.win_dict = {k: k for k in df.index}
        if self.split_time_id == -1:
            self.train_idcs = df.index
            self.val_idcs = df.loc[df.time_id == df.time_id.max()].index
        else:
            self.train_idcs = df.loc[df.time_id <= self.split_time_id].index
            self.val_idcs = df.loc[df.time_id > self.split_time_id].index
        self.test_idcs = self.val_idcs

        # time aggregate mapping
        print(self.dset_type)
        if self.dset_type == "time_ds":
            print("time ds")
            time_df = df.groupby("time_id")[self.feature_cols].mean()
            self.time_ids = df.time_id.values
            self.time_map = {k: v for v, k in enumerate(df.time_id.unique())}
            self.time_features = time_df.astype("float32").values
        else:
            self.time_ids, self.time_map, self.time_features = None, None, None

        dset_cls = TimeDS if self.dset_type == "time_ds" else WinDS
        self.train_dset = dset_cls(
            self.features,
            self.time_features,
            self.train_idcs,
            self.win_dict,
            time_ids=self.time_ids,
            time_map=self.time_map,
            targets=self.targets,
            win_len=self.win_len,
            time_win_len=self.time_win_len,
        )
        self.valid_dset = dset_cls(
            self.features,
            self.time_features,
            self.val_idcs,
            self.win_dict,
            time_ids=self.time_ids,
            time_map=self.time_map,
            targets=self.targets,
            win_len=self.win_len,
            time_win_len=self.time_win_len,
        )
        self.test_dset = dset_cls(
            self.features,
            self.time_features,
            self.val_idcs,
            self.win_dict,
            time_ids=self.time_ids,
            time_map=self.time_map,
            targets=None,
            win_len=self.win_len,
            time_win_len=self.time_win_len,
        )

        gc.collect()

    def train_dataloader(self):
        return DataLoader(
            self.train_dset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size",
            type=int,
            default=256,
            help="Number of examples to operate on per forward step.",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=0,
            help="Number of additional processes to load data.",
        )
        parser.add_argument(
            "--df_path",
            type=str,
            default='train_low_mem.parquet',
            help="training dataframe",
        )
        parser.add_argument(
            "--num_features", type=int, default=20, help="number of features"
        )
        parser.add_argument(
            "--win_len", type=int, default=1, help="length of considered investment_id window"
        )
        parser.add_argument(
            "--time_win_len", type=int, default=0, help="length of considered time_id window"
        )
        parser.add_argument(
            "--split_time_id", type=int, default=1100, help="time_id split point for validation, -1 for full data")
        parser.add_argument(
                "--small_df", dest='small_df', default=False, action="store_true"
                )
        parser.add_argument(
            "--n_folds", type=int, default=1, help="end index of test set"
        )
        parser.add_argument(
            "--fold", type=int, default=1, help="end index of test set"
        )
        parser.add_argument(
            "--dset_type", type=str, default='time_ds', help="dataset type"
        )
        parser.add_argument(
                "--pin_memory", dest='pin_memory', default=False, action="store_true"
                )
        # parser.add_argument(
        #     "--augments", type=str, nargs='+', action='append',
        #     default=AUGMENTS, help='tfms for the groups list of lists, use like --augments noise scale --augments all --augments integer_noise --augments all'
        # )
        parser.add_argument(
            "--data_dir",
            type=str,
            default='./',
            help="directory of the input dataframe",
        )
        parser.add_argument(
            "--log_dir", type=str, default='./lightning_logs', help="directory of the log files"
        )
        return parser
