import pandas as pd 
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack
from torch.nn.utils.rnn import pad_sequence
from scipy.stats import pearsonr

# from tsai.all import *
from tqdm import tqdm

NUM_FEATURES = 20

def feature_cols(n_features=NUM_FEATURES):
    col_subset = [f'f_{i}' for i in range(0, n_features)]
    return col_subset

def idx_cols():
    col_subset = ['time_id','investment_id']
    return col_subset

def load_df(idx_cols, feature_cols, target='target', fn='train_low_mem.parquet'):
    train = pd.read_parquet(fn, columns=idx_cols+feature_cols+['target'])
    return train


def create_time_series_tensors(df, with_targets=True):
    '''
    create a dictionary of tensors for all investment_ids
    '''
    groups = df.groupby('investment_id').groups
    tensors = {k:torch.tensor(df.loc[v][feature_cols()].values) for k,v in groups.items()}
    if with_targets:
        targets = {k:torch.tensor(df.loc[v]['target'].values) for k,v in groups.items()}
    else:
        targets=None
    return tensors, targets


def create_data(df, time_id_split=1000, small=True):
    '''
    create a train/valid tensors
    '''
    if small:
        df = df.loc[df.investment_id<=500]
        df.reset_index(inplace=True, drop=True)

    x_train, y_train = create_time_series_tensors(df.loc[df.time_id<=time_id_split])
    x_valid, y_valid = create_time_series_tensors(df.loc[df.time_id>time_id_split])

    dl_train = data.DataLoader(UDataset(x_train, y_train), collate_fn=collate_fn, batch_size=4,
            pin_memory=True)
    dl_valid = data.DataLoader(UDataset(x_valid, y_valid, prepend_features=x_train, prepend_targets=y_train),
            collate_fn=collate_fn, batch_size=4, pin_memory=True)
    return dl_train, dl_valid



class UDataset(data.Dataset):
    
    def __init__(self, features:dict, targets:dict, prepend_features:dict=None,
            prepend_targets:dict=None):
        assert features.keys() == targets.keys()
        # if prepend_features is not None:
            # assert features.keys() == prepend_features.keys()

        self.features, self.targets = features, targets
        self.keys = list(self.targets.keys())

        self.prepend_limits = {k: 0 for k in self.keys} 
        if prepend_features is not None:
            for k in self.keys:
                # print(k)
                if prepend_features.get(k) is not None:
                    prep_feats, prep_targets = prepend_features[k], prepend_targets[k]
                    self.prepend_limits[k]=prep_feats.shape[0]
                    self.features[k] = torch.cat([prep_feats, features[k]])
                    self.targets[k] = torch.cat([prep_targets, targets[k]])



    def __getitem__(self, i):
        return self.features.get(self.keys[i]), self.targets.get(self.keys[i]), self.prepend_limits.get(self.keys[i])

    def __len__(self):
        return len(self.keys)

def pad_tensor(t, max_len, value=-1.):
    if len(t.shape)==2:
        return F.pad(t, (0, 0, 0, max_len-t.shape[0]))
    else:
        return F.pad(t, (0, max_len-t.shape[0]))

def collate_fn(batch, max_len=1500):
    # print([b[0].shape for b in batch])
    seq_lens = torch.LongTensor([b[0].shape[0] for b in batch])
    ## pad first xb to max_len
    # batch[0][0] = F.pad(batch[0][0], (0, 0, 0, max_len-batch[0][0].shape[0]), value=-1)
    # xb=pad_sequence([b[0] for b in batch], batch_first=True, padding_value=-1)
    # yb=pad_sequence([b[1] for b in batch], batch_first=True, padding_value=-1)
    xb=torch.stack([pad_tensor(b[0], max_len) for b in batch])
    yb=torch.stack([pad_tensor(b[1], max_len) for b in batch])
    # print(xb.shape)
    # xb_packed = pack(xb, seq_lens.cpu().numpy(), enforce_sorted=False, batch_first=True)
    # yb_packed = pack(yb, seq_lens.cpu().numpy(), enforce_sorted=False, batch_first=True)
    limits = torch.LongTensor([b[2] for b in batch])
    # return xb_packed, yb_packed, seq_lens, limits
    return xb, yb, seq_lens, limits


class ULSTM(nn.Module):

    def __init__(self, n_features=20, n_hidden=124):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, n_hidden)
        # self.lstm2 = nn.LSTM(n_hidden, n_hidden)
        # self.lstm3 = nn.LSTM(n_hidden, n_hidden)
        # self.lstm4 = nn.LSTM(n_hidden, n_hidden)
        # self.lin1 = nn.Linear(n_features, n_hidden)
        # self.transformer = nn.TransformerEncoderLayer(n_features, dim_feedforward=128)
        self.fc1 = nn.Linear(n_hidden, 50)
        self.dropout = nn.Dropout(p=.2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(50,1)

    def forward(self, xb):
        '''
        xb: packed seqeuence
        '''
        res = xb
        res, _= self.lstm1(xb)
        # res, _= self.lstm2(res)
        # res, _= self.lstm3(res)
        # res, _= self.lstm4(res)
        # res = self.lin1(res.datu)
        res = self.fc1(res)
        res = self.dropout(res)
        res = self.act(res)
        res = self.fc2(res.data)
        return res
    
    # if len(batch) == 1:
    #         sigs, labels = batch[0][0], batch[0][1]
    #         sigs = sigs.t()
    #         lengths = [sigs.size(0)]
    #         sigs.unsqueeze_(0)
    #         labels.unsqueeze_(0)
    # if len(batch) > 1:
    #     sigs, labels, lengths = zip(*[(a.t(), b, a.size(1)) for (a,b) in sorted(batch, key=lambda x: x[0].size(1), reverse=True)])
    #     max_len, n_feats = sigs[0].size()
    #     sigs = [torch.cat((s, torch.zeros(max_len - s.size(0), n_feats)), 0) if s.size(0) != max_len else s for s in sigs]
    #     print(sigs)
    #     sigs = torch.stack(sigs, 0)
    #     labels = torch.stack(labels, 0)
    # packed_batch = pack(Variable(sigs), lengths, batch_first=True)
    # return packed_batch, labels
    
    

def create_mask(xt, lengths):
    xmask = torch.zeros((xt.shape[0], xt.shape[1]))
    for i in range(lengths.shape[0]):
        xmask[i, lengths[i]:] = 1
    return xmask.bool()



class UTransformer(nn.Module):

    def __init__(self, n_features=20, n_hidden=124):
        super().__init__()
        transformer_layer=nn.TransformerEncoderLayer(n_features, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=3)
        self.fc1 = nn.Linear(n_features, 128)
        self.dropout = nn.Dropout(p=.2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(128,1)

    def forward(self, xb, lengths):
        '''
        xb: packed seqeuence
        '''

        res = xb
        # xt, lengths = unpack(xb, batch_first=True)
        xmask = create_mask(xb, lengths)
        res = self.transformer(xb, src_key_padding_mask=xmask)
        # res, _= self.lstm1(xb)
        # res, _= self.lstm2(res)
        # res, _= self.lstm3(res)
        # res, _= self.lstm4(res)
        # res = self.lin1(res.datu)
        res = self.fc1(res.data)
        res = self.dropout(res)
        res = self.act(res)
        res = self.fc2(res.data)
        return res
    
    # if len(batch) == 1:
    #         sigs, labels = batch[0][0], batch[0][1]
    #         sigs = sigs.t()
    #         lengths = [sigs.size(0)]
    #         sigs.unsqueeze_(0)
    #         labels.unsqueeze_(0)
    # if len(batch) > 1:
    #     sigs, labels, lengths = zip(*[(a.t(), b, a.size(1)) for (a,b) in sorted(batch, key=lambda x: x[0].size(1), reverse=True)])
    #     max_len, n_feats = sigs[0].size()
    #     sigs = [torch.cat((s, torch.zeros(max_len - s.size(0), n_feats)), 0) if s.size(0) != max_len else s for s in sigs]
    #     print(sigs)
    #     sigs = torch.stack(sigs, 0)
    #     labels = torch.stack(labels, 0)
    # packed_batch = pack(Variable(sigs), lengths, batch_first=True)
    # return packed_batch, labels
def get_tensors(train, y=True, prepend_df=None):
    if prepend_df is not None:
        investment_ids = list(train.investment_id.unique())
        train = pd.concat([prepend_df.loc[prepend_df['investment_id'].isin(investment_ids)], train])
        train.reset_index(inplace=True, drop=True)
    # return train
    groups = train.groupby('investment_id').groups
    # print(len(groups))
    # print(train)
    # return train, None
    features = list(train.columns[:])
    Xs = [train.iloc[group_idxs][features].values for group_idxs in list(groups.values())]
    Xs = [torch.tensor(x) for x in Xs]
    Xs = [F.pad(x, (0,0,512-x.shape[0],0)) for x in Xs]
    Xs = torch.stack(Xs)
    # return Xs, None
    print(Xs.shape)
    if y:
        print('with target')
        return Xs[:, :, :-1].permute(0,2,1), Xs[:, :, -1]
    else:
        return Xs[:, :, :].permute(0,2,1), None

def create_test_dl(learn, x_test):
    test_dl = learn.dls[1].new(TSDataset(learn.dls.add_test(x_test)))
    return test_dl

def get_preds(x_test, df_test, learn):
    x_test_std = learn.dls.after_batch(x_test).float()
    # x_test_std = x_test
    return learn.get_X_preds(x_test_std, with_decoded=False)[0][:, -1]


class SimpleLSTM(nn.Module):
    def __init__(self, c_in, hidden=128):
        super().__init__()
        self.c_in = 7
        self.lstm = nn.LSTM(input_size=c_in, hidden_size=hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden*2, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

def weighted_mse(preds, y_true, weights=None):

    if weights is None:
        weights = torch.ones_like(y_true)

    loss = (preds.squeeze() - y_true.squeeze()).pow_(2)
    # print(loss.shape)
    loss = loss*weights
    loss = (preds.squeeze() - y_true.squeeze()).pow_(2) * weights.squeeze()
    return loss.sum()/weights.sum()

def calc_weights(yb, seq_lens, seq_limits, max_len=1500):
    weights = torch.ones_like(yb)
    cursor=0
    for i, (total_length, limit) in enumerate(zip(seq_lens, seq_limits)):
        weights[i, 0:limit] = 0
        weights[i, total_length: max_len]= 0
    return weights

def run_training(model, train_dl, valid_dl):
    opt = torch.optim.AdamW(model.parameters(), 1e-2)
    loss_fn = weighted_mse
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-5, epochs=20, steps_per_epoch=1)
    for i in range(50):
        losses=[]
        for xb, yb, seq_lens, seq_limits in tqdm(train_dl):
            opt.zero_grad()
            # logits = model(xb)
            logits = model(xb, seq_lens)
            weights = calc_weights(yb, seq_lens, seq_limits)
            # print(weights.shape)
            loss = loss_fn(logits.squeeze(-1), yb.data.squeeze(), weights=weights.squeeze())
            losses.append(loss)
            loss.backward()
            opt.step()
        # scheduler.step()
        # print(opt.state_dict)
        # print(scheduler.get_last_lr())
        with torch.no_grad():
            print(f'train loss {torch.tensor(losses).mean()}')
            # val_losses = []
            # for xb, yb, seq_lens, seq_limits in valid_dl:
            #     logits = model(xb)
            #     weights = calc_weights(yb.data, seq_lens, seq_limits)
            #     val_losses.append([loss_fn(logits, yb.data, weights)])
            preds, y_true = predict(model, valid_dl)
            print(f'val_loss: {weighted_mse(preds, y_true)}, pearson: {pearsonr(preds.detach().numpy(), y_true.detach().numpy())[0]}')
            # print(f'valid loss {torch.tensor(val_losses).mean()}')

def predict(model, test_dl):
    model.eval()
    val_preds, ys = [], []
    for xb, yb, seq_lens, seq_limits in test_dl:
        # print(xb.data, seq_limits)
        # logits = model(xb.float())
        logits = model(xb.float(), seq_lens)
        # print(logits,shape)
        weights = calc_weights(yb.data, seq_lens, seq_limits)
        # print(weights.shape, logits.shape)
        # print(weights[-5:])
        val_preds.append(logits.squeeze()[weights.squeeze().bool()])
        ys.append(yb.data.squeeze()[weights.squeeze().bool()])
    return torch.cat(val_preds), torch.cat(ys)


            


def pearson_coef(data):
    return data.corr()['target']['preds']

def comp_metric(valid_df):
    return np.mean(valid_df[['time_id', 'target', 'preds']].groupby('time_id').apply(pearson_coef))

def combine_tensors(x_train, x_test):
    # keys = x_test.investment_id
    return {k:torch.cat([x_train[k], x_test[k]]) for k in x_test.keys()}

def create_dummy_targets(x:dict, value=-1):
    return {k:torch.ones(x[k].shape[0])*value for k in x.keys()}
    

def predict_test(x_train, df_test, model):
    all_preds=[]
    for time_id in df_test.time_id.unique():
        df = df_test.loc[df_test.time_id==time_id].copy()
        df.reset_index(inplace=True, drop=True)
        # print(df.shape)
        x_test, _ = create_time_series_tensors(df, with_targets=False)
        # print(x_test[1])
        y_train, y_test = create_dummy_targets(x_train), create_dummy_targets(x_test)
        test_dset = UDataset(x_test, y_test, prepend_features=x_train, prepend_targets=y_train)
        test_dl = data.DataLoader(test_dset, collate_fn=collate_fn, batch_size=4, shuffle=False)
        preds, ys = predict(model, test_dl)
        all_preds.append(preds)
    return torch.cat(all_preds)


