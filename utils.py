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


    #     max_len, n_feats = sigs[0].size()
    #     sigs = [torch.cat((s, torch.zeros(max_len - s.size(0), n_feats)), 0) if s.size(0) != max_len else s for s in sigs]
    #     print(sigs)
    #     sigs = torch.stack(sigs, 0)
    #     labels = torch.stack(labels, 0)
    # packed_batch = pack(Variable(sigs), lengths, batch_first=True)
    # return packed_batch, labels
    
    

