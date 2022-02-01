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

from data import *


def get_experiment_dsets(dset_type = WinDataset, small=True):
    df = load_df(idx_cols(), feature_cols()) 
    if small:
        df = df.loc[df.investment_id<=100]
    df_train = df.loc[df.time_id<=1000]
    df_valid = df.loc[df.time_id>1000]
    df_train.reset_index(drop=True, inplace=True)
    df_valid.reset_index(drop=True, inplace=True)

    return dset_type(df_train), dset_type(df_valid), dset_type(df_valid.drop(columns='target'))
    


    #     max_len, n_feats = sigs[0].size()
    #     sigs = [torch.cat((s, torch.zeros(max_len - s.size(0), n_feats)), 0) if s.size(0) != max_len else s for s in sigs]
    #     print(sigs)
    #     sigs = torch.stack(sigs, 0)
    #     labels = torch.stack(labels, 0)
    # packed_batch = pack(Variable(sigs), lengths, batch_first=True)
    # return packed_batch, labels
    
    

