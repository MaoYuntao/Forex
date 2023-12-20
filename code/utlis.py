import random
import numpy as np
import torch
import os
import pandas as pd


def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_dataset(path: str):
    df = pd.read_csv(path)
    df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
    data_train = df.iloc[:int(df.shape[0] * 0.7)]
    data_val = df.iloc[int(df.shape[0] * 0.7):]
    data_train.to_csv('data/train_data.csv', index=False)
    data_val.to_csv('data/val_data.csv', index=False)
