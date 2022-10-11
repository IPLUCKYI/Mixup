#@title Load packages and data


import torch as th
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt


from IPython.display import clear_output
from sktime.datasets import load_gunpoint
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import KNeighborsClassifier


def to_np(x):
    return x.cpu().detach().numpy()

#@title load data and convert to numpy array

def toarray(x,y):
    x = pd.DataFrame(x).to_numpy()
    y = pd.DataFrame(y).to_numpy()

    x = np.array(np.ndarray.tolist(x), dtype=np.float32)
    y = np.array(np.ndarray.tolist(y), dtype=np.int32)
    return x,y


#@title create dataset

class MyDataset(Dataset):
    def __init__(self, x, y):

        device = 'cuda' if th.cuda.is_available() else 'cpu'
        self.x = th.tensor(x, dtype=th.float, device=device)
        self.y = th.tensor(y, dtype=th.long, device=device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]