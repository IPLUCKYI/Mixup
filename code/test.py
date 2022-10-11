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

#@title model evaluation


def test_model(model, training_set, test_set):

    model.eval()

    N_tr = len(training_set.x)
    N_te = len(test_set.x)

    training_generator = DataLoader(training_set, batch_size=1,
                                    shuffle=True, drop_last=False)
    test_generator = DataLoader(test_set, batch_size= 1,
                                    shuffle=True, drop_last=False)

    H_tr = th.zeros((N_tr, 128))
    y_tr = th.zeros((N_tr), dtype=th.long)

    H_te = th.zeros((N_te, 128))
    y_te = th.zeros((N_te), dtype=th.long)

    for idx_tr, (x_tr, y_tr_i) in enumerate(training_generator):
        with th.no_grad():
            _, H_tr_i = model(x_tr)
            H_tr[idx_tr] = H_tr_i
            y_tr[idx_tr] = y_tr_i

    H_tr = to_np(nn.functional.normalize(H_tr))
    y_tr = to_np(y_tr)


    for idx_te, (x_te, y_te_i) in enumerate(test_generator):
        with th.no_grad():
            _, H_te_i = model(x_te)
            H_te[idx_te] = H_te_i
            y_te[idx_te] = y_te_i

    H_te = to_np(nn.functional.normalize(H_te))
    y_te = to_np(y_te)

    clf = KNeighborsClassifier(n_neighbors=1).fit(H_tr, y_tr)

    return clf.score(H_te, y_te)