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

#@title Experiment number of epochs

device = 'cuda' if th.cuda.is_available() else 'cpu'
epochs = 10

alpha = 1.0

training_set = MyDataset(x_tr, y_tr)
test_set = MyDataset(x_te, y_te)

model = FCN(training_set.x.shape[1]).to(device)

optimizer = th.optim.Adam(model.parameters())
LossListM, AccListM = train_mixup_model_epoch(model, training_set, test_set,
                                              optimizer, alpha, epochs)


print(f"Score for alpha = {alpha}: {AccListM[-1]}")