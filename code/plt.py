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

plt.figure(1, figsize=(8, 8))
plt.subplot(121)
plt.plot(LossListM)
plt.title('Loss')
plt.subplot(122)
plt.plot(AccListM)
plt.title('Accuracy')
plt.show()