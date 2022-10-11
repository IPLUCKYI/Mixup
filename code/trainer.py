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

#@title mixup model trainer per epoch


def train_mixup_model_epoch(model, training_set, test_set, optimizer, alpha, epochs):

    device = 'cuda' if th.cuda.is_available() else 'cpu'
    batch_size_tr = len(training_set.x)

    LossList, AccList = [] , []
    criterion = MixUpLoss(device, batch_size_tr)

    training_generator = DataLoader(training_set, batch_size=batch_size_tr,
                                    shuffle=True, drop_last=True)

    for epoch in range(epochs):

        for x, y in training_generator:

            model.train()

            optimizer.zero_grad()

            x_1 = x
            x_2 = x[th.randperm(len(x))]

            lam = np.random.beta(alpha, alpha)

            x_aug = lam * x_1 + (1-lam) * x_2

            z_1, _ = model(x_1)
            z_2, _ = model(x_2)
            z_aug, _ = model(x_aug)

            loss= criterion(z_aug, z_1, z_2, lam)
            loss.backward()
            optimizer.step()
            LossList.append(loss.item())


        AccList.append(test_model(model, training_set, test_set))

        print(f"Epoch number: {epoch}")
        print(f"Loss: {LossList[-1]}")
        print(f"Accuracy: {AccList[-1]}")
        print("-"*50)

        if epoch % 10 == 0 and epoch != 0: clear_output()
            
    return LossList, AccList