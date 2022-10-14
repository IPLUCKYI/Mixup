from enum import unique
import pandas as pd
import numpy as np
import os
from scipy.io.arff import loadarff

def load_ucr_data(filename, typename):
    path = r'./data/UCRArchive_2018/%s/'%(filename)  
    for name in os.listdir(path):
        if typename.upper() in name:
            sj = pd.read_csv(path + name, sep='\t', header=None)
            break

    x = sj.iloc[:,1:]
    kong = []
    for rows in range(x.shape[0]):
        kong.append( list(x.iloc[rows,:]) )
    x.iloc[:,0] = kong
    x = x.iloc[:,:1]
    y = np.array( sj.iloc[:,0] )
    return x,y

def shift_y(x):
    new = []
    uniq = list(set(x))
    for i in x:
        n = 0
        for j in uniq:
            if i == j:
                new.append(n)
            else:
                n += 1
    return new

def load_uea_data(filename, typename):
    path = r'./data/Multivariate_arff/' + filename + r'/' + filename + '_' + typename.upper() + r'.arff'
    data = loadarff(path)[0]

    x = [list(sa[0]) for sa in data]
    x = [[list(y) for y in i] for i in x]
    y = [i[1] for i in data]
    y = shift_y(y)
    return x,y

