import pandas as pd
import numpy as np
import os

def load_data(filename, typename):
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