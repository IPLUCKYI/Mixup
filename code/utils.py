import numpy as np
import torch as th

def to_np(x):
    return x.cpu().detach().numpy()

def set_global_seed(seed:int):
    th.manual_seed(seed)
