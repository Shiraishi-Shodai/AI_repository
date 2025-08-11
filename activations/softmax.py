import numpy as np
import torch
from matplotlib import pyplot as plt
import japanize_matplotlib
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F

x = torch.tensor([0.3, 2.9, 4.0])

def softmax(x):
    c = torch.max(x)
    exp_x = torch.exp(x - c)
    sum_exp_x = torch.sum(exp_x)
    y = exp_x / sum_exp_x
    return y

def sigmoid(self, u):
    return 1 / (1 + np.exp(-u))

def dsigmoid(self, dout, z):
    return dout * (1 - z) * z


def ReLU(self, u):
    """ここで受け取るuは行列である。

    u: u > 0
    0: u <= 0
    """
    return np.where(u > 0, u, 0)

def dReLU(self, u, dz):
    """ここで受け取るuとdは行列である。
        また、dzはuが属する層におけるdL/dzを意味する。
        つまり、zにおける損失関数の勾配を表す。

        例：）1層のニューロンにおけるduを求めたい。
        u:  u1
        dz：z1における損失関数の勾配(dL/dz1)

        ReLUの微分では、活性化したニューロン(出力が0以上)における勾配をそのまま返す。

        1: u >= 0
        0: u < 0
    """
    d_rel_mask =  np.where(u > 0, 1, 0) # uが0以上なら1, それ以下なら0

    return dz * d_rel_mask 