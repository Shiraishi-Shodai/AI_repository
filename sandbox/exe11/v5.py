"""勾配が学習中に消失している説。
活性化関数に、ReLUではなく、LeakyReLUを使う。

各重みのL2ノルムを描画
各勾配のL2ノルムが小さい。つまり、各勾配の更新量や方向が小さい。
勾配消失が起きている可能性が高い。

初期化時の重みを工夫する
https://zero2one.jp/learningblog/strategy-for-deep-learning-weight-initialization/?srsltid=AfmBOorRHSo1ZSdeV_jO6yBVZBlB3D0IbyWu6us-zrX5N3UfzcBQOJKf

"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import japanize_matplotlib
import random

random.seed(0)

class NeuralNetwork():

    """
    3層のニューラルネットワーク
    入力層：1
    中間層：(2, 3)
    出力層：1
    活性化関数：ReLU(出力層は恒等関数)
    損失関数：1/2を掛けて、勾配を求めやすくした疑似平均二乗誤差
    勾配の求め方：誤差逆伝播法
    最適化手法：勾配降下法
    """

    def __init__(self, data_len, input_dim=1, hidden1=2, hidden2=3, out_dim=1, epochs=1000, lr=0.001):

        # 損失関数に平均二乗誤差を使うため、データの個数は使用頻度が高くなると予想される。
        # そのため、予めデータの個数を取得しておく
        self.data_len = data_len 

        self.input_dim = input_dim # 説明変数の次元数
        self.hidden1 = hidden1  # 1層目のニューロンの個数
        self.hidden2 = hidden2  # 2層目のニューロンの個数
        self.out_dim = out_dim  # 3層目のニューロンの個数
        self.epochs = epochs
        self.lr = lr

        self.loss_history = []

        self.initialize_parameters()

    def initialize_parameters(self):
        np.random.seed(42)

        self.params = {
            "w1": np.random.normal(loc=0, scale=1, size=(self.input_dim, self.hidden1)),
            "b1": np.random.normal(loc=0, scale=1, size=(1, self.hidden1)),
            "w2": np.random.normal(loc=0, scale=1, size=(self.hidden1, self.hidden2)),
            "b2": np.random.normal(loc=0, scale=1, size=(1, self.hidden2)),
            "w3": np.random.normal(loc=0, scale=1, size=(self.hidden2, self.out_dim)),
            "b3": np.random.normal(loc=0, scale=1, size=(1, self.out_dim))
        }

        self.grad_norms = {
            "dw1": [],
            "db1": [],
            "dw2": [],
            "db2": [],
            "dw3": [],
            "db3": []
        }

    
    def loss_func(self, y_pred, y):
        """ここでは、ディープラーニングでよく利用される。1/2を掛けて勾配を求めやすくした
            損失関数を用いる。疑似平均二乗誤差関数
        """

        # loss = 1/self.data_len * 1/2  * ((y_pred - y)**2)
        loss = np.mean(1/2  * ((y_pred - y)**2))

        return loss
    
    # def ReLU(self, u):
    #     """ここで受け取るuは行列である。

    #     u: u > 0
    #     0: u <= 0
    #     """
    #     return np.where(u > 0, u, 0)

    def LeakyReLU(self, u):
        a = 0.01
        # return a * min(0, u) + max(0, u)
        return np.maximum(a * u, u)
    
    def dLeakyReLU(self, u, dz):
        a = 0.01
        d_mask = np.where(u > 0, 1, a)  # u > 0 → 1, u <= 0 → a
        return dz * d_mask
    
    # def dReLU(self, u, dz):
    #     """ここで受け取るuとdは行列である。
    #         また、dzはuが属する層におけるdL/dzを意味する。
    #         つまり、zにおける損失関数の勾配を表す。

    #         例：）1層のニューロンにおけるduを求めたい。
    #         u:  u1
    #         dz：z1における損失関数の勾配(dL/dz1)

    #         ReLUの微分では、活性化したニューロン(出力が0以上)における勾配をそのまま返す。

    #         1: u >= 0
    #         0: u < 0
    #     """
    #     d_rel_mask =  np.where(u > 0, 1, 0) # uが0以上なら1, それ以下なら0

    #     return dz * d_rel_mask 

    def foward(self, X):
        self.X = X

        self.u1 = X @ self.params["w1"] + self.params["b1"] # (50, 2)
        self.z1 = self.LeakyReLU(self.u1) # (50, 2)

        self.u2 = self.z1 @ self.params["w2"] + self.params["b2"] # (50, 3)
        self.z2 = self.LeakyReLU(self.u2) # (50, 3)

        self.u3 = self.z2 @ self.params["w3"] + self.params["b3"] # (50, 1)
        self.z3 =self. u3 # (50, 1)

        return self.z3 # 目的変数の形に合わせる

    
    def backward(self, y_pred,  y):
        z3 = y_pred

        dz3 = 1/self.data_len * (z3 - y) # (50, 1)
        du3 = dz3 * 1 # (50, 1)

        dw3 = self.z2.T @ du3 # (3, 1)
        db3 = np.sum(du3, axis=0) # (1, 1) 縦に合計

        dz2 = du3 @ self.params["w3"].T # (50, 3)
        du2 = self.dLeakyReLU(self.u2, dz2) # (50, 3)

        dw2 = self.z1.T @ du2 # (2, 3)
        db2 = np.sum(du2, axis=0) # (1, 3)

        # dz1は入力値が少しずれると、損失関数にどれくらい影響するかを表すため、本来は計算不要
        dz1 = du2 @ self.params["w2"].T # (50, 2)
        du1 = self.dLeakyReLU(self.u1, dz1) # (50, 2)

        dw1 = self.X.T @ du1  # (1, 2)
        db1 = np.sum(du1, axis=0) # (1, 2)

        grads = {
            "dw1": dw1,
            "db1": db1,
            "dw2": dw2,
            "db2": db2,
            "dw3": dw3,
            "db3": db3,
        }

        return grads

    def fit(self, X_train, y_train):

        for epoch in range(1, self.epochs + 1):
            y_pred = self.foward(X_train)
            loss = self.loss_func(y_pred, y_train)
            grads = self.backward(y_pred, y_train)

            self.loss_history.append(loss)

            # 勾配を計算
            for k in self.grad_norms.keys():
                grad_norms = np.linalg.norm(grads[k], ord=2)
                self.grad_norms[k].append(grad_norms)

            # 重みを更新(zip: パラメータのキー, 各バラメータの勾配)
            for w_key, dw_value in zip(self.params.keys(), grads.values()):
                self.params[w_key] = self.params[w_key] - dw_value * self.lr
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch} loss {loss}")
            
            if loss  <= 0.001:
                break

    def predict(self, X_test):
        y_pred = self.foward(X_test)
        return y_pred

def main():
    df = pd.read_csv("../../csv/exe11.csv")
    X = df["x"].to_numpy().reshape(-1, 1)
    y = df["y"].to_numpy().reshape(-1, 1)

    nn = NeuralNetwork(data_len=X.shape[0], input_dim=X.shape[1], hidden1=5, hidden2=2, out_dim=1, epochs=1000, lr=0.0012)

    nn.fit(X_train=X, y_train=y)
    y_pred = nn.predict(X_test=X)

    # 描画
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(np.arange(0, len(nn.loss_history)), nn.loss_history)
    ax1.set_title("損失関数の遷移")
    ax1.set_xlabel("損失回数計算回数")
    ax1.set_ylabel("損失関数")

    ax2.scatter(X.flatten(), y, label="元データ")
    ax2.plot(X.flatten(), y_pred.flatten(), label="予測")
    ax2.grid()    
    ax2.legend()
    ax2.set_title("予測")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    plt.savefig("v5_result.png")

    # 勾配のL2ノルムを描画
    # 描画用 figure を用意
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    # 各勾配のノルムの推移を描画
    for key, values in nn.grad_norms.items():
        ax.plot(values, label=f"{key}のL2ノルム")

    ax.set_title("各重みの勾配のL2ノルムの推移")
    ax.set_xlabel("エポック数")
    ax.set_ylabel("L2ノルム")
    ax.legend()
    ax.grid()
    fig.savefig("v5_l2_norm.png")

if __name__ == "__main__":
    main()
