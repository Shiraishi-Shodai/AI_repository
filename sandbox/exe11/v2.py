"""
v1.py + ミニバッチ勾配降下法
"""

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import japanize_matplotlib
import traceback
import random

random.seed(0)

def create_minibatches(X, y, batch_size, shuffle=True):
    """
    Xとyを受け取って、ミニバッチに分割して返す関数

    Parameters:
        X (np.ndarray): 入力データ (例: (N, 特徴量数))
        y (np.ndarray): ラベルデータ (例: (N, 出力次元))
        batch_size (int): バッチサイズ
        shuffle (bool): シャッフルするかどうか（デフォルト: True）

    Returns:
        List[Tuple[np.ndarray, np.ndarray]]: (X_batch, y_batch) のリスト
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    X_shuffled = X[indices]
    y_shuffled = y[indices]

    minibatches = []

    for start_idx in range(0, n_samples, batch_size):
        end_idx = start_idx + batch_size
        X_batch = X_shuffled[start_idx:end_idx]
        y_batch = y_shuffled[start_idx:end_idx]
        minibatches.append((X_batch, y_batch))

    return minibatches



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

        """
        変更点①：勾配をクラス変数で持つように変更。
        """
        self.grads = {
            "dw1": np.random.normal(loc=0, scale=1, size=(self.input_dim, self.hidden1)),
            "db1": np.random.normal(loc=0, scale=1, size=(1, self.hidden1)),
            "dw2": np.random.normal(loc=0, scale=1, size=(self.hidden1, self.hidden2)),
            "db2": np.random.normal(loc=0, scale=1, size=(1, self.hidden2)),
            "dw3": np.random.normal(loc=0, scale=1, size=(self.hidden2, self.out_dim)),
            "db3": np.random.normal(loc=0, scale=1, size=(1, self.out_dim))
        }
    
    def loss_func(self, y_pred, y):
        """ここでは、ディープラーニングでよく利用される。1/2を掛けて勾配を求めやすくした
            損失関数を用いる。疑似平均二乗誤差関数
        """

        # loss = 1/self.data_len * 1/2  * ((y_pred - y)**2)
        loss = np.mean(1/2  * ((y_pred - y)**2))

        return loss
    
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

    

    def foward(self, X):
        self.X = X

        self.u1 = X @ self.params["w1"] + self.params["b1"] # (50, 2)
        self.z1 = self.ReLU(self.u1) # (50, 2)

        self.u2 = self.z1 @ self.params["w2"] + self.params["b2"] # (50, 3)
        self.z2 = self.ReLU(self.u2) # (50, 3)

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
        du2 = self.dReLU(self.u2, dz2) # (50, 3)

        dw2 = self.z1.T @ du2 # (2, 3)
        db2 = np.sum(du2, axis=0) # (1, 3)

        # dz1は入力値が少しずれると、損失関数にどれくらい影響するかを表すため、本来は計算不要
        dz1 = du2 @ self.params["w2"].T # (50, 2)
        du1 = self.dReLU(self.u1, dz1) # (50, 2)

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

            """
            変更点③：ミニバッチ化
            """
            batch_size=10
            batches =  create_minibatches(X_train, y_train, batch_size=batch_size)
            batch_num = 0

            for (X_train_batches, y_train_batches) in batches:
                batch_num += 1
                y_pred = self.foward(X_train_batches)
                loss = self.loss_func(y_pred, y_train_batches)
                grads = self.backward(y_pred, y_train_batches)

                """
                変更点④：勾配をクラス変数のgradに合計する
                """
                self.grads = {k: self.grads[k] + grads[k] for k in self.grads}


            """
            変更点⑤：各勾配の平均を計算
            """
            self.grads = {k: self.grads[k] / batch_num for k in self.grads}
            self.loss_history.append(loss)

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

    nn = NeuralNetwork(data_len=X.shape[0], input_dim=X.shape[1], hidden1=2, hidden2=3, out_dim=1, epochs=1000, lr=0.008)

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

    plt.savefig("v2_result.png")

if __name__ == "__main__":
    main()
