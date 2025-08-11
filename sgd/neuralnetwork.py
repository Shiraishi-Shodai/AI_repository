import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import japanize_matplotlib
import traceback
import random

"""
活性化関数：LeakeyReLU
最適化手法：ミニバッチ勾配降下法
Epoch 1000, Loss 50.86218781869696, grad_norm2 0.07130531831041349
各重みの勾配
"""
random.seed(0)

def get_mini_batch_indexes(data_len,  batch_size=10):
    """データのサイズを受け取り、バッチサイズごとにデータのインデックスを区切ったイテレータを返す。
    なお、各バッチで使用するインデックスは重複なく、網羅的に抽出する
    """
    try:
        if data_len % batch_size != 0:
            raise Exception("データサイズがバッチサイズで割り切れません")
    except:
        traceback.print_exc()


    iter_num = int(data_len / batch_size)
    mini_batch_indexies = np.full((iter_num, batch_size), None) # 最終的に返す値
    available_index = list(range(0, data_len)) # 使用可能なインデックスから各バッチで使用するインデックスを抽出する  

    for i in range(0, iter_num):
        mini_batch_index = np.array(random.sample(available_index, batch_size))
        available_index = list(filter(lambda x: x not in mini_batch_index, available_index)) # 使用可能なインデックスを更新
        mini_batch_indexies[i] = mini_batch_index # バッチインデックスを追加

    return mini_batch_indexies

def ReLU(u):
    r, c = u.shape
    z = np.zeros((r, c))

    for r_n in range(r):
        for c_n in range(c):
            element = u[r_n, c_n]
            if element > 0:
                z[r_n, c_n] = element
    return z


def LeakeyReLU(u, a=0.001):
    r, c = u.shape
    z = np.zeros((r, c))

    for r_n in range(r):
        for c_n in range(c):
            element = u[r_n, c_n]
            z[r_n, c_n] = a * min(0, element) + max(0, element)
    return z


class NeuralNetwork():
    """
    scikit-learn風にfit, predictで実装する
    3層のニューラルネットワーク：中間層は(3, 2), 入出力層(1)
    まずはミニバッチなしで実装する
    """

    def initialize_parameters(self):
        np.random.seed(42)

        self.params = {
            "w1": np.random.randn(self.input_dim, self.hidden1),
            "b1": np.zeros((1, self.hidden1)),
            "w2": np.random.randn(self.hidden1, self.hidden2),
            "b2": np.zeros((1, self.hidden2)),
            "w3": np.random.randn(self.hidden2, self.output_dim),
            "b3": np.zeros((1, self.output_dim)),
        }

        self.grads_history = {
                "w1": [],
                "b1": [],
                "w2": [],
                "b2": [],
                "w3": [],
                "b3": [],
        }

    def __init__(self, input_dim, hidden1, hidden2, output_dim, epochs=1000, η=0.001):
        self.η = η
        self.min_loss = float("inf")
        self.loss_history = []
        self.input_dim = input_dim
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.output_dim = output_dim
        self.epochs = epochs

        self.initialize_parameters()

    def loss_fn(self, mini_batch_X, mini_batch_y):
        w1, b1 = self.params["w1"], self.params["b1"]
        w2, b2 = self.params["w2"], self.params["b2"]
        w3, b3 = self.params["w3"], self.params["b3"]

        u1 = mini_batch_X @ w1 + b1
        z1 = LeakeyReLU(u1)
        u2 = z1 @ w2 + b2
        z2 = LeakeyReLU(u2)
        u3 = z2 @ w3 + b3
        z3 = u3 # 恒等関数
        mini_batch_y_hat = z3
        mse = np.mean((mini_batch_y - mini_batch_y_hat.flatten())**2)
        return mse
        
    def get_grad(self, mini_batch_X, mini_batch_y, param_name, h=0.001):
        """こここでは、勾配を計算。重み(w, b)は二次元のnumpy配列であるものとする"""
        grad = np.zeros_like(self.params[param_name]) # 勾配をゼロで初期化
        row_num, column_num = self.params[param_name].shape # wの形状を取得

        for rn in range(row_num): # range(0, 行数)
            for cn in range(column_num): # range(0, 列数)

                original_element = self.params[param_name][rn, cn].copy()
                self.params[param_name][rn, cn] = original_element + h
                foward_loss = self.loss_fn(mini_batch_X, mini_batch_y)
                self.params[param_name][rn, cn] = original_element - h
                back_loss = self.loss_fn(mini_batch_X, mini_batch_y)

                # 中心差分近似で微分係数を計算
                grad[rn, cn]  = (foward_loss - back_loss) / (2 * h)
                self.params[param_name][rn, cn] = original_element
            
        return grad

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.data_len = self.X_train.shape[0]

            # エポックを回す
        for epoch in range(1, self.epochs + 1):

            self.update()
            loss = self.loss_fn(X_train, y_train)
            self.loss_history = loss

            # 全ての重みの勾配を求め、一次元の配列に結合した後にL2ノルムを計算
            # grad_all = []
            # for key in self.params.keys():
            #     key_grad = self.get_grad(X_train, y_train, key)

            #     grad_all.append(key_grad.flatten())
            
            # grad_vec = np.concatenate(grad_all)
            # grad_norm2 = np.linalg.norm(grad_vec, ord=2)
            
            if epoch % 50 == 0:
                # print(f"Epoch {epoch}, Loss {loss}, grad_norm2 {grad_norm2}")
                print(f"Epoch {epoch}, Loss {loss}")

            # 勾配のL2ノルムが0.0001未満の時、学習を終了する
            # if grad_norm2 < 0.0001:
            #     break

    def update(self):

        # 各重みの入れれーたを回す
        for key in self.params.keys():
            batch_size = 10
            mini_batch_indexes = get_mini_batch_indexes(self.y_train.shape[0], batch_size=10) # ミニバッチインデックスを生成
            key_total_grad = 0

            # ミニバッチデータを使って重みを更新
            for batch_index in mini_batch_indexes:

                mini_batch_X = self.X_train[[list(batch_index)], 0].reshape(-1, 1) # 説明変数のミニバッチデータを取得
                mini_batch_y = self.y_train[[list(batch_index)]] # 目的変数のミニバッチデータを取得
                grad = self.get_grad(mini_batch_X, mini_batch_y, key) # キーで指定した勾配をミニバッチデータで計算
                key_total_grad += grad # あるキーのミニバッチ勾配を加算
            
            key_avg_grad = key_total_grad / batch_size # ミニバッチの勾配をバッチサイズで平均化

            self.params[key] -= self.η * key_avg_grad # 重みを更新

            key_grad_norm2 = np.linalg.norm(key_total_grad, ord=2)
            self.grads_history[key].append(key_grad_norm2)

    def predict(self, X_test):
        w1, b1 = self.params["w1"], self.params["b1"]
        w2, b2 = self.params["w2"], self.params["b2"]
        w3, b3 = self.params["w3"], self.params["b3"]

        u1 = X_test @ w1 + b1
        z1 = LeakeyReLU(u1)
        u2 = z1 @ w2 + b2
        z2 = LeakeyReLU(u2)
        u3 = z2 @ w3 + b3
        z3 = u3 # 恒等関数
        y_hat = z3

        return y_hat

def main():
    df = pd.read_csv("exe11.csv")
    X_train = df["x"].to_numpy().reshape(-1, 1)
    y_train = df["y"].to_numpy()

    nn = NeuralNetwork(input_dim=1, hidden1=3, hidden2=2, output_dim=1, epochs=1000, η=0.009)
    nn.fit(X_train, y_train)

    y_hat = nn.predict(X_train)

    # plt.scatter(X_train, y_train)
    # plt.plot(X_train, y_hat)

    # plt.show()

    for key, history in nn.grads_history.items():
        plt.plot(history, label=key)

    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm (L2)")
    plt.title("勾配ノルムの推移（各パラメータ）")
    plt.legend()
    plt.grid()
    plt.savefig(f"grad_history.png")

if __name__ == "__main__":
    main()