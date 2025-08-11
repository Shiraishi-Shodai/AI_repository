import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import japanize_matplotlib
import torch
import seaborn as sns

class NeuralNetwork():

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):

        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.loss_history = []

        self.params = {
            "w1": np.random.normal(loc=0, scale=np.sqrt(2 / self.input_size), size=(self.input_size, self.hidden_size1)),
            "b1": np.zeros((1, self.hidden_size1)),
            "w2": np.random.normal(loc=0, scale=np.sqrt(2 / self.hidden_size1), size=(self.hidden_size1, self.hidden_size2)),
            "b2": np.zeros((1, self.hidden_size2)),
            "w3": np.random.normal(loc=0, scale=np.sqrt(2 / self.hidden_size2), size=(self.hidden_size2, self.output_size)),
            "b3": np.zeros((1, self.output_size))
        }

    def ReLU(self, u):
        """ここで受け取るuは行列である。

        u: u > 0
        0: u <= 0
        """
        return np.where(u > 0, u, 0)
    
    def dReLU(self, u, dz):
        return dz * np.where(u > 0, 1, 0)
    
    def loss_fn(self, y_hat, y):
        return np.mean(1/2 * ((y_hat - y)**2))
    
    def foward(self, X, training=True):
        u1 = X @ self.params["w1"] + self.params["b1"]
        z1 = self.ReLU(u1)

        u2 = z1 @ self.params["w2"] + self.params["b2"]
        z2 = self.ReLU(u2)

        u3 = z2 @ self.params["w3"] + self.params["b3"]
        z3 = u3

        y_hat = z3

        if training:
            self.u1 = u1
            self.z1 = z1
            self.u2 = u2
            self.z2 = z2
            self.u3 = u3
            self.z3 = z3

        return y_hat

    def backward(self):
        dz3 = (1 / self.y_train.shape[0]) * (self.z3 - self.y_train)
        du3 = dz3 * 1
        dw3 = self.z2.T @ du3
        db3 = np.sum(du3, axis=0)

        dz2 = du3 @ self.params["w3"].T
        du2 = self.dReLU(self.u2, dz2)
        dw2 = self.u1.T @ du2
        db2 = np.sum(du2, axis=0)

        dz1 = du2 @ self.params["w2"].T
        du1 = self.dReLU(self.u1, dz1)
        dw1 = self.X_train.T @ du1
        db1 = np.sum(du1, axis=0)

        grads = {
            "dw1": dw1,
            "db1": db1,
            "dw2": dw2,
            "db2": db2,
            "dw3": dw3,
            "db3": db3,
        }

        return grads
    
    def training(self, X_train, y_train, epochs, lr):
        self.X_train = X_train
        self.y_train = y_train
        self.epochs = epochs
        self.lr = lr

        for epoch in np.arange(1, epochs + 1):
            y_hat = self.foward(self.X_train)
            loss = self.loss_fn(y_hat, self.y_train)
            grads = self.backward()
            
            # 重みを更新
            for key, grad in zip(self.params.keys(), grads.values()):
                self.params[key] = self.params[key] - self.lr * grad

            self.loss_history.append(loss)

            if epoch % 50 == 0:
                print(f"Epoch {epoch} loss {loss}")
            
            if loss  <= 0.001:
                break

    
    def test(self, X_test):
        y_hat = self.foward(X_test, training=False)
        return y_hat

def main():
    df = pd.read_csv("../../csv/california_housing.csv")
    X = df.iloc[:, :-1].to_numpy()
    y = df["Price"].to_numpy().reshape(-1, 1)

    print(df.describe(), df.isnull().sum(), df.shape)

    # sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    # plt.title("カリフォルアの家賃データ")
    # plt.savefig("heat_map.png")

    nn = NeuralNetwork(input_size=X.shape[1], hidden_size1=10, hidden_size2=3, output_size=1)

    nn.training(X_train=X, y_train=y, epochs=1000, lr=0.0009)
    y_pred = nn.test(X_test=X)
    print(f"平均二乗誤差：{nn.loss_fn(y_pred, y)}")

    # 描画
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(np.arange(0, len(nn.loss_history)), nn.loss_history)
    ax1.set_title("損失関数の遷移")
    ax1.set_xlabel("損失回数計算回数")
    ax1.set_ylabel("損失関数")

    ax2.scatter(y[:100], y_pred[:100])
    ax2.grid()    
    # ax2.legend()
    ax2.set_title("予測")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    plt.savefig("california_housing_train.png")
    # plt.savefig("california_housing_test.png")

if __name__ == "__main__":
    main()