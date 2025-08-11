import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import japanize_matplotlib
import traceback
import random

random.seed(0)
df = pd.read_csv("exe11.csv")
X = df["x"].to_numpy().reshape(1, -1)
y = df["y"].to_numpy()

# print(df.head())

# plt.scatter(df["x"], df["y"])
# plt.grid()
# plt.title("元データ")
# plt.xlabel("x")
# plt.ylabel("y")

def get_grad(mini_batch_X, mini_batch_y, vec, loss_fn, h=0.001):
    grad = np.zeros_like(vec)
    for i in range(vec.shape[0]): # (1, 3) range(0, 3)
        original_element = vec[i]
        vec[i] = original_element + h
        foward_loss = loss_fn(mini_batch_X, mini_batch_y, vec)
        vec[i] = original_element - h
        back_loss = loss_fn(mini_batch_X, mini_batch_y, vec)

        grad[i]  = (foward_loss - back_loss) / (2 * h)
        vec[i] = original_element
        
    return grad

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

def loss_fn(mini_batch_X, mini_batch_y, vec):
    mini_batch_y_hat = vec[0]  + vec[1]* np.sin(3.1416*mini_batch_X/5) + vec[2]* mini_batch_X
    mse = np.mean((mini_batch_y - mini_batch_y_hat.flatten())**2)
    return mse

epochs = 1000
best_η = None
best_w = np.ones(3)
η_scope = np.linspace(0.001, 0.01, 10)
min_loss = float("inf")
loss_history = []

# 最適な学習率を探す
for η in η_scope:

    w = best_w.copy()

    # エポックを回す
    for epoch in range(1, epochs + 1):
        batch_size = 10
        mini_batch_indexes = get_mini_batch_indexes(y.shape[0], batch_size=10) # ミニバッチインデックスを生成
        total_grad = 0

        # ミニバッチごとに重みを更新
        for batch_index in mini_batch_indexes:
            mini_batch_X = X[0, [list(batch_index)]]
            mini_batch_y = y[[list(batch_index)]]
            grad = get_grad(mini_batch_X, mini_batch_y, w, loss_fn)
            total_grad += grad

            # 全てのデータを使うと計算量が膨大になるので、ミンバッチデータのみで損失価数を計算
            loss = loss_fn(mini_batch_X, mini_batch_y, w) 

            if epoch % 100 == 0:
                # ここで表示している損失関数はミニバッチデータのみの損失関数
                print(f"epoch: {epoch}, loss : {loss}, grad: {grad.flatten()}, w : {w}")

        avg_grad = total_grad / batch_size # バッチごとに求めた勾配の合計をバッチサイズで平均化する
        w -= η * avg_grad

        if np.sqrt(np.sum([g**2 for g in grad])) < 0.001:
            break

        # 1エポック終了後に全体の損失を評価
        full_loss = loss_fn(X, y, w)
        loss_history.append(full_loss)

        if full_loss < min_loss:
            min_loss = full_loss
            best_w = w.copy()
            best_η = η



print(f"best_w {best_w}, best_η {best_η}, min_loss {min_loss}")
y_hat = best_w[0]  + best_w[1]* np.sin(3.1416*X/5) + best_w[2]* X


fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
# print(loss_history)
ax1.plot(np.arange(0, len(loss_history)), loss_history)
ax1.set_title("損失関数の遷移")
ax1.set_xlabel("損失回数計算回数")
ax1.set_ylabel("損失関数")

ax2.scatter(X.flatten(), y, label="元データ")
ax2.plot(X.flatten(), y_hat.flatten(), label="予測")
ax2.grid()    
ax2.legend()
ax2.set_title("予測")
ax2.set_xlabel("x")
ax2.set_ylabel("y")

plt.savefig("result.png")