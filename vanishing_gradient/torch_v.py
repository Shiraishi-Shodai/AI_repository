import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import japanize_matplotlib

# 再現性のためのシード固定
torch.manual_seed(0)
np.random.seed(0)

class NeuralNetwork(nn.Module):
    """
    PyTorchのnn.Moduleを継承してニューラルネットワークを定義します。
    
    可変層のニューラルネットワーク
    入力層：1
    中間層：hidden_sizesで指定された数の層とニューロン数
    出力層：1
    活性化関数：ReLU(出力層は恒等関数)
    """

    def __init__(self, input_dim=1, hidden_sizes=[2, 3], out_dim=1):
        # 親クラス(nn.Module)のコンストラクタを呼び出します。
        super(NeuralNetwork, self).__init__()

        # ネットワークの各層を定義します。
        # nn.Linearは全結合層（線形変換）を表します。
        # (入力次元, 出力次元)を指定します。
        
        # 隠れ層を格納するためのリスト
        layers = []
        
        # 最初の隠れ層の入力次元はinput_dim
        current_input_dim = input_dim
        
        # hidden_sizesリストに基づいて隠れ層を動的に生成
        for hidden_size in hidden_sizes:
            # 線形層を追加
            layers.append(nn.Linear(current_input_dim, hidden_size))
            # ReLU活性化関数を追加
            layers.append(nn.ReLU())
            # 次の層の入力次元を更新
            current_input_dim = hidden_size
            
        # nn.ModuleListに隠れ層をまとめて登録
        # これにより、モデルがこれらの層を認識し、パラメータを適切に管理します。
        self.hidden_layers = nn.ModuleList(layers)
        
        # 出力層を定義
        # 出力層の入力次元は最後の隠れ層の出力次元
        self.output_layer = nn.Linear(current_input_dim, out_dim)

    def forward(self, x):
        """
        順伝播の計算を定義します。
        入力xがネットワークをどのように流れるかを記述します。
        """
        # 隠れ層を順番に適用
        for layer in self.hidden_layers:
            x = layer(x)
        
        # 出力層を適用
        y_pred = self.output_layer(x)
        
        return y_pred

def main():
    # データの読み込みと前処理
    df = pd.read_csv("exe11.csv")
    # NumPy配列をPyTorchのテンソルに変換します。
    # .reshape(-1, 1)で列ベクトルに変換し、.float()でデータ型を浮動小数点数に変換します。
    X = torch.from_numpy(df["x"].to_numpy().reshape(-1, 1)).float()
    y = torch.from_numpy(df["y"].to_numpy().reshape(-1, 1)).float()

    # ハイパーパラメータ
    input_dim = X.shape[1]
    epochs = 3000
    lr = 0.0012

    # ここでhidden_sizesリストを変更することで、隠れ層の数と各層のニューロン数を調整できます。
    # 例: hidden_sizes=[10, 10] は、ニューロン数10の隠れ層が2つあるネットワークを意味します。
    # 例: hidden_sizes=[5, 10, 5] は、ニューロン数5, 10, 5の隠れ層が3つあるネットワークを意味します。
    hidden_sizes = [100, 140, 100] # 元の構成
    # hidden_sizes = [10, 10] # 例: 隠れ層2つ、各10ニューロン
    # hidden_sizes = [5, 10, 5] # 例: 隠れ層3つ、各5, 10, 5ニューロン

    # モデルのインスタンス化
    model = NeuralNetwork(input_dim=input_dim, hidden_sizes=hidden_sizes, out_dim=1)
    
    # 損失関数の定義
    # 平均二乗誤差(Mean Squared Error)を使用します。
    criterion = nn.MSELoss()
    
    # 最適化手法の定義
    # 確率的勾配降下法(SGD)を使用します。
    # model.parameters()でモデルの全パラメータ（重みとバイアス）をオプティマイザに渡します。
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # 訓練ループ
    loss_history = []
    for epoch in range(1, epochs + 1):
        # 1. 順伝播: モデルで予測値を計算
        y_pred = model(X)
        
        # 2. 損失の計算
        # criterion(予測値, 正解値)の形で損失を計算します。
        loss = criterion(y_pred, y)
        
        # 3. 勾配のリセット
        # 前のループで計算された勾配が残らないように、毎回リセットします。
        optimizer.zero_grad()
        
        # 4. 誤差逆伝播: 損失に対する各パラメータの勾配を計算
        loss.backward()
        
        # 5. パラメータの更新: 計算された勾配を元にパラメータを更新
        optimizer.step()
        
        # 損失の履歴を保存
        # .item()でテンソルからPythonのスカラー値を取得します。
        loss_history.append(loss.item())
        
        # 50エポックごとに損失を表示
        if epoch % 50 == 0:
            print(f"Epoch {epoch} loss {loss.item()}")
        
        # 損失が一定以下になったら学習を終了
        if loss.item() <= 0.001:
            break

    # 予測
    # torch.no_grad()ブロック内で実行することで、勾配計算を無効にし、メモリ消費を抑え、計算を高速化します。
    with torch.no_grad():
        y_pred_final = model(X)

    # 描画
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.plot(np.arange(len(loss_history)), loss_history)
    ax1.set_title("損失関数の遷移")
    ax1.set_xlabel("エポック数")
    ax1.set_ylabel("損失関数 (MSE)")

    # テンソルをNumPy配列に変換してプロット
    ax2.scatter(X.numpy(), y.numpy(), label="元データ")
    ax2.plot(X.numpy(), y_pred_final.numpy(), label="予測", color="red")
    ax2.grid()    
    ax2.legend()
    ax2.set_title("予測")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    plt.savefig("torch_result.png")
    print("PyTorchバージョンの結果を torch_result.png に保存しました。")

if __name__ == "__main__":
    main()