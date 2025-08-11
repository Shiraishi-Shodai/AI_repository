import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import japanize_matplotlib
from matplotlib.animation import FuncAnimation 
print(matplotlib.get_backend())
print(matplotlib.matplotlib_fname())  # 設定ファイルのパスを表示
# matplotlib.use("TkAgg")

# 関数 f(x) = x^2 - 4x + 1
def f(x: np.float16) -> np.float16:
    return x**2 - 4*x + 1

# 導関数 f'(x) = 2x - 4
def df(x: np.float16) -> np.float16:
    return 2*x - 4

# 接線を表示するx軸のnumpyとy軸のnumpyを返す
def generate_linear(sample_a : np.float16) -> tuple[np.ndarray, np.ndarray]:
    # sample_x = sample_a
    target_y = f(sample_a)
    target_slope = df(sample_a)
    linear_length = 10

    linear_scope = np.linspace(sample_a - 0.5, sample_a + 0.5, linear_length) # (10,)
    target_linear = target_slope * (linear_scope - sample_a) + target_y # (10,)

    return linear_scope, target_linear, target_slope

# animationを初期化する関数
def init():
    print("plot start !!")

def update(frame, ax, x, y):
    # 既存のプロットを削除（remove() を使用）
    for line in ax.lines[:]:  # コピーを作成してループ中に削除
        line.remove()
    for collection in ax.collections[:]:  # scatter で作成された点を削除
        collection.remove()

    sample_a = frame
    linear_scope, target_linear, target_slope = generate_linear(sample_a)

    # 曲線を再描画
    ax.plot(x, y, label="関数 $f(x) = x^2 - 4x + 1$", c="red")
    # 接線を描画
    ax.plot(linear_scope, target_linear, linestyle="dashed", label=f"接線 (x={sample_a}) \n 傾き：{target_slope}", c="blue")
    # 接点をプロット
    ax.scatter(sample_a, f(sample_a), color="black", zorder=3) 

    ax.legend()

    return ax.lines + ax.collections  # blit=True に対応するための返り値

def main():
    x = np.linspace(0, 5, 100)
    y = x**2 - 4*x + 1

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.grid()
    ax.set_title("曲線上のある点における接線")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ani = FuncAnimation(fig=fig, func=update, init_func=init, frames=x, repeat=False, blit=False, fargs=(ax, x, y))

    #保存
    ani.save("sample.gif", writer="pillow")

if __name__ == "__main__":
    main()