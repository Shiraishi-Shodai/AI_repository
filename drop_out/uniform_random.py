import numpy as np
import matplotlib.pyplot as plt

# 一様乱数を10000個生成
samples = np.random.rand(10000)

# ヒストグラムで可視化
plt.hist(samples, bins=50, edgecolor='black')
plt.title("Uniform Random Numbers (0 to 1)")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
