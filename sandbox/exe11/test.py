import numpy as np

# a = np.arange(1, 12)
# b = np.arange(1000, 12000, 1000)

# asx = np.mean(a**2) - np.mean(a)**2
# bsx = np.mean(b**2) - np.mean(b)**2

# astd = np.sqrt(asx)
# bstd = np.sqrt(bsx)
# print(astd, bstd)

# an = (a - np.mean(a)) / astd
# bn = (b - np.mean(b)) / bstd

# print(an, bn)

import numpy as np

arr = np.array([[1, 2, 3],
                [4, 5, 6]])

total = np.sum(arr)
print(total)  # 出力: 21
print(np.linalg.norm(arr, ord=2))
