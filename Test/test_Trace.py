# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/5/25 15:34
import numpy as np
A = np.arange(1, 10).reshape(3, 3)
B = np.arange(1, 10)[-1::-1].reshape(3, 3)
print("A:", A)
print("B:", B)
print("A × B = ", np.matmul(A, B))
print("B × A = ", np.matmul(B, A))