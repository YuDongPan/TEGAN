# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2021/11/24 0:19
import numpy as np
import torch
a = torch.arange(24).reshape(24, 1)
print(a)
b = a.repeat(1, 2).reshape(-1, 1)
print("b:", b, b.shape)

x = np.array([[[1, 2], [3, 4]]])
print("x:", x)
y = np.repeat(x, 2, axis=0)
print("y:", y)


z = torch.arange(16).reshape(4, 1, 2, 2)
print("z:", z)

q = z.repeat(1, 2, 1, 1)
print("q:", q)