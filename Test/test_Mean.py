# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/12/15 22:51
import numpy as np

x = np.arange(6).reshape(2, 3)
y = np.mean(x)
z = y.repeat(2*3).reshape(2, 3)


print("x:", x)
print("y:", y)
print("z:", z)
