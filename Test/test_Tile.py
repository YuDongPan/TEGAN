# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/12/1 13:38
import numpy as np

num_targs = 12

label = np.tile(np.arange(num_targs).reshape(-1, 1), (1, 15))

print("label:", label, label.shape)