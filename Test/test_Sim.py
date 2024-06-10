# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/5/27 15:49
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
x = np.array([3, 5, 4, 1])
y = np.array([3, 3, 5, 2])
sim1 = cosine_similarity([x, y])
sim2 = pearsonr(x, y)[0]
print("similarity1:", sim1)
print("similarity2:", sim2)


