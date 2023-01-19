# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/1/26 18:32
import numpy as np
from sklearn import preprocessing

def MaxAbs(signal_data):
    signal_data = preprocessing.MaxAbsScaler().fit_transform(signal_data.reshape(-1, 1))
    signal_data = signal_data.squeeze(-1)
    # maxAbs = np.max(np.abs(signal_data), axis=0)
    # signal_data = signal_data / maxAbs
    return signal_data

def MaxMin(signal_data):
    min = np.min(signal_data)
    max = np.max(signal_data)
    signal_data = (signal_data - min) / (max - min)
    return signal_data

def Z_Score(signal_data):
    expectation = np.mean(signal_data, axis=0)
    variance = np.std(signal_data, axis=0)
    signal_data = (signal_data - expectation) / variance
    return signal_data

def DeNorm_MaxAbs(src_data, tar_data):
    maxAbs = np.max(np.abs(tar_data), axis=0)
    src_data = src_data * maxAbs
    return src_data

def DeNorm_MaxMin(src_data, tar_data):
    min = np.argmin(tar_data, axis=0)
    max = np.argmax(tar_data, axis=0)
    src_data = src_data * (max - min) + min
    return src_data

def DeNorm_Z_score(src_data, tar_data):
    expectation = np.mean(tar_data, axis=0)
    variance = np.std(tar_data, axis=0)
    src_data = src_data * variance + expectation
    return src_data

'''
# a simple test example
x = np.array([1, 2, 3, 4, -1, -2, -3, -4])
# norm_x = MaxAbs(x)
# norm_x = MaxMin(x)
norm_x = Z_Score(x)
# denorm_x = DeNorm_MaxAbs(norm_x, x)
# denorm_x = DeNorm_MaxMin(norm_x, x)
denorm_x = DeNorm_Z_score(norm_x, x)
print("x:", x)
print("norm_x:", norm_x)
print("denorm_x:", denorm_x)
'''