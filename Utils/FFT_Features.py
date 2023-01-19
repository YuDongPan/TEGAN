# 设计师:Pan YuDong
# 编写者:God's hand
# 时间:2022/9/9 16:21
import torch
import numpy as np

def complex_features(segmented_data):
    segmented_data = segmented_data.numpy()
    Nt = segmented_data.shape[-1]
    fft_result = np.fft.fft(segmented_data, axis=-1) / (Nt / 2)
    real_part = np.real(fft_result[:, :, :, :Nt // 2])
    imag_part = np.imag(fft_result[:, :, :, :Nt // 2])
    features_data = np.concatenate([real_part, imag_part], axis=-1)
    features_data = torch.from_numpy(features_data)
    return features_data

def magnitude_features(segmented_data):
    segmented_data = segmented_data.numpy()
    Nt = segmented_data.shape[-1]
    fft_result = np.fft.fft(segmented_data, axis=-1) / (Nt / 2)
    fft_result = fft_result[:, :, :, :Nt // 2]
    features_data = np.abs(fft_result)
    features_data = torch.from_numpy(features_data)
    return features_data

