# Designer:Yudong Pan
# Coder:God's hand
# Time:2023/3/15 11:15
import numpy as np

itcca_mean = [42.44, 48.38, 54.14, 59.44, 63.38, 67.01]
itcca_aug_mean = [73.24, 74.65, 77.39, 79.92, 81.54, 84.61]
trca_mean = [64.99, 68.53, 73.08, 75.79, 79.72, 82.93]
trca_aug_mean = [71.53, 73.25, 76.35, 78.42, 79.00, 82.44]
eegnet_mean = [35.44, 39.34, 45.87, 48.02, 53.72, 55.63]
eegnet_aug_mean = [72.48, 73.95, 76.82, 79.02, 80.77, 83.43]
c_cnn_mean = [55.46, 57.75, 61.14, 64.36, 66.44, 70.42]
c_cnn_aug_mean = [73.21, 74.80, 77.40, 79.78, 81.16, 84.15]

itcca_improve = [round(acc2 - acc1, 2) for acc1, acc2 in zip(itcca_mean, itcca_aug_mean)]
trca_improve = [round(acc2 - acc1, 2) for acc1, acc2 in zip(trca_mean, trca_aug_mean)]
eegnet_improve = [round(acc2 - acc1, 2) for acc1, acc2 in zip(eegnet_mean, eegnet_aug_mean)]
c_cnn_improve = [round(acc2 - acc1, 2) for acc1, acc2 in zip(c_cnn_mean, c_cnn_aug_mean)]

print("itcca_improve:", itcca_improve)
print("trca_improve:", trca_improve)
print("eegnet_improve:", eegnet_improve)
print("c_cnn_improve:", c_cnn_improve)

